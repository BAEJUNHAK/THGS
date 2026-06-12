"""
P1-B — Absence query experiment (pre-registered R13, see
md/hypotheses/strategy/p1_problem_experiments.md — read-only registration).

Problem being concretized: every paper inherits LERF's canon-contrast score
({"object","things","stuff","texture"}); LERF-OVS is positives-only, so the
score's behavior on ABSENT objects has never been measured.

Vocabulary (~30/scene): LVIS v1 categories (detectron2 dump), minus any
category sharing a token stem with the scene's GT prompts, preferring a
curated clearly-absent pool + 3 semantically-near absent categories/scene.
Provenance recorded per entry (p1b_absence_vocab.csv).

Rules scored on both methods (THGS / ReLaGS dumps):
  mean   : canon over sai_nag features                (pipeline baseline)
  top5   : query-top-5 portion-weighted agg           (bag/view-selection)
  gm_w   : weighted geometric median                  (robust-stat family)
  hybrid : stage4 v2 rule, config a0.5 k5 tv5 c0.1    (ours)

Metrics per (method, rule):
  confident-hit rate = share of absent queries whose Z-filtered top-1 score
                       >= 25th percentile of PRESENT-EASY top-1 scores
  signal AUROC (present vs absent): mean margin (top1-top2), mean top-1
                       confidence, regime-agreement (mean top1 == top5 top1)

Outputs:
  output/diagnostics/p1b_absence_vocab.csv
  output/diagnostics/p1b_absence_scores.csv   (query level, both methods)
  output/diagnostics/p1b_top1_selections.pkl  (for montage rendering)
"""

import os
import re
import sys
import ast
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, query_topk_agg, LEVELS
from stage3_2_common import make_vlm
from p1a_competitor_autopsy import geometric_median, METHODS, SCENES

HYBRID = dict(alpha=0.5, k=5, tau_v=5, tau_c=0.10)

ABSENT_SAFE = [
    'zebra', 'giraffe', 'gazelle', 'rhinoceros', 'hippopotamus', 'camel',
    'gorilla', 'ostrich', 'penguin', 'dolphin', 'shark', 'alligator',
    'canoe', 'kayak', 'jet_plane', 'helicopter', 'ambulance', 'fire_engine',
    'school_bus', 'motorcycle', 'tractor', 'parking_meter', 'traffic_light',
    'fire_hydrant', 'surfboard', 'snowboard', 'ski', 'skateboard',
    'tennis_racket', 'volleyball', 'basketball', 'football_helmet',
    'parachute', 'tent', 'lawn_mower', 'chainsaw', 'anvil', 'harmonica',
    'trombone', 'saxophone', 'accordion', 'typewriter', 'mailbox',
    'birdhouse', 'scarecrow', 'snowman', 'canteen', 'stirrup', 'wheelchair',
    'stop_sign',
]
NEAR = {
    'figurines': ['fork', 'banana', 'wine_bottle'],
    'ramen': ['fork', 'pizza', 'sushi'],
    'teatime': ['fork', 'wine_bottle', 'laptop_computer'],
    'waldo_kitchen': ['pizza', 'hamburger', 'laptop_computer'],
}
N_PER_SCENE = 30
STOP = {'a', 'an', 'the', 'of', 'with'}


def load_lvis_names(path='/tmp/lvis_v1_categories.py'):
    src = open(path).read()
    m = re.search(r'^LVIS_CATEGORIES\s*=\s*', src, re.M)
    start = src.index('[', m.end() - 1)
    lst = ast.literal_eval(src[start:src.rindex('}') + 2])
    names = []
    for c in lst:
        n = (c.get('synonyms') or [c['name']])[0]
        n = re.sub(r'\(.*?\)', '', n).replace('_', ' ').strip()
        if n:
            names.append(n)
    return sorted(set(names))


def stems(text):
    out = set()
    for t in re.split(r'[^a-z]+', text.lower()):
        if t and t not in STOP:
            out.add(t[:-2] if t.endswith('es') else t[:-1] if t.endswith('s') else t)
    return out


def build_vocab(prompts_by_scene, lvis_names):
    lvis_set = {n.replace(' ', '_'): n for n in lvis_names}
    rows = []
    for scene, prompts in prompts_by_scene.items():
        banned = set()
        for p in prompts:
            banned |= stems(p)
        chosen, seen = [], set()

        def ok(name):
            return not (stems(name) & banned) and name not in seen

        for key in NEAR[scene]:
            name = lvis_set.get(key, key.replace('_', ' '))
            if name in lvis_names and ok(name):
                chosen.append((name, 'near_curated'))
                seen.add(name)
        for key in ABSENT_SAFE:
            if len(chosen) >= N_PER_SCENE:
                break
            name = lvis_set.get(key)
            if name and ok(name):
                chosen.append((name, 'absent_safe'))
                seen.add(name)
        for name in lvis_names:                      # deterministic fill
            if len(chosen) >= N_PER_SCENE:
                break
            if ok(name) and len(name.split()) <= 2:
                chosen.append((name, 'lvis_fill'))
                seen.add(name)
        for name, src in chosen:
            rows.append({'scene': scene, 'query': name, 'source': src,
                         'in_lvis': True})
    return pd.DataFrame(rows)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['thgs', 'relags'])
    args = ap.parse_args()

    lvis_names = load_lvis_names()
    print(f"LVIS names: {len(lvis_names)}")

    b7 = pd.read_csv('output/diagnostics/b7_a4_combined.csv')
    ref = b7[b7['is_ref_frame'] == 1]
    prompts_by_scene = {s: sorted(ref[ref.scene == s]['prompt']) for s in SCENES}
    vocab = build_vocab(prompts_by_scene, lvis_names)
    vocab.to_csv('output/diagnostics/p1b_absence_vocab.csv', index=False)
    print(vocab.groupby(['scene', 'source']).size())

    vlm = make_vlm()
    all_rows, top1_sel = [], {}
    for method in args.methods:
        cfg = METHODS[method]
        mref = pd.read_csv(cfg['ref_csv'])
        mref = mref[mref['is_ref_frame'] == 1]
        for scene in SCENES:
            dump = load_dump(cfg['dump_tpl'].format(scene))
            lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}
            entries = [(l, i) for l in LEVELS for i in range(lv[l][0].shape[1])]
            nval = np.concatenate([lv[l][4] for l in LEVELS]).astype(int)
            has_view = nval > 0
            nag = torch.load(cfg['nag_tpl'].format(scene))
            base_feat = torch.cat([
                F.normalize(nag['nag_feat'][l - 1].cuda().float(), p=2, dim=-1)
                for l in LEVELS])
            gm_w = torch.cat([geometric_median(*lv[l][:3], weighted=True)
                              for l in LEVELS])

            sc_ref = mref[mref.scene == scene]
            easy_set = set(sc_ref[sc_ref.oracle_rank <= 3]['prompt'])
            present = list(sc_ref['prompt'])
            absent = list(vocab[vocab.scene == scene]['query'])
            queries = [(q, 0) for q in present] + [(q, 1) for q in absent]
            print(f"=== [{method}] {scene}: {len(present)} present + "
                  f"{len(absent)} absent ===", flush=True)

            for q, is_absent in queries:
                vlm.encode_text(q)
                text = vlm.text_feature[0].float()
                s_mean = vlm.compute_similarity(base_feat).cpu().numpy()
                top5 = torch.cat([query_topk_agg(*lv[l][:3], text, k=5)
                                  for l in LEVELS])
                s_top5 = vlm.compute_similarity(top5).cpu().numpy()
                s_gm = vlm.compute_similarity(gm_w).cpu().numpy()

                sm_f = np.where(has_view, s_mean, -1e9)
                srt = np.sort(sm_f)[::-1]
                mean_margin = float(srt[0] - srt[1])
                if mean_margin >= HYBRID['tau_c']:
                    s_hy = sm_f
                else:
                    top_eff = np.where(nval < HYBRID['tau_v'], s_mean, s_top5)
                    s_hy = np.where(has_view,
                                    HYBRID['alpha'] * s_mean
                                    + (1 - HYBRID['alpha']) * top_eff, -1e9)

                row = {'method': method, 'scene': scene, 'query': q,
                       'is_absent': is_absent,
                       'is_easy_present': int(q in easy_set and not is_absent),
                       'mean_margin': mean_margin,
                       'ghost_top1_plain_mean': int(
                           has_view[int(np.argmax(s_mean))] == 0)}
                t1 = {}
                for rule, s in [('mean', sm_f),
                                ('top5', np.where(has_view, s_top5, -1e9)),
                                ('gm_w', np.where(has_view, s_gm, -1e9)),
                                ('hybrid', s_hy)]:
                    i1 = int(np.argmax(s))
                    row[f'top1_{rule}'] = float(s[i1])
                    t1[rule] = entries[i1]
                row['agree_mean_top5'] = int(t1['mean'] == t1['top5'])
                all_rows.append(row)
                top1_sel[(method, scene, q, is_absent)] = t1
            del lv, base_feat, gm_w
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv('output/diagnostics/p1b_absence_scores.csv', index=False)
    with open('output/diagnostics/p1b_top1_selections.pkl', 'wb') as f:
        pickle.dump(top1_sel, f)

    def auroc(pos, neg):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float)
        if len(pos) == 0 or len(neg) == 0:
            return float('nan')
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        return (gt + 0.5 * eq) / (len(pos) * len(neg))

    print("\n" + "=" * 78)
    print("P1-B SUMMARY — confident-hit rate (vs present-easy q25) / AUROC")
    print("=" * 78)
    for method in args.methods:
        m = df[df.method == method]
        ez = m[(m.is_easy_present == 1)]
        ab = m[m.is_absent == 1]
        pr = m[m.is_absent == 0]
        print(f"\n[{method}] absent n={len(ab)}, present n={len(pr)} "
              f"(easy {len(ez)})")
        for rule in ['mean', 'top5', 'gm_w', 'hybrid']:
            thr = np.percentile(ez[f'top1_{rule}'], 25)
            rate = float((ab[f'top1_{rule}'] >= thr).mean())
            print(f"  {rule:6s}: q25(easy)={thr:.4f}  "
                  f"confident-hit={rate * 100:5.1f}%")
        for sig, col in [('margin(mean)', 'mean_margin'),
                         ('top1-conf(mean)', 'top1_mean'),
                         ('agree(mean,top5)', 'agree_mean_top5')]:
            a = auroc(pr[col].values, ab[col].values)
            print(f"  AUROC {sig:18s}: {a:.3f}")
        gr = float(ab['ghost_top1_plain_mean'].mean())
        print(f"  ghost-as-top1 on absent (plain mean, no Z): {gr*100:.1f}%")
    print("\nWrote p1b_absence_vocab.csv, p1b_absence_scores.csv")


if __name__ == '__main__':
    main()
