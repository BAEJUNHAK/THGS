"""
P1-C — E1 phantom direction bias (catalog-registered decision rule; see
extended_failure_hypotheses.md E1 + section 3, and
md/hypotheses/strategy/p1_problem_experiments.md §2).

Per phantom (THGS thgs_class=='phantom' n=21, ReLaGS relags_class=='phantom'
n=20; pooled primary + per-method):
  direction (c) = normalize(f_aggregated - f_target_text)        [default]
  robustness (a) = f_aggregated itself
  robustness (b) = portion-weighted mean of per-view directions
                   normalize(f_v - f_text)  (post-B8, pre-across-view)

Vocabulary: LVIS 1203 (detectron2 dump, scene-disjoint by construction for
bucket tests; cleaned names) -> CLIP text embeddings -> top-5 nearest
categories per direction.

Patterns (pre-registered):
  P1 small-object -> background bucket   (small = gt_pixels < pooled median)
  P2 food -> vessel bucket               (curated food prompt set)
  P4 transparent -> background bucket    (curated transparent set)
  stat = hit-rate difference between pattern group and complement;
  permutation null = group-label shuffle N=1000; sig: p<0.05 & |effect|>0.3
  (power-adjusted variant: |effect|>0.5; P1+P4 aggregate also tested)
  P3 character -> nearest figure (figurines, instance level):
  nearest-prompt of direction among other figurines prompts; observed mean
  normalized 3D-centroid-distance rank of the chosen target vs random-pairing
  null N=1000 (low rank = spatially nearer than chance).

Outputs:
  output/diagnostics/p1c_e1_directions.csv
  output/diagnostics/p1c_e1_patterns.csv
"""

import os
import re
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, LEVELS
from stage3_2_common import make_vlm
from p1b_absence_query import load_lvis_names
from p1a_competitor_autopsy import METHODS, SCENES

K_NEAR = 5
N_PERM = 1000
BG_TOKENS = {'table', 'desk', 'countertop', 'tablecloth', 'rug', 'carpet',
             'mat', 'curtain', 'shelf', 'drawer', 'cabinet', 'counter',
             'wall', 'floor'}
VESSEL_TOKENS = {'bowl', 'plate', 'cup', 'mug', 'glass', 'saucer', 'pitcher',
                 'pot', 'pan', 'jar', 'bottle', 'container', 'tray', 'dish',
                 'teapot', 'kettle', 'vase'}
FOOD_PROMPTS = {'onion segments', 'egg', 'kamaboko', 'corn', 'wavy noodles',
                'nori'}
TRANSPARENT_PROMPTS = {'glass of water'}


def toks(name):
    return set(re.split(r'[^a-z]+', name.lower())) - {''}


def bucket_flags(names):
    bg = any(toks(n) & BG_TOKENS for n in names)
    vs = any(toks(n) & VESSEL_TOKENS for n in names)
    return int(bg), int(vs)


@torch.no_grad()
def lvis_text_emb(vlm, names, bs=256):
    embs = []
    for i in range(0, len(names), bs):
        t = vlm.tokenizer(names[i:i + bs]).to('cuda')
        e = vlm.clip_pretrained.encode_text(t).float()
        embs.append(F.normalize(e, p=2, dim=-1))
    return torch.cat(embs)


def load_xyz(nag_tpl, scene):
    try:
        from plyfile import PlyData
    except ImportError:
        return None
    root = os.path.dirname(nag_tpl.format(scene))
    cands = sorted(glob.glob(os.path.join(
        nag_tpl.format(scene).replace('sai_nag.pt', ''),
        'point_cloud', 'iteration_*', 'point_cloud.ply')))
    if not cands:
        return None
    ply = PlyData.read(cands[-1])
    v = ply['vertex']
    return np.stack([v['x'], v['y'], v['z']], axis=1)


def perm_test(flags, group, n_perm=N_PERM, rng=None):
    """flags: 0/1 hit per item; group: 0/1 membership. Returns effect, p."""
    rng = rng or np.random.default_rng(0)
    flags, group = np.asarray(flags), np.asarray(group)
    if group.sum() == 0 or group.sum() == len(group):
        return float('nan'), float('nan')
    obs = flags[group == 1].mean() - flags[group == 0].mean()
    cnt = 0
    for _ in range(n_perm):
        g = rng.permutation(group)
        st = flags[g == 1].mean() - flags[g == 0].mean()
        if st >= obs:
            cnt += 1
    return float(obs), (cnt + 1) / (n_perm + 1)


@torch.no_grad()
def main():
    vlm = make_vlm()
    lvis = load_lvis_names()
    lvis_emb = lvis_text_emb(vlm, lvis)
    print(f"LVIS emb: {lvis_emb.shape}")

    cm = pd.read_csv('output/diagnostics/cross_method_d2_decomposition.csv')
    rows = []
    for method in ['thgs', 'relags']:
        cfg = METHODS[method]
        ref = pd.read_csv(cfg['ref_csv'])
        ref = ref[ref['is_ref_frame'] == 1]
        cls_col = 'thgs_class' if method == 'thgs' else 'relags_class'
        pset = set(map(tuple, cm[cm[cls_col] == 'phantom'][['scene', 'prompt']].values))

        for scene in SCENES:
            sc = ref[ref.scene == scene]
            ph = sc[sc.apply(lambda r: (scene, r['prompt']) in pset, axis=1)]
            if len(ph) == 0:
                continue
            nag = torch.load(cfg['nag_tpl'].format(scene))
            feats_lvl = {l: F.normalize(nag['nag_feat'][l - 1].cuda().float(),
                                        p=2, dim=-1) for l in LEVELS}
            dump = load_dump(cfg['dump_tpl'].format(scene))
            lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}

            for _, r in ph.iterrows():
                prompt, lvl, sp = r['prompt'], int(r['oracle_lvl']), int(r['oracle_sp_id'])
                vlm.encode_text(prompt)
                text = vlm.text_feature[0].float()
                f_agg = feats_lvl[lvl][sp]
                # (b) view-mixed mean direction
                fv = lv[lvl][0][:, sp, :].float()             # (V,512)
                por = lv[lvl][1][:, sp]
                val = lv[lvl][2][:, sp]
                w = (por * val)
                if w.sum() > 0:
                    dirs_v = F.normalize(fv - text.unsqueeze(0), p=2, dim=-1)
                    d_b = F.normalize((dirs_v * w.unsqueeze(-1)).sum(0), p=2, dim=0)
                else:
                    d_b = torch.zeros(512, device='cuda')
                variants = {
                    'c': F.normalize(f_agg - text, p=2, dim=0),
                    'a': f_agg,
                    'b': d_b,
                }
                row = {'method': method, 'scene': scene, 'prompt': prompt,
                       'gt_pixels': int(r['gt_pixels'])}
                for vname, d in variants.items():
                    sims = lvis_emb @ d
                    top = sims.topk(K_NEAR).indices.cpu().numpy()
                    names = [lvis[i] for i in top]
                    bg, vs = bucket_flags(names)
                    row[f'top5_{vname}'] = '|'.join(names)
                    row[f'bg_{vname}'] = bg
                    row[f'vessel_{vname}'] = vs
                rows.append(row)
            del lv, feats_lvl
            torch.cuda.empty_cache()
        print(f"[{method}] phantoms processed: "
              f"{len([x for x in rows if x['method'] == method])}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('output/diagnostics/p1c_e1_directions.csv', index=False)

    # ---- pattern tests (pooled primary, per-method secondary) ----
    pat_rows = []
    for scope, dscope in [('pooled', df)] + [(m, df[df.method == m])
                                             for m in ['thgs', 'relags']]:
        med = dscope['gt_pixels'].median()
        small = (dscope['gt_pixels'] < med).astype(int).values
        food = dscope['prompt'].isin(FOOD_PROMPTS).astype(int).values
        transp = dscope['prompt'].isin(TRANSPARENT_PROMPTS).astype(int).values
        for vname in ['c', 'a', 'b']:
            bg = dscope[f'bg_{vname}'].values
            vs = dscope[f'vessel_{vname}'].values
            for pat, grp, flags in [('P1_small_bg', small, bg),
                                    ('P2_food_vessel', food, vs),
                                    ('P4_transp_bg', transp, bg),
                                    ('P1+P4_bg', np.maximum(small, transp), bg)]:
                n_g = int(grp.sum())
                if n_g < 3:
                    pat_rows.append({'scope': scope, 'variant': vname,
                                     'pattern': pat, 'n_group': n_g,
                                     'effect': float('nan'), 'p': float('nan'),
                                     'verdict': 'insufficient_n'})
                    continue
                eff, p = perm_test(flags, grp)
                sig03 = bool(p < 0.05 and abs(eff) > 0.3)
                sig05 = bool(p < 0.05 and abs(eff) > 0.5)
                pat_rows.append({'scope': scope, 'variant': vname,
                                 'pattern': pat, 'n_group': n_g,
                                 'effect': round(eff, 3), 'p': round(p, 4),
                                 'verdict': ('sig@0.3' if sig03 else 'ns')
                                            + ('|sig@0.5' if sig05 else '')})

    # ---- P3: figurines instance-level (per method) ----
    rng = np.random.default_rng(0)
    for method in ['thgs', 'relags']:
        cfg = METHODS[method]
        ref = pd.read_csv(cfg['ref_csv'])
        ref = ref[(ref['is_ref_frame'] == 1) & (ref.scene == 'figurines')]
        sub = df[(df.method == method) & (df.scene == 'figurines')]
        if len(sub) < 3:
            pat_rows.append({'scope': method, 'variant': 'c', 'pattern': 'P3_near_fig',
                             'n_group': len(sub), 'effect': float('nan'),
                             'p': float('nan'), 'verdict': 'insufficient_n'})
            continue
        xyz = load_xyz(cfg['nag_tpl'], 'figurines')
        nag = torch.load(cfg['nag_tpl'].format('figurines'))
        labels = [l.long().cpu().numpy() for l in nag['nag']]
        cent = {}
        for _, r in ref.iterrows():
            g = np.where(labels[int(r['oracle_lvl'])] == int(r['oracle_sp_id']))[0]
            if xyz is not None and len(g) > 0:
                cent[r['prompt']] = xyz[g].mean(axis=0)
        prompts = [p for p in ref['prompt'] if p in cent]
        pe = {p: None for p in prompts}
        for p in prompts:
            vlm.encode_text(p)
            pe[p] = vlm.text_feature[0].float()
        obs_ranks = []
        for _, r in sub.iterrows():
            p0 = r['prompt']
            if p0 not in cent:
                continue
            nag2 = torch.load(cfg['nag_tpl'].format('figurines'))
            f_agg = F.normalize(nag2['nag_feat'][
                int(ref[ref.prompt == p0]['oracle_lvl'].iloc[0]) - 1].cuda().float(),
                p=2, dim=-1)[int(ref[ref.prompt == p0]['oracle_sp_id'].iloc[0])]
            d = F.normalize(f_agg - pe[p0], p=2, dim=0)
            others = [p for p in prompts if p != p0]
            sims = torch.stack([pe[p] @ d for p in others]).cpu().numpy()
            j = others[int(np.argmax(sims))]
            dists = np.array([np.linalg.norm(cent[p0] - cent[p]) for p in others])
            rank01 = float((dists < dists[others.index(j)]).mean())
            obs_ranks.append(rank01)
        if len(obs_ranks) >= 3:
            obs = float(np.mean(obs_ranks))
            null = [float(np.mean(rng.uniform(0, 1, len(obs_ranks))))
                    for _ in range(N_PERM)]
            p = (sum(1 for x in null if x <= obs) + 1) / (N_PERM + 1)
            pat_rows.append({'scope': method, 'variant': 'c',
                             'pattern': 'P3_near_fig', 'n_group': len(obs_ranks),
                             'effect': round(0.5 - obs, 3), 'p': round(p, 4),
                             'verdict': 'sig@0.3' if (p < 0.05 and (0.5 - obs) > 0.3)
                                        else ('sig' if p < 0.05 else 'ns')})
        else:
            pat_rows.append({'scope': method, 'variant': 'c', 'pattern': 'P3_near_fig',
                             'n_group': len(obs_ranks), 'effect': float('nan'),
                             'p': float('nan'), 'verdict': 'insufficient_n'})

    pdf = pd.DataFrame(pat_rows)
    pdf.to_csv('output/diagnostics/p1c_e1_patterns.csv', index=False)
    print("\n" + "=" * 78)
    print(pdf.to_string(index=False))
    sig = pdf[(pdf.scope == 'pooled') & (pdf.variant == 'c')
              & pdf.verdict.str.startswith('sig')]
    n_sig = len(sig[sig.pattern.isin(['P1_small_bg', 'P2_food_vessel',
                                      'P4_transp_bg'])])
    print(f"\n[E1 decision rule] significant patterns (pooled, variant c, "
          f"P1/P2/P4): {n_sig}")
    print("  2+ -> main contribution | 1 -> moderate | 0 -> negative finding")


if __name__ == '__main__':
    main()
