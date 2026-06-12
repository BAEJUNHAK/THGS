"""
Stage 4 / Phase A — taxonomy-derived combined scorer (training-free, query-time).

score_i = alpha * canon(mean_feat_i) + (1-alpha) * canon(topk_q_feat_i)
signals:
  Z  zero-norm filter : SPs with no valid view are excluded from the pool
                        (kills the 150 D1 ghosts; canon(0)=0.5 never competes)
  E  min-evidence     : n_valid_views < tau_v  -> top-k term replaced by the
                        mean term (few-view opportunists lose their jump)
  G  g1 guard         : (top_i - mean_i) > tau_g -> top term clamped to
                        mean_i + tau_g  (lucky-view jump dampening)
  P  parent-union     : after top-3 selection, NAG parent/child of a selected
                        SP joins the union if its score >= sel_score - EPS_PU
                        (granularity cases; affects masks, not ranks)

Grid: alpha x k x tau_v x g1 x parent_union  (216 configs; rank math is
array-only — s_mean once, s_top per k — so the grid is cheap).

Consistency gates (MUST pass before anything else is trusted):
  ① alpha=1, signals off  -> reproduce stage3_3 baseline ranks
  ② alpha=0, signals off  -> reproduce stage3_3 rank_p

Outputs:
  output/diagnostics/stage4_grid_ranks.csv     (per prompt x config: oracle rank)
  output/diagnostics/stage4_selections.pkl     {(scene,prompt): {config: [pairs]}}
  output/diagnostics/stage4_scorer_meta.json   (config list, gate results)
"""

import os
import sys
import json
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, query_topk_agg, LEVELS
from stage3_4_nag_kinship import load_labels, relation
from stage3_2_common import make_vlm

ALPHAS = [0.2, 0.3, 0.4, 0.5]
KS = [3, 5, 10]
TAU_VS = [0, 5, 10]
G1S = [None, 0.10, 0.20]          # None = off; value = tau_g (canon units)
PUS = [False, True]
EPS_PU = 0.05
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def config_name(a, k, tv, g1, pu):
    return (f"a{a}_k{k}_tv{tv}_g{'off' if g1 is None else g1}"
            f"_pu{'on' if pu else 'off'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_ranks', default='output/diagnostics/stage4_grid_ranks.csv')
    ap.add_argument('--out_sel', default='output/diagnostics/stage4_selections.pkl')
    ap.add_argument('--out_meta', default='output/diagnostics/stage4_scorer_meta.json')
    args = ap.parse_args()

    b7 = pd.read_csv('output/diagnostics/b7_a4_combined.csv')
    ref = b7[b7['is_ref_frame'] == 1]
    ph = pd.read_csv('output/diagnostics/persistent_phantoms_17.csv')
    pset = set(map(tuple, ph[['scene', 'prompt']].values))
    fp = pd.read_csv('output/diagnostics/stage3_3_fullpool_ranks.csv')
    fp_idx = fp.set_index(['scene', 'prompt'])

    vlm = make_vlm()
    configs = [(a, k, tv, g1, pu) for a, k, tv, g1, pu in
               itertools.product(ALPHAS, KS, TAU_VS, G1S, PUS)]
    rows, selections = [], {}
    gate1_fail = gate2_fail = 0
    gate_n = 0

    for scene in SCENES:
        dump = load_dump(f'output/diagnostics/stage3_3_allsp_{scene}.pkl')
        lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}
        entries = [(l, i) for l in LEVELS for i in range(lv[l][0].shape[1])]
        index = {e: i for i, e in enumerate(entries)}
        nval = np.concatenate([lv[l][4] for l in LEVELS]).astype(int)
        has_view = nval > 0
        nagd = torch.load(f'output/lerf/{scene}/sai_nag.pt')
        base_feat = torch.cat([
            torch.nn.functional.normalize(nagd['nag_feat'][l - 1].cuda().float(),
                                          p=2, dim=-1) for l in LEVELS])
        labels = load_labels(scene)

        # precompute parent/child candidate map per entry (NAG containment)
        # parent at lvl+1: the SP at coarser level containing >=80% of gaussians
        # child(ren) at lvl-1 within LEVELS: SPs whose gaussians are >=80% inside.
        # Cheap via label arrays.
        parent_of = {}
        children_of = {}
        for (l, s) in entries:
            g = np.where(labels[l] == s)[0]
            if len(g) == 0:
                continue
            if l + 1 <= 3 and (l + 1) in LEVELS:
                par_ids, cnt = np.unique(labels[l + 1][g], return_counts=True)
                j = int(np.argmax(cnt))
                if cnt[j] / len(g) >= 0.8:
                    parent_of[(l, s)] = (l + 1, int(par_ids[j]))
        for ch, par in parent_of.items():
            children_of.setdefault(par, []).append(ch)

        sc_ref = ref[ref['scene'] == scene]
        print(f"\n=== {scene}: {len(sc_ref)} prompts, pool={len(entries)} ===", flush=True)
        for _, r in sc_ref.iterrows():
            prompt = r['prompt']
            okey = (int(r['oracle_lvl']), int(r['oracle_sp_id']))
            oi = index[okey]
            vlm.encode_text(prompt)
            text = vlm.text_feature[0].float()
            s_mean = vlm.compute_similarity(base_feat).cpu().numpy()
            s_top = {}
            for k in KS:
                aggs = torch.cat([query_topk_agg(*lv[l][:3], text, k=k)
                                  for l in LEVELS])
                s_top[k] = vlm.compute_similarity(aggs).cpu().numpy()

            # ---- consistency gates (once per prompt)
            gate_n += 1
            r_base = int((s_mean[has_view if False else slice(None)] > s_mean[oi]).sum()) + 1
            r_base = int((s_mean > s_mean[oi]).sum()) + 1
            if abs(r_base - int(fp_idx.loc[(scene, prompt), 'baseline_rank'])) > 2:
                gate1_fail += 1
            s_p = s_top[5].copy()
            r_p = int((s_p > s_p[oi]).sum()) + 1
            if abs(r_p - int(fp_idx.loc[(scene, prompt), 'rank_p'])) > 0:
                gate2_fail += 1

            sel_pp = {}
            row = {'scene': scene, 'prompt': prompt,
                   'category': ('phantom17' if (scene, prompt) in pset
                                else 'easy' if int(r['oracle_rank']) <= 3 else 'other'),
                   'baseline_rank': int(r['oracle_rank'])}
            for (a, k, tv, g1, pu) in configs:
                top_eff = s_top[k].copy()
                if tv > 0:
                    top_eff = np.where(nval < tv, s_mean, top_eff)
                if g1 is not None:
                    top_eff = np.minimum(top_eff, s_mean + g1)
                s = a * s_mean + (1 - a) * top_eff
                s = np.where(has_view, s, -1e9)          # Z filter (always on)
                rank = int((s > s[oi]).sum()) + 1
                cname = config_name(a, k, tv, g1, pu)
                row[f'rank_{cname}'] = rank
                order = np.argsort(-s)[:3]
                pairs = [entries[i] for i in order]
                if pu:
                    extra = []
                    for p_, i_ in zip(pairs, order):
                        thr = s[i_] - EPS_PU
                        cand = []
                        if p_ in parent_of:
                            cand.append(parent_of[p_])
                        cand += children_of.get(p_, [])
                        for c in cand:
                            ci = index[c]
                            if s[ci] >= thr and c not in pairs and c not in extra:
                                extra.append(c)
                    pairs = pairs + extra[:3]
                sel_pp[cname] = pairs
            selections[(scene, prompt)] = sel_pp
            rows.append(row)
        del lv, base_feat
        torch.cuda.empty_cache()
        print(f"  done {scene}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_ranks, index=False)
    with open(args.out_sel, 'wb') as f:
        pickle.dump(selections, f)
    meta = {'configs': [config_name(*c) for c in configs],
            'gate1_baseline_mismatch': gate1_fail,
            'gate2_rankp_mismatch': gate2_fail, 'n_prompts': gate_n,
            'eps_pu': EPS_PU}
    with open(args.out_meta, 'w') as f:
        json.dump(meta, f, indent=1)

    print("\n" + "=" * 70)
    print(f"GATE ①  alpha=1 ≙ baseline (±2): mismatches = {gate1_fail}/{gate_n}")
    print(f"GATE ②  alpha=0,k=5 ≙ rank_p (exact): mismatches = {gate2_fail}/{gate_n}")
    print("=" * 70)
    print(f"{len(configs)} configs × {gate_n} prompts -> {args.out_ranks}")


if __name__ == '__main__':
    main()
