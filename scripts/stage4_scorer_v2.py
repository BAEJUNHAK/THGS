"""
Stage 4 / re-search #1 (allowed once by pre-registered R9 PARTIAL branch).

Changes vs v1, each justified by prior measurements (not post-hoc fishing):
  - g1 global clamp REMOVED  — held-out ablation showed it suppresses oracle
    jumps as much as impostor jumps (figurines fold: G_off +8.7pt).
  - g2 ADDED (the 3.4-B guard that v1 omitted): per-PROMPT adaptive alpha.
    If the mean regime is already confident — canon margin between its top-1
    and top-2 >= tau_c — the prompt keeps the pure-mean ranking (easy prompts
    are exactly the mean-confident ones; 3.4-B: 60% of regressions had the
    oracle at mean rank 1). Otherwise the hybrid applies.

Grid v2: alpha x k x tau_v x tau_c x pu  (2*3*2*4*2 = 96 configs)
Same outputs/format as v1 (suffix _v2), same LOSO driver reusable.
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
from stage3_4_nag_kinship import load_labels
from stage3_2_common import make_vlm

ALPHAS = [0.3, 0.5]
KS = [3, 5, 10]
TAU_VS = [5, 10]
TAU_CS = [0.01, 0.03, 0.05, 0.10]
PUS = [False, True]
EPS_PU = 0.05
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def config_name(a, k, tv, tc, pu):
    return f"a{a}_k{k}_tv{tv}_c{tc}_pu{'on' if pu else 'off'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_ranks', default='output/diagnostics/stage4_grid_ranks_v2.csv')
    ap.add_argument('--out_sel', default='output/diagnostics/stage4_selections_v2.pkl')
    args = ap.parse_args()

    b7 = pd.read_csv('output/diagnostics/b7_a4_combined.csv')
    ref = b7[b7['is_ref_frame'] == 1]
    ph = pd.read_csv('output/diagnostics/persistent_phantoms_17.csv')
    pset = set(map(tuple, ph[['scene', 'prompt']].values))

    vlm = make_vlm()
    configs = list(itertools.product(ALPHAS, KS, TAU_VS, TAU_CS, PUS))
    rows, selections = [], {}

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
        parent_of = {}
        children_of = {}
        for (l, s) in entries:
            g = np.where(labels[l] == s)[0]
            if len(g) == 0 or l + 1 > 3 or (l + 1) not in LEVELS:
                continue
            par_ids, cnt = np.unique(labels[l + 1][g], return_counts=True)
            j = int(np.argmax(cnt))
            if cnt[j] / len(g) >= 0.8:
                parent_of[(l, s)] = (l + 1, int(par_ids[j]))
        for ch, par in parent_of.items():
            children_of.setdefault(par, []).append(ch)

        sc_ref = ref[ref['scene'] == scene]
        print(f"=== {scene}: {len(sc_ref)} prompts ===", flush=True)
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

            # g2 confidence of the mean regime (Z-filtered pool)
            sm_f = np.where(has_view, s_mean, -1e9)
            srt = np.sort(sm_f)[::-1]
            mean_margin = float(srt[0] - srt[1])

            sel_pp = {}
            row = {'scene': scene, 'prompt': prompt,
                   'category': ('phantom17' if (scene, prompt) in pset
                                else 'easy' if int(r['oracle_rank']) <= 3 else 'other'),
                   'baseline_rank': int(r['oracle_rank']),
                   'mean_margin': round(mean_margin, 4)}
            for (a, k, tv, tc, pu) in configs:
                if mean_margin >= tc:                  # g2: confident mean wins
                    s = sm_f
                else:
                    top_eff = np.where(nval < tv, s_mean, s_top[k])
                    s = np.where(has_view, a * s_mean + (1 - a) * top_eff, -1e9)
                rank = int((s > s[oi]).sum()) + 1
                cname = config_name(a, k, tv, tc, pu)
                row[f'rank_{cname}'] = rank
                order = np.argsort(-s)[:3]
                pairs = [entries[i] for i in order]
                if pu:
                    extra = []
                    for p_, i_ in zip(pairs, order):
                        cand = ([parent_of[p_]] if p_ in parent_of else []) \
                               + children_of.get(p_, [])
                        for c in cand:
                            if s[index[c]] >= s[i_] - EPS_PU and c not in pairs \
                                    and c not in extra:
                                extra.append(c)
                    pairs = pairs + extra[:3]
                sel_pp[cname] = pairs
            selections[(scene, prompt)] = sel_pp
            rows.append(row)
        del lv, base_feat
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(args.out_ranks, index=False)
    with open(args.out_sel, 'wb') as f:
        pickle.dump(selections, f)
    print(f"{len(configs)} configs -> {args.out_ranks}")


if __name__ == '__main__':
    main()
