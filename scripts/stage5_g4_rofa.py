"""
Stage 5 / G4 — ROFA's REAL net effect inside ReLaGS (new, ReLaGS-specific).

From the same per-view dump, reconstruct every SP's final feature twice:
  ROFA-on  : tau=2 keep-mask -> mean of kept scaled feats -> normalize
             (= the actual ReLaGS pipeline; fidelity-gated)
  ROFA-off : plain visibility-weighted mean (= THGS-style aggregation)
and compare each prompt's oracle pool rank under the two pools.

This settles Stage 2B's simulated verdict ("ROFA is innocent; 0%
outlier_handled") on the real pipeline distribution: how many ranks does
ROFA actually move, for phantoms vs easy?

Output: output/diagnostics/stage5_g4_rofa.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, LEVELS
from stage3_2_common import make_vlm
from stage3_b8_replay import aggregate_views

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def recon_pools(lv):
    """(rofa_on_feats, rofa_off_feats) over the concatenated pool."""
    on, off = [], []
    for l in LEVELS:
        feats, por, valid, _, _ = lv[l]
        V, S, D = feats.shape
        ft = feats.float()
        pt = por
        nz = valid
        for s_ in range(S):
            m = nz[:, s_]
            if not m.any():
                on.append(torch.zeros(D, device='cuda'))
                off.append(torch.zeros(D, device='cuda'))
                continue
            scaled = ft[m, s_, :] * pt[m, s_].unsqueeze(-1)
            on.append(aggregate_views(scaled, 'relags'))
            off.append(F.normalize(scaled.sum(dim=0), p=2, dim=-1))
    return torch.stack(on), torch.stack(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_tpl', default='output/diagnostics/stage5_relags_allsp_{}.pkl')
    ap.add_argument('--out_csv', default='output/diagnostics/stage5_g4_rofa.csv')
    args = ap.parse_args()

    cm = pd.read_csv('output/diagnostics/cross_method_d2_decomposition.csv')
    phantom = set(map(tuple, cm[cm.relags_class == 'phantom'][['scene', 'prompt']].values))
    rb = pd.read_csv('output/diagnostics/b7_a4_combined_relags.csv')
    ref = rb[rb['is_ref_frame'] == 1]
    vlm = make_vlm()
    rows = []
    for scene in SCENES:
        dump = load_dump(args.dump_tpl.format(scene))
        lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}
        entries = [(l, i) for l in LEVELS for i in range(lv[l][0].shape[1])]
        index = {e: i for i, e in enumerate(entries)}
        pool_on, pool_off = recon_pools(lv)
        print(f"=== {scene}: pools reconstructed ({len(entries)} SPs) ===", flush=True)
        for _, r in ref[ref.scene == scene].iterrows():
            prompt = r['prompt']
            oi = index[(int(r['oracle_lvl']), int(r['oracle_sp_id']))]
            vlm.encode_text(prompt)
            s_on = vlm.compute_similarity(pool_on).cpu().numpy()
            s_off = vlm.compute_similarity(pool_off).cpu().numpy()
            r_on = int((s_on > s_on[oi]).sum()) + 1
            r_off = int((s_off > s_off[oi]).sum()) + 1
            cat = ('phantom' if (scene, prompt) in phantom
                   else 'easy' if int(r['oracle_rank']) <= 3 else 'other')
            rows.append({'scene': scene, 'prompt': prompt, 'category': cat,
                         'rank_rofa_on': r_on, 'rank_rofa_off': r_off,
                         'rofa_effect': r_off - r_on})
        del lv, pool_on, pool_off
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("\n" + "=" * 70)
    print("[G4] ROFA net effect on oracle rank (rofa_off_rank - rofa_on_rank; "
          ">0 = ROFA 가 도움)")
    for grp in ['phantom', 'easy', 'other']:
        g = df[df.category == grp]
        print(f"  {grp:8s} (n={len(g):2d}): median effect={g['rofa_effect'].median():+.0f}, "
              f"helped {int((g['rofa_effect'] > 0).sum())} / hurt "
              f"{int((g['rofa_effect'] < 0).sum())} / neutral "
              f"{int((g['rofa_effect'] == 0).sum())}")
        ch = g[g['rofa_effect'] != 0].nlargest(3, 'rofa_effect')
        for _, x in ch.iterrows():
            print(f"      {x['prompt']:24s} off={x['rank_rofa_off']:4d} -> "
                  f"on={x['rank_rofa_on']:4d}")
    ph = df[df.category == 'phantom']
    n_resc = int(((ph['rank_rofa_off'] > 3) & (ph['rank_rofa_on'] <= 3)).sum())
    print(f"\n[G4] ROFA 가 phantom 을 rank<=3 으로 구한 수: {n_resc}/{len(ph)} "
          f"[2B 시뮬 예측: outlier_handled 0%]")
    print(f"Wrote {args.out_csv}")


if __name__ == '__main__':
    main()
