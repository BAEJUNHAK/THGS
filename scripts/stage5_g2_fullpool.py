"""
Stage 5 / G2 — zero-sum replication on ReLaGS (rank level, full pool).

Plain query-top5 re-scoring of the ENTIRE ReLaGS pool (no frozen competitors)
vs the ReLaGS baseline (mean/sai_nag scoring). Categories from the ReLaGS
side: phantom = cross_method relags_class, easy = relags oracle_rank <= 3.

G2 verdict: phantom recoveries AND easy regressions co-occur (THGS: +6 / -10).

Output: output/diagnostics/stage5_g2_fullpool.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, query_topk_agg, LEVELS
from stage3_2_common import make_vlm

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_root', default='ReLaGS/output/lerf_hf/scenes/LeRF')
    ap.add_argument('--dump_tpl', default='output/diagnostics/stage5_relags_allsp_{}.pkl')
    ap.add_argument('--out_csv', default='output/diagnostics/stage5_g2_fullpool.csv')
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
        nval = np.concatenate([lv[l][4] for l in LEVELS]).astype(int)
        nag = torch.load(f'{args.model_root}/{scene}/sai_nag.pt')
        base_feat = torch.cat([
            torch.nn.functional.normalize(nag['nag_feat'][l - 1].cuda().float(),
                                          p=2, dim=-1) for l in LEVELS])
        sc_ref = ref[ref['scene'] == scene]
        print(f"=== {scene}: {len(sc_ref)} prompts, pool={len(entries)} ===", flush=True)
        for _, r in sc_ref.iterrows():
            prompt = r['prompt']
            okey = (int(r['oracle_lvl']), int(r['oracle_sp_id']))
            oi = index[okey]
            vlm.encode_text(prompt)
            text = vlm.text_feature[0].float()
            s_mean = vlm.compute_similarity(base_feat).cpu().numpy()
            aggs = torch.cat([query_topk_agg(*lv[l][:3], text, k=5) for l in LEVELS])
            s_top = vlm.compute_similarity(aggs).cpu().numpy()
            r_base = int((s_mean > s_mean[oi]).sum()) + 1
            r_p = int((s_top > s_top[oi]).sum()) + 1
            cat = ('phantom' if (scene, prompt) in phantom
                   else 'easy' if int(r['oracle_rank']) <= 3 else 'other')
            rows.append({'scene': scene, 'prompt': prompt, 'category': cat,
                         'relags_a4_rank': int(r['oracle_rank']),
                         'rank_base_mean': r_base, 'rank_top5': r_p,
                         'oracle_n_views': int(nval[oi])})
            print(f"  {cat:7s} {prompt:26s} base={r_base:4d} top5={r_p:4d}", flush=True)
        del lv, base_feat
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("\n" + "=" * 70)
    for grp in ['phantom', 'easy', 'other']:
        g = df[df.category == grp]
        nb = int((g['rank_base_mean'] <= 3).sum())
        np_ = int((g['rank_top5'] <= 3).sum())
        print(f"  {grp:8s} (n={len(g):2d}): base {nb}<=3 -> top5 {np_}<=3  (Δ{np_-nb:+d})")
    ph = df[df.category == 'phantom']
    ez = df[df.category == 'easy']
    rec = int(((ph['rank_base_mean'] > 3) & (ph['rank_top5'] <= 3)).sum())
    reg = int(((ez['rank_base_mean'] <= 3) & (ez['rank_top5'] > 3)).sum())
    print(f"\n[G2] phantom 회복 {rec} / easy 역행 {reg}  [THGS: +6 / -10]")
    print(f"[G2 VERDICT] {'제로섬 구조 재현 ✅ (회복·역행 동시 발생)' if rec > 0 and reg > 0 else '구조 비재현'}")
    print(f"Wrote {args.out_csv}")


if __name__ == '__main__':
    main()
