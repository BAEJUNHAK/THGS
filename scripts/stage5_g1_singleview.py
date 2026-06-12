"""
Stage 5 / G1 — minority-dilution signature on ReLaGS's OWN pipeline.

For each ReLaGS phantom (~20) and easy control (16): every view's mixed
feature ALONE replaces the SP's entry in the ReLaGS pool (levels [2,3],
canon-contrast) -> single-view ranks; plus the fraction of views with
rank <= 3 ("good-view fraction").

G1 verdict (pre-registered): best-single <= 3 for >= 60% of phantoms AND
good-view fraction phantom < easy (Mann-Whitney p < 0.05).
THGS reference: 88% / 18% vs 50%.

Output: output/diagnostics/stage5_g1_singleview.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, LEVELS
from stage3_2_common import make_vlm

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_root', default='ReLaGS/output/lerf_hf/scenes/LeRF')
    ap.add_argument('--dump_tpl', default='output/diagnostics/stage5_relags_allsp_{}.pkl')
    ap.add_argument('--targets_csv', default='output/diagnostics/stage5_relags_targets.csv')
    ap.add_argument('--out_csv', default='output/diagnostics/stage5_g1_singleview.csv')
    args = ap.parse_args()

    targets = pd.read_csv(args.targets_csv)
    vlm = make_vlm()
    rows = []
    for scene in SCENES:
        t_sc = targets[targets.scene == scene]
        if len(t_sc) == 0:
            continue
        dump = load_dump(args.dump_tpl.format(scene))
        lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}
        entries = [(l, i) for l in LEVELS for i in range(lv[l][0].shape[1])]
        index = {e: i for i, e in enumerate(entries)}
        nag = torch.load(f'{args.model_root}/{scene}/sai_nag.pt')
        pool_feat = torch.cat([
            torch.nn.functional.normalize(nag['nag_feat'][l - 1].cuda().float(),
                                          p=2, dim=-1) for l in LEVELS])
        print(f"=== {scene}: pool={len(entries)} ===", flush=True)
        for _, t in t_sc.iterrows():
            prompt, lvl, sp = t['prompt'], int(t['oracle_lvl']), int(t['oracle_sp_id'])
            oi = index[(lvl, sp)]
            vlm.encode_text(prompt)
            pool_canon = vlm.compute_similarity(pool_feat).cpu().numpy()
            feats, por, valid, _, _ = lv[lvl]
            col = lvl_local = sp                     # feat column index within level
            f_v = feats[:, col, :].float()
            ok = valid[:, col]
            if not ok.any():
                rows.append({'scene': scene, 'prompt': prompt,
                             'category': t['category'], 'n_views': 0})
                continue
            cand = torch.nn.functional.normalize(f_v[ok], p=2, dim=-1)
            c_canon = vlm.compute_similarity(cand).cpu().numpy()
            others = np.delete(pool_canon, oi)
            ranks = np.array([(others > c).sum() + 1 for c in c_canon])
            rows.append({'scene': scene, 'prompt': prompt, 'category': t['category'],
                         'n_views': int(ok.sum()),
                         'single_best': int(ranks.min()),
                         'single_median': float(np.median(ranks)),
                         'frac_le3': float((ranks <= 3).mean())})
            print(f"  {t['category'][:14]:14s} {prompt:24s} best={ranks.min():3d} "
                  f"med={np.median(ranks):5.0f} %<=3={(ranks <= 3).mean()*100:3.0f}%",
                  flush=True)
        del lv, pool_feat
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    ph = df[df.category == 'relags_phantom'].dropna(subset=['single_best'])
    ez = df[df.category == 'relags_easy'].dropna(subset=['single_best'])
    frac_best = (ph['single_best'] <= 3).mean()
    mw = stats.mannwhitneyu(ph['frac_le3'], ez['frac_le3'], alternative='less')
    print("\n" + "=" * 70)
    print(f"[G1] phantom best-single <=3 : {int((ph['single_best'] <= 3).sum())}/{len(ph)} "
          f"({frac_best*100:.0f}%)  [기준 >=60%; THGS 88%]")
    print(f"[G1] good-view frac : phantom {ph['frac_le3'].mean()*100:.0f}% vs "
          f"easy {ez['frac_le3'].mean()*100:.0f}%  (MW one-sided p={mw.pvalue:.4f})"
          f"  [THGS 18% vs 50%]")
    ok = frac_best >= 0.6 and mw.pvalue < 0.05
    print(f"[G1 VERDICT] {'재현 ✅' if ok else 'NOT 재현'}")
    print(f"Wrote {args.out_csv}")


if __name__ == '__main__':
    main()
