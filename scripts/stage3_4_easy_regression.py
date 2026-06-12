"""
Stage 3.4-B — anatomy of the 10 easy regressions under query-top5.

For each easy prompt whose rank went baseline<=3 -> p>3: identify the SPs that
newly beat the oracle under top5, run identity analysis (GT IoU / kinship /
coherence / n_views), and tabulate GUARD-SIGNAL candidates that could let a
hybrid know when NOT to trust top-k:
  g1 winner_gap   = winner_top5_score - winner_mean_score   (lucky-view jump)
  g2 oracle_already_top = oracle mean-rank <= 1
  g3 winner_n_views (few-view opportunists?)
  g4 winner kinship vs oracle (parts of the same object are harmless)

Output: output/diagnostics/stage3_4_easy_regression.csv
        output/diagnostics/plots/stage3_4_easyreg_montage.png
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import make_vlm
from stage3_4_common import SceneCtx, classify_winner


def main():
    fr = pd.read_csv('output/diagnostics/stage3_3_fullpool_ranks.csv')
    reg = fr[(fr['category'] == 'easy') & (fr['baseline_rank'] <= 3) & (fr['rank_p'] > 3)]
    print(f"easy regressions: {len(reg)}")
    b7 = pd.read_csv('output/diagnostics/b7_a4_combined.csv')
    ref = b7[b7['is_ref_frame'] == 1]
    oracle_of = {(r['scene'], r['prompt']): (int(r['oracle_lvl']), int(r['oracle_sp_id']))
                 for _, r in ref.iterrows()}

    vlm = make_vlm()
    ctxs = {}
    rows, panels = [], []
    for _, r in reg.iterrows():
        sc, prompt = r['scene'], r['prompt']
        if sc not in ctxs:
            print(f"[load] {sc}", flush=True)
            ctxs[sc] = SceneCtx(sc, vlm)
        ctx = ctxs[sc]
        oracle = oracle_of[(sc, prompt)]
        s_mean, s_top = ctx.scores(prompt)
        oi = ctx.index[oracle]
        oracle_mean_rank = int((s_mean > s_mean[oi]).sum()) + 1
        winners, orank = ctx.winners_above(prompt, oracle, 'top5', topn=3)
        print(f"\n=== {sc}/{prompt} base={r['baseline_rank']} -> p={orank} ===", flush=True)
        for wrank, (w, score) in enumerate(winners, 1):
            ident = ctx.identity(prompt, oracle, w)
            cls = classify_winner(ident)
            wi = ctx.index[w]
            gap = float(s_top[wi] - s_mean[wi])
            rows.append({'scene': sc, 'prompt': prompt,
                         'baseline_rank': int(r['baseline_rank']), 'p_rank': orank,
                         'oracle_mean_rank': oracle_mean_rank,
                         'winner_rank': wrank, 'class': cls,
                         'g1_winner_gap_top5_minus_mean': round(gap, 4),
                         'g2_oracle_was_rank1': oracle_mean_rank == 1,
                         **ident})
            print(f"  #{wrank} {ident['winner']:8s} {cls:16s} gap={gap:+.3f} "
                  f"gt_iou={ident['best_gt_iou']:.2f} kin={ident['kinship']:12s} "
                  f"nv={ident['n_valid_views']} coh={ident['coherence']:.2f}", flush=True)
            if wrank == 1:
                c = ctx.crop(prompt, w)
                if c is not None:
                    panels.append((f"{sc[:5]}/{prompt[:14]}\n{cls} gap={gap:+.2f}", c))

    df = pd.DataFrame(rows)
    df.to_csv('output/diagnostics/stage3_4_easy_regression.csv', index=False)

    print("\n=== GUARD-SIGNAL summary (new winners under top5) ===")
    print("class distribution:\n" + df['class'].value_counts().to_string())
    print(f"\ng1 winner gap (top5-mean): median={df['g1_winner_gap_top5_minus_mean'].median():.3f} "
          f"(>0 = lucky-view jump)")
    print(f"g2 oracle was mean-rank-1: {df.groupby(['scene','prompt'])['g2_oracle_was_rank1'].first().mean()*100:.0f}% of cases")
    print(f"g3 winner n_views: median={df['n_valid_views'].median():.0f}")
    print(f"g4 harmless kin (child/parent/overlap_gt): "
          f"{(df['class'].isin(['kin_child_of_b','kin_parent_of_b','overlaps_gt'])).mean()*100:.0f}%")

    if panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(panels)
        cols = 5
        rn = (n + cols - 1) // cols
        fig, axes = plt.subplots(rn, cols, figsize=(3 * cols, 2.6 * rn))
        axes = np.atleast_2d(axes)
        for i, (label, img) in enumerate(panels):
            axes[i // cols, i % cols].imshow(img)
            axes[i // cols, i % cols].set_title(label, fontsize=7)
        for ax in axes.flat:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig('output/diagnostics/plots/stage3_4_easyreg_montage.png', dpi=110)
    print("Wrote stage3_4_easy_regression.csv + montage")


if __name__ == '__main__':
    main()
