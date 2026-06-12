"""
Stage 3.4-A — anatomy of the "unrescuable" phantoms.

Targets: never_good (waldo spoon, cabinet) + query-aware-worsened
(sake cup, sink, onion segments). For each, identify the SPs that BEAT the
oracle under BOTH regimes (mean / query-top5), then run identity analysis:
max IoU vs the prompt's GT across all labeled frames, NAG kinship vs oracle,
coherence / n_views, and an RGB crop montage as visual evidence.

Output: output/diagnostics/stage3_4_losers.csv
        output/diagnostics/plots/stage3_4_loser_winners_montage.png
"""

import os
import sys
import csv
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import make_vlm
from stage3_4_common import SceneCtx, classify_winner

TARGETS = [
    ('waldo_kitchen', 'spoon'), ('waldo_kitchen', 'cabinet'),
    ('ramen', 'sake cup'), ('waldo_kitchen', 'sink'), ('ramen', 'onion segments'),
]


def main():
    b7 = pd.read_csv('output/diagnostics/b7_a4_combined.csv')
    ref = b7[b7['is_ref_frame'] == 1]
    oracle_of = {(r['scene'], r['prompt']): (int(r['oracle_lvl']), int(r['oracle_sp_id']))
                 for _, r in ref.iterrows()}
    vlm = make_vlm()
    ctxs = {}
    rows, panels = [], []

    for sc, prompt in TARGETS:
        if sc not in ctxs:
            print(f"[load] {sc}", flush=True)
            ctxs[sc] = SceneCtx(sc, vlm)
        ctx = ctxs[sc]
        oracle = oracle_of[(sc, prompt)]
        for regime in ('mean', 'top5'):
            winners, orank = ctx.winners_above(prompt, oracle, regime, topn=10)
            print(f"\n=== {sc}/{prompt} [{regime}] oracle_rank={orank}, "
                  f"{len(winners)} winners ===", flush=True)
            for wrank, (w, score) in enumerate(winners, 1):
                ident = ctx.identity(prompt, oracle, w)
                cls = classify_winner(ident)
                rows.append({'scene': sc, 'prompt': prompt, 'regime': regime,
                             'oracle_rank': orank, 'winner_rank': wrank,
                             'score': round(score, 4), 'class': cls, **ident})
                print(f"  #{wrank} {ident['winner']:8s} {cls:16s} "
                      f"gt_iou={ident['best_gt_iou']:.2f} rec={ident['best_gt_recall']:.2f} "
                      f"kin={ident['kinship']:12s} coh={ident['coherence']:.2f} "
                      f"nv={ident['n_valid_views']}", flush=True)
                if regime == 'top5' and wrank <= 2:
                    c = ctx.crop(prompt, w)
                    if c is not None:
                        panels.append((f"{sc[:5]}/{prompt[:14]} top5#{wrank}\n"
                                       f"{cls} iou={ident['best_gt_iou']:.2f}", c))

    df = pd.DataFrame(rows)
    df.to_csv('output/diagnostics/stage3_4_losers.csv', index=False)
    print("\n=== winner class distribution (top5 regime) ===")
    print(df[df.regime == 'top5']['class'].value_counts().to_string())

    if panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(panels)
        cols = 5
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(3 * cols, 2.6 * rows_n))
        axes = np.atleast_2d(axes)
        for i, (label, img) in enumerate(panels):
            ax = axes[i // cols, i % cols]
            ax.imshow(img)
            ax.set_title(label, fontsize=7)
        for j in range(n, rows_n * cols):
            axes[j // cols, j % cols].axis('off')
        for ax in axes.flat:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig('output/diagnostics/plots/stage3_4_loser_winners_montage.png', dpi=110)
    print("Wrote stage3_4_losers.csv + montage")


if __name__ == '__main__':
    main()
