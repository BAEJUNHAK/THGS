"""
Stage 3.4-D — multi-instance / annotation audit for generic prompts.

For generic-noun prompts appearing in the phantom-17 / easy-regression sets,
collect the top-10 SPs under the query-top5 regime whose IoU with the prompt's
GT is < 0.1 in EVERY labeled frame, and dump their RGB crops as a montage.
These are "suspected unannotated same-category instances" — graded
conservatively (suspect / unlikely by size+views heuristics); final visual
confirmation is left to the montage (evidence, not verdict).

Output: output/diagnostics/stage3_4_multi_instance.csv
        output/diagnostics/plots/stage3_4_multi_instance_montage.png
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import make_vlm
from stage3_4_common import SceneCtx, classify_winner

GENERIC = [
    ('waldo_kitchen', 'spoon'), ('waldo_kitchen', 'cabinet'),
    ('waldo_kitchen', 'plate'), ('waldo_kitchen', 'sink'),
    ('ramen', 'bowl'), ('ramen', 'plate'), ('ramen', 'napkin'),
    ('ramen', 'sake cup'), ('waldo_kitchen', 'knife'),
    ('waldo_kitchen', 'dark cup'), ('ramen', 'spoon'),
]


def main():
    b7 = pd.read_csv('output/diagnostics/b7_a4_combined.csv')
    ref = b7[b7['is_ref_frame'] == 1]
    oracle_of = {(r['scene'], r['prompt']): (int(r['oracle_lvl']), int(r['oracle_sp_id']))
                 for _, r in ref.iterrows()}
    vlm = make_vlm()
    ctxs = {}
    rows, panels = [], []
    for sc, prompt in GENERIC:
        if (sc, prompt) not in oracle_of:
            continue
        if sc not in ctxs:
            print(f"[load] {sc}", flush=True)
            ctxs[sc] = SceneCtx(sc, vlm)
        ctx = ctxs[sc]
        oracle = oracle_of[(sc, prompt)]
        winners, orank = ctx.winners_above(prompt, oracle, 'top5', topn=10)
        n_suspect = 0
        for wrank, (w, score) in enumerate(winners, 1):
            ident = ctx.identity(prompt, oracle, w)
            cls = classify_winner(ident)
            if cls != 'external':
                continue
            # heuristic grade: enough views + non-trivial size -> suspect
            grade = ('suspect' if ident['n_valid_views'] >= 10 else 'weak_evidence')
            n_suspect += int(grade == 'suspect')
            rows.append({'scene': sc, 'prompt': prompt, 'winner_rank': wrank,
                         'grade': grade, **ident})
            if wrank <= 3:
                c = ctx.crop(prompt, w)
                if c is not None:
                    panels.append((f"{sc[:5]}/{prompt[:12]} #{wrank}\n"
                                   f"{grade} nv={ident['n_valid_views']}", c))
        print(f"  {sc}/{prompt}: oracle_rank={orank}, external winners={sum(1 for r in rows if r['scene']==sc and r['prompt']==prompt)}, suspects={n_suspect}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('output/diagnostics/stage3_4_multi_instance.csv', index=False)
    per = df[df.grade == 'suspect'].groupby(['scene', 'prompt']).size()
    print("\n=== suspected unannotated-instance counts (visual confirm via montage) ===")
    print(per.to_string() if len(per) else "(none)")

    if panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(panels)
        cols = 6
        rn = (n + cols - 1) // cols
        fig, axes = plt.subplots(rn, cols, figsize=(2.8 * cols, 2.6 * rn))
        axes = np.atleast_2d(axes)
        for i, (label, img) in enumerate(panels):
            axes[i // cols, i % cols].imshow(img)
            axes[i // cols, i % cols].set_title(label, fontsize=7)
        for ax in axes.flat:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig('output/diagnostics/plots/stage3_4_multi_instance_montage.png', dpi=110)
    print("Wrote stage3_4_multi_instance.csv + montage")


if __name__ == '__main__':
    main()
