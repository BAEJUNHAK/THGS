"""
Stage 3.2 / B1.C — pool-competition decomposition: oracle SP vs its wrong-top1 SP.

Paired comparison (17 phantoms) on per-view replay dumps:
  - coherence = || sum(normalize(f_i) * w_i) || / sum(w_i)
      resultant length of the weighted unit-vector sum; 1.0 = all views point
      the same way, small = dispersed (averaging washes out the direction)
  - per-view raw cos to the (phantom's) prompt: mean/std
  - portion (visibility) profile: sum/mean/n_views
  - canon-contrast vs raw-cos pool rank of the aggregate (canon sensitivity)

R3: Wilcoxon paired one-sided — oracle coherence < wrongtop1 coherence?

Output: output/diagnostics/stage3_2_competition.csv
        output/diagnostics/plots/stage3_2_coherence_paired.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import (load_replay, visible_views, baseline_aggregate,
                             ScenePool, make_vlm)


def side_stats(rec, pool, prompt):
    views = visible_views(rec)
    if not views:
        return None
    feats = np.stack([v['mixed_feat'].astype(np.float32) for v in views])
    w = np.array([v['portion'] for v in views])
    resultant = (feats.astype(np.float64) * w[:, None]).sum(axis=0)
    coherence = float(np.linalg.norm(resultant) / max(w.sum(), 1e-9))
    _, _, text = pool.prompt_scores(prompt)
    pcos = feats @ text
    agg = baseline_aggregate(views)
    rc, rr, _, _ = pool.ranks_batch(prompt, rec['oracle_lvl'], rec['oracle_sp_id'], agg[None])
    return {
        'n_views': len(views),
        'coherence': coherence,
        'pcos_mean': float(pcos.mean()), 'pcos_std': float(pcos.std()),
        'pcos_max': float(pcos.max()),
        'portion_sum': float(w.sum()), 'portion_mean': float(w.mean()),
        'agg_canon_rank': int(rc[0]), 'agg_raw_rank': int(rr[0]),
        'mean_pairwise_cos': float((feats @ feats.T).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle_pkl', default='output/diagnostics/stage3_b8_replay_perview.pkl')
    ap.add_argument('--wrong_pkl', default='output/diagnostics/stage3_2_wrongtop1_perview.pkl')
    ap.add_argument('--persistent_csv', default='output/diagnostics/persistent_phantoms_17.csv')
    ap.add_argument('--out_csv', default='output/diagnostics/stage3_2_competition.csv')
    ap.add_argument('--plot_dir', default='output/diagnostics/plots')
    args = ap.parse_args()

    oracle = load_replay(args.oracle_pkl)
    wrong = load_replay(args.wrong_pkl)
    ph = pd.read_csv(args.persistent_csv)
    phantom_keys = [(r['scene'], r['prompt']) for _, r in ph.iterrows()]

    vlm = make_vlm()
    pools = {}
    rows = []
    for (sc, prompt) in sorted(phantom_keys):
        if (sc, prompt) not in oracle or (sc, prompt) not in wrong:
            print(f"  [skip] missing dump for {sc}/{prompt}")
            continue
        if sc not in pools:
            pools[sc] = ScenePool(f"output/lerf/{sc}", vlm)
        o = side_stats(oracle[(sc, prompt)], pools[sc], prompt)
        wt = side_stats(wrong[(sc, prompt)], pools[sc], prompt)
        if o is None or wt is None:
            continue
        row = {'scene': sc, 'prompt': prompt,
               'wrong_fidelity': wrong[(sc, prompt)]['fidelity_cos']}
        row.update({f'oracle_{k}': v for k, v in o.items()})
        row.update({f'wrong_{k}': v for k, v in wt.items()})
        rows.append(row)
        print(f"  {sc:13s} {prompt:24s} coh O={o['coherence']:.3f} W={wt['coherence']:.3f} "
              f"| pcos O={o['pcos_mean']:.3f} W={wt['pcos_mean']:.3f} "
              f"| portion_sum O={o['portion_sum']:.1f} W={wt['portion_sum']:.1f} "
              f"| n_views O={o['n_views']} W={wt['n_views']}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("\n" + "=" * 78)
    print("R3 — paired tests (oracle vs wrong-top1)")
    print("=" * 78)
    for metric, alt, label in [
            ('coherence', 'less', 'oracle coherence < wrong (dispersion-defeat)'),
            ('pcos_mean', 'greater', 'oracle prompt-cos > wrong (semantic edge exists)'),
            ('portion_sum', 'less', 'oracle visibility mass < wrong (visibility-defeat)'),
            ('n_views', 'less', 'oracle seen in fewer views')]:
        a = df[f'oracle_{metric}'].values
        b = df[f'wrong_{metric}'].values
        try:
            w = stats.wilcoxon(a, b, alternative=alt)
            print(f"  {label:55s} O_med={np.median(a):8.3f} W_med={np.median(b):8.3f} "
                  f"p={w.pvalue:.4f} {'**' if w.pvalue < 0.05 else ''}")
        except ValueError as e:
            print(f"  {label}: {e}")

    canon_pen_o = df['oracle_agg_canon_rank'] - df['oracle_agg_raw_rank']
    print(f"\n  canon-contrast penalty (oracle agg): median canon-raw rank diff = "
          f"{np.median(canon_pen_o):.0f} (>0 = canon 이 더 깎음)")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df['oracle_coherence'], df['wrong_coherence'], s=70, color='tab:red')
    lim = [min(df['oracle_coherence'].min(), df['wrong_coherence'].min()) - 0.02,
           max(df['oracle_coherence'].max(), df['wrong_coherence'].max()) + 0.02]
    ax.plot(lim, lim, 'k--', lw=0.8)
    for _, r in df.iterrows():
        ax.annotate(r['prompt'][:12], (r['oracle_coherence'], r['wrong_coherence']),
                    fontsize=6, alpha=0.7)
    ax.set_xlabel('oracle SP coherence')
    ax.set_ylabel('wrong-top1 SP coherence')
    ax.set_title('B1.C — above diagonal = wrong-top1 more coherent')
    fig.tight_layout()
    os.makedirs(args.plot_dir, exist_ok=True)
    fig.savefig(os.path.join(args.plot_dir, 'stage3_2_coherence_paired.png'), dpi=120)
    print(f"\nWrote {args.out_csv}, plot.")


if __name__ == '__main__':
    main()
