"""
Stage 3.2 / B1.B — aggregation ablation on the same per-view feature sets
(counterfactual fixes, no pipeline re-run).

Variants per target SP (visible views, mixed features from the B8 replay):
  (a) uniform        : mean(normalize(f_i))
  (b) visweight      : sum(normalize(f_i) * portion_i)        [pipeline baseline]
  (c) topk_portion_K : visibility-weighted mean over top-K views by portion
  (d) topk_query_K   : visibility-weighted mean over top-K views by raw cos to
                       the prompt text  ** QUERY-CONDITIONED — upper-bound
                       diagnostic (uses the query); not a training-free method
                       unless made query-time **
  (e) mode_cluster   : GMM(k=2, 512D float32) on views; visibility-weighted
                       mean of the MAJORITY cluster (first real implementation
                       of hypothesis 6.1's feature-space GMM). N<5 -> baseline.
  (f) rofa2          : ROFA keep-mask (tau=2, mean pairwise cos) then
                       visibility-weighted mean of kept views.

Metric: canon-contrast pool rank (entry-replacement, same as B1.A).
  - phantom recovered  : rank <= 3
  - easy regression    : easy whose baseline rank <= 3 but variant rank > 3

Output: output/diagnostics/stage3_2_ablation.csv
        output/diagnostics/plots/stage3_2_ablation_recovery.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import load_replay, visible_views, ScenePool, make_vlm

K_SET = [1, 3, 5, 10]


def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def agg_weighted(feats, weights):
    acc = (feats.astype(np.float64) * np.asarray(weights)[:, None]).sum(axis=0)
    return norm(acc).astype(np.float32)


def variants(views, text_feat):
    feats = np.stack([v['mixed_feat'].astype(np.float32) for v in views])
    portions = np.array([v['portion'] for v in views])
    out = {}
    out['a_uniform'] = agg_weighted(feats, np.ones(len(views)))
    out['b_visweight'] = agg_weighted(feats, portions)
    for k in K_SET:
        idx = np.argsort(-portions)[:k]
        out[f'c_topk_portion_{k}'] = agg_weighted(feats[idx], portions[idx])
    qcos = feats @ text_feat
    for k in K_SET:
        idx = np.argsort(-qcos)[:k]
        out[f'd_topk_query_{k}'] = agg_weighted(feats[idx], portions[idx])
    if len(views) >= 5:
        gm = GaussianMixture(n_components=2, covariance_type='diag',
                             random_state=0, n_init=2).fit(feats.astype(np.float64))
        lab = gm.predict(feats.astype(np.float64))
        maj = int(np.bincount(lab).argmax())
        sel = lab == maj
        out['e_mode_cluster'] = agg_weighted(feats[sel], portions[sel])
        out['_e_majority_frac'] = float(sel.mean())
    else:
        out['e_mode_cluster'] = out['b_visweight']
        out['_e_majority_frac'] = np.nan
    cos = feats @ feats.T
    n = len(views)
    if n > 1:
        mean_sim = (cos.sum(axis=1) - 1) / (n - 1)
        keep = mean_sim > (mean_sim.mean() - 2.0 * mean_sim.std())
        if not keep.any():
            keep[np.argmax(mean_sim)] = True
    else:
        keep = np.ones(1, dtype=bool)
    out['f_rofa2'] = agg_weighted(feats[keep], portions[keep])
    out['_f_n_dropped'] = int((~keep).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay_pkl', default='output/diagnostics/stage3_b8_replay_perview.pkl')
    ap.add_argument('--out_csv', default='output/diagnostics/stage3_2_ablation.csv')
    ap.add_argument('--plot_dir', default='output/diagnostics/plots')
    args = ap.parse_args()

    replay = load_replay(args.replay_pkl)
    vlm = make_vlm()
    pools = {}
    rows = []

    for (sc, prompt), rec in sorted(replay.items()):
        if sc not in pools:
            pools[sc] = ScenePool(f"output/lerf/{sc}", vlm)
        pool = pools[sc]
        lvl, sp = rec['oracle_lvl'], rec['oracle_sp_id']
        views = visible_views(rec)
        if not views:
            continue
        _, _, text = pool.prompt_scores(prompt)
        vs = variants(views, text)
        names = [k for k in vs if not k.startswith('_')]
        feats = np.stack([vs[k] for k in names])
        rc, rr, _, _ = pool.ranks_batch(prompt, lvl, sp, feats)
        row = {'scene': sc, 'prompt': prompt, 'category': rec['category'],
               'n_views': len(views),
               'e_majority_frac': vs['_e_majority_frac'],
               'f_n_dropped': vs['_f_n_dropped']}
        for nme, c, r in zip(names, rc, rr):
            row[f'rank_{nme}'] = int(c)
            row[f'rawrank_{nme}'] = int(r)
        rows.append(row)
        print(f"  {rec['category'][:7]:7s} {sc:13s} {prompt:24s} " +
              " ".join(f"{nme.split('_')[0]}{nme.split('_')[-1] if nme[0] in 'cd' else ''}="
                       f"{int(c):3d}" for nme, c in zip(names, rc)), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    ph = df[df['category'] == 'phantom17']
    ez = df[df['category'] == 'easy_sample']
    ez_base_ok = ez[ez['rank_b_visweight'] <= 3]
    names = [c[5:] for c in df.columns if c.startswith('rank_')]

    print("\n" + "=" * 78)
    print(f"R2 TABLE — phantom recovered (rank<=3, n={len(ph)}) / easy regression "
          f"(baseline<=3 -> >3, n={len(ez_base_ok)})")
    print("=" * 78)
    summary = []
    for nme in names:
        rec_n = int((ph[f'rank_{nme}'] <= 3).sum())
        reg_n = int((ez_base_ok[f'rank_{nme}'] > 3).sum())
        qflag = ' [QUERY-COND]' if nme.startswith('d_') else ''
        summary.append((nme, rec_n, reg_n))
        print(f"  {nme:18s} recovered={rec_n:2d}/17  easy_regression={reg_n:2d}{qflag}")

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = np.arange(len(summary))
    ax.bar(xs - 0.2, [s[1] for s in summary], width=0.4, label='phantom recovered (<=3)',
           color='tab:green')
    ax.bar(xs + 0.2, [s[2] for s in summary], width=0.4, label='easy regression',
           color='tab:red')
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in summary], rotation=45, ha='right', fontsize=8)
    ax.axhline(6, color='green', ls='--', lw=0.8)
    ax.axhline(1, color='red', ls='--', lw=0.8)
    ax.set_title('B1.B — aggregation ablation (pool rank, canon-contrast)')
    ax.legend()
    fig.tight_layout()
    os.makedirs(args.plot_dir, exist_ok=True)
    fig.savefig(os.path.join(args.plot_dir, 'stage3_2_ablation_recovery.png'), dpi=120)
    print(f"\nWrote {args.out_csv}, plot.")


if __name__ == '__main__':
    main()
