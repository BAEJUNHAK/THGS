"""
Stage 3.2 / B1.A — single-view pool ranks (true A3-postB8) + cumulative trajectory.

Per target SP (17 phantom + 16 easy):
  1. FIDELITY GATE: visibility-weighted baseline reconstructed from the replay
     pkl -> pool rank must match b7_a4's oracle_rank (+/-2).
  2. Single-view pool rank: each view's mixed feature ALONE replacing the SP's
     pool entry -> canon-contrast rank (and raw-cos rank).
  3. Cumulative trajectory: accumulate normalize(f_i)*portion_i in pipeline
     view order; rank after each k.
  4. Curve classification:
     - never_good : best single-view canon rank > 3
     - gradual    : best single <= 3, trajectory degrades without a single
                    dominant drop (no step contributing > 50% of total loss)
     - sudden     : best single <= 3 and one accumulation step contributes
                    > 50% of the total rank loss

Outputs:
  output/diagnostics/stage3_2_singleview_rank.csv  (per SP)
  output/diagnostics/stage3_2_trajectory.csv       (per SP per k)
  output/diagnostics/plots/stage3_2_trajectory_curves.png
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import (load_replay, visible_views, baseline_aggregate,
                             ScenePool, make_vlm)


def cum_aggregates(views):
    """Cumulative normalized aggregates after each visible view (pipeline order)."""
    acc = np.zeros(512, dtype=np.float64)
    out = []
    for v in sorted(views, key=lambda x: x['view_idx']):
        acc += v['mixed_feat'].astype(np.float64) * v['portion']
        n = np.linalg.norm(acc)
        out.append((acc / n if n > 0 else acc).astype(np.float32))
    return np.stack(out, axis=0)


def classify_curve(best_single, traj_ranks):
    if best_single > 3:
        return 'never_good_in_pool'
    if len(traj_ranks) < 2:
        return 'insufficient'
    best_cum = int(np.min(traj_ranks))
    final = int(traj_ranks[-1])
    loss = final - best_cum
    if loss <= 0:
        return 'no_degradation'
    steps = np.diff(traj_ranks.astype(float))
    worst_step = float(np.max(steps)) if len(steps) else 0.0
    return 'sudden_drop' if worst_step > 0.5 * loss else 'gradual_dilution'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay_pkl', default='output/diagnostics/stage3_b8_replay_perview.pkl')
    ap.add_argument('--b7_csv', default='output/diagnostics/b7_a4_combined.csv')
    ap.add_argument('--out_single', default='output/diagnostics/stage3_2_singleview_rank.csv')
    ap.add_argument('--out_traj', default='output/diagnostics/stage3_2_trajectory.csv')
    ap.add_argument('--plot_dir', default='output/diagnostics/plots')
    args = ap.parse_args()

    replay = load_replay(args.replay_pkl)
    b7 = pd.read_csv(args.b7_csv)
    ref = b7[b7['is_ref_frame'] == 1]
    a4_rank = {(r['scene'], r['prompt']): int(r['oracle_rank']) for _, r in ref.iterrows()}

    vlm = make_vlm()
    pools = {}

    single_rows, traj_rows = [], []
    gate_pass = gate_total = 0
    print("=" * 90)
    print("FIDELITY GATE — baseline pool rank vs b7_a4 oracle_rank (tolerance +/-2)")
    print("=" * 90)

    for (sc, prompt), rec in sorted(replay.items()):
        if sc not in pools:
            pools[sc] = ScenePool(f"output/lerf/{sc}", vlm)
        pool = pools[sc]
        lvl, sp = rec['oracle_lvl'], rec['oracle_sp_id']
        views = visible_views(rec)
        if not views:
            continue

        # --- gate
        base = baseline_aggregate(views)
        rc, rr, _, _ = pool.ranks_batch(prompt, lvl, sp, base[None])
        expect = a4_rank.get((sc, prompt), -1)
        ok = abs(int(rc[0]) - expect) <= 2
        gate_total += 1
        gate_pass += int(ok)
        print(f"  {rec['category'][:7]:7s} {sc:13s} {prompt:24s} baseline_rank={int(rc[0]):4d} "
              f"a4={expect:4d} {'PASS' if ok else '** FAIL **'}")

        # --- single-view ranks
        feats = np.stack([v['mixed_feat'].astype(np.float32) for v in views])
        s_canon, s_raw, _, _ = pool.ranks_batch(prompt, lvl, sp, feats)

        # --- cumulative trajectory
        cums = cum_aggregates(views)
        t_canon, t_raw, t_cscore, _ = pool.ranks_batch(prompt, lvl, sp, cums)

        order = sorted(views, key=lambda x: x['view_idx'])
        for k, (v, tc, tr) in enumerate(zip(order, t_canon, t_raw), start=1):
            traj_rows.append({'scene': sc, 'prompt': prompt, 'category': rec['category'],
                              'k': k, 'view_idx': v['view_idx'], 'portion': v['portion'],
                              'cum_canon_rank': int(tc), 'cum_raw_rank': int(tr)})

        best_single = int(s_canon.min())
        cls = classify_curve(best_single, t_canon)
        single_rows.append({
            'scene': sc, 'prompt': prompt, 'category': rec['category'],
            'oracle_lvl': lvl, 'oracle_sp_id': sp, 'n_views': len(views),
            'gate_baseline_rank': int(rc[0]), 'gate_a4_rank': expect, 'gate_pass': ok,
            'single_best_canon': best_single,
            'single_median_canon': float(np.median(s_canon)),
            'single_frac_le3': float((s_canon <= 3).mean()),
            'single_best_raw': int(s_raw.min()),
            'single_median_raw': float(np.median(s_raw)),
            'traj_best_canon': int(t_canon.min()),
            'traj_final_canon': int(t_canon[-1]),
            'curve_class': cls,
        })
        print(f"      single: best={best_single} med={np.median(s_canon):.0f} "
              f"%<=3={(s_canon <= 3).mean()*100:.0f}%  traj: best={int(t_canon.min())} "
              f"final={int(t_canon[-1])}  class={cls}", flush=True)

    sdf = pd.DataFrame(single_rows)
    tdf = pd.DataFrame(traj_rows)
    os.makedirs(os.path.dirname(args.out_single), exist_ok=True)
    sdf.to_csv(args.out_single, index=False)
    tdf.to_csv(args.out_traj, index=False)

    print(f"\nGATE: {gate_pass}/{gate_total} pass")
    ph = sdf[sdf['category'] == 'phantom17']
    print("\n=== R1 inputs ===")
    print(f"phantoms with best SINGLE-view pool rank <= 3 : "
          f"{int((ph['single_best_canon'] <= 3).sum())}/{len(ph)} "
          f"({(ph['single_best_canon'] <= 3).mean()*100:.0f}%)")
    print(f"phantoms with best CUMULATIVE rank <= 3       : "
          f"{int((ph['traj_best_canon'] <= 3).sum())}/{len(ph)}")
    print(f"phantoms final rank <= 3                      : "
          f"{int((ph['traj_final_canon'] <= 3).sum())}/{len(ph)}")
    print("\ncurve classes (phantom):")
    print(ph['curve_class'].value_counts().to_string())

    # plot trajectories
    fig, axes = plt.subplots(3, 6, figsize=(22, 10), sharex=False)
    axes = axes.flatten()
    for i, (_, r) in enumerate(ph.iterrows()):
        if i >= len(axes):
            break
        d = tdf[(tdf['scene'] == r['scene']) & (tdf['prompt'] == r['prompt'])]
        ax = axes[i]
        ax.plot(d['k'], d['cum_canon_rank'], color='tab:red', lw=1.5, label='cumulative')
        ax.axhline(3, color='green', ls='--', lw=0.8)
        ax.axhline(r['single_best_canon'], color='tab:blue', ls=':', lw=1, label='best single')
        ax.set_yscale('log')
        ax.set_title(f"{r['scene'][:4]}/{r['prompt'][:16]}\n[{r['curve_class']}]", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(ph), len(axes)):
        axes[j].axis('off')
    fig.suptitle("B1.A — cumulative pool-rank trajectories (17 phantoms, log scale)")
    fig.tight_layout()
    os.makedirs(args.plot_dir, exist_ok=True)
    fig.savefig(os.path.join(args.plot_dir, 'stage3_2_trajectory_curves.png'), dpi=120)
    print(f"\nWrote {args.out_single}, {args.out_traj}, plot.")


if __name__ == '__main__':
    main()
