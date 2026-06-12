"""
Stage 2B — F2.B + F2.C: ROFA subtype classification + keep_mask analysis.

For each 17 phantom, using the H2 lite per-view features:
  1. Simulate ROFA: compute mean_sim per view, derive keep_mask with tau=1.0
  2. Classify subtype using feature-space statistics:
     - outlier_phantom    : keep_mask drops ≥ 1 view AND remaining variance low
     - bimodal_balanced   : 2-component GMM BIC < 1-component AND minority cluster ≥ 30%
                          AND mean_sim std < tau * overall_std (ROFA-blind condition)
     - mean_dilution      : 1-component GMM AND mean direction is far from prompt
                          (all views ~consistent but in wrong direction)
  3. Analyze keep_mask geometry: are kept views the ones where SP looks like prompt?
     - kept_cos_mean vs dropped_cos_mean
     - If dropped views have HIGHER cos with prompt → ROFA dropped the good views
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


TAU = 1.0  # ROFA default


def simulate_rofa(features, tau=TAU):
    """Return (keep_mask, mean_sim_per_view, mu, sigma)."""
    N = features.shape[0]
    if N == 1:
        return np.array([True]), np.array([1.0]), 1.0, 0.0
    cos_sim = features @ features.T  # (N, N)
    mean_sim = (cos_sim.sum(axis=1) - 1) / (N - 1)
    mu = mean_sim.mean()
    sigma = mean_sim.std()
    keep_mask = mean_sim > (mu - tau * sigma)
    return keep_mask, mean_sim, mu, sigma


def classify_subtype(features, cos_with_prompt, kept_mask, mean_sim,
                      mu, sigma, prompt_threshold=0.05):
    """Classify phantom by cos-distribution and ROFA keep pattern (simpler than GMM in 512D).

    Returns (subtype, cos_q25, cos_q75, n_dropped, notes).
    """
    N = features.shape[0]
    if N < 5:
        return 'insufficient_evidence', np.nan, np.nan, 0, 'N<5'

    cos_mean = float(np.mean(cos_with_prompt))
    cos_std = float(np.std(cos_with_prompt))
    cos_q25 = float(np.percentile(cos_with_prompt, 25))
    cos_q75 = float(np.percentile(cos_with_prompt, 75))
    cos_iqr = cos_q75 - cos_q25
    n_dropped = int((~kept_mask).sum())

    kept_cos_mean = float(cos_with_prompt[kept_mask].mean()) if kept_mask.any() else np.nan
    dropped_cos_mean = float(cos_with_prompt[~kept_mask].mean()) if (~kept_mask).any() else np.nan

    # Bimodal — cos distribution has wide spread (IQR > 0.06) AND clear gap
    if cos_iqr > 0.06 and cos_std > 0.04:
        # check if ROFA filter is blind (kept set still has spread)
        kept_std = float(np.std(cos_with_prompt[kept_mask])) if kept_mask.sum() > 1 else 0.0
        if kept_std > 0.03:
            notes = f'cos_iqr={cos_iqr:.3f} kept_std={kept_std:.3f}'
            return 'bimodal_balanced', cos_q25, cos_q75, n_dropped, notes
        # ROFA caught the high-low split
        if not np.isnan(dropped_cos_mean) and dropped_cos_mean < kept_cos_mean - 0.04:
            return 'outlier_handled', cos_q25, cos_q75, n_dropped, 'ROFA dropped low-cos views'
        return 'outlier_partial', cos_q25, cos_q75, n_dropped, ''

    # Mean dilution: low signal across all views (CLIP doesn't recognize prompt in clean crops)
    if cos_mean < 0.20:
        return 'mean_dilution', cos_q25, cos_q75, n_dropped, f'cos_mean={cos_mean:.2f} low'

    # Strong signal in CLIP space — phantom must be from elsewhere (within-view mixing, SAM, etc.)
    return 'strong_signal_phantom', cos_q25, cos_q75, n_dropped, f'cos_mean={cos_mean:.2f} but SP-CLIP phantom'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_pkl", default="output/diagnostics/_h2_lite_perview.pkl")
    parser.add_argument("--out_csv", default="output/diagnostics/stage2b_rofa_subtypes.csv")
    parser.add_argument("--out_keep_csv", default="output/diagnostics/stage2b_rofa_keep_mask.csv")
    parser.add_argument("--out_dir", default="output/diagnostics/plots")
    args = parser.parse_args()

    with open(args.in_pkl, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} phantoms from {args.in_pkl}")

    rows = []
    keep_rows = []
    for (sc, prompt), views in data.items():
        if not views:
            continue
        features = np.stack([v['feature'] for v in views], axis=0)
        cos_with_prompt = np.array([v['cos_with_prompt'] for v in views])
        n_views = len(views)

        kept, mean_sim, mu, sigma = simulate_rofa(features, tau=TAU)
        n_dropped_rofa = int((~kept).sum())
        subtype, cos_q25, cos_q75, n_dropped, notes = classify_subtype(
            features, cos_with_prompt, kept, mean_sim, mu, sigma)

        # keep_mask analysis: cos with prompt for kept vs dropped views
        kept_cos_mean = float(cos_with_prompt[kept].mean()) if kept.any() else float('nan')
        dropped_cos_mean = float(cos_with_prompt[~kept].mean()) if (~kept).any() else float('nan')
        rofa_pathology = (
            'ROFA_dropped_good_views' if not np.isnan(dropped_cos_mean) and dropped_cos_mean > kept_cos_mean
            else 'ROFA_kept_good_views' if not np.isnan(dropped_cos_mean) and dropped_cos_mean <= kept_cos_mean
            else 'no_drops')

        rows.append({
            'scene': sc, 'prompt': prompt,
            'n_views': n_views,
            'cos_mean': float(cos_with_prompt.mean()),
            'cos_min': float(cos_with_prompt.min()),
            'cos_max': float(cos_with_prompt.max()),
            'cos_std': float(cos_with_prompt.std()),
            'cos_q25': float(cos_q25),
            'cos_q75': float(cos_q75),
            'mean_sim_mu': float(mu), 'mean_sim_sigma': float(sigma),
            'n_dropped_by_rofa': n_dropped_rofa,
            'subtype': subtype,
            'notes': notes,
        })
        keep_rows.append({
            'scene': sc, 'prompt': prompt,
            'n_views': n_views, 'n_dropped': n_dropped,
            'kept_cos_mean': kept_cos_mean,
            'dropped_cos_mean': dropped_cos_mean,
            'cos_delta_kept_minus_dropped': (
                kept_cos_mean - dropped_cos_mean if not np.isnan(dropped_cos_mean) else 0.0),
            'rofa_pathology': rofa_pathology,
        })
        print(f"  {sc:14s} {prompt:25s} N={n_views:2d} subtype={subtype:22s} "
              f"n_dropped={n_dropped} kept_cos={kept_cos_mean:.3f} "
              f"dropped_cos={dropped_cos_mean if not np.isnan(dropped_cos_mean) else 'N/A':>7} "
              f"pathology={rofa_pathology}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    pd.DataFrame(keep_rows).to_csv(args.out_keep_csv, index=False)
    print(f"\nWrote {len(df)} rows to {args.out_csv}")
    print(f"Wrote {len(keep_rows)} rows to {args.out_keep_csv}")

    # Summary
    print("\n=== Subtype distribution ===")
    print(df['subtype'].value_counts())
    print()
    print("=== ROFA pathology distribution ===")
    print(pd.DataFrame(keep_rows)['rofa_pathology'].value_counts())

    # Plot 1 — subtype distribution
    os.makedirs(args.out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['subtype'].value_counts()
    colors_map = {
        'outlier': 'tab:green',
        'bimodal_balanced': 'tab:red',
        'mean_dilution': 'tab:orange',
        'insufficient_evidence': 'tab:gray',
        'compact_no_dilution': 'tab:blue',
        'undetermined': 'tab:purple',
    }
    colors = [colors_map.get(s, 'tab:gray') for s in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("# phantoms (of 17)")
    ax.set_title("F2 ROFA subtype distribution")
    ax.tick_params(axis='x', rotation=20)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha='center', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "stage2b_rofa_subtype_distribution.png"), dpi=120)
    plt.close(fig)

    # Plot 2 — keep_mask geometry
    fig, ax = plt.subplots(figsize=(9, 5))
    kdf = pd.DataFrame(keep_rows)
    valid = kdf[~kdf['dropped_cos_mean'].isna()]
    ax.bar(range(len(valid)), valid['kept_cos_mean'].values, width=0.4,
           label='kept views cos_mean', color='tab:blue', alpha=0.7)
    ax.bar(np.arange(len(valid)) + 0.4, valid['dropped_cos_mean'].values, width=0.4,
           label='dropped views cos_mean', color='tab:red', alpha=0.7)
    ax.set_xticks(np.arange(len(valid)) + 0.2)
    ax.set_xticklabels([f"{r['scene'][:4]}/{r['prompt']}" for _, r in valid.iterrows()],
                       rotation=90, fontsize=8)
    ax.set_ylabel("cos with prompt")
    ax.set_title("F2.C — ROFA keep_mask analysis: kept vs dropped views")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "stage2b_keep_mask_geometry.png"), dpi=120)
    plt.close(fig)

    # Plot 3 — per-view cos distribution per phantom
    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.flatten()
    for i, (k, v) in enumerate(data.items()):
        if i >= 20:
            break
        ax = axes[i]
        sc, prompt = k
        cos = np.array([d['cos_with_prompt'] for d in v])
        ax.hist(cos, bins=12, color='tab:blue', edgecolor='white')
        # mark mean_sim mu line
        features = np.stack([d['feature'] for d in v], axis=0)
        kept, mean_sim, mu, sigma = simulate_rofa(features, tau=TAU)
        ax.axvline(cos.mean(), color='black', linewidth=2, label=f"mean={cos.mean():.2f}")
        ax.set_title(f"{sc[:4]}/{prompt}\nN={len(v)} cos∈[{cos.min():.2f},{cos.max():.2f}]", fontsize=9)
        ax.tick_params(labelsize=7)
    for i in range(len(data), 20):
        axes[i].axis('off')
    fig.suptitle("Per-view cos with prompt (17 phantoms)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "stage2b_per_view_cos_dists.png"), dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
