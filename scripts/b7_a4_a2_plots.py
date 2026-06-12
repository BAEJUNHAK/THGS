"""
Generate distribution + stratification plots for B7, A4, A2.

Reads:
  - output/diagnostics/b7_a4_combined.csv  (208 rows, per (prompt, eval_frame))
  - output/diagnostics/a2_image_clip_ceiling.csv (268 rows, per (prompt, policy))

Saves PNG plots under output/diagnostics/plots/.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def fig_savetight(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----- B7 plots -----------------------------------------------------------
def plot_b7(df, out_dir):
    """Distributions of purity/completeness/fragmentation + stratification."""
    # Use ref-frame rows (1 per prompt) to avoid weighting by frame count
    ref = df[df["is_ref_frame"] == 1].copy()
    print(f"  B7: {len(ref)} unique (prompt, ref_frame) entries", flush=True)

    # 1. Purity / completeness / fragmentation distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(ref["oracle_purity_ref"], bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Oracle SP purity")
    axes[0].set_ylabel("# prompts")
    axes[0].set_title("B7-purity distribution")
    axes[0].axvline(0.5, color="red", linestyle="--", label="purity=0.5")
    axes[0].axvline(0.8, color="orange", linestyle="--", label="purity=0.8")
    axes[0].legend(fontsize=8)

    axes[1].hist(ref["oracle_completeness_ref"], bins=20, range=(0, 1), color="seagreen", edgecolor="white")
    axes[1].set_xlabel("Oracle SP completeness")
    axes[1].set_ylabel("# prompts")
    axes[1].set_title("B7-completeness distribution")
    axes[1].axvline(0.5, color="red", linestyle="--", label="comp=0.5")
    axes[1].legend(fontsize=8)

    axes[2].hist(ref["fragmentation"], bins=np.arange(0.5, 6.5, 1.0), color="darkorange", edgecolor="white")
    axes[2].set_xlabel("Fragmentation (SPs in greedy-union)")
    axes[2].set_ylabel("# prompts")
    axes[2].set_title("B7-fragmentation distribution")
    fig_savetight(fig, os.path.join(out_dir, "b7_distributions.png"))

    # 2. Stratified by scene
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax_idx, (col, label) in enumerate([
        ("oracle_purity_ref", "Oracle purity"),
        ("oracle_completeness_ref", "Oracle completeness"),
        ("fragmentation", "Fragmentation"),
    ]):
        for scene_name, g in ref.groupby("scene"):
            axes[ax_idx].scatter(
                g[col] + (np.random.RandomState(42).uniform(-0.02, 0.02, len(g)) if col == "fragmentation" else 0),
                np.full(len(g), {"figurines": 0, "ramen": 1, "teatime": 2, "waldo_kitchen": 3}[scene_name])
                + np.random.RandomState(0).uniform(-0.1, 0.1, len(g)),
                alpha=0.6, s=30, label=scene_name)
        axes[ax_idx].set_xlabel(label)
        axes[ax_idx].set_yticks([0, 1, 2, 3])
        axes[ax_idx].set_yticklabels(["figurines", "ramen", "teatime", "waldo_kitchen"])
        axes[ax_idx].set_title(f"{label} by scene")
        axes[ax_idx].grid(True, axis="x", alpha=0.3)
    fig_savetight(fig, os.path.join(out_dir, "b7_by_scene.png"))

    # 3. Purity vs Completeness scatter (per prompt, ref-frame)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"figurines": "tab:blue", "ramen": "tab:orange",
              "teatime": "tab:green", "waldo_kitchen": "tab:red"}
    for sc, g in ref.groupby("scene"):
        ax.scatter(g["oracle_purity_ref"], g["oracle_completeness_ref"],
                   c=colors[sc], label=sc, alpha=0.7, s=40)
    ax.set_xlabel("Purity = |SP∩GT|/|SP|")
    ax.set_ylabel("Completeness = |SP∩GT|/|GT|")
    ax.set_title("Oracle SP purity vs completeness (per prompt @ ref_frame)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_savetight(fig, os.path.join(out_dir, "b7_purity_vs_completeness.png"))

    # 4. Cross-view stability of oracle SP
    # For prompts with >1 frame, see how purity/completeness varies across frames
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    multi = df.groupby(["scene", "prompt"])["frame"].count().reset_index(name="n_frames")
    multi = multi[multi["n_frames"] > 1].set_index(["scene", "prompt"])
    rows_multi = df.set_index(["scene", "prompt"]).join(multi, how="inner").reset_index()
    for col, ax, label in [
        ("oracle_purity_eval", axes[0], "Per-frame purity (eval)"),
        ("oracle_completeness_eval", axes[1], "Per-frame completeness (eval)"),
    ]:
        agg = rows_multi.groupby(["scene", "prompt"])[col].agg(["mean", "std", "min", "max"]).reset_index()
        ax.errorbar(range(len(agg)), agg["mean"], yerr=[agg["mean"] - agg["min"], agg["max"] - agg["mean"]],
                    fmt="o", alpha=0.6, capsize=3)
        ax.set_xlabel("prompt index (multi-frame only)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: mean + [min, max] range")
        ax.grid(True, alpha=0.3)
    fig_savetight(fig, os.path.join(out_dir, "b7_cross_view_stability.png"))


# ----- A4 plots -----------------------------------------------------------
def plot_a4(df, out_dir):
    ref = df[df["is_ref_frame"] == 1].copy()
    # 1. Rank, z-margin, percentile-margin distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(ref["oracle_rank"], bins=30, color="steelblue", edgecolor="white")
    axes[0, 0].set_xlabel("Oracle CLIP rank")
    axes[0, 0].set_ylabel("# prompts")
    axes[0, 0].set_title(f"A4-oracle rank (median={ref['oracle_rank'].median():.0f}, "
                         f"75th={ref['oracle_rank'].quantile(0.75):.0f})")
    axes[0, 0].set_yscale("log")

    axes[0, 1].hist(ref["raw_margin"], bins=30, color="seagreen", edgecolor="white")
    axes[0, 1].set_xlabel("Raw margin = cos(top)−cos(oracle)")
    axes[0, 1].set_ylabel("# prompts")
    axes[0, 1].set_title("A4-raw margin (within-prompt only)")

    axes[1, 0].hist(ref["z_margin"], bins=30, color="darkorange", edgecolor="white")
    axes[1, 0].set_xlabel("Z-margin = (cos_top − cos_oracle)/std_pool")
    axes[1, 0].set_ylabel("# prompts")
    axes[1, 0].set_title("A4-z-margin (cross-prompt comparable)")
    axes[1, 0].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="z=0.5")
    axes[1, 0].axvline(1.5, color="orange", linestyle="--", alpha=0.5, label="z=1.5")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].hist(ref["percentile_margin"], bins=30, color="purple", edgecolor="white")
    axes[1, 1].set_xlabel("Percentile margin (0=top, 100=bottom)")
    axes[1, 1].set_ylabel("# prompts")
    axes[1, 1].set_title("A4-percentile margin (scale-free)")
    fig_savetight(fig, os.path.join(out_dir, "a4_distributions.png"))

    # 2. Rank vs z-margin scatter — D2 sub-classification heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(ref["oracle_rank"], ref["z_margin"],
                    c=ref["oracle_purity_ref"], cmap="RdYlGn",
                    alpha=0.7, s=60, edgecolors="black", linewidths=0.3, vmin=0, vmax=1)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Oracle purity")
    ax.set_xlabel("Oracle CLIP rank (log scale)")
    ax.set_ylabel("Z-margin (cross-prompt)")
    ax.set_xscale("log")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.axhline(1.5, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(3.5, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(10.5, color="grey", linestyle=":", alpha=0.5)
    ax.set_title("A4 rank × z-margin (color = oracle purity)")
    fig_savetight(fig, os.path.join(out_dir, "a4_rank_zmargin_scatter.png"))

    # 3. By-scene rank box-plot
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [ref[ref["scene"] == sc]["oracle_rank"].values
            for sc in ["figurines", "ramen", "teatime", "waldo_kitchen"]]
    ax.boxplot(data, tick_labels=["figurines", "ramen", "teatime", "waldo_kitchen"], showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("Oracle CLIP rank (log)")
    ax.set_title("A4 oracle rank by scene")
    ax.grid(True, axis="y", alpha=0.3)
    fig_savetight(fig, os.path.join(out_dir, "a4_rank_by_scene.png"))


# ----- A2 plots -----------------------------------------------------------
def plot_a2(df, out_dir):
    # 1. Image-CLIP rank distribution per policy
    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    for i, policy in enumerate(["tight", "mask", "context", "method"]):
        d = df[df["policy"] == policy]
        ranks = pd.to_numeric(d["raw_rank_true"], errors="coerce").dropna()
        axes[i].hist(ranks, bins=range(1, int(ranks.max()) + 2),
                     color="steelblue", edgecolor="white")
        axes[i].set_xlabel("Image-CLIP rank of true prompt")
        axes[i].set_title(f"{policy} — top1: {(ranks == 1).sum()}/{len(ranks)} "
                          f"({(ranks == 1).mean()*100:.1f}%)")
        axes[i].set_yscale("log")
    axes[0].set_ylabel("# prompts (log)")
    fig_savetight(fig, os.path.join(out_dir, "a2_rank_per_policy.png"))

    # 2. Top-1 hit rate per policy per scene
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    scenes = sorted(df["scene"].unique())
    policies = ["tight", "mask", "context", "method"]
    x = np.arange(len(scenes))
    for i, policy in enumerate(policies):
        rates = []
        for sc in scenes:
            d = df[(df["scene"] == sc) & (df["policy"] == policy)]
            r = pd.to_numeric(d["raw_rank_true"], errors="coerce").dropna()
            rates.append(0.0 if len(r) == 0 else (r == 1).mean() * 100)
        ax.bar(x + (i - 1.5) * width, rates, width, label=policy)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("Top-1 hit rate (%)")
    ax.set_title("A2 image-CLIP top-1 hit rate × policy × scene")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig_savetight(fig, os.path.join(out_dir, "a2_top1_per_scene_policy.png"))

    # 3. Median rank per policy
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = []
    for policy in policies:
        d = df[df["policy"] == policy]
        ranks = pd.to_numeric(d["raw_rank_true"], errors="coerce").dropna()
        box_data.append(ranks.values)
    ax.boxplot(box_data, tick_labels=policies, showfliers=True)
    ax.set_ylabel("Image-CLIP rank")
    ax.set_yscale("log")
    ax.set_title("A2 rank distribution per crop policy")
    ax.grid(True, axis="y", alpha=0.3)
    fig_savetight(fig, os.path.join(out_dir, "a2_rank_box.png"))


# ----- Joint plot: A2-encoded limit vs B7/A4 ranks -----------------------
def plot_joint(b7a4, a2, out_dir):
    """Identify D2.real-encoder-limit: image-CLIP fails AND SP-CLIP rank > 3."""
    ref = b7a4[b7a4["is_ref_frame"] == 1].copy()
    # take A2 with "mask" policy as the ceiling
    a2_mask = a2[a2["policy"] == "mask"][["scene", "prompt", "raw_rank_true"]].rename(
        columns={"raw_rank_true": "a2_mask_rank"})
    a2_mask["a2_mask_rank"] = pd.to_numeric(a2_mask["a2_mask_rank"], errors="coerce")
    joined = ref.merge(a2_mask, on=["scene", "prompt"], how="left")

    fig, ax = plt.subplots(figsize=(8, 7))
    # quadrant logic:
    #   A2 rank > 3 AND SP rank > 3: D2.real (encoder limit)
    #   A2 rank ≤ 3 AND SP rank > 3: D2.phantom candidate (aggregation problem)
    #   A2 rank > 3 AND SP rank ≤ 3: ?? rare
    #   A2 rank ≤ 3 AND SP rank ≤ 3: easy/success
    a2r = joined["a2_mask_rank"].fillna(99).astype(float)
    spr = joined["oracle_rank"].astype(float)

    colors = []
    labels_data = []
    for a, s in zip(a2r, spr):
        if a > 3 and s > 3:
            colors.append("tab:red"); labels_data.append("D2.real candidate")
        elif a <= 3 and s > 3:
            colors.append("tab:orange"); labels_data.append("D2.phantom candidate")
        elif a > 3 and s <= 3:
            colors.append("tab:purple"); labels_data.append("rare: SP > A2")
        else:
            colors.append("tab:green"); labels_data.append("easy")

    ax.scatter(a2r, spr, c=colors, alpha=0.7, s=50, edgecolors="black", linewidths=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("A2 image-CLIP rank (mask crop)")
    ax.set_ylabel("Oracle SP CLIP rank")
    ax.axhline(3.5, color="grey", linestyle=":", alpha=0.5)
    ax.axvline(3.5, color="grey", linestyle=":", alpha=0.5)
    # legend manually
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(color="tab:red", label="D2.real candidate (encoder limit)"),
        Patch(color="tab:orange", label="D2.phantom candidate (aggregation)"),
        Patch(color="tab:green", label="easy (both ≤ 3)"),
        Patch(color="tab:purple", label="rare (SP top-3 but A2 fails)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9)
    ax.set_title("D2.real vs D2.phantom decomposition\n(A2 ceiling × SP rank)")
    ax.grid(True, alpha=0.3)
    fig_savetight(fig, os.path.join(out_dir, "joint_d2real_vs_phantom.png"))

    # Print breakdown
    print(f"\n=== D2.real vs D2.phantom breakdown ===", flush=True)
    counts = {}
    for k in ["D2.real candidate", "D2.phantom candidate", "easy", "rare"]:
        counts[k] = labels_data.count(k)
    for k, v in counts.items():
        print(f"  {k}: {v}/{len(labels_data)} ({v/len(labels_data)*100:.1f}%)", flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b7_a4_csv", default="output/diagnostics/b7_a4_combined.csv")
    parser.add_argument("--a2_csv", default="output/diagnostics/a2_image_clip_ceiling.csv")
    parser.add_argument("--out_dir", default="output/diagnostics/plots")
    args = parser.parse_args()
    ensure_dir(args.out_dir)

    print(f"Loading {args.b7_a4_csv}", flush=True)
    b7a4 = pd.read_csv(args.b7_a4_csv)
    print(f"Loading {args.a2_csv}", flush=True)
    a2 = pd.read_csv(args.a2_csv)

    plot_b7(b7a4, args.out_dir)
    plot_a4(b7a4, args.out_dir)
    plot_a2(a2, args.out_dir)
    counts = plot_joint(b7a4, a2, args.out_dir)

    print(f"\nAll plots in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
