"""
Cross-method B7+A4+A2 comparison: THGS vs ReLaGS.

Reads:
  output/diagnostics/b7_a4_combined.csv         (THGS)
  output/diagnostics/b7_a4_combined_relags.csv  (ReLaGS)
  output/diagnostics/a2_image_clip_ceiling.csv  (method-agnostic, shared)

Produces:
  - cross_method_d2_decomposition.csv  (per-prompt class for both methods + transition)
  - plots/cross_method_*.png

Print summary of transition matrix and key cross-method findings.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def classify(a2_rank, sp_rank):
    if a2_rank <= 3 and sp_rank <= 3:
        return "easy"
    if a2_rank <= 3 and sp_rank > 3:
        return "phantom"
    if a2_rank > 3 and sp_rank > 3:
        return "real"
    return "rare"


def load_and_classify(b7a4_csv, a2_csv):
    b7a4 = pd.read_csv(b7a4_csv)
    a2 = pd.read_csv(a2_csv)
    ref = b7a4[b7a4["is_ref_frame"] == 1].copy()
    a2_mask = a2[a2["policy"] == "mask"][["scene", "prompt", "raw_rank_true"]].rename(
        columns={"raw_rank_true": "a2_mask_rank"})
    a2_mask["a2_mask_rank"] = pd.to_numeric(a2_mask["a2_mask_rank"], errors="coerce")
    j = ref.merge(a2_mask, on=["scene", "prompt"], how="left")
    j["a2_mask_rank"] = j["a2_mask_rank"].fillna(99).astype(float)
    j["d2_class"] = [classify(a, s) for a, s in zip(j["a2_mask_rank"], j["oracle_rank"])]
    return j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thgs_csv", default="output/diagnostics/b7_a4_combined.csv")
    parser.add_argument("--relags_csv", default="output/diagnostics/b7_a4_combined_relags.csv")
    parser.add_argument("--a2_csv", default="output/diagnostics/a2_image_clip_ceiling.csv")
    parser.add_argument("--out_dir", default="output/diagnostics/plots")
    parser.add_argument("--out_csv", default="output/diagnostics/cross_method_d2_decomposition.csv")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    thgs = load_and_classify(args.thgs_csv, args.a2_csv)
    relags = load_and_classify(args.relags_csv, args.a2_csv)

    # Merge per-prompt for transition
    cols_t = ["scene", "prompt", "oracle_rank", "z_margin", "oracle_purity_ref",
              "oracle_completeness_ref", "fragmentation", "a2_mask_rank", "d2_class"]
    a = thgs[cols_t].rename(columns={
        "oracle_rank": "thgs_rank", "z_margin": "thgs_z",
        "oracle_purity_ref": "thgs_purity", "oracle_completeness_ref": "thgs_comp",
        "fragmentation": "thgs_frag", "d2_class": "thgs_class"})
    b = relags[cols_t].drop(columns=["a2_mask_rank"]).rename(columns={
        "oracle_rank": "relags_rank", "z_margin": "relags_z",
        "oracle_purity_ref": "relags_purity", "oracle_completeness_ref": "relags_comp",
        "fragmentation": "relags_frag", "d2_class": "relags_class"})
    joined = a.merge(b, on=["scene", "prompt"], how="inner")
    joined.to_csv(args.out_csv, index=False)
    print(f"\nSaved {args.out_csv} ({len(joined)} rows)")

    # === Class distribution per method ===
    print("\n=== D2 class distribution (THGS vs ReLaGS) ===")
    classes = ["easy", "phantom", "real", "rare"]
    for cls in classes:
        nt = (thgs["d2_class"] == cls).sum()
        nr = (relags["d2_class"] == cls).sum()
        print(f"  {cls:9s}: THGS {nt:3d}/{len(thgs)} ({nt/len(thgs)*100:5.1f}%)  "
              f"ReLaGS {nr:3d}/{len(relags)} ({nr/len(relags)*100:5.1f}%)  "
              f"Δ = {nr - nt:+d}")

    # === Transition matrix ===
    print("\n=== THGS → ReLaGS transition matrix (counts) ===")
    label = "(THGS->ReLaGS)"
    print(f"  {label:>15}", *[f"{c:>9}" for c in classes])
    for t in classes:
        row = [(joined["thgs_class"] == t) & (joined["relags_class"] == r) for r in classes]
        counts = [int(x.sum()) for x in row]
        print(f"  {t:>15}", *[f"{c:>9d}" for c in counts])

    # === Rank improvement on THGS phantoms ===
    thgs_phantom = joined[joined["thgs_class"] == "phantom"]
    n_recovered = (thgs_phantom["relags_class"] == "easy").sum()
    n_real = (thgs_phantom["relags_class"] == "real").sum()
    n_still_phantom = (thgs_phantom["relags_class"] == "phantom").sum()
    print(f"\n=== THGS phantoms (n={len(thgs_phantom)}) → ReLaGS outcomes ===")
    print(f"  Easy   : {n_recovered} ({n_recovered/len(thgs_phantom)*100:.0f}%)")
    print(f"  Real   : {n_real} ({n_real/len(thgs_phantom)*100:.0f}%)")
    print(f"  Phantom: {n_still_phantom} ({n_still_phantom/len(thgs_phantom)*100:.0f}%)")
    if len(thgs_phantom) > 0:
        print(f"  median rank: THGS {thgs_phantom['thgs_rank'].median():.0f} → ReLaGS {thgs_phantom['relags_rank'].median():.0f}")
        print(f"  median z   : THGS {thgs_phantom['thgs_z'].median():.2f} → ReLaGS {thgs_phantom['relags_z'].median():.2f}")

    # === ReLaGS phantoms that came from where? ===
    relags_phantom = joined[joined["relags_class"] == "phantom"]
    print(f"\n=== ReLaGS phantoms (n={len(relags_phantom)}) — origin ===")
    for src in classes:
        cnt = (relags_phantom["thgs_class"] == src).sum()
        if cnt > 0:
            print(f"  from THGS-{src:9s}: {cnt}")

    # === Easy in THGS but lost in ReLaGS (regressions) ===
    regressions = joined[(joined["thgs_class"] == "easy") & (joined["relags_class"] != "easy")]
    print(f"\n=== Regressions: THGS easy → ReLaGS not easy (n={len(regressions)}) ===")
    if len(regressions) > 0:
        print(regressions[["scene", "prompt", "thgs_class", "relags_class",
                          "thgs_rank", "relags_rank"]].to_string(index=False))

    # === Per-scene class shifts ===
    print("\n=== Per-scene D2 class shift ===")
    for sc, g in joined.groupby("scene"):
        print(f"\n  --- {sc} (n={len(g)}) ---")
        for cls in classes:
            nt = (g["thgs_class"] == cls).sum()
            nr = (g["relags_class"] == cls).sum()
            print(f"    {cls:9s}: THGS {nt:2d} → ReLaGS {nr:2d}  Δ={nr-nt:+d}")

    # === Plots ===
    # Plot 1: per-prompt rank scatter (THGS vs ReLaGS)
    fig, ax = plt.subplots(figsize=(8, 7))
    color_map = {"easy": "tab:green", "phantom": "tab:orange",
                 "real": "tab:red", "rare": "tab:purple"}
    for cls in classes:
        g = joined[joined["thgs_class"] == cls]
        if len(g):
            ax.scatter(g["thgs_rank"], g["relags_rank"], c=color_map[cls],
                       alpha=0.75, s=55, edgecolors="black", linewidths=0.3,
                       label=f"THGS {cls} (n={len(g)})")
    rmax = max(joined["thgs_rank"].max(), joined["relags_rank"].max())
    ax.plot([1, rmax], [1, rmax], "k--", alpha=0.3, label="y=x (no change)")
    ax.axhline(3.5, color="grey", linestyle=":", alpha=0.4)
    ax.axvline(3.5, color="grey", linestyle=":", alpha=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("THGS oracle rank")
    ax.set_ylabel("ReLaGS oracle rank")
    ax.set_title("Cross-method rank: every prompt is one point\n(below diagonal = ReLaGS better)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_method_rank_scatter.png"), dpi=120)
    plt.close(fig)

    # Plot 2: D2 class stacked bar per method
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    bottoms_t, bottoms_r = 0, 0
    for cls in classes:
        nt = (thgs["d2_class"] == cls).sum()
        nr = (relags["d2_class"] == cls).sum()
        ax.bar([0], [nt], width, bottom=bottoms_t, color=color_map[cls], label=cls if bottoms_t == 0 else None)
        ax.bar([1], [nr], width, bottom=bottoms_r, color=color_map[cls])
        ax.text(0, bottoms_t + nt / 2, f"{cls}\n{nt}", ha="center", va="center", fontsize=9)
        ax.text(1, bottoms_r + nr / 2, f"{cls}\n{nr}", ha="center", va="center", fontsize=9)
        bottoms_t += nt
        bottoms_r += nr
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["THGS", "ReLaGS"])
    ax.set_ylabel("# prompts")
    ax.set_title("D2 decomposition by method (n=67 each)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_method_d2_stacked.png"), dpi=120)
    plt.close(fig)

    # Plot 3: per-scene side-by-side stacked
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, sc in enumerate(sorted(joined["scene"].unique())):
        g = joined[joined["scene"] == sc]
        bottoms = [0, 0]
        for cls in classes:
            nt = (g["thgs_class"] == cls).sum()
            nr = (g["relags_class"] == cls).sum()
            axes[i].bar(0, nt, 0.5, bottom=bottoms[0], color=color_map[cls],
                        label=cls if i == 0 else None)
            axes[i].bar(1, nr, 0.5, bottom=bottoms[1], color=color_map[cls])
            bottoms[0] += nt; bottoms[1] += nr
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["THGS", "ReLaGS"])
        axes[i].set_title(f"{sc} (n={len(g)})")
        axes[i].grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("# prompts")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-scene D2 decomposition: THGS vs ReLaGS")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_method_per_scene_stacked.png"), dpi=120)
    plt.close(fig)

    # Plot 4: transition heatmap
    cls_idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((4, 4), dtype=int)
    for _, row in joined.iterrows():
        M[cls_idx[row["thgs_class"]], cls_idx[row["relags_class"]]] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(classes)
    ax.set_yticks(range(4)); ax.set_yticklabels(classes)
    ax.set_xlabel("ReLaGS class")
    ax.set_ylabel("THGS class")
    ax.set_title("Transition matrix: THGS → ReLaGS (n=67)")
    for i in range(4):
        for j in range(4):
            t = M[i, j]
            ax.text(j, i, str(t), ha="center", va="center",
                    color="white" if t > M.max() * 0.5 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cross_method_transition_matrix.png"), dpi=120)
    plt.close(fig)

    print(f"\nPlots in {args.out_dir}/")


if __name__ == "__main__":
    main()
