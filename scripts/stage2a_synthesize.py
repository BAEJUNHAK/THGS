"""
Synthesize Stage 2A results into:
  - output/diagnostics/phantom_anatomy.csv   — all layers per phantom
  - output/diagnostics/phantom_attribution.csv — mechanism attribution table
  - output/diagnostics/plots/phantom_b8_mixrate.png
  - output/diagnostics/plots/phantom_wrong_top1_types.png
  - output/diagnostics/plots/phantom_d3_sweep.png
  - output/diagnostics/plots/phantom_per_view_trajectory.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    out_dir = "output/diagnostics"
    plots_dir = "output/diagnostics/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # === Load all Stage 2A data ===
    persistent = pd.read_csv("output/diagnostics/persistent_phantoms_17.csv")
    layer1 = pd.read_csv("output/diagnostics/stage2a_layer1.csv")
    layer2_for = pd.read_csv("output/diagnostics/stage2a_layer2_forensic.csv")
    layer2_traj = pd.read_csv("output/diagnostics/stage2a_layer2_trajectory.csv")
    layer3_summary = pd.read_csv("output/diagnostics/stage2a_layer3_d3_summary.csv")
    b7a4 = pd.read_csv("output/diagnostics/b7_a4_combined.csv")
    relags = pd.read_csv("output/diagnostics/b7_a4_combined_relags.csv")
    a2 = pd.read_csv("output/diagnostics/a2_image_clip_ceiling.csv")
    cross = pd.read_csv("output/diagnostics/cross_method_d2_decomposition.csv")

    # Use ref-frame for THGS
    ref = b7a4[b7a4['is_ref_frame'] == 1].copy()
    ref_relags = relags[relags['is_ref_frame'] == 1].copy()
    a2_mask = a2[a2['policy'] == 'mask'][['scene', 'prompt', 'raw_rank_true']]
    a2_mask = a2_mask.rename(columns={'raw_rank_true': 'a2_mask_rank'})
    a2_mask['a2_mask_rank'] = pd.to_numeric(a2_mask['a2_mask_rank'], errors='coerce')

    # === Build phantom_anatomy.csv ===
    rows = []
    for _, p in persistent.iterrows():
        sc, prompt = p['scene'], p['prompt']
        b7 = ref[(ref['scene'] == sc) & (ref['prompt'] == prompt)].iloc[0]
        b7r = ref_relags[(ref_relags['scene'] == sc) & (ref_relags['prompt'] == prompt)].iloc[0]
        a2r = a2_mask[(a2_mask['scene'] == sc) & (a2_mask['prompt'] == prompt)]
        a2_rank_val = float(a2r['a2_mask_rank'].iloc[0]) if len(a2r) else float('nan')
        l1 = layer1[(layer1['scene'] == sc) & (layer1['prompt'] == prompt) & (layer1['category'] == 'phantom17')]
        l2f = layer2_for[(layer2_for['scene'] == sc) & (layer2_for['prompt'] == prompt)]
        l2t = layer2_traj[(layer2_traj['scene'] == sc) & (layer2_traj['prompt'] == prompt)]
        l3 = layer3_summary[(layer3_summary['scene'] == sc) & (layer3_summary['prompt'] == prompt)]

        # Layer 2 trajectory aggregates
        if len(l2t) > 0:
            min_view_rank = int(l2t['rank_in_clip_pool'].min())
            max_view_rank = int(l2t['rank_in_clip_pool'].max())
            mean_view_rank = float(l2t['rank_in_clip_pool'].mean())
            same_oracle_frac = float(l2t['same_as_ref_oracle'].mean())
        else:
            min_view_rank = max_view_rank = -1
            mean_view_rank = float('nan')
            same_oracle_frac = float('nan')

        rows.append({
            'scene': sc,
            'prompt': prompt,
            # B7
            'oracle_purity': float(b7['oracle_purity_ref']),
            'oracle_completeness': float(b7['oracle_completeness_ref']),
            'fragmentation': int(b7['fragmentation']),
            # A2 ceiling
            'a2_mask_rank': a2_rank_val,
            # THGS rank/margin
            'thgs_rank': int(b7['oracle_rank']),
            'thgs_z_margin': float(b7['z_margin']),
            # ReLaGS rank
            'relags_rank': int(b7r['oracle_rank']),
            'relags_z_margin': float(b7r['z_margin']),
            # Layer 1 — B8 visual proxy
            'mix_view_frac': float(l1['mix_view_frac'].iloc[0]) if len(l1) else float('nan'),
            'mean_num_components': float(l1['mean_num_components'].iloc[0]) if len(l1) else float('nan'),
            'mean_bbox_fill': float(l1['mean_bbox_fill'].iloc[0]) if len(l1) else float('nan'),
            # Layer 2 — wrong top-1
            'wrong_top1_best_match_prompt': l2f['wrong_top1_best_match_prompt'].iloc[0] if len(l2f) else '',
            'wrong_top1_best_match_iou': float(l2f['wrong_top1_best_match_iou'].iloc[0]) if len(l2f) else float('nan'),
            'wrong_top1_overlap_with_gt': float(l2f['wrong_top1_overlap_with_gt_iou'].iloc[0]) if len(l2f) else float('nan'),
            # Layer 2 — trajectory
            'min_view_rank': min_view_rank,
            'max_view_rank': max_view_rank,
            'mean_view_rank': mean_view_rank,
            'same_oracle_frac': same_oracle_frac,
            # Layer 3 — D3 sweep
            'iou_k1': float(l3['iou_k1'].iloc[0]) if len(l3) else float('nan'),
            'iou_k2': float(l3['iou_k2'].iloc[0]) if len(l3) else float('nan'),
            'iou_k3': float(l3['iou_k3'].iloc[0]) if len(l3) else float('nan'),
            'iou_k5': float(l3['iou_k5'].iloc[0]) if len(l3) else float('nan'),
            'iou_k10': float(l3['iou_k10'].iloc[0]) if len(l3) else float('nan'),
            'best_k': int(l3['best_k'].iloc[0]) if len(l3) else -1,
            'best_iou': float(l3['best_iou'].iloc[0]) if len(l3) else float('nan'),
        })

    anatomy = pd.DataFrame(rows)
    anatomy.to_csv(os.path.join(out_dir, "phantom_anatomy.csv"), index=False)
    print(f"Saved {out_dir}/phantom_anatomy.csv ({len(anatomy)} rows)")

    # === Mechanism attribution ===
    # Rules:
    #   - over_union: wrong_top1_overlap_with_gt >= 0.30 OR best_k=1 with iou>0.3 and original k3 iou < best_iou-0.1
    #   - instance_confusion: wrong_top1_best_match_iou >= 0.30
    #   - background_drift: wrong_top1 has no match (both iou < 0.10)
    #   - target_dilution (cross-view): min_view_rank <= 3 but max_view_rank > 10 (per-view rescue)
    #   - structural_phantom: all view ranks > 10 AND best_iou < 0.10 (no top-k recovers)
    #   - geometry_fragmented: mix_view_frac > 0.4 (B8 high)
    #   - encoder_hidden: a2_mask_rank > 1 AND other signals weak

    def attribute(r):
        tags = []
        if r['wrong_top1_overlap_with_gt'] >= 0.30:
            tags.append('over_union')
        if r['wrong_top1_best_match_iou'] >= 0.30:
            tags.append('instance_confusion')
        if r['wrong_top1_overlap_with_gt'] < 0.10 and r['wrong_top1_best_match_iou'] < 0.10:
            tags.append('background_drift')
        if r['min_view_rank'] > 0 and r['min_view_rank'] <= 3 and r['max_view_rank'] > 10:
            tags.append('target_dilution')
        if r['best_iou'] >= 0.30 and r['iou_k3'] + 0.10 <= r['best_iou']:
            tags.append('d3_recoverable')
        if r['best_iou'] < 0.10:
            tags.append('structural_phantom')
        if r['mix_view_frac'] > 0.40:
            tags.append('geometry_fragmented')
        if r['a2_mask_rank'] > 1:
            tags.append('encoder_hidden')
        # primary label (most decisive)
        primary = 'unknown'
        if 'd3_recoverable' in tags and 'over_union' in tags:
            primary = 'D3_over_union'
        elif 'instance_confusion' in tags:
            primary = 'instance_confusion'
        elif 'target_dilution' in tags:
            primary = 'target_dilution'
        elif 'd3_recoverable' in tags:
            primary = 'D3_deep_pool'
        elif 'structural_phantom' in tags:
            primary = 'structural_ROFA'
        elif 'background_drift' in tags:
            primary = 'background_drift'
        elif 'geometry_fragmented' in tags:
            primary = 'B8_fragmented'
        return pd.Series({'primary_mechanism': primary, 'all_tags': ';'.join(tags)})

    attr = anatomy.apply(attribute, axis=1)
    attribution = pd.concat([anatomy[['scene', 'prompt']], attr,
                              anatomy[['mix_view_frac', 'min_view_rank', 'max_view_rank',
                                       'wrong_top1_best_match_iou', 'wrong_top1_overlap_with_gt',
                                       'best_k', 'best_iou', 'a2_mask_rank']]], axis=1)
    attribution.to_csv(os.path.join(out_dir, "phantom_attribution.csv"), index=False)
    print(f"Saved {out_dir}/phantom_attribution.csv")
    print("\n=== Primary mechanism distribution ===")
    print(attribution['primary_mechanism'].value_counts())
    print("\n=== Per-phantom attribution ===")
    print(attribution[['scene', 'prompt', 'primary_mechanism', 'all_tags']].to_string(index=False))

    # === Plots ===
    # 1. B8 mix-rate distribution: phantom17 vs easy_sample
    fig, ax = plt.subplots(figsize=(8, 5))
    for cat, color in [('phantom17', 'tab:red'), ('easy_sample', 'tab:green')]:
        d = layer1[layer1['category'] == cat]
        ax.hist(d['mix_view_frac'], bins=np.linspace(0, 1, 11), alpha=0.6,
                label=f"{cat} (n={len(d)}, mean={d['mix_view_frac'].mean():.2f})",
                color=color, edgecolor='white')
    ax.set_xlabel("mix_view_frac (fraction of views with ≥2 components)")
    ax.set_ylabel("# SPs")
    ax.set_title("B8 within-view mixing proxy: phantom17 vs sampled easy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(plots_dir, "phantom_b8_mixrate.png"))

    # 2. Wrong top-1 type distribution
    type_counts = {'background_drift': 0, 'over_union': 0, 'instance_confusion': 0}
    for _, r in anatomy.iterrows():
        if r['wrong_top1_overlap_with_gt'] >= 0.30:
            type_counts['over_union'] += 1
        elif r['wrong_top1_best_match_iou'] >= 0.30:
            type_counts['instance_confusion'] += 1
        else:
            type_counts['background_drift'] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    cats = list(type_counts.keys())
    vals = [type_counts[k] for k in cats]
    bars = ax.bar(cats, vals, color=['tab:gray', 'tab:purple', 'tab:orange'])
    ax.set_ylabel("# phantoms (of 17)")
    ax.set_title("Wrong top-1 SP types (Layer 2 forensic)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, str(v), ha='center', fontsize=11)
    fig_save(fig, os.path.join(plots_dir, "phantom_wrong_top1_types.png"))

    # 3. D3 top-k sweep — per-phantom IoU curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ks = [1, 2, 3, 5, 10]
    for _, r in anatomy.iterrows():
        ious = [r[f'iou_k{k}'] for k in ks]
        ax.plot(ks, ious, marker='o', alpha=0.6, label=f"{r['scene'][:4]}/{r['prompt']}")
    ax.set_xlabel("top-k")
    ax.set_ylabel("Mean IoU")
    ax.set_xscale("log")
    ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_title("D3 top-k sweep per phantom (17 prompts)")
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(plots_dir, "phantom_d3_sweep.png"))

    # 4. Per-view rank trajectory — min vs max per phantom
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(anatomy))
    sorted_anat = anatomy.sort_values('thgs_rank').reset_index(drop=True)
    ax.scatter(x, sorted_anat['min_view_rank'], color='tab:green', label='min view rank', s=60)
    ax.scatter(x, sorted_anat['max_view_rank'], color='tab:red', label='max view rank', s=60)
    ax.scatter(x, sorted_anat['mean_view_rank'], color='tab:blue', label='mean view rank', s=40, alpha=0.7)
    for i, r in sorted_anat.iterrows():
        ax.plot([i, i], [r['min_view_rank'], r['max_view_rank']], color='gray', alpha=0.4, linewidth=1)
    ax.axhline(3.5, color='black', linestyle='--', alpha=0.4, label='rank=3 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['scene'][:4]}/{r['prompt']}"
                        for _, r in sorted_anat.iterrows()], rotation=90, fontsize=8)
    ax.set_ylabel("CLIP rank (log)")
    ax.set_yscale("log")
    ax.set_title("Per-view rank trajectory: min vs max per phantom\n"
                 "(min ≤ 3 with high max = target dilution evidence)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_save(fig, os.path.join(plots_dir, "phantom_per_view_trajectory.png"))

    print(f"\nAll plots saved to {plots_dir}/")
    return attribution


if __name__ == "__main__":
    main()
