"""
Stage 2A — Layer 1: B8 within-view mixing visual proxy.

For each target SP (17 phantoms + sampled easy), render the SP mask at multiple
training views and compute:

  - num_components (≥10% area): if SP projection breaks into multiple
        large disconnected components, the SP likely covers multiple objects.
        High evidence for B8 within-view mixing.
  - largest_component_frac: ratio of biggest component to total SP pixels.
        Low → multi-object SP. High (≈1) → compact single object.
  - bbox_fill_ratio = total_pixels / bbox_area: low = spread out / multi-blob.
  - aspect_ratio of mask bbox.

Per target SP, aggregate across views:
  - mix_view_frac = fraction of views with num_components ≥ 2
  - mean / median num_components
  - mean / min largest_component_frac
  - mean bbox_fill_ratio

Compares 17 persistent phantoms vs sampled easy oracles.
"""

import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
import cv2
from argparse import ArgumentParser

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams


@torch.no_grad()
def render_sp_mask(cam, gaussians, pipe, background, snag, level, sp_id, thresh):
    point_valid = (snag.labels[level].long() == sp_id).float()
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


def mask_geometry(mask_np, min_area_frac=0.10):
    """Return (num_significant_components, largest_frac, bbox_fill, aspect_ratio, total_pix).

    Components are counted if they have ≥ min_area_frac of total mask.
    """
    total = int(mask_np.sum())
    if total == 0:
        return 0, 1.0, 1.0, 1.0, 0
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        mask_np.astype(np.uint8), connectivity=8)
    if n_lbl <= 1:
        return 0, 1.0, 1.0, 1.0, total
    comp_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_frac = float(comp_areas.max() / total)
    valid = int((comp_areas >= total * min_area_frac).sum())
    # bbox fill ratio: full mask bbox vs total mask pixels
    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        return valid, largest_frac, 1.0, 1.0, total
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    bbox_area = h * w
    bbox_fill = total / max(bbox_area, 1)
    aspect = max(h, w) / max(min(h, w), 1)
    return valid, largest_frac, bbox_fill, aspect, total


CSV_HEADER = [
    'scene', 'prompt', 'category',
    'oracle_lvl', 'oracle_sp_id', 'oracle_purity_ref',
    'n_views_sampled', 'n_views_visible',
    'mix_view_frac', 'mean_num_components', 'median_num_components', 'max_num_components',
    'mean_largest_frac', 'min_largest_frac',
    'mean_bbox_fill', 'min_bbox_fill',
    'mean_aspect', 'max_aspect',
    'mean_sp_pixels', 'max_sp_pixels',
]


@torch.no_grad()
def run_scene(dataset, pipe, args, target_list):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    targets_here = [t for t in target_list if t['scene'] == scene_name]
    if not targets_here:
        return []
    print(f"\n=== {scene_name}: {len(targets_here)} targets ===", flush=True)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    iter_arg = args.iteration if args.iteration > 0 else -1
    scene = Scene(dataset, gaussians, iter_arg, load_sem=False, shuffle=False)
    if args.iteration == 0:
        ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_0", "point_cloud.ply")
        if os.path.exists(ply_path):
            gaussians.load_ply(ply_path)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

    train_cams = scene.getTrainCameras()
    step = max(1, len(train_cams) // args.view_subsample)
    sampled_cams = train_cams[::step][:args.view_subsample]
    print(f"  {len(train_cams)} train cams -> sampling {len(sampled_cams)}", flush=True)

    rows = []
    for t in targets_here:
        sp_id = int(t['oracle_sp_id'])
        lvl = int(t['oracle_lvl'])
        per_view = []
        for cam in sampled_cams:
            try:
                m = render_sp_mask(cam, gaussians, pipe, background, snag, lvl, sp_id, args.thresh)
            except Exception:
                continue
            total_pix = int(m.sum().item())
            if total_pix < args.min_pixels:
                continue
            mask_np = m.cpu().numpy()
            n_comp, largest_frac, bbox_fill, aspect, tot = mask_geometry(
                mask_np, min_area_frac=0.10)
            per_view.append({
                'n_comp': n_comp, 'largest_frac': largest_frac,
                'bbox_fill': bbox_fill, 'aspect': aspect, 'pix': tot,
            })
            del m
        torch.cuda.empty_cache()
        if not per_view:
            print(f"    [skip] no visible views for {t['prompt']}", flush=True)
            continue
        df = pd.DataFrame(per_view)
        rows.append([
            scene_name, t['prompt'], t['category'],
            lvl, sp_id, f"{t.get('oracle_purity_ref', 0):.4f}",
            len(sampled_cams), len(df),
            f"{(df['n_comp'] >= 2).mean():.4f}",
            f"{df['n_comp'].mean():.3f}",
            f"{df['n_comp'].median():.2f}",
            int(df['n_comp'].max()),
            f"{df['largest_frac'].mean():.4f}",
            f"{df['largest_frac'].min():.4f}",
            f"{df['bbox_fill'].mean():.4f}",
            f"{df['bbox_fill'].min():.4f}",
            f"{df['aspect'].mean():.3f}",
            f"{df['aspect'].max():.3f}",
            f"{df['pix'].mean():.1f}",
            int(df['pix'].max()),
        ])
        print(f"  {t['prompt']:25s} mix_view_frac={float((df['n_comp']>=2).mean()):.2f} "
              f"mean_nc={df['n_comp'].mean():.2f} mean_largest={df['largest_frac'].mean():.2f} "
              f"mean_bbox_fill={df['bbox_fill'].mean():.2f}", flush=True)
    return rows


def build_target_list(b7_csv, persistent_csv, easy_per_scene=4):
    b7 = pd.read_csv(b7_csv)
    ref = b7[b7['is_ref_frame'] == 1].copy()
    phantoms = pd.read_csv(persistent_csv)
    phantom_set = set((r['scene'], r['prompt']) for _, r in phantoms.iterrows())
    targets = []
    for _, r in ref.iterrows():
        if (r['scene'], r['prompt']) in phantom_set:
            targets.append({
                'scene': r['scene'], 'prompt': r['prompt'],
                'category': 'phantom17',
                'oracle_lvl': int(r['oracle_lvl']), 'oracle_sp_id': int(r['oracle_sp_id']),
                'oracle_purity_ref': float(r['oracle_purity_ref']),
            })
    rng = np.random.RandomState(42)
    for sc, g in ref.groupby('scene'):
        easy = g[g['oracle_rank'] <= 3]
        n = min(easy_per_scene, len(easy))
        if n == 0:
            continue
        idx = rng.choice(len(easy), size=n, replace=False)
        for i in idx:
            r = easy.iloc[i]
            targets.append({
                'scene': r['scene'], 'prompt': r['prompt'],
                'category': 'easy_sample',
                'oracle_lvl': int(r['oracle_lvl']), 'oracle_sp_id': int(r['oracle_sp_id']),
                'oracle_purity_ref': float(r['oracle_purity_ref']),
            })
    return targets


@torch.no_grad()
def main(dataset, pipe, args):
    targets = build_target_list(args.b7_csv, args.persistent_csv, args.easy_per_scene)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new_header = (args.append == 0 or not os.path.exists(args.out_csv))
    mode = 'w' if (args.append == 0) else 'a'
    f_csv = open(args.out_csv, mode, newline='')
    writer = csv.writer(f_csv)
    if new_header:
        writer.writerow(CSV_HEADER)
    rows = run_scene(dataset, pipe, args, targets)
    for r in rows:
        writer.writerow(r)
    f_csv.flush()
    f_csv.close()
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_csv", type=str,
                        default="output/diagnostics/stage2a_layer1.csv")
    parser.add_argument("--b7_csv", type=str,
                        default="output/diagnostics/b7_a4_combined.csv")
    parser.add_argument("--persistent_csv", type=str,
                        default="output/diagnostics/persistent_phantoms_17.csv")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--view_subsample", type=int, default=20)
    parser.add_argument("--min_pixels", type=int, default=50)
    parser.add_argument("--easy_per_scene", type=int, default=4)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
    print("Done.")
