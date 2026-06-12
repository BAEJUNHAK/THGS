"""
SAM-only Oracle v4 for LERF-Mask — Greedy Union (budget-aware optimal selection).

For each prompt:
  1. At the ref view, render each NAG superpoint individually and cache its 2D mask.
  2. Greedy union: at each step, pick the SP that *maximizes* the union's IoU with GT.
  3. Stop at given budgets or when IoU no longer improves (unlimited).
  4. Render the union of selected SPs' Gaussians at every test view.

Fair comparison with CLIP-based (test_lerf_mask.py, which uses topk=3 at level=[2,3]):
  - Same model (sai_nag.pt)
  - Same SP candidate pool (level=[2,3])
  - Same budget (default sweep includes K=3, matching CLIP)
  - Selection: greedy approximation of "best K-SP subset by GT IoU"

Outputs: <path_pred>_budget{K}/<scene>/<view>/<prompt>.png  (and 'unlimited' variant)
"""

import os
import sys
import cv2
import torch
import numpy as np
from argparse import ArgumentParser

from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams

from test_lerf_mask import estimate_sim3, transform_pose, build_camera


@torch.no_grad()
def render_sp_mask(cam, gaussians, pipe, background, snag, level, sp_id, thresh):
    point_valid = (snag.labels[level].long() == sp_id).float()
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


@torch.no_grad()
def greedy_union_select(cache_masks, gt, max_budget):
    """Greedy union: at each step pick the SP whose union with current selection maximizes GT IoU.
    Returns (selected_indices_in_order, ious_at_each_step).
    Stops early if IoU doesn't improve (suitable for 'unlimited' budget).
    """
    selected = []
    ious_at = []
    current_union = torch.zeros_like(cache_masks[0])
    current_iou = 0.0

    for _ in range(max_budget):
        candidate_unions = current_union[None] | cache_masks
        inter = (candidate_unions & gt[None]).sum(dim=(1, 2)).float()
        union = (candidate_unions | gt[None]).sum(dim=(1, 2)).float()
        ious = inter / union.clamp_min(1)
        if selected:
            ious_clone = ious.clone()
            for i in selected:
                ious_clone[i] = -1.0
            best_idx = ious_clone.argmax().item()
            new_iou = ious_clone[best_idx].item()
        else:
            best_idx = ious.argmax().item()
            new_iou = ious[best_idx].item()
        if new_iou <= current_iou + 1e-6:
            break
        selected.append(best_idx)
        ious_at.append(new_iou)
        current_union = current_union | cache_masks[best_idx]
        current_iou = new_iou
    return selected, ious_at


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Scene: {scene_name} (LERF-Mask Oracle v4 greedy, budgets={args.budget_sweep}) ===")

    gaussians = GaussianModel(dataset.sh_degree, 20)
    Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

    mask_root = os.path.join(args.path_mask, scene_name)
    src_sparse = os.path.join(mask_root, "sparse", "0")
    dst_sparse = os.path.join(dataset.source_path, "sparse", "0")
    sim3 = estimate_sim3(src_sparse, dst_sparse)
    print(f"  sim3 inliers={sim3['inliers']}/{sim3['n']}  mean_err={sim3['inlier_mean']:.5f}")

    mask_imgs = read_extrinsics_binary(os.path.join(src_sparse, "images.bin"))
    mask_cams = read_intrinsics_binary(os.path.join(src_sparse, "cameras.bin"))
    gt_root = os.path.join(mask_root, "test_mask")
    test_cams = {}
    for view_idx in sorted(os.listdir(gt_root)):
        if not os.path.isdir(os.path.join(gt_root, view_idx)):
            continue
        test_jpg = f"test_{view_idx}.jpg"
        match = next((im for im in mask_imgs.values() if im.name == test_jpg), None)
        if match is None:
            continue
        intr = mask_cams[match.camera_id]
        R_w2c_msk = qvec2rotmat(match.qvec)
        t_w2c_msk = np.array(match.tvec, dtype=np.float64)
        R_ovs, t_ovs = transform_pose(R_w2c_msk, t_w2c_msk, sim3)
        cam = build_camera(R_ovs, t_ovs, intr, intr.width, intr.height,
                           image_name=f"test_{view_idx}", uid=int(view_idx),
                           data_device=dataset.data_device)
        test_cams[view_idx] = cam
    view_keys = sorted(test_cams.keys(), key=lambda x: int(x))
    ref_view = args.ref_view if args.ref_view in view_keys else view_keys[0]
    ref_cam = test_cams[ref_view]
    H, W = ref_cam.image_height, ref_cam.image_width
    print(f"  ref view: {ref_view}, total views: {view_keys}, resolution: {H}x{W}")

    # Cache per-SP masks at ref view (same as v3)
    print(f"  caching SP masks at ref view...")
    cache_lvl, cache_sp, cache_mask = [], [], []
    for lvl in args.levels:
        sp_ids = snag.labels[lvl].long().unique()
        for sp_id in sp_ids:
            m = render_sp_mask(ref_cam, gaussians, pipe, background, snag, lvl, sp_id.item(), args.thresh)
            cache_lvl.append(lvl)
            cache_sp.append(int(sp_id.item()))
            cache_mask.append(m)
    cache_masks = torch.stack(cache_mask, dim=0)
    print(f"    cached {cache_masks.shape[0]} SP masks")

    # max budget for greedy unrolling (separately handle 'unlimited' by max=100)
    explicit_budgets = [b for b in args.budget_sweep if isinstance(b, int)]
    max_explicit = max(explicit_budgets) if explicit_budgets else 0
    max_unroll = 100  # cap for unlimited greedy

    ref_gt_dir = os.path.join(gt_root, ref_view)
    prompts = [f.rsplit(".", 1)[0] for f in sorted(os.listdir(ref_gt_dir)) if f.endswith(".png")]
    print(f"  {len(prompts)} prompts from ref-view GT")

    for prompt in prompts:
        gt_path = os.path.join(ref_gt_dir, prompt + ".png")
        gt_mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 128
        if not gt_mask_np.any():
            continue
        gt = torch.from_numpy(gt_mask_np).cuda()

        # Greedy unroll up to max_unroll (truncate as needed for each budget)
        sel_order, ious_at = greedy_union_select(cache_masks, gt, max_unroll)
        n_picked = len(sel_order)

        # Diagnostic
        seq_str = ' → '.join([f'{iou:.3f}' for iou in ious_at[:5]])
        print(f"    {prompt}: greedy picked {n_picked} SPs, ref-view IoU progression: {seq_str}{'...' if len(ious_at)>5 else ''}", flush=True)

        # For each budget in sweep
        for b in args.budget_sweep:
            if b == 'unlimited':
                effective_k = n_picked
                label = 'unlimited'
            else:
                effective_k = min(b, n_picked)
                label = f'budget{b}'

            sel_idx = sel_order[:effective_k]
            point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
            for j in sel_idx:
                lvl = cache_lvl[j]
                sp_id = cache_sp[j]
                picked = snag.labels[lvl].long() == sp_id
                point_valid[picked] = 1.0

            if point_valid.sum() == 0:
                continue
            gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()

            for v_idx, cam in test_cams.items():
                embd_sim = render(cam, gaussians, pipe, background)["semantics"]
                mask = (embd_sim.reshape(20, -1)[0] > args.thresh).reshape(
                    cam.image_height, cam.image_width)
                out_view = os.path.join(args.path_pred + f"_{label}", scene_name, v_idx)
                os.makedirs(out_view, exist_ok=True)
                cv2.imwrite(os.path.join(out_view, prompt + ".png"),
                            mask.cpu().numpy().astype(np.uint8) * 255)


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--path_pred", type=str, default="output/render/lerf_mask_sam_oracle_v4")
    parser.add_argument("--path_mask", type=str, default="data/lerf_mask")
    parser.add_argument("--ref_view", type=str, default="0")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    # budget sweep: int or 'unlimited'
    parser.add_argument("--budget_sweep", nargs="+",
                        default=[1, 2, 3, 5, 'unlimited'],
                        type=lambda x: int(x) if x.isdigit() else x)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
