"""
SAM-only Oracle v3 — Best-IoU upper bound.

For each prompt:
  1. At the ref view, render *each NAG superpoint individually* and cache its 2D mask.
  2. Compute per-SP IoU with the prompt's GT mask.
  3. Rank SPs by per-SP IoU and select top-k (sweep over multiple k).
  4. Render the union of selected SPs' Gaussians at every test view.

This measures THGS's *theoretical ceiling*: how well can the existing superpoints represent
a given GT object if we were free to pick the optimal subset?

Saves predictions to <path_pred>_topk{K}/<scene>/<view>/<prompt>.png for each k.
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
    """Render binary mask of a single SP at the given view."""
    point_valid = (snag.labels[level].long() == sp_id).float()
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Scene: {scene_name} (SAM Oracle v3, ref_view={args.ref_view}, topk={args.topk_sweep}) ===")

    # 1) load Gaussians + NAG
    gaussians = GaussianModel(dataset.sh_degree, 20)
    Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

    # 2) sim3 + cameras
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
    print(f"  reference view: {ref_view}, total views: {view_keys}")
    print(f"  ref view resolution: {H}x{W}")

    # 3) Cache per-SP masks at ref view (across selected levels)
    # Layout: lists [ (lvl, sp_id, mask(H,W) bool) ]
    print(f"  caching per-SP masks at ref view...")
    cache_lvl, cache_sp, cache_mask = [], [], []
    for lvl in args.levels:
        sp_ids = snag.labels[lvl].long().unique()
        n_sp = sp_ids.numel()
        mb = (n_sp * H * W) // (1024 * 1024)
        print(f"    level {lvl}: {n_sp} SPs  (~{mb} MB cache)")
        for sp_id in sp_ids:
            m = render_sp_mask(ref_cam, gaussians, pipe, background, snag, lvl, sp_id.item(), args.thresh)
            cache_lvl.append(lvl)
            cache_sp.append(int(sp_id.item()))
            cache_mask.append(m)
    cache_masks = torch.stack(cache_mask, dim=0)  # (total_sp, H, W) bool
    print(f"    total cached: {cache_masks.shape[0]} SP masks, dtype={cache_masks.dtype}")

    # 4) per prompt: compute per-SP IoU with GT, rank, topk sweep
    ref_gt_dir = os.path.join(gt_root, ref_view)
    prompts = [f.rsplit(".", 1)[0] for f in sorted(os.listdir(ref_gt_dir)) if f.endswith(".png")]
    print(f"  {len(prompts)} prompts from ref-view GT")

    for prompt in prompts:
        gt_path = os.path.join(ref_gt_dir, prompt + ".png")
        gt_mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 128
        if not gt_mask_np.any():
            print(f"    [empty GT] {prompt}")
            continue
        gt = torch.from_numpy(gt_mask_np.astype(np.bool_)).cuda()

        # vectorized IoU across cached SPs
        inter = (cache_masks & gt[None]).sum(dim=(1, 2)).float()
        union = (cache_masks | gt[None]).sum(dim=(1, 2)).float()
        ious = inter / union.clamp_min(1)
        order = torch.argsort(ious, descending=True)
        top1_iou = ious[order[0]].item()

        for topk in args.topk_sweep:
            sel_idx = order[:topk].cpu().tolist()
            point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
            for j in sel_idx:
                lvl = cache_lvl[j]
                sp_id = cache_sp[j]
                picked = snag.labels[lvl].long() == sp_id
                point_valid[picked] = 1.0
            if point_valid.sum() == 0:
                for v_idx, cam in test_cams.items():
                    zero_mask = np.zeros((cam.image_height, cam.image_width), dtype=np.uint8)
                    out_view = os.path.join(args.path_pred + f"_topk{topk}", scene_name, v_idx)
                    os.makedirs(out_view, exist_ok=True)
                    cv2.imwrite(os.path.join(out_view, prompt + ".png"), zero_mask)
                continue

            gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
            for v_idx, cam in test_cams.items():
                embd_sim = render(cam, gaussians, pipe, background)["semantics"]
                mask = (embd_sim.reshape(20, -1)[0] > args.thresh).reshape(
                    cam.image_height, cam.image_width)
                out_view = os.path.join(args.path_pred + f"_topk{topk}", scene_name, v_idx)
                os.makedirs(out_view, exist_ok=True)
                cv2.imwrite(os.path.join(out_view, prompt + ".png"),
                            mask.cpu().numpy().astype(np.uint8) * 255)

        # diagnostic: top single SP IoU on ref view
        print(f"    {prompt}: best single-SP IoU={top1_iou:.3f}  (top-5 IoUs: {[round(ious[i].item(),3) for i in order[:5].cpu().tolist()]})")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--path_pred", type=str, default="output/render/lerf_mask_sam_oracle_v3")
    parser.add_argument("--path_mask", type=str, default="data/lerf_mask")
    parser.add_argument("--ref_view", type=str, default="0")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--topk_sweep", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
