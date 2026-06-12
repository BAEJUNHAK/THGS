"""
SAM-only Oracle ablation for LERF-Mask.

Replaces THGS's CLIP-text matching with a GT-mask oracle:
  1. Use the FIRST test view's GT mask to identify which superpoints "belong" to each prompt
     (Gaussians that project into the GT region -> their superpoint IDs at level 2 & 3)
  2. Render those superpoints' Gaussians at ALL test views (including the ref view itself)
  3. Save predicted masks for evaluation with scripts/eval_lerf_mask.py

The resulting score isolates SAM-side (superpoint quality + view consistency),
removing CLIP-side influence. Contrast with test_lerf_mask.py which uses CLIP relevancy.
"""

import os
import sys
import cv2
import torch
import numpy as np
from argparse import ArgumentParser

from gaussian_renderer import render, render_point
from scene import Scene, GaussianModel
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams

from test_lerf_mask import estimate_sim3, transform_pose, build_camera


@torch.no_grad()
def find_gaussians_in_mask(view_cam, gaussians, pipe, background, gt_mask_np, weight_thresh=0.01):
    """Identify Gaussian indices whose 2D projection falls inside the GT mask."""
    render_pkg = render_point(view_cam, gaussians, pipe, background)
    weight = render_pkg["weight"]
    means2D = render_pkg["means2D"]

    gau_alive = weight > weight_thresh
    gau_indices = torch.where(gau_alive)[0]
    m2d = means2D[gau_alive]

    H, W = gt_mask_np.shape
    gt = torch.from_numpy(gt_mask_np.astype(np.bool_)).to(m2d.device)
    bx = m2d[:, 0].clamp(0, W - 1).long()
    by = m2d[:, 1].clamp(0, H - 1).long()
    in_mask = gt[by, bx]
    return gau_indices[in_mask]


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Scene: {scene_name} (SAM-only Oracle, ref_view={args.ref_view}) ===")

    # 1) load trained Gaussians in LERF-OVS world
    gaussians = GaussianModel(dataset.sh_degree, 20)
    Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    print(f"  loaded {snag.gaussian_num} Gaussians, NAG levels={[l.unique().numel() for l in snag.labels]}")

    # 2) sim3 LERF-Mask world -> LERF-OVS world
    mask_root = os.path.join(args.path_mask, scene_name)
    src_sparse = os.path.join(mask_root, "sparse", "0")
    dst_sparse = os.path.join(dataset.source_path, "sparse", "0")
    sim3 = estimate_sim3(src_sparse, dst_sparse)
    print(f"  sim3 inliers={sim3['inliers']}/{sim3['n']}  mean_err={sim3['inlier_mean']:.5f}")

    # 3) build cameras for every test view
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
            print(f"  [skip] {test_jpg} not in LERF-Mask COLMAP")
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
    if args.ref_view not in view_keys:
        print(f"  [warn] ref_view={args.ref_view} not available; falling back to {view_keys[0]}")
        ref_view = view_keys[0]
    else:
        ref_view = args.ref_view
    print(f"  reference view: {ref_view}, total views: {view_keys}")

    out_root = os.path.join(args.path_pred, scene_name)
    os.makedirs(out_root, exist_ok=True)

    # 4) for each prompt, oracle-select superpoints at ref view, then propagate
    ref_gt_dir = os.path.join(gt_root, ref_view)
    prompts = [f.rsplit(".", 1)[0] for f in sorted(os.listdir(ref_gt_dir)) if f.endswith(".png")]
    print(f"  {len(prompts)} prompts from ref-view GT")

    for prompt in prompts:
        gt_path = os.path.join(ref_gt_dir, prompt + ".png")
        gt_mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 128
        if not gt_mask_np.any():
            print(f"    [empty GT] {prompt}")
            continue

        sel_gau = find_gaussians_in_mask(test_cams[ref_view], gaussians, pipe, background, gt_mask_np)
        if sel_gau.numel() == 0:
            print(f"    [no Gaussians fall in GT] {prompt}")
            continue

        # Union of all Gaussians whose superpoint at any selected level matches a chosen SP
        point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device="cuda")
        sp_counts = []
        for lvl in args.levels:
            sp_ids = snag.labels[lvl][sel_gau].unique()
            sp_counts.append(int(sp_ids.numel()))
            picked = torch.isin(snag.labels[lvl], sp_ids)
            point_valid[picked] = 1.0
        if point_valid.sum() == 0:
            print(f"    [empty superpoint expansion] {prompt}")
            continue

        gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()

        for v_idx, cam in test_cams.items():
            embd_sim = render(cam, gaussians, pipe, background)["semantics"]
            mask = (embd_sim.reshape(20, -1)[0] > args.thresh).reshape(
                cam.image_height, cam.image_width
            )
            out_view = os.path.join(out_root, v_idx)
            os.makedirs(out_view, exist_ok=True)
            cv2.imwrite(os.path.join(out_view, prompt + ".png"),
                        mask.cpu().numpy().astype(np.uint8) * 255)
        print(f"    {prompt}: gau={int(sel_gau.numel())}, sp_per_lvl={sp_counts}, "
              f"propagated_gau={int(point_valid.sum().item())}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--path_pred", type=str, default="output/render/lerf_mask_sam_oracle")
    parser.add_argument("--path_mask", type=str, default="data/lerf_mask")
    parser.add_argument("--ref_view", type=str, default="0",
                        help="Index of the test view to use as the GT-mask oracle source.")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3],
                        help="NAG levels to take superpoint membership from.")
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
