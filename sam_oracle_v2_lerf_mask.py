"""
SAM-only Oracle v2 — Majority-threshold variant.

For each prompt at the ref view:
  1. Render Gaussian centers (render_point) and locate each visible Gaussian's 2D position.
  2. For each NAG superpoint at the chosen levels, compute the fraction of its visible
     Gaussians whose 2D projection falls inside the GT mask.
  3. Select only superpoints with fraction >= tau (sweep over multiple tau values).
  4. Render the union of selected superpoints' Gaussians at every test view.

Compared to v1 (which took union of every SP containing at least one GT-inside Gaussian,
leading to massive over-expansion), v2 demands a SP be *majority* inside the GT before
being selected. This more accurately reflects "the SP belongs to this object".

Saves predictions to <path_pred>_tau{TAU}/<scene>/<view>/<prompt>.png for each tau,
so eval_lerf_mask.py can be run on each.
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
def compute_sp_fractions(view_cam, gaussians, pipe, background, gt_mask_np,
                         snag, levels, weight_thresh=0.01):
    """For each chosen NAG level, return (sp_ids, fractions) — the fraction of each SP's
    visible Gaussians whose 2D projection falls inside gt_mask_np."""
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
    in_mask = gt[by, bx].float()

    result = []
    for lvl in levels:
        sp_label = snag.labels[lvl][gau_indices].long()
        sp_ids, inv = sp_label.unique(return_inverse=True)
        sums = torch.zeros(sp_ids.numel(), device=m2d.device)
        counts = torch.zeros(sp_ids.numel(), device=m2d.device)
        sums.scatter_add_(0, inv, in_mask)
        counts.scatter_add_(0, inv, torch.ones_like(in_mask))
        fractions = sums / counts.clamp_min(1)
        result.append((lvl, sp_ids, fractions))
    return result


def build_indicator(snag, level_sp_pairs):
    """level_sp_pairs: list of (lvl, sp_ids selected). Return (N,) float indicator."""
    point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
    for lvl, sel_sp in level_sp_pairs:
        if sel_sp.numel() == 0:
            continue
        picked = torch.isin(snag.labels[lvl].long(), sel_sp.long())
        point_valid[picked] = 1.0
    return point_valid


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Scene: {scene_name} (SAM Oracle v2, ref_view={args.ref_view}, taus={args.tau_sweep}) ===")

    # 1) load Gaussians + NAG
    gaussians = GaussianModel(dataset.sh_degree, 20)
    Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    print(f"  loaded {snag.gaussian_num} Gaussians, NAG levels={[l.unique().numel() for l in snag.labels]}")

    # 2) sim3 alignment
    mask_root = os.path.join(args.path_mask, scene_name)
    src_sparse = os.path.join(mask_root, "sparse", "0")
    dst_sparse = os.path.join(dataset.source_path, "sparse", "0")
    sim3 = estimate_sim3(src_sparse, dst_sparse)
    print(f"  sim3 inliers={sim3['inliers']}/{sim3['n']}  mean_err={sim3['inlier_mean']:.5f}")

    # 3) build test cameras
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
    print(f"  reference view: {ref_view}, total views: {view_keys}")

    # 4) per prompt: compute SP fractions once, then sweep tau
    ref_gt_dir = os.path.join(gt_root, ref_view)
    prompts = [f.rsplit(".", 1)[0] for f in sorted(os.listdir(ref_gt_dir)) if f.endswith(".png")]
    print(f"  {len(prompts)} prompts from ref-view GT")

    for prompt in prompts:
        gt_path = os.path.join(ref_gt_dir, prompt + ".png")
        gt_mask_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 128
        if not gt_mask_np.any():
            print(f"    [empty GT] {prompt}")
            continue

        fractions_per_level = compute_sp_fractions(
            test_cams[ref_view], gaussians, pipe, background,
            gt_mask_np, snag, args.levels,
        )

        # Stats for diagnosis
        diag_strs = []
        for lvl, sp_ids, fr in fractions_per_level:
            diag_strs.append(f"L{lvl}: visible={sp_ids.numel()}, max_frac={fr.max().item():.3f}")
        diag = ", ".join(diag_strs)

        per_tau_sp_count = []
        for tau in args.tau_sweep:
            sel_pairs = [(lvl, sp_ids[fr >= tau]) for (lvl, sp_ids, fr) in fractions_per_level]
            point_valid = build_indicator(snag, sel_pairs)
            sp_counts = [int(sp.numel()) for (_, sp) in sel_pairs]
            per_tau_sp_count.append((tau, sp_counts, int(point_valid.sum().item())))

            if point_valid.sum() == 0:
                # render zero mask so eval still finds the file
                for v_idx, cam in test_cams.items():
                    H, W = cam.image_height, cam.image_width
                    zero_mask = np.zeros((H, W), dtype=np.uint8)
                    out_view = os.path.join(args.path_pred + f"_tau{tau:.1f}", scene_name, v_idx)
                    os.makedirs(out_view, exist_ok=True)
                    cv2.imwrite(os.path.join(out_view, prompt + ".png"), zero_mask)
                continue

            gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
            for v_idx, cam in test_cams.items():
                embd_sim = render(cam, gaussians, pipe, background)["semantics"]
                mask = (embd_sim.reshape(20, -1)[0] > args.thresh).reshape(
                    cam.image_height, cam.image_width
                )
                out_view = os.path.join(args.path_pred + f"_tau{tau:.1f}", scene_name, v_idx)
                os.makedirs(out_view, exist_ok=True)
                cv2.imwrite(os.path.join(out_view, prompt + ".png"),
                            mask.cpu().numpy().astype(np.uint8) * 255)

        sweep_str = " | ".join([f"tau{t}:sp={sc},gau={gc}" for (t, sc, gc) in per_tau_sp_count])
        print(f"    {prompt}: [{diag}] | {sweep_str}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--path_pred", type=str, default="output/render/lerf_mask_sam_oracle_v2")
    parser.add_argument("--path_mask", type=str, default="data/lerf_mask")
    parser.add_argument("--ref_view", type=str, default="0")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--tau_sweep", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
