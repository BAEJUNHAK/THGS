"""
THGS inference on the LERF-Mask benchmark (Gaussian Grouping, ECCV 2024).

Uses the same trained sai_nag.pt as test_lerf.py, but evaluates on LERF-Mask's
novel-view test images (test_0..N.jpg) instead of LERF-OVS train-view frames.

Because LERF-Mask and LERF-OVS were reconstructed with separate COLMAP runs,
their world frames differ by a sim3. We estimate that sim3 from the common
training-frame camera centers (RANSAC) and use it to bring LERF-Mask test-view
camera poses into the LERF-OVS frame, where the trained Gaussians live.
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from argparse import ArgumentParser

from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams


def _cam_pose(im):
    R_w2c = qvec2rotmat(im.qvec)
    t = np.array(im.tvec, dtype=np.float64)
    C = -R_w2c.T @ t
    return R_w2c, t, C


def _umeyama(P, Q):
    """Estimate similarity transform (s, R, t) mapping P -> Q (rows are points)."""
    muP = P.mean(0); muQ = Q.mean(0)
    X = P - muP; Y = Q - muQ
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / (X * X).sum()
    t = muQ - s * R @ muP
    return s, R, t


def estimate_sim3(src_dir, dst_dir, inlier_thresh=0.05, iters=2000, seed=0):
    """RANSAC sim3 from src COLMAP world -> dst COLMAP world (using shared frame names)."""
    src = {im.name: _cam_pose(im) for im in read_extrinsics_binary(os.path.join(src_dir, 'images.bin')).values()}
    dst = {im.name: _cam_pose(im) for im in read_extrinsics_binary(os.path.join(dst_dir, 'images.bin')).values()}
    common = sorted(set(src) & set(dst))
    Cs = np.stack([src[k][2] for k in common])
    Cd = np.stack([dst[k][2] for k in common])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(common), 4, replace=False)
        s, R, t = _umeyama(Cs[idx], Cd[idx])
        pred = (s * (R @ Cs.T)).T + t
        err = np.linalg.norm(pred - Cd, axis=1)
        inl = err < inlier_thresh
        if inl.sum() < 4:
            continue
        score = (-int(inl.sum()), float(err[inl].mean()))
        if best is None or score < best[0]:
            best = (score, inl)
    if best is None:
        s, R, t = _umeyama(Cs, Cd)
        inl = np.ones(len(common), bool)
    else:
        inl = best[1]
        s, R, t = _umeyama(Cs[inl], Cd[inl])
    final_err = np.linalg.norm((s * (R @ Cs.T)).T + t - Cd, axis=1)
    return dict(s=s, R=R, t=t, n=len(common), inliers=int(inl.sum()),
                inlier_mean=float(final_err[inl].mean()))


def transform_pose(R_w2c_src, t_w2c_src, sim3):
    """Move a (R_w2c, t_w2c) camera from src world frame into dst world frame."""
    s, Ra, ta = sim3['s'], sim3['R'], sim3['t']
    C_src = -R_w2c_src.T @ t_w2c_src
    C_dst = s * Ra @ C_src + ta
    R_w2c_dst = R_w2c_src @ Ra.T
    t_w2c_dst = -R_w2c_dst @ C_dst
    return R_w2c_dst, t_w2c_dst


def build_camera(R_w2c, t_w2c, intr, width, height, image_name, uid, data_device):
    """Build a THGS Camera with no GT image (semantic-only render)."""
    R_c2w = R_w2c.T
    if intr.model in ('PINHOLE', 'OPENCV'):
        fx, fy, cx, cy = intr.params[:4]
    elif intr.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'):
        fx = fy = intr.params[0]
        cx, cy = intr.params[1], intr.params[2]
    else:
        raise ValueError(f'Unsupported COLMAP camera model: {intr.model}')
    FoVx = focal2fov(fx, intr.width)
    FoVy = focal2fov(fy, intr.height)
    return Camera(
        colmap_id=uid, R=R_c2w.astype(np.float32), T=t_w2c.astype(np.float32),
        FoVx=FoVx, FoVy=FoVy,
        image=None, gt_alpha_mask=None,
        image_name=image_name, uid=uid,
        width=width, height=height,
        data_device=data_device,
    )


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f'\n=== Scene: {scene_name} ===')

    # 1) load trained Gaussians (in LERF-OVS world frame)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    nag = torch.load(os.path.join(dataset.model_path, 'sai_nag.pt'))
    vlm = ClipSimMeasure(); vlm.load_model()
    snag = SemanticNAG(nag['nag'], nag['nag_feat'])

    # 2) sim3 from LERF-Mask world -> LERF-OVS world
    mask_root = os.path.join(args.path_mask, scene_name)
    src_sparse = os.path.join(mask_root, 'sparse', '0')
    dst_sparse = os.path.join(dataset.source_path, 'sparse', '0')
    sim3 = estimate_sim3(src_sparse, dst_sparse)
    print(f'  sim3 inliers={sim3["inliers"]}/{sim3["n"]}  mean err={sim3["inlier_mean"]:.5f}  scale={sim3["s"]:.6f}')

    # 3) gather LERF-Mask camera intrinsics + test-view extrinsics
    mask_imgs = read_extrinsics_binary(os.path.join(src_sparse, 'images.bin'))
    from scene.colmap_loader import read_intrinsics_binary
    mask_cams = read_intrinsics_binary(os.path.join(src_sparse, 'cameras.bin'))

    out_root = os.path.join(args.path_pred, scene_name)
    os.makedirs(out_root, exist_ok=True)

    # GT layout: data/lerf_mask/<scene>/test_mask/<view_idx>/<prompt>.png
    gt_root = os.path.join(mask_root, 'test_mask')
    for view_idx in sorted(os.listdir(gt_root)):
        gt_view_dir = os.path.join(gt_root, view_idx)
        if not os.path.isdir(gt_view_dir):
            continue
        test_jpg = f'test_{view_idx}.jpg'
        # find this image in COLMAP extrinsics
        match = next((im for im in mask_imgs.values() if im.name == test_jpg), None)
        if match is None:
            print(f'  [skip] {test_jpg} not in LERF-Mask COLMAP')
            continue
        intr = mask_cams[match.camera_id]
        R_w2c_msk = qvec2rotmat(match.qvec)
        t_w2c_msk = np.array(match.tvec, dtype=np.float64)
        R_w2c_ovs, t_w2c_ovs = transform_pose(R_w2c_msk, t_w2c_msk, sim3)
        cam = build_camera(R_w2c_ovs, t_w2c_ovs, intr, intr.width, intr.height,
                            image_name=f'test_{view_idx}', uid=int(view_idx),
                            data_device=dataset.data_device)

        out_view = os.path.join(out_root, view_idx)
        os.makedirs(out_view, exist_ok=True)

        prompts = [f.rsplit('.', 1)[0] for f in os.listdir(gt_view_dir) if f.endswith('.png')]
        print(f'  view {view_idx}: {len(prompts)} prompts')
        for prompt in prompts:
            vlm.encode_text(prompt)
            point_valid = snag.get_related_gaussian(
                [vlm.compute_similarity(f) for f in snag.feat],
                topk=args.topk, level=list(args.levels),
            )
            point_valid = point_valid.expand(-1, 20).cuda()
            gaussians._semantics = point_valid
            embd_sim = render(cam, gaussians, pipe, background)['semantics']
            mask = (embd_sim.reshape(20, -1)[0] > args.thresh).reshape(cam.image_height, cam.image_width)
            out_png = os.path.join(out_view, prompt + '.png')
            cv2.imwrite(out_png, mask.cpu().numpy().astype(np.uint8) * 255)


if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--path_pred', type=str, default='output/render/lerf_mask')
    parser.add_argument('--path_mask', type=str, default='data/lerf_mask',
                        help='LERF-Mask dataset root containing <scene>/test_mask/<view>/*.png')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--levels', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--thresh', type=float, default=0.5)
    args = parser.parse_args(sys.argv[1:])

    safe_state(True)
    run(lp.extract(args), pp.extract(args), args)
    print('\nDone.')
