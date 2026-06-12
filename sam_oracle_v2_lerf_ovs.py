"""
SAM-only Oracle v2 for LERF-OVS — Majority threshold variant.

For each prompt in the scene's GT:
  1. Use that prompt's *first* GT frame as the reference view.
  2. Build the polygon-rasterized GT mask at ref frame.
  3. For each NAG superpoint, compute fraction of its visible Gaussians falling inside GT.
  4. For each tau in tau_sweep, select SPs with fraction >= tau.
  5. Render the selected SPs' Gaussians at *every* frame where this prompt's GT exists.

Output:
  <path_pred>_tau{TAU}/<scene>/<frame>/<prompt>.png        (pred)
  <path_pred>_tau{TAU}/<scene>/<frame>/<prompt>_gt.png     (polygon-rasterized GT)

Differences from sam_oracle_v2_lerf_mask.py:
  - Uses scene.getTrainCameras() instead of LERF-Mask test cams (sim3 unnecessary).
  - GT is polygon → raster (test_lerf.py style) instead of pre-rasterized PNG.
  - Per-prompt ref frame (first frame containing GT for that prompt).
  - Propagation: only to frames where the prompt's GT exists.
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

from gaussian_renderer import render, render_point
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


@torch.no_grad()
def compute_sp_fractions(view_cam, gaussians, pipe, background, gt_mask_np,
                         snag, levels, weight_thresh=0.01):
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
    print(f"\n=== Scene: {scene_name} (LERF-OVS Oracle v2, taus={args.tau_sweep}) ===")

    # Load Gaussians + NAG
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    print(f"  loaded {snag.gaussian_num} Gaussians, NAG levels={[l.unique().numel() for l in snag.labels]}")

    # Load GT (test_lerf.py style)
    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    img_list = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')])

    train_cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    frame_data = {}
    for im in img_list:
        image_name = im.split('.')[0]
        js_file = os.path.join(data_path, image_name + '.json')
        anno = json.load(open(js_file))
        if image_name not in train_cams_by_name:
            print(f"  [skip] no train cam for {image_name}")
            continue
        cam = train_cams_by_name[image_name]
        frame_data[image_name] = {
            'cam': cam,
            'objects': anno['objects'],
            'h': cam.image_height,
            'w': cam.image_width,
        }

    # Build prompt → list of frame_names
    prompt_to_frames = defaultdict(list)
    for frame_name, fd in frame_data.items():
        prompt_set = set(obj['category'] for obj in fd['objects'])
        for prompt in prompt_set:
            prompt_to_frames[prompt].append(frame_name)
    for prompt in prompt_to_frames:
        prompt_to_frames[prompt] = sorted(prompt_to_frames[prompt])

    print(f"  {len(frame_data)} GT frames, {len(prompt_to_frames)} unique prompts")

    def get_gt_for(prompt, frame_name):
        fd = frame_data[frame_name]
        h, w = fd['h'], fd['w']
        mask = np.zeros((h, w), dtype=np.uint8)
        for obj in fd['objects']:
            if obj['category'] == prompt:
                m = polygon_to_mask((h, w), obj['segmentation'])
                mask = np.maximum(mask, m)
        return mask > 0

    for prompt in sorted(prompt_to_frames.keys()):
        frames = prompt_to_frames[prompt]
        ref_frame = frames[0]
        gt_mask_np = get_gt_for(prompt, ref_frame)
        if not gt_mask_np.any():
            print(f"    [empty GT] {prompt} at {ref_frame}")
            continue

        fractions_per_level = compute_sp_fractions(
            frame_data[ref_frame]['cam'], gaussians, pipe, background,
            gt_mask_np, snag, args.levels,
        )

        for tau in args.tau_sweep:
            sel_pairs = [(lvl, sp_ids[fr >= tau]) for (lvl, sp_ids, fr) in fractions_per_level]
            point_valid = build_indicator(snag, sel_pairs)
            prompt_safe = prompt.replace(' ', '_')

            for frame_name in frames:
                fd = frame_data[frame_name]
                out_dir = os.path.join(args.path_pred + f"_tau{tau:.1f}", scene_name, frame_name)
                os.makedirs(out_dir, exist_ok=True)

                if point_valid.sum() == 0:
                    pred_mask = np.zeros((fd['h'], fd['w']), dtype=np.uint8)
                else:
                    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
                    embd_sim = render(fd['cam'], gaussians, pipe, background)["semantics"]
                    mask = (embd_sim.reshape(20, -1)[0] > args.thresh).reshape(fd['h'], fd['w'])
                    pred_mask = mask.cpu().numpy().astype(np.uint8) * 255

                cv2.imwrite(os.path.join(out_dir, prompt_safe + '.png'), pred_mask)
                gt_mask = get_gt_for(prompt, frame_name)
                cv2.imwrite(os.path.join(out_dir, prompt_safe + '_gt.png'),
                            gt_mask.astype(np.uint8) * 255)

        sp_counts = [(t, [int((fr >= t).sum().item()) for (_, _, fr) in fractions_per_level])
                     for t in args.tau_sweep]
        print(f"    {prompt}: ref={ref_frame}, frames={len(frames)}, sp_per_tau={sp_counts}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--path_pred", type=str, default="output/render/lerf_ovs_sam_oracle_v2")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--tau_sweep", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
