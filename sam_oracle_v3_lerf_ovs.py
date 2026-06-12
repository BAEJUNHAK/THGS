"""
SAM-only Oracle v3 for LERF-OVS — Best-IoU topk variant.

For each prompt:
  1. Use the prompt's *first* GT frame as the reference view.
  2. At the ref frame, render each NAG superpoint individually and cache its 2D mask.
     (Cache is per ref frame; prompts that share a ref frame reuse the cache.)
  3. Compute per-SP IoU with prompt's polygon-rasterized GT.
  4. For each k in topk_sweep, select top-k SPs by IoU and render their union at all
     frames containing the prompt's GT.

Outputs same layout as v2:
  <path_pred>_topk{K}/<scene>/<frame>/<prompt>.png      (pred)
  <path_pred>_topk{K}/<scene>/<frame>/<prompt>_gt.png   (raster GT)
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

from gaussian_renderer import render
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
def render_sp_mask(cam, gaussians, pipe, background, snag, level, sp_id, thresh):
    point_valid = (snag.labels[level].long() == sp_id).float()
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


@torch.no_grad()
def cache_sp_masks_at(cam, gaussians, pipe, background, snag, levels, thresh):
    """For a given camera, render each SP at each chosen level into a stacked tensor."""
    H, W = cam.image_height, cam.image_width
    cache_lvl, cache_sp, cache_mask = [], [], []
    for lvl in levels:
        sp_ids = snag.labels[lvl].long().unique()
        for sp_id in sp_ids:
            m = render_sp_mask(cam, gaussians, pipe, background, snag, lvl, sp_id.item(), thresh)
            cache_lvl.append(lvl)
            cache_sp.append(int(sp_id.item()))
            cache_mask.append(m)
    cache_masks = torch.stack(cache_mask, dim=0)
    return cache_lvl, cache_sp, cache_masks


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Scene: {scene_name} (LERF-OVS Oracle v3, topk={args.topk_sweep}) ===")

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    print(f"  loaded {snag.gaussian_num} Gaussians, NAG levels={[l.unique().numel() for l in snag.labels]}")

    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    img_list = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')])
    train_cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    frame_data = {}
    for im in img_list:
        image_name = im.split('.')[0]
        js_file = os.path.join(data_path, image_name + '.json')
        anno = json.load(open(js_file))
        if image_name not in train_cams_by_name:
            continue
        cam = train_cams_by_name[image_name]
        frame_data[image_name] = {
            'cam': cam, 'objects': anno['objects'],
            'h': cam.image_height, 'w': cam.image_width,
        }

    prompt_to_frames = defaultdict(list)
    for frame_name, fd in frame_data.items():
        for prompt in set(o['category'] for o in fd['objects']):
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

    # Cache SP masks per ref frame (reuse across prompts sharing the frame)
    ref_frame_cache = {}  # frame_name -> (cache_lvl, cache_sp, cache_masks_HW_bool)

    for prompt in sorted(prompt_to_frames.keys()):
        frames = prompt_to_frames[prompt]
        ref_frame = frames[0]
        gt_mask_np = get_gt_for(prompt, ref_frame)
        if not gt_mask_np.any():
            print(f"    [empty GT] {prompt} at {ref_frame}")
            continue

        if ref_frame not in ref_frame_cache:
            print(f"  caching SP masks at frame {ref_frame}...", flush=True)
            ref_frame_cache[ref_frame] = cache_sp_masks_at(
                frame_data[ref_frame]['cam'], gaussians, pipe, background,
                snag, args.levels, args.thresh,
            )
        cache_lvl, cache_sp, cache_masks = ref_frame_cache[ref_frame]

        gt = torch.from_numpy(gt_mask_np).cuda()
        inter = (cache_masks & gt[None]).sum(dim=(1, 2)).float()
        union = (cache_masks | gt[None]).sum(dim=(1, 2)).float()
        ious = inter / union.clamp_min(1)
        order = torch.argsort(ious, descending=True)
        top1 = ious[order[0]].item()

        prompt_safe = prompt.replace(' ', '_')
        for k in args.topk_sweep:
            sel_idx = order[:k].cpu().tolist()
            point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
            for j in sel_idx:
                lvl = cache_lvl[j]
                sp_id = cache_sp[j]
                picked = snag.labels[lvl].long() == sp_id
                point_valid[picked] = 1.0

            for frame_name in frames:
                fd = frame_data[frame_name]
                out_dir = os.path.join(args.path_pred + f"_topk{k}", scene_name, frame_name)
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

        print(f"    {prompt}: ref={ref_frame}, frames={len(frames)}, top1_iou={top1:.3f}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--path_pred", type=str, default="output/render/lerf_ovs_sam_oracle_v3")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--topk_sweep", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
