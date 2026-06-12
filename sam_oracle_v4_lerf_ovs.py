"""
SAM-only Oracle v4 for LERF-OVS — Greedy Union (budget-aware optimal selection).

For each prompt:
  1. Use the prompt's first GT frame as ref. Cache each SP's mask at ref view.
  2. Greedy union: at each step pick the SP whose union-with-current maximizes GT IoU.
  3. Stop at given budgets or when IoU no longer improves.
  4. Render the union of selected SPs' Gaussians at every frame containing the prompt.

Fair comparison with CLIP-based (test_lerf.py, topk=3 at level=[2,3]).

Outputs: <path_pred>_budget{K}/<scene>/<frame>/<prompt>.png + <prompt>_gt.png
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
    cache_lvl, cache_sp, cache_mask = [], [], []
    for lvl in levels:
        sp_ids = snag.labels[lvl].long().unique()
        for sp_id in sp_ids:
            m = render_sp_mask(cam, gaussians, pipe, background, snag, lvl, sp_id.item(), thresh)
            cache_lvl.append(lvl)
            cache_sp.append(int(sp_id.item()))
            cache_mask.append(m)
    return cache_lvl, cache_sp, torch.stack(cache_mask, dim=0)


@torch.no_grad()
def greedy_union_select(cache_masks, gt, max_budget):
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
    print(f"\n=== Scene: {scene_name} (LERF-OVS Oracle v4 greedy, budgets={args.budget_sweep}) ===")

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

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
    for fr, fd in frame_data.items():
        for p in set(o['category'] for o in fd['objects']):
            prompt_to_frames[p].append(fr)
    for p in prompt_to_frames:
        prompt_to_frames[p] = sorted(prompt_to_frames[p])

    def get_gt_for(prompt, frame_name):
        fd = frame_data[frame_name]
        h, w = fd['h'], fd['w']
        mask = np.zeros((h, w), dtype=np.uint8)
        for obj in fd['objects']:
            if obj['category'] == prompt:
                m = polygon_to_mask((h, w), obj['segmentation'])
                mask = np.maximum(mask, m)
        return mask > 0

    ref_frame_cache = {}
    max_unroll = 100

    for prompt in sorted(prompt_to_frames.keys()):
        frames = prompt_to_frames[prompt]
        ref_frame = frames[0]
        gt_mask_np = get_gt_for(prompt, ref_frame)
        if not gt_mask_np.any():
            continue

        if ref_frame not in ref_frame_cache:
            print(f"  caching SP masks at frame {ref_frame}...", flush=True)
            ref_frame_cache[ref_frame] = cache_sp_masks_at(
                frame_data[ref_frame]['cam'], gaussians, pipe, background,
                snag, args.levels, args.thresh,
            )
        cache_lvl, cache_sp, cache_masks = ref_frame_cache[ref_frame]

        gt = torch.from_numpy(gt_mask_np).cuda()
        sel_order, ious_at = greedy_union_select(cache_masks, gt, max_unroll)
        n_picked = len(sel_order)

        prompt_safe = prompt.replace(' ', '_')
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

            for frame_name in frames:
                fd = frame_data[frame_name]
                out_dir = os.path.join(args.path_pred + f"_{label}", scene_name, frame_name)
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

        seq = ' → '.join([f'{iou:.3f}' for iou in ious_at[:5]])
        print(f"    {prompt}: ref={ref_frame}, frames={len(frames)}, picked={n_picked}, IoU prog: {seq}{'...' if len(ious_at)>5 else ''}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--path_pred", type=str, default="output/render/lerf_ovs_sam_oracle_v4")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--budget_sweep", nargs="+",
                        default=[1, 2, 3, 5, 'unlimited'],
                        type=lambda x: int(x) if x.isdigit() else x)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
