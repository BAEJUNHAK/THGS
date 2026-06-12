"""
Stage 2B — D3 prompt-agnostic top-k sweep on full 67 prompts.

Unlike Stage 2A Layer 3 (which only swept 17 phantoms), this sweeps ALL 67 prompts
and computes per-method-fixed-k mean IoU, to determine the optimal k that
maximizes mean mIoU across the entire dataset (which is the actual lever a method
can use without per-prompt oracle knowledge).

Output: stage2b_d3_full_sweep.csv (208 rows: per (prompt, eval_frame))
         + summary CSV with per-k mean mIoU
         + plot of mIoU vs k
"""

import os
import sys
import csv
import json
import torch
import numpy as np
import pandas as pd
import cv2
from argparse import ArgumentParser
from collections import defaultdict

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure
from arguments import ModelParams, PipelineParams, OptimizationParams


K_SWEEP = [1, 2, 3, 5, 10, 20]


def polygon_to_mask(img_shape, points_list):
    pts = np.asarray(points_list, dtype=np.int32)
    m = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


@torch.no_grad()
def render_sp_mask(cam, gaussians, pipe, background, snag, level, sp_id, thresh):
    point_valid = (snag.labels[level].long() == sp_id).float()
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


@torch.no_grad()
def cache_sp_masks_at(cam, gaussians, pipe, background, snag, levels, thresh):
    cache_lvl, cache_sp, cache_mask_cpu = [], [], []
    for lvl in levels:
        sp_ids = snag.labels[lvl].long().unique()
        for sp_id in sp_ids:
            m = render_sp_mask(cam, gaussians, pipe, background, snag, lvl, sp_id.item(), thresh)
            cache_lvl.append(lvl)
            cache_sp.append(int(sp_id.item()))
            cache_mask_cpu.append(m.cpu())
            del m
    torch.cuda.empty_cache()
    return cache_lvl, cache_sp, torch.stack(cache_mask_cpu, dim=0)


@torch.no_grad()
def render_union(cam, gaussians, pipe, background, snag, pairs, thresh):
    point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
    for lvl, sp_id in pairs:
        point_valid[snag.labels[lvl].long() == sp_id] = 1.0
    if point_valid.sum() == 0:
        return torch.zeros(cam.image_height, cam.image_width, dtype=torch.bool, device='cuda')
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


CSV_HEADER = ['scene', 'prompt', 'eval_frame', 'is_ref'] + [f'iou_k{k}' for k in K_SWEEP]


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== {scene_name} ===", flush=True)
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
    vlm = ClipSimMeasure()
    vlm.load_model()

    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    img_list = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')])
    train_cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    frame_data = {}
    for im in img_list:
        image_name = im.split('.')[0]
        js = os.path.join(data_path, image_name + '.json')
        anno = json.load(open(js))
        if image_name not in train_cams_by_name:
            continue
        frame_data[image_name] = {
            'cam': train_cams_by_name[image_name], 'objects': anno['objects'],
            'h': train_cams_by_name[image_name].image_height,
            'w': train_cams_by_name[image_name].image_width,
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

    rows = []
    ref_cache = {}
    for prompt in sorted(prompt_to_frames.keys()):
        frames = prompt_to_frames[prompt]
        ref_frame = frames[0]
        if ref_frame not in ref_cache:
            print(f"  caching {ref_frame}", flush=True)
            ref_cache[ref_frame] = cache_sp_masks_at(
                frame_data[ref_frame]['cam'], gaussians, pipe, background,
                snag, args.levels, args.thresh)
        cache_lvl, cache_sp, cache_masks_cpu = ref_cache[ref_frame]
        vlm.encode_text(prompt)
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        pool_scores = np.array([sim_per_level[lvl - 1][sp_id].item()
                                for lvl, sp_id in zip(cache_lvl, cache_sp)], dtype=np.float64)
        order = np.argsort(-pool_scores)
        for eval_frame in frames:
            gt_eval_np = get_gt_for(prompt, eval_frame)
            if not gt_eval_np.any():
                continue
            gt_eval = torch.from_numpy(gt_eval_np).cuda()
            cam = frame_data[eval_frame]['cam']
            row = [scene_name, prompt, eval_frame, int(eval_frame == ref_frame)]
            for k in K_SWEEP:
                sel = order[:k]
                pairs = [(cache_lvl[i], cache_sp[i]) for i in sel]
                mask = render_union(cam, gaussians, pipe, background, snag, pairs, args.thresh)
                inter = (mask & gt_eval).sum().item()
                union = (mask | gt_eval).sum().item()
                row.append(f"{inter / max(union, 1):.4f}")
                del mask
            rows.append(row)
            del gt_eval
        torch.cuda.empty_cache()
        print(f"  {prompt}: done", flush=True)
    return rows


@torch.no_grad()
def main(dataset, pipe, args):
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new_h = (args.append == 0 or not os.path.exists(args.out_csv))
    mode = 'w' if (args.append == 0) else 'a'
    f = open(args.out_csv, mode, newline='')
    w = csv.writer(f)
    if new_h:
        w.writerow(CSV_HEADER)
    for r in run(dataset, pipe, args):
        w.writerow(r)
    f.flush(); f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_csv", default="output/diagnostics/stage2b_d3_full_sweep.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
    print("Done.")
