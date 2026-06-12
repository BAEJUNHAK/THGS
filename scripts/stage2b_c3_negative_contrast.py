"""
Stage 2B — C3 quick check: negative prompt contrast on 3 instance confusion phantoms.

For each instance-confusion phantom (pikachu, rubber duck with hat, onion segments):
  - Build score = cos(SP, target) - lambda * max_i cos(SP, negatives_i)
  - negatives = other prompts in scene
  - lambda ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
  - Measure: does oracle SP rank improve?
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


LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
INSTANCE_CONFUSION = [
    ('figurines', 'pikachu'),
    ('figurines', 'rubber duck with hat'),
    ('ramen', 'onion segments'),
]


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


CSV_HEADER = ['scene', 'prompt', 'lambda', 'oracle_rank', 'cos_oracle',
              'cos_top', 'top1_prompt_match']


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    targets_here = [(s, p) for (s, p) in INSTANCE_CONFUSION if s == scene_name]
    if not targets_here:
        print(f"[skip] no instance confusion targets for {scene_name}"); return []
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
    all_prompts = sorted(prompt_to_frames.keys())

    def get_gt_for(prompt, frame_name):
        fd = frame_data[frame_name]
        h, w = fd['h'], fd['w']
        mask = np.zeros((h, w), dtype=np.uint8)
        for obj in fd['objects']:
            if obj['category'] == prompt:
                m = polygon_to_mask((h, w), obj['segmentation'])
                mask = np.maximum(mask, m)
        return mask > 0

    # Pre-encode all prompt similarities at every level
    prompt_sims = {}  # prompt -> [sim_lvl0, sim_lvl1, ...]
    for p in all_prompts:
        vlm.encode_text(p)
        prompt_sims[p] = [vlm.compute_similarity(f).cpu().numpy() for f in snag.feat]

    rows = []
    for (_, target_prompt) in targets_here:
        ref_frame = prompt_to_frames[target_prompt][0]
        gt_ref_np = get_gt_for(target_prompt, ref_frame)
        gt_ref = torch.from_numpy(gt_ref_np).cuda()
        cam = frame_data[ref_frame]['cam']

        # Cache SP masks at ref frame
        cache_lvl, cache_sp, cache_masks_cpu = cache_sp_masks_at(
            cam, gaussians, pipe, background, snag, args.levels, args.thresh)
        # Identify oracle SP via mask IoU
        per_sp_iou = np.empty(len(cache_lvl), dtype=np.float32)
        chunk = 200
        for s in range(0, len(cache_lvl), chunk):
            e = min(s + chunk, len(cache_lvl))
            ch = cache_masks_cpu[s:e].cuda(non_blocking=True)
            inter = (ch & gt_ref[None]).sum(dim=(1, 2)).float()
            union = (ch | gt_ref[None]).sum(dim=(1, 2)).float()
            per_sp_iou[s:e] = (inter / union.clamp_min(1)).cpu().numpy()
            del ch
        oracle_idx = int(np.argmax(per_sp_iou))
        oracle_lvl = cache_lvl[oracle_idx]
        oracle_sp = cache_sp[oracle_idx]

        # Build pool scores for target prompt
        target_pool = np.array([prompt_sims[target_prompt][lvl - 1][sp_id]
                                for lvl, sp_id in zip(cache_lvl, cache_sp)], dtype=np.float64)

        # Build per-SP MAX-cos over negative prompts
        negatives = [p for p in all_prompts if p != target_prompt]
        if not negatives:
            continue
        max_neg = np.full(len(cache_lvl), -1.0)
        for neg in negatives:
            scores = np.array([prompt_sims[neg][lvl - 1][sp_id]
                               for lvl, sp_id in zip(cache_lvl, cache_sp)], dtype=np.float64)
            max_neg = np.maximum(max_neg, scores)

        for lam in LAMBDAS:
            adjusted = target_pool - lam * max_neg
            order = np.argsort(-adjusted)
            oracle_rank = int(np.where(order == oracle_idx)[0][0] + 1)
            cos_oracle = float(target_pool[oracle_idx])
            cos_top = float(target_pool[int(order[0])])
            # match the top1 prompt against scene's prompts
            top1_idx = int(order[0])
            rows.append([scene_name, target_prompt, lam,
                         oracle_rank, f"{cos_oracle:.4f}",
                         f"{cos_top:.4f}",
                         f"{adjusted[top1_idx]:.4f}"])
            print(f"  {target_prompt:25s} lambda={lam:.2f}: oracle_rank={oracle_rank} "
                  f"(was {int(np.where(np.argsort(-target_pool) == oracle_idx)[0][0] + 1)} at lambda=0)",
                  flush=True)
        del gt_ref
        torch.cuda.empty_cache()
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
    parser.add_argument("--out_csv", default="output/diagnostics/stage2b_c3_quickcheck.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
    print("Done.")
