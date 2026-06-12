"""
Stage 2A — Layer 3: D3 top-k sweep + cross-frame oracle persistence.

For each phantom prompt at ref_frame:
  - Build CLIP pool of SP scores
  - For k ∈ {1, 2, 3, 5, 10}:
      - Take top-k SPs by CLIP score
      - Render union at ref_frame
      - Compute IoU vs GT@ref
      - For each eval frame:
        - Render union at eval frame, IoU vs GT@eval
      - Record per-prompt mean actual_iou across eval frames

Key question: does any k recover the phantoms?
  - If smaller k (1) raises mIoU → over-union (D3) is real
  - If larger k (5, 10) raises mIoU → over-union not the issue; oracle is just buried in pool
  - If no k recovers → phantom is genuinely SP-feature failure

Also computes oracle-rank under TOP-K policy (different from rank-1):
  - "phantom rank in top-K" = whether oracle is included in top-K
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
def render_union(cam, gaussians, pipe, background, snag, sel_lvl_sp_pairs, thresh):
    point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
    for lvl, sp_id in sel_lvl_sp_pairs:
        picked = snag.labels[lvl].long() == sp_id
        point_valid[picked] = 1.0
    if point_valid.sum() == 0:
        return torch.zeros(cam.image_height, cam.image_width, dtype=torch.bool, device='cuda')
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


K_SWEEP = [1, 2, 3, 5, 10]


CSV_HEADER = ['scene', 'prompt', 'eval_frame', 'is_ref'] + \
             [f'iou_k{k}' for k in K_SWEEP] + \
             ['oracle_in_top10', 'oracle_in_top5', 'oracle_in_top3', 'oracle_in_top1']


@torch.no_grad()
def run_scene(dataset, pipe, args, target_set):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    if scene_name not in {t[0] for t in target_set}:
        return []
    targets_here = [(s, p) for (s, p) in target_set if s == scene_name]
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

    for (_, prompt) in targets_here:
        frames = prompt_to_frames.get(prompt, [])
        if not frames:
            continue
        ref_frame = frames[0]
        if ref_frame not in ref_cache:
            print(f"  caching {ref_frame}...", flush=True)
            ref_cache[ref_frame] = cache_sp_masks_at(
                frame_data[ref_frame]['cam'], gaussians, pipe, background,
                snag, args.levels, args.thresh)
        cache_lvl, cache_sp, cache_masks_cpu = ref_cache[ref_frame]

        # CLIP pool
        vlm.encode_text(prompt)
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        pool_scores = np.array([sim_per_level[lvl - 1][sp_id].item()
                                for lvl, sp_id in zip(cache_lvl, cache_sp)], dtype=np.float64)
        order = np.argsort(-pool_scores)

        # Oracle at ref_frame for "oracle_in_topK" check
        gt_ref_np = get_gt_for(prompt, ref_frame)
        gt_ref = torch.from_numpy(gt_ref_np).cuda()
        # per-SP IoU
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
        oracle_rank = int(np.where(order == oracle_idx)[0][0] + 1)

        oracle_in_top1 = int(oracle_rank <= 1)
        oracle_in_top3 = int(oracle_rank <= 3)
        oracle_in_top5 = int(oracle_rank <= 5)
        oracle_in_top10 = int(oracle_rank <= 10)
        del gt_ref

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
                mask_pred = render_union(cam, gaussians, pipe, background, snag, pairs, args.thresh)
                inter = (mask_pred & gt_eval).sum().item()
                union = (mask_pred | gt_eval).sum().item()
                iou = inter / max(union, 1)
                row.append(f"{iou:.4f}")
                del mask_pred
            row += [oracle_in_top10, oracle_in_top5, oracle_in_top3, oracle_in_top1]
            rows.append(row)
            del gt_eval
        torch.cuda.empty_cache()
        print(f"  {prompt:25s} oracle_rank={oracle_rank}, "
              f"in_top(1/3/5/10)=({oracle_in_top1}/{oracle_in_top3}/{oracle_in_top5}/{oracle_in_top10})",
              flush=True)
    return rows


def build_target_list(persistent_csv):
    df = pd.read_csv(persistent_csv)
    return [(r['scene'], r['prompt']) for _, r in df.iterrows()]


@torch.no_grad()
def main(dataset, pipe, args):
    targets = set(build_target_list(args.persistent_csv))
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new_h = (args.append == 0 or not os.path.exists(args.out_csv))
    mode = 'w' if (args.append == 0) else 'a'
    f = open(args.out_csv, mode, newline='')
    w = csv.writer(f)
    if new_h:
        w.writerow(CSV_HEADER)
    rows = run_scene(dataset, pipe, args, targets)
    for r in rows:
        w.writerow(r)
    f.flush(); f.close()
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_csv", default="output/diagnostics/stage2a_layer3_d3.csv")
    parser.add_argument("--persistent_csv", default="output/diagnostics/persistent_phantoms_17.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
    print("Done.")
