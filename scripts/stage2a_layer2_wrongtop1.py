"""
Stage 2A — Layer 2: Wrong-top1 forensics + per-frame rank trajectory.

For each phantom prompt, the CLIP top-1 SP is "what the method actually picks".
Question: what IS that SP?

Per phantom:
  - clip_top1 SP id (already in stage 1 csv)
  - render clip_top1 SP at ref_frame
  - measure its IoU with EACH OF THE OTHER prompts' GT in the same scene
  - identify the strongest match → "wrong_match_prompt"
  - record wrong_match_iou, wrong_match_prompt
  - record overlap with all scene GTs (for E1 direction analysis)

Per-frame rank trajectory:
  - For each (prompt, GT_frame), re-define oracle SP at THAT frame's GT
  - Compute rank of that frame's oracle SP in the SAME CLIP pool
  - Yields rank_at_frame[1..N] per phantom
  - "Per-frame rank trajectory" = how stable is the rank across views?
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
def per_sp_iou_chunked(masks_cpu, gt_gpu, chunk=200):
    n = masks_cpu.shape[0]
    out = torch.empty(n, dtype=torch.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        chunk_gpu = masks_cpu[s:e].cuda(non_blocking=True)
        inter = (chunk_gpu & gt_gpu[None]).sum(dim=(1, 2)).float()
        union = (chunk_gpu | gt_gpu[None]).sum(dim=(1, 2)).float()
        out[s:e] = (inter / union.clamp_min(1)).cpu()
        del chunk_gpu
    torch.cuda.empty_cache()
    return out.numpy()


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


FORENSIC_HEADER = [
    'scene', 'prompt',
    # wrong top-1 forensics
    'wrong_top1_lvl', 'wrong_top1_sp_id',
    'wrong_top1_pixels',
    'wrong_top1_best_match_prompt', 'wrong_top1_best_match_iou',
    'wrong_top1_overlap_with_gt_iou',  # IoU with the *correct* prompt's GT
    'wrong_top1_n_match_prompts',  # how many other prompts have iou ≥ 0.10
    'wrong_top1_top3_matches',  # comma-separated top-3 wrongprompt:iou pairs
]


TRAJECTORY_HEADER = [
    'scene', 'prompt', 'frame', 'is_ref',
    'oracle_lvl_at_frame', 'oracle_sp_id_at_frame', 'oracle_iou_at_frame',
    'oracle_purity_at_frame', 'oracle_completeness_at_frame',
    'rank_in_clip_pool',  # rank of this frame's oracle in the same CLIP pool
    'same_as_ref_oracle',
    'cos_oracle_at_frame', 'cos_top', 'z_margin_at_frame',
]


@torch.no_grad()
def run_scene(dataset, pipe, args, target_list):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    targets_here = [t for t in target_list if t['scene'] == scene_name]
    if not targets_here:
        return [], []
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

    ref_cache = {}
    forensic_rows = []
    trajectory_rows = []

    for t in targets_here:
        prompt = t['prompt']
        frames = prompt_to_frames.get(prompt, [])
        if not frames:
            continue
        ref_frame = frames[0]
        gt_ref_np = get_gt_for(prompt, ref_frame)
        gt_ref = torch.from_numpy(gt_ref_np).cuda()
        if ref_frame not in ref_cache:
            print(f"  caching SP masks at {ref_frame}...", flush=True)
            ref_cache[ref_frame] = cache_sp_masks_at(
                frame_data[ref_frame]['cam'], gaussians, pipe, background,
                snag, args.levels, args.thresh)
        cache_lvl, cache_sp, cache_masks_cpu = ref_cache[ref_frame]

        # CLIP pool
        vlm.encode_text(prompt)
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        pool_scores = np.array([sim_per_level[lvl - 1][sp_id].item()
                                for lvl, sp_id in zip(cache_lvl, cache_sp)], dtype=np.float64)
        pool_size = len(pool_scores)
        pool_mean = float(pool_scores.mean())
        pool_std = float(pool_scores.std(ddof=1) if pool_size > 1 else 0.0)
        cos_top = float(pool_scores.max())

        # === Forensic of wrong top-1 ===
        order = np.argsort(-pool_scores)
        clip_top1_idx = int(order[0])
        top1_lvl = cache_lvl[clip_top1_idx]
        top1_sp = cache_sp[clip_top1_idx]
        top1_mask_cpu = cache_masks_cpu[clip_top1_idx]
        top1_pixels = int(top1_mask_cpu.sum().item())

        # IoU of top1 mask vs each OTHER prompt's GT at ref_frame
        # (use this prompt's ref_frame; for other prompts, take their GT if exists at this frame)
        top1_mask_gpu = top1_mask_cpu.cuda()
        wrong_match = []
        for p2 in prompt_to_frames:
            if p2 == prompt:
                continue
            # Check if p2 has GT at this scene's ref_frame
            gt2_np = get_gt_for(p2, ref_frame)
            if not gt2_np.any():
                continue
            gt2 = torch.from_numpy(gt2_np).cuda()
            inter = (top1_mask_gpu & gt2).sum().item()
            union = (top1_mask_gpu | gt2).sum().item()
            iou = inter / max(union, 1)
            wrong_match.append((p2, iou))
            del gt2
        # also note IoU with the correct prompt's GT
        inter_self = (top1_mask_gpu & gt_ref).sum().item()
        union_self = (top1_mask_gpu | gt_ref).sum().item()
        overlap_self = inter_self / max(union_self, 1)
        del top1_mask_gpu
        torch.cuda.empty_cache()

        wrong_match.sort(key=lambda x: -x[1])
        best_match = wrong_match[0] if wrong_match else ('', 0.0)
        n_above = sum(1 for _, iou in wrong_match if iou >= 0.10)
        top3 = ";".join(f"{p}:{iou:.2f}" for p, iou in wrong_match[:3])

        forensic_rows.append([
            scene_name, prompt,
            top1_lvl, top1_sp,
            top1_pixels,
            best_match[0], f"{best_match[1]:.4f}",
            f"{overlap_self:.4f}",
            n_above, top3,
        ])
        print(f"  {prompt:25s} wrong_top1={top1_lvl}.{top1_sp} "
              f"best_match=({best_match[0]}, iou={best_match[1]:.2f}) "
              f"overlap_self={overlap_self:.2f}", flush=True)

        # === Per-frame trajectory: re-define oracle at each frame ===
        # For each GT frame for this prompt, define oracle SP at THAT frame and
        # measure rank in the SAME CLIP pool (the pool is prompt-level, not frame-level).
        for frame_name in frames:
            gt_at_frame_np = get_gt_for(prompt, frame_name)
            if not gt_at_frame_np.any():
                continue
            cam = frame_data[frame_name]['cam']
            if frame_name not in ref_cache:
                ref_cache[frame_name] = cache_sp_masks_at(
                    cam, gaussians, pipe, background,
                    snag, args.levels, args.thresh)
            cache_lvl2, cache_sp2, cache_masks2 = ref_cache[frame_name]
            gt_at_frame = torch.from_numpy(gt_at_frame_np).cuda()
            per_sp_iou_at_frame = per_sp_iou_chunked(cache_masks2, gt_at_frame, chunk=200)
            oracle_idx_at = int(np.argmax(per_sp_iou_at_frame))
            oracle_lvl_at = cache_lvl2[oracle_idx_at]
            oracle_sp_at = cache_sp2[oracle_idx_at]
            iou_at = float(per_sp_iou_at_frame[oracle_idx_at])

            # purity/comp at this frame
            sp_mask_at = cache_masks2[oracle_idx_at].cuda()
            inter_at = (sp_mask_at & gt_at_frame).sum().item()
            sp_sum_at = sp_mask_at.sum().item()
            gt_sum_at = gt_at_frame.sum().item()
            pur_at = inter_at / max(sp_sum_at, 1)
            comp_at = inter_at / max(gt_sum_at, 1)
            del sp_mask_at

            # Rank of THIS frame's oracle in the SHARED CLIP pool of THIS prompt
            # Note: pool is built from the ref_frame's SP masks (same SPs at lvl=[2,3])
            # We map this frame's oracle_sp_at to its index in the ref_frame pool
            # The pool indices are by (level, sp_id) — should match since same SPs.
            try:
                pool_idx_for_oracle = next(
                    i for i in range(len(cache_lvl))
                    if cache_lvl[i] == oracle_lvl_at and cache_sp[i] == oracle_sp_at)
                cos_at_frame_oracle = float(pool_scores[pool_idx_for_oracle])
                rank_at = int(np.where(order == pool_idx_for_oracle)[0][0] + 1)
            except StopIteration:
                cos_at_frame_oracle = float('nan')
                rank_at = -1

            z_at = (cos_top - cos_at_frame_oracle) / pool_std if pool_std > 1e-9 else 0.0
            same_as_ref = int(oracle_lvl_at == cache_lvl[int(np.argmax(per_sp_iou_chunked(cache_masks_cpu, gt_ref)))]
                              and oracle_sp_at == cache_sp[int(np.argmax(per_sp_iou_chunked(cache_masks_cpu, gt_ref)))])

            trajectory_rows.append([
                scene_name, prompt, frame_name, int(frame_name == ref_frame),
                oracle_lvl_at, oracle_sp_at, f"{iou_at:.4f}",
                f"{pur_at:.4f}", f"{comp_at:.4f}",
                rank_at, same_as_ref,
                f"{cos_at_frame_oracle:.6f}", f"{cos_top:.6f}", f"{z_at:.4f}",
            ])
            del gt_at_frame
        del gt_ref

    return forensic_rows, trajectory_rows


def build_target_list(persistent_csv):
    phantoms = pd.read_csv(persistent_csv)
    targets = []
    for _, r in phantoms.iterrows():
        targets.append({'scene': r['scene'], 'prompt': r['prompt']})
    return targets


@torch.no_grad()
def main(dataset, pipe, args):
    targets = build_target_list(args.persistent_csv)
    os.makedirs(os.path.dirname(args.forensic_csv), exist_ok=True)
    # forensic
    new_f = (args.append == 0 or not os.path.exists(args.forensic_csv))
    mode_f = 'w' if (args.append == 0) else 'a'
    f1 = open(args.forensic_csv, mode_f, newline='')
    w1 = csv.writer(f1)
    if new_f:
        w1.writerow(FORENSIC_HEADER)
    # trajectory
    new_t = (args.append == 0 or not os.path.exists(args.trajectory_csv))
    mode_t = 'w' if (args.append == 0) else 'a'
    f2 = open(args.trajectory_csv, mode_t, newline='')
    w2 = csv.writer(f2)
    if new_t:
        w2.writerow(TRAJECTORY_HEADER)

    forensic_rows, trajectory_rows = run_scene(dataset, pipe, args, targets)
    for r in forensic_rows:
        w1.writerow(r)
    for r in trajectory_rows:
        w2.writerow(r)
    f1.flush(); f1.close()
    f2.flush(); f2.close()
    print(f"\nWrote {len(forensic_rows)} forensic rows + {len(trajectory_rows)} trajectory rows")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--forensic_csv",
                        default="output/diagnostics/stage2a_layer2_forensic.csv")
    parser.add_argument("--trajectory_csv",
                        default="output/diagnostics/stage2a_layer2_trajectory.csv")
    parser.add_argument("--persistent_csv",
                        default="output/diagnostics/persistent_phantoms_17.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
    print("Done.")
