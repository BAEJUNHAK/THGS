"""
LERF-OVS per-prompt failure diagnostic.

For each (scene, prompt) at the prompt's ref frame (= first GT frame containing prompt):
  1. Cache per-SP masks at ref view (levels [2, 3]).
  2. Per-SP oracle IoU vs GT@ref.
  3. Per-SP CLIP relevancy score (canon-contrast, same as test_lerf.py).
  4. Correct SP = greedy v4 first pick (highest oracle IoU).
  5. Oracle SPs = greedy union budget=3 (matches sam_oracle_v4 ceiling).
  6. Actual SPs = unified top-3 by CLIP score (matches test_lerf.py topk=3 at level=[2,3]).
  7. Render Oracle / Actual at ref view, compute IoU + pixel-level precision/recall/FP/FN.
  8. Append CSV row.

Output: <out_csv> with columns documented in header.

This is the data source for md/lerf_ovs_failure_analysis_plan.md Steps 1-4.
"""

import os
import sys
import csv
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
    """Render every SP at given levels at the camera view; store masks on CPU.

    Returns (cache_lvl: list[int], cache_sp: list[int], masks_cpu: torch.Tensor on CPU).
    """
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
def per_sp_iou_chunked(masks_cpu, gt_gpu, chunk=200):
    """Compute IoU(SP, gt) for all SPs by chunking masks from CPU to GPU.

    masks_cpu: (N, H, W) bool tensor on CPU.
    gt_gpu:    (H, W) bool tensor on GPU.
    Returns:   (N,) float32 IoU array on CPU.
    """
    n = masks_cpu.shape[0]
    out = torch.empty(n, dtype=torch.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        chunk_gpu = masks_cpu[s:e].cuda(non_blocking=True)
        inter = (chunk_gpu & gt_gpu[None]).sum(dim=(1, 2)).float()
        union = (chunk_gpu | gt_gpu[None]).sum(dim=(1, 2)).float()
        out[s:e] = (inter / union.clamp_min(1)).cpu()
        del chunk_gpu, inter, union
    torch.cuda.empty_cache()
    return out.numpy()


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


CSV_HEADER = [
    'scene', 'prompt', 'frame', 'is_ref_frame', 'gt_pixels', 'image_pixels',
    'oracle_iou', 'actual_iou',
    'oracle_sp_count',
    'correct_sp_lvl', 'correct_sp_id', 'correct_sp_clip_rank',
    'clip_top1_lvl', 'clip_top1_sp_id', 'clip_top1_oracle_iou',
    'tp_pixels', 'tn_pixels', 'fp_pixels', 'fn_pixels',
    'precision', 'recall',
    'pool_size',
]


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Scene: {scene_name} (LERF-OVS diagnostic) ===", flush=True)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    # iteration=0 is treated as falsy by Scene; use -1 to bypass then load_ply manually
    iter_arg = args.iteration if args.iteration > 0 else -1
    scene = Scene(dataset, gaussians, iter_arg, load_sem=False, shuffle=False)
    # If requested iteration is 0, Scene falls back to input.ply due to truthy check bug. Force-load
    if args.iteration == 0:
        ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_0", "point_cloud.ply")
        if os.path.exists(ply_path):
            print(f"  Overriding scene load: gaussians from {ply_path}", flush=True)
            gaussians.load_ply(ply_path)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    print(f"  Gaussians={snag.gaussian_num}, levels per SP count: "
          f"{[l.unique().numel() for l in snag.labels]}", flush=True)

    vlm = ClipSimMeasure()
    vlm.load_model()

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
    print(f"  {len(frame_data)} GT frames, {len(prompt_to_frames)} unique prompts", flush=True)

    def get_gt_for(prompt, frame_name):
        fd = frame_data[frame_name]
        h, w = fd['h'], fd['w']
        mask = np.zeros((h, w), dtype=np.uint8)
        for obj in fd['objects']:
            if obj['category'] == prompt:
                m = polygon_to_mask((h, w), obj['segmentation'])
                mask = np.maximum(mask, m)
        return mask > 0

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new_file = not os.path.exists(args.out_csv)
    f_csv = open(args.out_csv, 'a', newline='')
    writer = csv.writer(f_csv)
    if new_file:
        writer.writerow(CSV_HEADER)

    ref_frame_cache = {}

    for prompt in sorted(prompt_to_frames.keys()):
        frames = prompt_to_frames[prompt]
        ref_frame = frames[0]
        gt_np = get_gt_for(prompt, ref_frame)
        if not gt_np.any():
            print(f"  [empty GT] skip {prompt}", flush=True)
            continue

        if ref_frame not in ref_frame_cache:
            print(f"  caching SP masks at frame {ref_frame}...", flush=True)
            ref_frame_cache[ref_frame] = cache_sp_masks_at(
                frame_data[ref_frame]['cam'], gaussians, pipe, background,
                snag, args.levels, args.thresh,
            )
        cache_lvl, cache_sp, cache_masks_cpu = ref_frame_cache[ref_frame]
        gt = torch.from_numpy(gt_np).cuda()

        per_sp_iou = per_sp_iou_chunked(cache_masks_cpu, gt, chunk=args.chunk)

        vlm.encode_text(prompt)
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        pool_scores = []
        for lvl, sp_id in zip(cache_lvl, cache_sp):
            score = sim_per_level[lvl - 1][sp_id].item()
            pool_scores.append(score)
        pool_scores_arr = np.array(pool_scores)

        topk_keep = min(args.topk_keep, len(per_sp_iou))
        top_idx = np.argpartition(-per_sp_iou, topk_keep - 1)[:topk_keep]
        top_idx = top_idx[np.argsort(-per_sp_iou[top_idx])]
        top_masks_gpu = cache_masks_cpu[top_idx].cuda()

        sel_top, ious_at = greedy_union_select(top_masks_gpu, gt, args.budget)
        del top_masks_gpu
        torch.cuda.empty_cache()
        oracle_sp_count = len(sel_top)
        if oracle_sp_count == 0:
            print(f"  [no oracle SP] skip {prompt}", flush=True)
            continue
        sel_order = [int(top_idx[i]) for i in sel_top]
        first_pick = sel_order[0]
        correct_sp_lvl = cache_lvl[first_pick]
        correct_sp_id = cache_sp[first_pick]

        order = np.argsort(-pool_scores_arr)
        correct_sp_rank = int(np.where(order == first_pick)[0][0] + 1)

        clip_top1_idx = int(order[0])
        clip_top1_lvl = cache_lvl[clip_top1_idx]
        clip_top1_sp_id = cache_sp[clip_top1_idx]
        clip_top1_oracle_iou = float(per_sp_iou[clip_top1_idx])

        oracle_pairs = [(cache_lvl[i], cache_sp[i]) for i in sel_order]
        actual_sel = order[:3].tolist()
        actual_pairs = [(cache_lvl[i], cache_sp[i]) for i in actual_sel]

        # Render Oracle and Actual at every frame containing this prompt
        per_frame_log = []
        for frame_name in frames:
            cam = frame_data[frame_name]['cam']
            gt_at_frame_np = get_gt_for(prompt, frame_name)
            if not gt_at_frame_np.any():
                continue
            gt_at_frame = torch.from_numpy(gt_at_frame_np).cuda()

            oracle_mask = render_union(
                cam, gaussians, pipe, background, snag,
                oracle_pairs, args.thresh)
            actual_mask = render_union(
                cam, gaussians, pipe, background, snag,
                actual_pairs, args.thresh)

            o_inter = (oracle_mask & gt_at_frame).sum().item()
            o_union = (oracle_mask | gt_at_frame).sum().item()
            oracle_iou = o_inter / max(o_union, 1)

            a_inter = (actual_mask & gt_at_frame).sum().item()
            a_union = (actual_mask | gt_at_frame).sum().item()
            actual_iou = a_inter / max(a_union, 1)

            tp = (actual_mask & gt_at_frame).sum().item()
            fp = (actual_mask & ~gt_at_frame).sum().item()
            fn = (~actual_mask & gt_at_frame).sum().item()
            tn = (~actual_mask & ~gt_at_frame).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            gt_pixels = int(gt_at_frame.sum().item())
            image_pixels = int(gt_at_frame.numel())

            writer.writerow([
                scene_name, prompt, frame_name,
                1 if frame_name == ref_frame else 0,
                gt_pixels, image_pixels,
                f'{oracle_iou:.6f}', f'{actual_iou:.6f}',
                oracle_sp_count,
                correct_sp_lvl, correct_sp_id, correct_sp_rank,
                clip_top1_lvl, clip_top1_sp_id, f'{clip_top1_oracle_iou:.6f}',
                tp, tn, fp, fn,
                f'{precision:.6f}', f'{recall:.6f}',
                len(pool_scores_arr),
            ])
            per_frame_log.append((frame_name, oracle_iou, actual_iou))
            del oracle_mask, actual_mask, gt_at_frame

        f_csv.flush()
        oracle_mean = np.mean([x[1] for x in per_frame_log]) if per_frame_log else 0
        actual_mean = np.mean([x[2] for x in per_frame_log]) if per_frame_log else 0
        print(f"  {prompt}: {len(per_frame_log)} frames | oracle={oracle_mean:.3f} "
              f"actual={actual_mean:.3f} (mean) | rank={correct_sp_rank}/{len(pool_scores_arr)}",
              flush=True)

    f_csv.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_csv", type=str,
                        default="output/diagnostics/lerf_ovs_per_prompt.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=200,
                        help="chunk size for per-SP IoU GPU computation")
    parser.add_argument("--topk_keep", type=int, default=30,
                        help="keep top-K SPs by single-SP IoU for greedy union")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="model iteration to load (THGS=30000, ReLaGS=0)")
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
