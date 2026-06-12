"""
B7 + A4 combined analysis on LERF-OVS.

Produces 1 CSV row per (scene, prompt, eval_frame) — matches existing
output/diagnostics/lerf_ovs_per_prompt.csv structure (208 rows for THGS LERF-OVS).

B7 (Oracle SP purity/completeness/fragmentation):
  - For each (scene, prompt): identify oracle SP at ref_frame (frames[0])
    as the single SP with highest IoU vs GT@ref (rendered-mask space).
  - For each eval_frame: re-render the oracle SP, measure
        purity_at_eval     = |SP_at_eval ∩ GT@eval| / |SP_at_eval|
        completeness_at_eval = |SP_at_eval ∩ GT@eval| / |GT@eval|
  - Also record fragmentation (= greedy-union SP count) — ref-frame-level, constant per prompt.
  - Top-3 alt oracles (also ref-frame-level).

A4 (Oracle-rank margin):
  - Pool cosines, oracle rank, z-score margin, percentile margin —
    prompt-level (SP-CLIP feature is view-invariant). Constant per (prompt, eval_frame).

Output: output/diagnostics/b7_a4_combined.csv
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
def per_sp_stats_chunked(masks_cpu, gt_gpu, chunk=200):
    """Per-SP IoU, purity (=|SP∩GT|/|SP|), completeness (=|SP∩GT|/|GT|) vs given GT mask.

    All three returned as np.float32 arrays of length N.
    """
    n = masks_cpu.shape[0]
    out_iou = torch.empty(n, dtype=torch.float32)
    out_pur = torch.empty(n, dtype=torch.float32)
    out_com = torch.empty(n, dtype=torch.float32)
    gt_sum = gt_gpu.sum().float().clamp_min(1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        chunk_gpu = masks_cpu[s:e].cuda(non_blocking=True)
        sp_sum = chunk_gpu.sum(dim=(1, 2)).float()
        inter = (chunk_gpu & gt_gpu[None]).sum(dim=(1, 2)).float()
        union = (chunk_gpu | gt_gpu[None]).sum(dim=(1, 2)).float()
        out_iou[s:e] = (inter / union.clamp_min(1)).cpu()
        out_pur[s:e] = (inter / sp_sum.clamp_min(1)).cpu()
        out_com[s:e] = (inter / gt_sum).cpu()
        del chunk_gpu, inter, union, sp_sum
    torch.cuda.empty_cache()
    return out_iou.numpy(), out_pur.numpy(), out_com.numpy()


@torch.no_grad()
def single_sp_stats(cam, gaussians, pipe, background, snag, level, sp_id, gt_gpu, thresh):
    """Render one SP at given cam, compute (iou, purity, completeness) vs gt_gpu."""
    sp_mask = render_sp_mask(cam, gaussians, pipe, background, snag, level, sp_id, thresh)
    inter = (sp_mask & gt_gpu).sum().item()
    sp_sum = sp_mask.sum().item()
    gt_sum = gt_gpu.sum().item()
    union = (sp_mask | gt_gpu).sum().item()
    iou = inter / max(union, 1)
    purity = inter / max(sp_sum, 1)
    completeness = inter / max(gt_sum, 1)
    return iou, purity, completeness, sp_sum


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


CSV_HEADER = [
    # identifiers
    'scene', 'prompt', 'frame', 'is_ref_frame', 'gt_pixels', 'image_pixels',
    # B7 — oracle SP identity (defined at ref_frame, constant per prompt)
    'oracle_lvl', 'oracle_sp_id', 'oracle_iou_ref', 'oracle_purity_ref', 'oracle_completeness_ref',
    # B7 — fragmentation (ref-frame level)
    'fragmentation', 'union_iou_ref',
    # B7 — top-3 alt oracles (ref-frame level)
    'alt2_lvl', 'alt2_sp_id', 'alt2_iou_ref', 'alt2_purity_ref', 'alt2_completeness_ref',
    'alt3_lvl', 'alt3_sp_id', 'alt3_iou_ref', 'alt3_purity_ref', 'alt3_completeness_ref',
    # B7 — per-eval-frame oracle behavior (re-rendered at this frame)
    'oracle_iou_eval', 'oracle_purity_eval', 'oracle_completeness_eval', 'oracle_sp_pixels_eval',
    # A4 — pool statistics (prompt-level, constant per (prompt, frame))
    'pool_size',
    'cos_top', 'cos_oracle',
    'pool_mean', 'pool_std', 'pool_min', 'pool_max', 'pool_entropy',
    'oracle_rank',
    'raw_margin', 'z_margin', 'percentile_margin',
    # CLIP top-1 SP identity + its ref-frame purity/completeness
    'clip_top1_lvl', 'clip_top1_sp_id', 'clip_top1_oracle_iou_ref',
    'clip_top1_purity_ref', 'clip_top1_completeness_ref',
]


def softmax_entropy(scores):
    s = torch.as_tensor(scores, dtype=torch.float64)
    p = torch.softmax(s, dim=0)
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Scene: {scene_name} (B7+A4 analysis) ===", flush=True)

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
    mode = 'a' if (os.path.exists(args.out_csv) and args.append) else 'w'
    new_header = (mode == 'w')
    f_csv = open(args.out_csv, mode, newline='')
    writer = csv.writer(f_csv)
    if new_header:
        writer.writerow(CSV_HEADER)

    ref_frame_cache = {}

    for prompt in sorted(prompt_to_frames.keys()):
        frames = prompt_to_frames[prompt]
        ref_frame = frames[0]
        gt_ref_np = get_gt_for(prompt, ref_frame)
        if not gt_ref_np.any():
            print(f"  [empty GT] skip {prompt}", flush=True)
            continue

        if ref_frame not in ref_frame_cache:
            print(f"  caching SP masks at frame {ref_frame}...", flush=True)
            ref_frame_cache[ref_frame] = cache_sp_masks_at(
                frame_data[ref_frame]['cam'], gaussians, pipe, background,
                snag, args.levels, args.thresh,
            )
        cache_lvl, cache_sp, cache_masks_cpu = ref_frame_cache[ref_frame]
        gt_ref = torch.from_numpy(gt_ref_np).cuda()

        per_sp_iou, per_sp_pur, per_sp_com = per_sp_stats_chunked(
            cache_masks_cpu, gt_ref, chunk=args.chunk)

        # CLIP pool scores
        vlm.encode_text(prompt)
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        pool_scores = []
        for lvl, sp_id in zip(cache_lvl, cache_sp):
            score = sim_per_level[lvl - 1][sp_id].item()
            pool_scores.append(score)
        pool_scores_arr = np.array(pool_scores, dtype=np.float64)
        pool_size = len(pool_scores_arr)
        cos_top = float(pool_scores_arr.max())
        pool_mean = float(pool_scores_arr.mean())
        pool_std = float(pool_scores_arr.std(ddof=1) if pool_size > 1 else 0.0)
        pool_min = float(pool_scores_arr.min())
        pool_max = float(pool_scores_arr.max())
        pool_entropy = softmax_entropy(pool_scores_arr)

        # Oracle SP = top single-SP IoU at ref_frame
        oracle_idx = int(np.argmax(per_sp_iou))
        oracle_lvl = cache_lvl[oracle_idx]
        oracle_sp_id = cache_sp[oracle_idx]
        oracle_iou_ref = float(per_sp_iou[oracle_idx])
        oracle_purity_ref = float(per_sp_pur[oracle_idx])
        oracle_completeness_ref = float(per_sp_com[oracle_idx])
        cos_oracle = float(pool_scores_arr[oracle_idx])

        # Greedy union -> fragmentation
        topk_keep = min(args.topk_keep, len(per_sp_iou))
        top_idx = np.argpartition(-per_sp_iou, topk_keep - 1)[:topk_keep]
        top_idx = top_idx[np.argsort(-per_sp_iou[top_idx])]
        top_masks_gpu = cache_masks_cpu[top_idx].cuda()
        sel_top, ious_at = greedy_union_select(top_masks_gpu, gt_ref, args.budget)
        del top_masks_gpu
        torch.cuda.empty_cache()
        fragmentation = len(sel_top)
        union_iou_ref = ious_at[-1] if ious_at else 0.0

        # Alt oracles (by ref-frame IoU)
        sorted_iou_idx = np.argsort(-per_sp_iou)
        alt2_idx = int(sorted_iou_idx[1]) if pool_size > 1 else oracle_idx
        alt3_idx = int(sorted_iou_idx[2]) if pool_size > 2 else oracle_idx
        alt2_lvl, alt2_sp_id = cache_lvl[alt2_idx], cache_sp[alt2_idx]
        alt2_iou_ref = float(per_sp_iou[alt2_idx])
        alt2_purity_ref = float(per_sp_pur[alt2_idx])
        alt2_completeness_ref = float(per_sp_com[alt2_idx])
        alt3_lvl, alt3_sp_id = cache_lvl[alt3_idx], cache_sp[alt3_idx]
        alt3_iou_ref = float(per_sp_iou[alt3_idx])
        alt3_purity_ref = float(per_sp_pur[alt3_idx])
        alt3_completeness_ref = float(per_sp_com[alt3_idx])

        # CLIP rank of oracle
        order = np.argsort(-pool_scores_arr)
        oracle_rank = int(np.where(order == oracle_idx)[0][0] + 1)
        clip_top1_idx = int(order[0])
        clip_top1_lvl, clip_top1_sp_id = cache_lvl[clip_top1_idx], cache_sp[clip_top1_idx]
        clip_top1_oracle_iou_ref = float(per_sp_iou[clip_top1_idx])
        clip_top1_purity_ref = float(per_sp_pur[clip_top1_idx])
        clip_top1_completeness_ref = float(per_sp_com[clip_top1_idx])

        # A4 margins
        raw_margin = cos_top - cos_oracle
        z_margin = raw_margin / pool_std if pool_std > 1e-9 else 0.0
        percentile_margin = (oracle_rank - 1) / max(pool_size - 1, 1) * 100.0

        # Per-eval-frame oracle re-rendering
        for frame_name in frames:
            gt_eval_np = get_gt_for(prompt, frame_name)
            if not gt_eval_np.any():
                continue
            gt_eval = torch.from_numpy(gt_eval_np).cuda()
            cam = frame_data[frame_name]['cam']
            ev_iou, ev_pur, ev_com, ev_sp_px = single_sp_stats(
                cam, gaussians, pipe, background, snag,
                oracle_lvl, oracle_sp_id, gt_eval, args.thresh)

            gt_pixels = int(gt_eval.sum().item())
            image_pixels = int(gt_eval.numel())
            is_ref = 1 if frame_name == ref_frame else 0

            writer.writerow([
                scene_name, prompt, frame_name, is_ref, gt_pixels, image_pixels,
                oracle_lvl, oracle_sp_id,
                f'{oracle_iou_ref:.6f}', f'{oracle_purity_ref:.6f}', f'{oracle_completeness_ref:.6f}',
                fragmentation, f'{union_iou_ref:.6f}',
                alt2_lvl, alt2_sp_id, f'{alt2_iou_ref:.6f}', f'{alt2_purity_ref:.6f}', f'{alt2_completeness_ref:.6f}',
                alt3_lvl, alt3_sp_id, f'{alt3_iou_ref:.6f}', f'{alt3_purity_ref:.6f}', f'{alt3_completeness_ref:.6f}',
                f'{ev_iou:.6f}', f'{ev_pur:.6f}', f'{ev_com:.6f}', ev_sp_px,
                pool_size,
                f'{cos_top:.6f}', f'{cos_oracle:.6f}',
                f'{pool_mean:.6f}', f'{pool_std:.6f}', f'{pool_min:.6f}', f'{pool_max:.6f}',
                f'{pool_entropy:.6f}',
                oracle_rank,
                f'{raw_margin:.6f}', f'{z_margin:.6f}', f'{percentile_margin:.4f}',
                clip_top1_lvl, clip_top1_sp_id, f'{clip_top1_oracle_iou_ref:.6f}',
                f'{clip_top1_purity_ref:.6f}', f'{clip_top1_completeness_ref:.6f}',
            ])
            del gt_eval
        f_csv.flush()
        print(
            f"  {prompt}: pur_ref={oracle_purity_ref:.3f} comp_ref={oracle_completeness_ref:.3f} "
            f"frag={fragmentation} rank={oracle_rank}/{pool_size} z_marg={z_margin:.3f} "
            f"pctile={percentile_margin:.1f}", flush=True)
        del gt_ref

    f_csv.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_csv", type=str,
                        default="output/diagnostics/b7_a4_combined.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--topk_keep", type=int, default=30)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
