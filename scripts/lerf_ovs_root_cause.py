"""
Phase 1.5 root-cause diagnostic — separate (a) per-SAM-mask CLIP limit,
(b) per-SP aggregation noise, (c) SAM mask coverage of GT object.

For one (scene, prompt) — default pikachu in figurines (A1, rank=64):

  For each training view:
    1. render_point() → per-gaussian visibility weight + 2D coordinate
    2. Visible gaussians in CORRECT SP (greedy first pick) and CLIP's top-1 SP
    3. At their 2D positions, look up SAM mask ID at the relevant level
    4. For each unique SAM mask hit:
       - Compute CLIP cos-sim with prompt text embedding
       - Compute canon-contrast relevancy (same as inference)
       - Compute SAM mask's IoU with GT polygon (does SAM capture the object?)

Output: per-view + aggregate table answering (a)/(b)/(c).

(a) per-SAM-mask CLIP limit:
    if the SAM masks that the correct SP's gaussians fall into have LOW similarity with the prompt
    → CLIP just doesn't recognize the object in 2D, no pipeline fix can recover.

(b) Per-SP aggregation noise:
    if SAM masks have HIGH similarity individually but SP feature is bad
    → aggregation in proj_gaussian_features() dilutes the signal.
    Detect: many unique SAM masks (with mixed alignment) per SP.

(c) SAM mask coverage:
    if SAM masks (containing the correct SP's gaussians) have LOW IoU with GT polygon
    → SAM doesn't properly segment the object — fix is at SAM level.
"""

import os
import sys
import csv
import cv2
import json
import torch
import numpy as np
from argparse import ArgumentParser

from gaussian_renderer import render_point
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


def iou(a_bool, b_bool):
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return inter / max(union, 1)


@torch.no_grad()
def diagnose_sp_at_view(view, gaussians, pipe, background, snag, sp_lvl, sp_id,
                       vlm, gt_polygon_pts, sam_level_idx):
    """For one SP at one view, return per-SAM-mask stats.

    Args:
        sp_lvl/sp_id: target SP in NAG (level 0-3).
        sam_level_idx: which SAM level's seg_map to use (0=default, 1=s, 2=m, 3=l).
                      For THGS pipeline, SP level 2 uses SAM level 2 ("m" masks),
                      SP level 3 uses SAM level 3 ("l" masks).
    """
    H, W = view.image_height, view.image_width
    # gaussians in this SP
    gau_in_sp = (snag.labels[sp_lvl].long() == sp_id)

    # render to get per-gaussian visibility and 2D coords
    render_pkg = render_point(view, gaussians, pipe, background)
    weight = render_pkg["weight"]
    means2D = render_pkg["means2D"]
    visible = weight > 0.01
    visible_in_sp = gau_in_sp & visible
    n_visible = int(visible_in_sp.sum().item())
    if n_visible == 0:
        return None

    # 2D coords of visible-in-SP gaussians
    pts_2d = means2D[visible_in_sp]
    y_coords = pts_2d[:, 1].long().clamp(0, H - 1).cpu().numpy()
    x_coords = pts_2d[:, 0].long().clamp(0, W - 1).cpu().numpy()

    # SAM seg map at the chosen level
    seg_map = view.semantic["seg_map"][sam_level_idx]  # (H', W') possibly different resolution
    fg_mask = view.semantic["fg_mask"][sam_level_idx]
    H_s, W_s = seg_map.shape
    # Scale 2D coords to seg_map resolution if different
    if (H_s != H) or (W_s != W):
        scale_y = H_s / H
        scale_x = W_s / W
        y_in_seg = np.clip((y_coords * scale_y).astype(int), 0, H_s - 1)
        x_in_seg = np.clip((x_coords * scale_x).astype(int), 0, W_s - 1)
    else:
        y_in_seg, x_in_seg = y_coords, x_coords

    sam_ids = seg_map[y_in_seg, x_in_seg].numpy()
    is_fg = fg_mask[y_in_seg, x_in_seg].numpy()
    # Filter to foreground
    sam_ids_fg = sam_ids[is_fg]
    n_fg = len(sam_ids_fg)
    if n_fg == 0:
        return {
            'n_visible': n_visible, 'n_fg': 0,
            'n_unique_masks': 0,
            'masks': [],
        }

    unique_ids, counts = np.unique(sam_ids_fg, return_counts=True)

    # Get GT polygon mask at this view (for SAM mask coverage check)
    gt_mask = polygon_to_mask((H, W), gt_polygon_pts) > 0  # bool (H, W)

    # For each unique SAM mask, compute stats
    sem_features = view.semantic["sem"].cuda()  # (N_total_masks, 512)
    rows = []
    for sid, cnt in zip(unique_ids, counts):
        sid = int(sid)
        feat = sem_features[sid]  # (512,)
        # Raw cosine sim with text positive embedding (=text_feature[0])
        # vlm.text_feature is shape (1+canon, 512), [0]=positive, [1:]=canon
        text_pos = vlm.text_feature[0]
        feat_n = feat / (feat.norm() + 1e-8)
        text_pos_n = text_pos / (text_pos.norm() + 1e-8)
        cos_sim = float((feat_n @ text_pos_n).item())
        # canon-contrast relevancy (same as inference)
        rel = vlm.compute_similarity(feat.unsqueeze(0))[0].item()

        # SAM mask spatial mask: where in this image is this mask?
        sam_pixel_mask = (seg_map.numpy() == sid)
        if (H_s != H) or (W_s != W):
            # Rescale to original res for IoU vs GT polygon
            sam_pixel_mask_resized = cv2.resize(
                sam_pixel_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ) > 0
        else:
            sam_pixel_mask_resized = sam_pixel_mask
        sam_gt_iou = iou(sam_pixel_mask_resized, gt_mask)
        sam_area = int(sam_pixel_mask_resized.sum())

        rows.append({
            'sam_id': sid,
            'gaussian_count': int(cnt),
            'cos_sim': cos_sim,
            'relevancy': rel,
            'sam_gt_iou': float(sam_gt_iou),
            'sam_area': sam_area,
        })

    return {
        'n_visible': n_visible,
        'n_fg': n_fg,
        'n_unique_masks': len(unique_ids),
        'masks': rows,
    }


@torch.no_grad()
def main(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Phase 1.5 Root Cause Diagnostic: {scene_name} / {args.prompt} ===\n")

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=True, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

    vlm = ClipSimMeasure()
    vlm.load_model()
    vlm.encode_text(args.prompt)
    print(f"VLM text encoded for prompt = '{args.prompt}'")
    print(f"vlm.canon = {vlm.canon}")

    # Load all training views, find frames with GT for this prompt
    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    train_cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    img_list = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')])
    gt_frames = []
    for im in img_list:
        image_name = im.split('.')[0]
        if image_name not in train_cams_by_name:
            continue
        js_file = os.path.join(data_path, image_name + '.json')
        anno = json.load(open(js_file))
        for obj in anno['objects']:
            if obj['category'] == args.prompt:
                gt_frames.append({
                    'image_name': image_name,
                    'cam': train_cams_by_name[image_name],
                    'polygon': obj['segmentation'],
                })
                break
    print(f"GT frames for '{args.prompt}': {[g['image_name'] for g in gt_frames]}\n")

    # SP level for THGS: 2 or 3 (level=[2,3] inference pool)
    # SAM seg_map indexing: level 0=default, 1=s, 2=m, 3=l. SP level i maps to SAM level i.
    correct_sp_lvl = args.correct_sp_lvl
    correct_sp_id = args.correct_sp_id
    sam_level_idx = correct_sp_lvl  # SP lvl 2 → SAM lvl 2 ("m"), SP lvl 3 → SAM lvl 3 ("l")

    # Print SP gaussian counts
    n_in_correct = int((snag.labels[correct_sp_lvl].long() == correct_sp_id).sum().item())
    print(f"Correct SP: lvl={correct_sp_lvl}, id={correct_sp_id}, n_gaussians={n_in_correct}")

    if args.clip_top1_sp_id is not None:
        clip_top1_lvl = args.clip_top1_sp_lvl
        clip_top1_id = args.clip_top1_sp_id
        n_in_top1 = int((snag.labels[clip_top1_lvl].long() == clip_top1_id).sum().item())
        print(f"CLIP top-1 SP: lvl={clip_top1_lvl}, id={clip_top1_id}, n_gaussians={n_in_top1}")
    print()

    # Diagnose at each GT frame
    print(f"=" * 100)
    print(f"--- CORRECT SP (the one greedy/oracle picks) ---")
    print(f"=" * 100)
    correct_aggregate = {'cos_sims': [], 'relevancies': [], 'sam_gt_ious': [],
                         'n_unique_per_view': [], 'mask_count_per_view': []}
    for g in gt_frames:
        print(f"\n>> view = {g['image_name']}")
        stats = diagnose_sp_at_view(
            g['cam'], gaussians, pipe, background, snag,
            correct_sp_lvl, correct_sp_id, vlm, g['polygon'], sam_level_idx)
        if stats is None:
            print(f"   no visible gaussians in SP")
            continue
        print(f"   visible gaussians in SP = {stats['n_visible']} (fg={stats['n_fg']}), "
              f"unique SAM masks they fall into = {stats['n_unique_masks']}")
        # Top-5 masks by gaussian count
        sorted_masks = sorted(stats['masks'], key=lambda x: -x['gaussian_count'])[:8]
        print(f"   {'sam_id':>8} {'gau':>5} {'cos_sim':>9} {'relevancy':>10} {'sam_gt_iou':>11} {'sam_area':>10}")
        for m in sorted_masks:
            print(f"   {m['sam_id']:>8} {m['gaussian_count']:>5} {m['cos_sim']:>9.4f} "
                  f"{m['relevancy']:>10.4f} {m['sam_gt_iou']:>11.4f} {m['sam_area']:>10}")
        # Aggregate (weighted by gaussian count, as merge_proj does)
        total_w = sum(m['gaussian_count'] for m in stats['masks'])
        weighted_cos = sum(m['cos_sim'] * m['gaussian_count'] for m in stats['masks']) / max(total_w, 1)
        weighted_rel = sum(m['relevancy'] * m['gaussian_count'] for m in stats['masks']) / max(total_w, 1)
        weighted_iou = sum(m['sam_gt_iou'] * m['gaussian_count'] for m in stats['masks']) / max(total_w, 1)
        print(f"   ► gaussian-weighted: cos={weighted_cos:.4f} rel={weighted_rel:.4f} sam_gt_iou={weighted_iou:.4f}")
        correct_aggregate['cos_sims'].append(weighted_cos)
        correct_aggregate['relevancies'].append(weighted_rel)
        correct_aggregate['sam_gt_ious'].append(weighted_iou)
        correct_aggregate['n_unique_per_view'].append(stats['n_unique_masks'])

    if args.clip_top1_sp_id is not None:
        print(f"\n{'='*100}")
        print(f"--- CLIP top-1 SP (the wrong one CLIP picks) ---")
        print(f"=" * 100)
        wrong_aggregate = {'cos_sims': [], 'relevancies': [], 'sam_gt_ious': [],
                           'n_unique_per_view': []}
        for g in gt_frames:
            print(f"\n>> view = {g['image_name']}")
            stats = diagnose_sp_at_view(
                g['cam'], gaussians, pipe, background, snag,
                args.clip_top1_sp_lvl, args.clip_top1_sp_id, vlm,
                g['polygon'], args.clip_top1_sp_lvl)
            if stats is None:
                print(f"   no visible gaussians in SP")
                continue
            print(f"   visible gaussians in SP = {stats['n_visible']} (fg={stats['n_fg']}), "
                  f"unique SAM masks they fall into = {stats['n_unique_masks']}")
            sorted_masks = sorted(stats['masks'], key=lambda x: -x['gaussian_count'])[:8]
            print(f"   {'sam_id':>8} {'gau':>5} {'cos_sim':>9} {'relevancy':>10} {'sam_gt_iou':>11} {'sam_area':>10}")
            for m in sorted_masks:
                print(f"   {m['sam_id']:>8} {m['gaussian_count']:>5} {m['cos_sim']:>9.4f} "
                      f"{m['relevancy']:>10.4f} {m['sam_gt_iou']:>11.4f} {m['sam_area']:>10}")
            total_w = sum(m['gaussian_count'] for m in stats['masks'])
            weighted_cos = sum(m['cos_sim'] * m['gaussian_count'] for m in stats['masks']) / max(total_w, 1)
            weighted_rel = sum(m['relevancy'] * m['gaussian_count'] for m in stats['masks']) / max(total_w, 1)
            weighted_iou = sum(m['sam_gt_iou'] * m['gaussian_count'] for m in stats['masks']) / max(total_w, 1)
            print(f"   ► gaussian-weighted: cos={weighted_cos:.4f} rel={weighted_rel:.4f} sam_gt_iou={weighted_iou:.4f}")
            wrong_aggregate['cos_sims'].append(weighted_cos)
            wrong_aggregate['relevancies'].append(weighted_rel)
            wrong_aggregate['sam_gt_ious'].append(weighted_iou)
            wrong_aggregate['n_unique_per_view'].append(stats['n_unique_masks'])

    print(f"\n{'='*100}")
    print(f"--- AGGREGATE SUMMARY ---")
    print(f"=" * 100)
    print(f"\nCorrect SP (lvl={correct_sp_lvl}, id={correct_sp_id}):")
    print(f"  cos_sim (avg across views): {np.mean(correct_aggregate['cos_sims']):.4f}")
    print(f"  relevancy (canon-contrast, avg): {np.mean(correct_aggregate['relevancies']):.4f}")
    print(f"  sam-gt IoU (avg): {np.mean(correct_aggregate['sam_gt_ious']):.4f}")
    print(f"  unique SAM masks per view: {correct_aggregate['n_unique_per_view']}")

    if args.clip_top1_sp_id is not None:
        print(f"\nCLIP top-1 SP (lvl={args.clip_top1_sp_lvl}, id={args.clip_top1_sp_id}):")
        print(f"  cos_sim (avg): {np.mean(wrong_aggregate['cos_sims']):.4f}")
        print(f"  relevancy (avg): {np.mean(wrong_aggregate['relevancies']):.4f}")
        print(f"  sam-gt IoU (avg): {np.mean(wrong_aggregate['sam_gt_ious']):.4f}")
        print(f"  unique SAM masks per view: {wrong_aggregate['n_unique_per_view']}")

    # Verdict
    print(f"\n--- VERDICT (preliminary) ---")
    correct_cos = np.mean(correct_aggregate['cos_sims'])
    correct_rel = np.mean(correct_aggregate['relevancies'])
    correct_iou = np.mean(correct_aggregate['sam_gt_ious'])

    if args.clip_top1_sp_id is not None:
        wrong_rel = np.mean(wrong_aggregate['relevancies'])
        if correct_rel < wrong_rel:
            print(f"  Correct SP의 SAM-masks relevancy ({correct_rel:.4f}) < CLIP top-1 SP의 relevancy "
                  f"({wrong_rel:.4f}) → SAM-CLIP feature 자체가 정답 SP에서 약함")
    if correct_cos < 0.2:
        print(f"  (a) Per-SAM-mask CLIP의 cos_sim 평균 {correct_cos:.4f} 매우 낮음 — "
              f"CLIP이 '{args.prompt}'를 2D에서도 못 알아봄. (a) 가설 강력 지지.")
    elif correct_cos < 0.3:
        print(f"  (a) Per-SAM-mask CLIP cos_sim 평균 {correct_cos:.4f} 약함. (a) 부분적 영향.")
    else:
        print(f"  (a) Per-SAM-mask CLIP cos_sim 평균 {correct_cos:.4f} 어느정도 OK — "
              f"SAM mask들이 prompt와 align됨. (b) aggregation 문제일 수도.")

    if correct_iou < 0.3:
        print(f"  (c) SAM mask들의 GT 폴리곤 평균 IoU = {correct_iou:.4f} 낮음 — "
              f"SAM이 객체를 잘 잡지 못함. (c) 가설 지지.")
    else:
        print(f"  (c) SAM mask GT IoU 평균 = {correct_iou:.4f} 적절 — SAM은 객체를 잡고 있음.")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--prompt", type=str, default="pikachu")
    parser.add_argument("--correct_sp_lvl", type=int, default=2)
    parser.add_argument("--correct_sp_id", type=int, default=4)
    parser.add_argument("--clip_top1_sp_lvl", type=int, default=2)
    parser.add_argument("--clip_top1_sp_id", type=int, default=34)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
