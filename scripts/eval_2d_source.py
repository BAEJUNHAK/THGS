"""
2D-only quality evaluator for THGS 2D-source ablation (Phase 1: A=SAM vs B=FastSAM).

This evaluator does NOT touch the 3D pipeline. It evaluates the *raw 2D semantic source*
(per-mask CLIP features + seg_map) by:
  - For each (frame, query) pair, broadcast the per-mask CLIP-text similarity to pixels
  - Compare against GT polygon masks where they exist
  - Report IoU, P, R, F1, PR-AUC, BF@2, BF@5

Inputs are A-format <frame>_s.npy (4,H,W) int32 + <frame>_f.npy (M, D) fp16,
which both A's image_encoding.py and B's encode_fastsam_clip.py produce identically.

Usage:
    python scripts/eval_2d_source.py \
        --feat_dir data/lerf-ovs/ramen/language_features \
        --gt_dir data/lerf-ovs/label/ramen \
        --img_dir data/lerf-ovs/ramen/images \
        --queries_json scripts/queries_ramen.json \
        --out_csv output/exp_2d_source/metrics_A.csv \
        --arm A --scale_mode s_only

    --scale_mode in {s_only, best4}: which seg_map level to use for pixel sim broadcast.
        s_only = level 1 (THGS default), best4 = max sim over all 4 levels per pixel.

Also writes an aggregate "proposer quality" CSV (--proposer_csv) measuring how well
the mask proposer covers each GT polygon (max-IoU mask per GT, no CLIP involved).
"""
import os
import json
import argparse
import csv
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vlm_utils import ClipSimMeasure


# ---- IO helpers ----

def polygon_to_mask(h, w, points):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask


def load_gt_masks_per_category(gt_dir, frame, target_hw=None):
    """Load all GT polygons for one frame, group by category (union per category).
    Returns dict: category -> binary mask (H, W) uint8."""
    jpath = os.path.join(gt_dir, frame + '.json')
    if not os.path.exists(jpath):
        return {}, None
    anno = json.load(open(jpath))
    W, H = anno['info']['width'], anno['info']['height']
    masks = {}
    for obj in anno['objects']:
        c = obj['category']
        m = polygon_to_mask(H, W, obj['segmentation'])
        if c in masks:
            masks[c] = np.maximum(masks[c], m)
        else:
            masks[c] = m
    if target_hw is not None and (H, W) != target_hw:
        # Resize all GT masks to target shape (when seg_map was downsized)
        H2, W2 = target_hw
        masks = {c: cv2.resize(m, (W2, H2), interpolation=cv2.INTER_NEAREST)
                 for c, m in masks.items()}
        H, W = H2, W2
    return masks, (H, W)


def load_features(feat_dir, frame):
    """Load (4,H,W) seg_map + (M,D) feat for one frame."""
    s_path = os.path.join(feat_dir, frame + '_s.npy')
    f_path = os.path.join(feat_dir, frame + '_f.npy')
    if not (os.path.exists(s_path) and os.path.exists(f_path)):
        return None, None
    seg_map = np.load(s_path).astype(np.int64)  # (4, H, W)
    feat = np.load(f_path)                       # (M, D)
    return seg_map, feat


# ---- Pixel similarity construction ----

def build_pixel_sim(seg_map_4hw, mask_sim_M, scale_mode='s_only'):
    """
    seg_map_4hw: (4, H, W) int64 mask ids in [0..M-1] or -1 for bg
    mask_sim_M:  (M,) float similarity score per mask (already canon-softmax)
    scale_mode:
      's_only'  -> use seg_map[1] only
      'best4'   -> for each pixel, take max sim over the 4 levels' mask ids
    Returns sim_map (H, W) float, with bg pixels = 0.
    """
    if scale_mode == 's_only':
        sm = seg_map_4hw[1]
        sim = np.where(sm >= 0, mask_sim_M[np.clip(sm, 0, len(mask_sim_M)-1)], 0.0)
        sim[sm < 0] = 0.0
        return sim
    elif scale_mode == 'best4':
        H, W = seg_map_4hw.shape[1:]
        sim_stack = np.zeros((4, H, W), dtype=np.float32)
        for j in range(4):
            sm = seg_map_4hw[j]
            sj = np.where(sm >= 0, mask_sim_M[np.clip(sm, 0, len(mask_sim_M)-1)], 0.0)
            sj[sm < 0] = 0.0
            sim_stack[j] = sj
        return sim_stack.max(axis=0)
    else:
        raise ValueError(f"unknown scale_mode {scale_mode}")


# ---- Metrics ----

def binary_metrics(pred, gt, eps=1e-9):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    iou = tp / (tp + fp + fn + eps)
    p   = tp / (tp + fp + eps)
    r   = tp / (tp + fn + eps)
    f1  = 2 * p * r / (p + r + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return dict(IoU=float(iou), P=float(p), R=float(r), F1=float(f1), Acc=float(acc),
                pred_area=int(pred.sum()), gt_area=int(gt.sum()))


def boundary_f_score(pred, gt, dilation_px=2):
    """DAVIS-style boundary F-score (precision/recall on dilated boundaries)."""
    pred = pred.astype(np.uint8); gt = gt.astype(np.uint8)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    # Extract boundaries via morph gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pred_b = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel) > 0
    gt_b = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    # Dilate
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_px + 1, 2 * dilation_px + 1))
    pred_d = cv2.dilate(pred_b.astype(np.uint8), d_kernel) > 0
    gt_d = cv2.dilate(gt_b.astype(np.uint8), d_kernel) > 0
    # Precision = pred_b ∩ gt_d / pred_b ; Recall = gt_b ∩ pred_d / gt_b
    prec = (pred_b & gt_d).sum() / max(pred_b.sum(), 1)
    rec = (gt_b & pred_d).sum() / max(gt_b.sum(), 1)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


# ---- Mask proposer quality (CLIP-free) ----

def proposer_quality_per_frame(seg_map_4hw, gt_masks_by_cat):
    """For each (GT category, scale level), find the max-IoU mask in seg_map[level] vs GT union.
    Returns list of records."""
    H, W = seg_map_4hw.shape[1:]
    records = []
    for cat, gt in gt_masks_by_cat.items():
        gt_b = gt.astype(bool)
        if gt_b.sum() == 0:
            continue
        for level in range(4):
            sm = seg_map_4hw[level]
            unique_ids = np.unique(sm)
            unique_ids = unique_ids[unique_ids >= 0]
            if len(unique_ids) == 0:
                records.append(dict(category=cat, level=level, best_iou=0.0, n_masks=0))
                continue
            best = 0.0
            for mid in unique_ids:
                m = (sm == mid)
                inter = np.logical_and(m, gt_b).sum()
                union = np.logical_or(m, gt_b).sum()
                if union == 0:
                    continue
                iou = inter / union
                if iou > best:
                    best = float(iou)
            records.append(dict(category=cat, level=level, best_iou=best, n_masks=len(unique_ids)))
    return records


# ---- Main eval loop ----

def evaluate(feat_dir, gt_dir, img_dir, queries_json, out_csv, arm, scale_mode,
             eval_frames=None, threshold=0.5, proposer_csv=None):
    with open(queries_json, 'r') as f:
        queries_by_level = json.load(f)
    # queries_by_level: {"L0": [{"q":..., "gt":...}, ...], "L1": [...], ...}

    if eval_frames is None:
        # auto-detect from gt_dir
        eval_frames = sorted([os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                              if f.endswith('.json')])

    vlm = ClipSimMeasure()
    vlm.load_model()

    rows = []
    proposer_rows = []

    for frame in tqdm(eval_frames, desc=f"eval[{arm}]"):
        seg_map, feat_M = load_features(feat_dir, frame)
        if seg_map is None:
            print(f"skip {frame}: no features")
            continue
        target_hw = seg_map.shape[1:]
        gt_masks, gt_hw = load_gt_masks_per_category(gt_dir, frame, target_hw=target_hw)

        if proposer_csv:
            for rec in proposer_quality_per_frame(seg_map, gt_masks):
                rec['frame'] = frame
                rec['arm'] = arm
                proposer_rows.append(rec)

        feat_t = torch.from_numpy(feat_M).float().cuda()  # (M, D)

        for level_name, items in queries_by_level.items():
            for it in items:
                q = it['q']
                gt_cat = it.get('gt')
                vlm.encode_text(q)
                with torch.no_grad():
                    mask_sim = vlm.compute_similarity(feat_t).cpu().numpy()  # (M,)

                sim_map = build_pixel_sim(seg_map, mask_sim, scale_mode=scale_mode)
                pred_bin = (sim_map > threshold).astype(np.uint8)

                row = dict(arm=arm, level=level_name, query=q, frame=frame,
                           gt_category=gt_cat, scale_mode=scale_mode,
                           pred_area=int(pred_bin.sum()),
                           sim_max=float(sim_map.max()),
                           sim_mean=float(sim_map.mean()))

                if gt_cat is not None and gt_cat in gt_masks:
                    gt = gt_masks[gt_cat]
                    if gt.shape != sim_map.shape:
                        gt = cv2.resize(gt, (sim_map.shape[1], sim_map.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    m = binary_metrics(pred_bin, gt)
                    row.update(m)
                    # PR-AUC (sklearn): need ground truth and continuous score per pixel
                    try:
                        row['PR_AUC'] = float(average_precision_score(gt.flatten().astype(int),
                                                                       sim_map.flatten()))
                    except Exception:
                        row['PR_AUC'] = float('nan')
                    row['BF2'] = boundary_f_score(pred_bin, gt, dilation_px=2)
                    row['BF5'] = boundary_f_score(pred_bin, gt, dilation_px=5)
                    row['has_gt'] = True
                else:
                    row['has_gt'] = False
                rows.append(row)

    # Write metrics CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
    fieldnames = ['arm', 'level', 'query', 'frame', 'gt_category', 'scale_mode', 'has_gt',
                  'IoU', 'P', 'R', 'F1', 'Acc', 'PR_AUC', 'BF2', 'BF5',
                  'pred_area', 'gt_area', 'sim_max', 'sim_mean']
    with open(out_csv, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    if proposer_csv and proposer_rows:
        os.makedirs(os.path.dirname(proposer_csv), exist_ok=True) if os.path.dirname(proposer_csv) else None
        with open(proposer_csv, 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=['arm', 'frame', 'category', 'level',
                                                'best_iou', 'n_masks'])
            w.writeheader()
            for r in proposer_rows:
                w.writerow(r)
        print(f"wrote {proposer_csv} ({len(proposer_rows)} rows)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', required=True,
                        help='e.g. data/lerf-ovs/ramen/language_features (A) or .._fastsam (B)')
    parser.add_argument('--gt_dir', required=True,
                        help='e.g. data/lerf-ovs/label/ramen')
    parser.add_argument('--img_dir', required=True,
                        help='e.g. data/lerf-ovs/ramen/images (only for size detection if needed)')
    parser.add_argument('--queries_json', required=True,
                        help='JSON file: {"L0": [{"q":..., "gt":...}], ...}')
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--arm', required=True, choices=['A', 'B'],
                        help='label written into rows')
    parser.add_argument('--scale_mode', default='s_only', choices=['s_only', 'best4'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--eval_frames', nargs='+', default=None,
                        help='if omitted, all frames with GT json')
    parser.add_argument('--proposer_csv', default=None,
                        help='if set, write CLIP-free mask coverage per (frame, GT cat, level)')
    args = parser.parse_args()

    evaluate(args.feat_dir, args.gt_dir, args.img_dir, args.queries_json,
             args.out_csv, args.arm, args.scale_mode,
             eval_frames=args.eval_frames, threshold=args.threshold,
             proposer_csv=args.proposer_csv)
