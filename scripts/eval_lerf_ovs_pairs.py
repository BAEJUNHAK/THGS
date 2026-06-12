"""
Evaluate LERF-OVS predictions against same-directory polygon-rasterized GT.

Layout assumed:
  <path_pred>/<scene>/<frame_name>/<prompt>.png       (pred)
  <path_pred>/<scene>/<frame_name>/<prompt>_gt.png    (raster GT, same dir)

Metrics: per (frame, prompt) IoU + Boundary-IoU. Per-scene flat mean → overall mean.
Boundary-IoU definition matches scripts/eval_lerf_mask.py (3x3 erode, iter=round(0.02*diag)).
"""

import os
import cv2
import numpy as np
from argparse import ArgumentParser


def boundary_mask(mask_bool, dilation_ratio=0.02):
    h, w = mask_bool.shape
    diag = np.sqrt(h * h + w * w)
    d = max(1, int(round(dilation_ratio * diag)))
    m_u8 = mask_bool.astype(np.uint8)
    pad = cv2.copyMakeBorder(m_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(pad, kernel, iterations=d)[1:-1, 1:-1]
    return (m_u8 - eroded).astype(bool)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float('nan')
    return inter / union


def evaluate(path_pred, scenes, dilation_ratio=0.02, verbose=False):
    per_scene = {}
    for scene in scenes:
        sc_root = os.path.join(path_pred, scene)
        if not os.path.isdir(sc_root):
            print(f'[warn] missing: {sc_root}')
            continue
        ious, bious = [], []
        for frame in sorted(os.listdir(sc_root)):
            fd = os.path.join(sc_root, frame)
            if not os.path.isdir(fd):
                continue
            preds = sorted([f for f in os.listdir(fd)
                            if f.endswith('.png') and not f.endswith('_gt.png')])
            for p in preds:
                base = p.rsplit('.', 1)[0]
                gt_path = os.path.join(fd, base + '_gt.png')
                pr_path = os.path.join(fd, p)
                if not os.path.exists(gt_path):
                    continue
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 128
                pr = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
                if pr is None:
                    pr = np.zeros_like(gt, dtype=np.uint8)
                if pr.shape != gt.shape:
                    pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                pr = pr > 128
                v_iou = iou(pr, gt)
                v_biou = iou(boundary_mask(pr, dilation_ratio),
                             boundary_mask(gt, dilation_ratio))
                if verbose:
                    print(f'  {scene}/{frame}/{base}: IoU={v_iou:.4f}  BIoU={v_biou:.4f}')
                if not np.isnan(v_iou):
                    ious.append(v_iou)
                if not np.isnan(v_biou):
                    bious.append(v_biou)
        miou = float(np.mean(ious)) if ious else float('nan')
        mbiou = float(np.mean(bious)) if bious else float('nan')
        per_scene[scene] = (miou, mbiou, len(ious))
        print(f'{scene}: mIoU={miou:.4f}  BIoU={mbiou:.4f}  (n={len(ious)} prompt-views)')
    if per_scene:
        all_miou = float(np.mean([v[0] for v in per_scene.values() if not np.isnan(v[0])]))
        all_biou = float(np.mean([v[1] for v in per_scene.values() if not np.isnan(v[1])]))
        print(f'\nOverall: mIoU={all_miou:.4f}  BIoU={all_biou:.4f}')
    return per_scene


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_pred', '-p', type=str, required=True)
    parser.add_argument('--scene_list', '-s', nargs='+',
                        default=['figurines', 'ramen', 'teatime', 'waldo_kitchen'])
    parser.add_argument('--dilation_ratio', type=float, default=0.02)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    evaluate(args.path_pred, args.scene_list, args.dilation_ratio, args.verbose)
