"""
Evaluate predicted masks against the LERF-Mask GT (Gaussian Grouping protocol).

Metrics: mean IoU and mean Boundary-IoU per scene, plus the overall mean.

Layout assumed:
  GT   : <path_gt>/<scene>/test_mask/<view_idx>/<prompt>.png      (binary 0/255)
  PRED : <path_pred>/<scene>/<view_idx>/<prompt>.png              (binary 0/255)
"""

import os
import cv2
import numpy as np
from argparse import ArgumentParser


def boundary_mask(mask_bool, dilation_ratio=0.02):
    """Return boundary pixels of a binary mask via the morphological-erosion trick.

    Boundary = mask AND NOT erode(mask, k) where k scales with image diagonal.
    Matches the Gaussian-Grouping LERF-Mask eval definition.
    """
    h, w = mask_bool.shape
    diag = np.sqrt(h * h + w * w)
    d = max(1, int(round(dilation_ratio * diag)))
    m_u8 = mask_bool.astype(np.uint8)
    # pad so that erosion at borders is correct
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


def evaluate(path_pred, path_gt, scenes, dilation_ratio=0.02, verbose=False):
    per_scene = {}
    for scene in scenes:
        gt_root = os.path.join(path_gt, scene, 'test_mask')
        pr_root = os.path.join(path_pred, scene)
        if not os.path.isdir(gt_root):
            print(f'[warn] missing GT: {gt_root}')
            continue
        view_ious, view_bious = [], []
        for view in sorted(os.listdir(gt_root)):
            gv = os.path.join(gt_root, view)
            pv = os.path.join(pr_root, view)
            if not os.path.isdir(gv):
                continue
            prompts = [f for f in os.listdir(gv) if f.endswith('.png')]
            for p in prompts:
                gt = cv2.imread(os.path.join(gv, p), cv2.IMREAD_GRAYSCALE)
                pr_p = os.path.join(pv, p)
                if not os.path.exists(pr_p):
                    if verbose:
                        print(f'  [miss] {scene}/{view}/{p}')
                    pr = np.zeros_like(gt)
                else:
                    pr = cv2.imread(pr_p, cv2.IMREAD_GRAYSCALE)
                    if pr.shape != gt.shape:
                        pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt_b = gt > 128
                pr_b = pr > 128
                v_iou = iou(pr_b, gt_b)
                v_biou = iou(boundary_mask(pr_b, dilation_ratio),
                             boundary_mask(gt_b, dilation_ratio))
                if verbose:
                    print(f'  {scene}/{view}/{p[:-4]}: IoU={v_iou:.4f}  BIoU={v_biou:.4f}')
                if not np.isnan(v_iou):
                    view_ious.append(v_iou)
                if not np.isnan(v_biou):
                    view_bious.append(v_biou)
        miou = float(np.mean(view_ious)) if view_ious else float('nan')
        mbiou = float(np.mean(view_bious)) if view_bious else float('nan')
        per_scene[scene] = (miou, mbiou, len(view_ious))
        print(f'{scene}: mIoU={miou:.4f}  Boundary-IoU={mbiou:.4f}   (n={len(view_ious)} prompt-views)')
    if per_scene:
        all_miou = float(np.mean([v[0] for v in per_scene.values() if not np.isnan(v[0])]))
        all_biou = float(np.mean([v[1] for v in per_scene.values() if not np.isnan(v[1])]))
        print(f'\nOverall: mIoU={all_miou:.4f}  Boundary-IoU={all_biou:.4f}')
    return per_scene


if __name__ == '__main__':
    parser = ArgumentParser('Evaluate LERF-Mask predictions.')
    parser.add_argument('--path_pred', '-p', type=str, required=True)
    parser.add_argument('--path_gt', '-g', type=str, default='data/lerf_mask')
    parser.add_argument('--scene_list', '-s', nargs='+', default=['figurines', 'ramen', 'teatime'])
    parser.add_argument('--dilation_ratio', type=float, default=0.02)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    evaluate(args.path_pred, args.path_gt, args.scene_list, args.dilation_ratio, args.verbose)
