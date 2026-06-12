"""
LERF-OVS evaluator under the OpenGaussian/OpenSplat3D ("galre B") protocol.

- mIoU              : mean per-(frame, prompt) IoU
- mAcc@0.25         : fraction of (frame, prompt) pairs with IoU >= 0.25
- mAcc@0.50 (bonus) : stricter threshold

Reuses the predicted PNGs already saved by test_lerf.py at:
    <path_pred>/<scene>/<frame>/<prompt>.png       (binary 0/255, our pred)
    <path_pred>/<scene>/<frame>/<prompt>_gt.png    (binary 0/255, polygon GT raster)
"""

import os
import cv2
import json
import numpy as np
from argparse import ArgumentParser


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float('nan') if union == 0 else inter / union


def evaluate(path_pred, scenes, path_gt=None, verbose=False, thresholds=(0.25, 0.50)):
    per_scene = {}
    for scene in scenes:
        pr_root = os.path.join(path_pred, scene)
        gt_root = path_gt and os.path.join(path_gt, scene)
        if not os.path.isdir(pr_root):
            print(f'[warn] missing pred dir: {pr_root}')
            continue
        ious = []
        for frame in sorted(os.listdir(pr_root)):
            fdir = os.path.join(pr_root, frame)
            if not os.path.isdir(fdir):
                continue
            # prompt list comes from the saved <prompt>.png files (excluding *_gt.png)
            prompts = [f[:-4] for f in os.listdir(fdir)
                       if f.endswith('.png') and not f.endswith('_gt.png')]
            for p in prompts:
                pr_path = os.path.join(fdir, p + '.png')
                gt_path = os.path.join(fdir, p + '_gt.png')
                if not os.path.exists(gt_path):
                    continue
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                pr = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
                if pr.shape != gt.shape:
                    pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                v = iou(pr > 128, gt > 128)
                if not np.isnan(v):
                    ious.append(v)
                if verbose:
                    print(f'  {scene}/{frame}/{p}: IoU={v:.4f}')
        ious = np.array(ious)
        if len(ious) == 0:
            print(f'{scene}: no valid (frame, prompt) pairs')
            continue
        miou = float(ious.mean())
        accs = {t: float((ious >= t).mean()) for t in thresholds}
        per_scene[scene] = (miou, accs, len(ious))
        acc_strs = '  '.join(f'mAcc@{t:.2f}={accs[t]:.4f}' for t in thresholds)
        print(f'{scene:<14}: mIoU={miou:.4f}   {acc_strs}   (n={len(ious)} prompt-frames)')
    if per_scene:
        all_miou = float(np.mean([v[0] for v in per_scene.values()]))
        all_accs = {t: float(np.mean([v[1][t] for v in per_scene.values()])) for t in thresholds}
        acc_strs = '  '.join(f'mAcc@{t:.2f}={all_accs[t]:.4f}' for t in thresholds)
        print(f'\n{"Overall":<14}: mIoU={all_miou:.4f}   {acc_strs}')
    return per_scene


if __name__ == '__main__':
    parser = ArgumentParser('Galre B (OpenGaussian/OpenSplat3D) evaluator for LERF-OVS.')
    parser.add_argument('--path_pred', '-p', type=str, default='output/render/lerf')
    parser.add_argument('--scene_list', '-s', nargs='+',
                        default=['figurines', 'ramen', 'teatime', 'waldo_kitchen'])
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    evaluate(args.path_pred, args.scene_list, verbose=args.verbose)
