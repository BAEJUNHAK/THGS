"""
Stage 2A — Layer 4: 17-prompt visual montage + per-phantom case study.

For each of 17 persistent phantoms, save a row with 4 panels:
  - GT crop (green)
  - oracle SP (cyan) — what we should pick
  - clip top-1 SP (red) — what method actually picks
  - top-k=10 union (yellow) — best D3 candidate

Run per scene. Panels are pickled to a temp file, finally combined.
"""

import os
import sys
import json
import pickle
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
def render_union(cam, gaussians, pipe, background, snag, pairs, thresh):
    point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
    for lvl, sp_id in pairs:
        point_valid[snag.labels[lvl].long() == sp_id] = 1.0
    if point_valid.sum() == 0:
        return torch.zeros(cam.image_height, cam.image_width, dtype=torch.bool, device='cuda')
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


def overlay_mask(image_rgb, mask, color=(0, 255, 0), alpha=0.45):
    out = image_rgb.copy()
    out[mask] = (out[mask].astype(np.float32) * (1 - alpha)
                 + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out


def crop_box(image, mask, padding_factor=1.4):
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    h, w = image.shape[:2]
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = (y1 - y0) * padding_factor / 2
    ww = (x1 - x0) * padding_factor / 2
    y0 = max(0, int(cy - hh)); y1 = min(h, int(cy + hh))
    x0 = max(0, int(cx - ww)); x1 = min(w, int(cx + ww))
    return image[y0:y1, x0:x1]


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    b7 = pd.read_csv(args.b7_csv)
    ref = b7[b7['is_ref_frame'] == 1].copy()
    phantoms = pd.read_csv(args.persistent_csv)
    phantom_set = set((r['scene'], r['prompt']) for _, r in phantoms.iterrows())
    targets = ref[ref.apply(lambda r: (r['scene'], r['prompt']) in phantom_set, axis=1)]
    targets = targets[targets['scene'] == scene_name]
    if len(targets) == 0:
        print(f"[skip] no targets for {scene_name}"); return
    print(f"\n=== {scene_name}: {len(targets)} phantom targets ===", flush=True)

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
    train_cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}

    image_root = os.path.join(dataset.source_path, 'images')
    out_pkl = args.panels_pkl

    panels_acc = {}
    if os.path.exists(out_pkl) and args.append:
        with open(out_pkl, "rb") as f:
            panels_acc = pickle.load(f)

    ref_cache = {}
    for _, r in targets.iterrows():
        prompt = r['prompt']
        ref_frame = r['frame']
        oracle_lvl = int(r['oracle_lvl']); oracle_sp = int(r['oracle_sp_id'])
        clip_top1_lvl = int(r['clip_top1_lvl']); clip_top1_sp = int(r['clip_top1_sp_id'])

        if ref_frame not in train_cams_by_name:
            print(f"  [skip] {prompt}: cam missing")
            continue
        cam = train_cams_by_name[ref_frame]

        img_path = os.path.join(image_root, ref_frame + '.jpg')
        if not os.path.exists(img_path):
            print(f"  [skip] {prompt}: image missing"); continue
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # resize image to cam resolution if different
        if img_rgb.shape[0] != cam.image_height or img_rgb.shape[1] != cam.image_width:
            img_rgb = cv2.resize(img_rgb, (cam.image_width, cam.image_height))

        js = os.path.join(data_path, ref_frame + '.json')
        anno = json.load(open(js))
        gt_np = np.zeros((cam.image_height, cam.image_width), dtype=np.uint8)
        for obj in anno['objects']:
            if obj['category'] == prompt:
                m = polygon_to_mask((cam.image_height, cam.image_width), obj['segmentation'])
                gt_np = np.maximum(gt_np, m)
        gt_mask = gt_np > 0

        if ref_frame not in ref_cache:
            print(f"  caching {ref_frame}...", flush=True)
            ref_cache[ref_frame] = cache_sp_masks_at(cam, gaussians, pipe, background,
                                                    snag, args.levels, args.thresh)
        cache_lvl, cache_sp, cache_masks_cpu = ref_cache[ref_frame]

        vlm.encode_text(prompt)
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        pool_scores = np.array([sim_per_level[lvl - 1][sid].item()
                                for lvl, sid in zip(cache_lvl, cache_sp)], dtype=np.float64)
        order = np.argsort(-pool_scores)

        oracle_mask = render_sp_mask(cam, gaussians, pipe, background, snag,
                                     oracle_lvl, oracle_sp, args.thresh).cpu().numpy()
        top1_mask = render_sp_mask(cam, gaussians, pipe, background, snag,
                                   clip_top1_lvl, clip_top1_sp, args.thresh).cpu().numpy()
        top10_pairs = [(cache_lvl[i], cache_sp[i]) for i in order[:10]]
        top10_mask = render_union(cam, gaussians, pipe, background, snag,
                                  top10_pairs, args.thresh).cpu().numpy()
        torch.cuda.empty_cache()

        combined = gt_mask | oracle_mask | top1_mask | top10_mask
        if not combined.any():
            print(f"  [skip] empty combined for {prompt}"); continue

        gt_panel = overlay_mask(img_rgb, gt_mask, color=(0, 220, 0), alpha=0.5)
        oracle_panel = overlay_mask(img_rgb, oracle_mask, color=(0, 200, 220), alpha=0.5)
        top1_panel = overlay_mask(img_rgb, top1_mask, color=(220, 30, 30), alpha=0.5)
        top10_panel = overlay_mask(img_rgb, top10_mask, color=(220, 220, 30), alpha=0.5)

        gt_panel = crop_box(gt_panel, combined, padding_factor=1.6)
        oracle_panel = crop_box(oracle_panel, combined, padding_factor=1.6)
        top1_panel = crop_box(top1_panel, combined, padding_factor=1.6)
        top10_panel = crop_box(top10_panel, combined, padding_factor=1.6)

        H_PANEL = 180
        def resize_h(im, h_target):
            h, w = im.shape[:2]
            scale = h_target / max(h, 1)
            return cv2.resize(im, (max(int(w * scale), 1), h_target))
        panels_acc[(scene_name, prompt)] = (
            resize_h(gt_panel, H_PANEL),
            resize_h(oracle_panel, H_PANEL),
            resize_h(top1_panel, H_PANEL),
            resize_h(top10_panel, H_PANEL),
        )
        print(f"  saved panels for {prompt}", flush=True)

    with open(out_pkl, "wb") as f:
        pickle.dump(panels_acc, f)
    print(f"\nWrote {len(panels_acc)} total panels to {out_pkl}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--b7_csv", default="output/diagnostics/b7_a4_combined.csv")
    parser.add_argument("--persistent_csv", default="output/diagnostics/persistent_phantoms_17.csv")
    parser.add_argument("--panels_pkl", default="output/diagnostics/_phantom_panels.pkl")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("Done.")
