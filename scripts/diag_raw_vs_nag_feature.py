"""
Diagnose A1 Type A failures: (a) CLIP encoder limit vs (b) registration mixing.

Without language_features on this machine, we use two indirect signals:

  Signal 1 — CLIP-limit ceiling (raw image crop CLIP).
    For each prompt, crop the ref-view RGB by GT bbox and feed it whole-image to CLIP.
    Score with canon-contrast (same metric as test_lerf.py).
    If this is high → CLIP encoder DOES recognize the object on the raw view.
    If this is low → CLIP encoder is fundamentally weak on this prompt → (a).

  Signal 2 — Registration mixing fingerprint (nag_feat self-comparison).
    Find Oracle's correct SP and CLIP top-1 SP (wrong pick).
    Compute cosine sim between their nag_feat vectors.
    If high (≥0.85) → correct SP's feature was contaminated to look like wrong-object → (b).
    If low → CLIP just genuinely prefers the wrong SP's distinct feature → (a)-ish.

  Plus baseline:
    GT crop score, CLIP top-1 SP's rendered region crop score (control), correct SP's region crop score.

Verdict logic:
  GT_crop_high AND correct_SP_nag_low → (b) mixing dominant
  GT_crop_low                          → (a) CLIP limit dominant
  Both moderate                        → both contribute
"""
import os
import sys
import csv
import json
import cv2
import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser

from gaussian_renderer import render, render_point
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure
from arguments import ModelParams, PipelineParams, OptimizationParams


A1_PROMPTS = {
    'figurines': ['pumpkin', 'pikachu', 'bag', 'pirate hat', 'miffy'],
    'waldo_kitchen': ['cabinet', 'ottolenghi', 'spoon', 'pour-over vessel'],
    'ramen': ['onion segments', 'hand'],
    'teatime': ['hooves'],
}


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
    cache_lvl, cache_sp, cache_mask = [], [], []
    for lvl in levels:
        sp_ids = snag.labels[lvl].long().unique()
        for sp_id in sp_ids:
            m = render_sp_mask(cam, gaussians, pipe, background, snag, lvl, sp_id.item(), thresh)
            cache_lvl.append(lvl)
            cache_sp.append(int(sp_id.item()))
            cache_mask.append(m.cpu())
    return cache_lvl, cache_sp, torch.stack(cache_mask, dim=0)


def bbox_from_mask(mask_np):
    """Return (y0, x0, y1, x1) bbox of True pixels, or None if empty."""
    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


@torch.no_grad()
def crop_and_clip_score(vlm, img_pil, bbox, pad_ratio=0.1):
    """Crop img by bbox (with optional padding), feed whole crop to CLIP, return canon-contrast score."""
    if bbox is None:
        return float('nan')
    y0, x0, y1, x1 = bbox
    H, W = img_pil.size[1], img_pil.size[0]
    h, w = y1 - y0, x1 - x0
    py, px = int(h * pad_ratio), int(w * pad_ratio)
    y0 = max(0, y0 - py); x0 = max(0, x0 - px)
    y1 = min(H, y1 + py); x1 = min(W, x1 + px)
    crop = img_pil.crop((x0, y0, x1, y1))
    # CLIP preprocess
    from torchvision import transforms as T
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    img_t = preprocess(crop).unsqueeze(0).cuda().half()
    feat = vlm.clip_pretrained.encode_image(img_t).type(torch.float32)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return float(vlm.compute_similarity(feat).item())


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    prompts = A1_PROMPTS.get(scene_name, [])
    if not prompts:
        print(f"[{scene_name}] no A1 prompts defined, skip")
        return
    print(f"\n=== {scene_name}: {len(prompts)} A1 prompts ===", flush=True)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    vlm = ClipSimMeasure()
    vlm.load_model()

    train_cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    images_dir = os.path.join(dataset.source_path, 'images')

    out_rows = []
    ref_cache = {}
    for prompt in prompts:
        ref_frame = None
        ref_anno = None
        for img in sorted(os.listdir(data_path)):
            if not img.endswith('.jpg'):
                continue
            name = img.split('.')[0]
            if name not in train_cams_by_name:
                continue
            anno = json.load(open(os.path.join(data_path, name + '.json')))
            if any(o['category'] == prompt for o in anno['objects']):
                ref_frame = name
                ref_anno = anno
                break
        if ref_frame is None:
            print(f"  [{prompt}] no frame found"); continue
        cam = train_cams_by_name[ref_frame]
        H, W = cam.image_height, cam.image_width
        gt_np = np.zeros((H, W), dtype=np.uint8)
        for obj in ref_anno['objects']:
            if obj['category'] == prompt:
                gt_np = np.maximum(gt_np, polygon_to_mask((H, W), obj['segmentation']))
        gt_t = torch.from_numpy(gt_np > 0).cuda()

        if ref_frame not in ref_cache:
            print(f"  caching SP masks at {ref_frame}...", flush=True)
            ref_cache[ref_frame] = cache_sp_masks_at(cam, gaussians, pipe, background, snag, args.levels, args.thresh)
        cache_lvl, cache_sp, cache_masks_cpu = ref_cache[ref_frame]

        cache_masks = cache_masks_cpu.cuda()
        inter = (cache_masks & gt_t[None]).sum(dim=(1, 2)).float()
        union = (cache_masks | gt_t[None]).sum(dim=(1, 2)).float()
        ious = (inter / union.clamp_min(1)).cpu().numpy()
        correct_idx = int(np.argmax(ious))
        correct_iou = float(ious[correct_idx])
        correct_lvl = cache_lvl[correct_idx]
        correct_sp_id = cache_sp[correct_idx]

        vlm.encode_text(prompt)
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        pool_scores = np.array([
            sim_per_level[lvl - 1][sp].item()
            for lvl, sp in zip(cache_lvl, cache_sp)
        ])
        nag_score_correct = float(pool_scores[correct_idx])
        nag_rank_correct = int((pool_scores > nag_score_correct).sum() + 1)
        nag_pool = len(pool_scores)
        clip_top1_idx = int(np.argmax(pool_scores))
        clip_top1_lvl = cache_lvl[clip_top1_idx]
        clip_top1_sp_id = cache_sp[clip_top1_idx]
        clip_top1_iou = float(ious[clip_top1_idx])

        # Cosine sim between correct SP nag_feat and CLIP top-1 SP nag_feat
        f_correct = snag.feat[correct_lvl - 1][correct_sp_id]
        f_top1 = snag.feat[clip_top1_lvl - 1][clip_top1_sp_id]
        cos_correct_top1 = float(torch.nn.functional.cosine_similarity(
            f_correct.unsqueeze(0), f_top1.unsqueeze(0)
        ).item())

        # Load ref image (PIL)
        # try .jpg first
        img_path = os.path.join(images_dir, ref_frame + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, ref_frame + '.png')
        img_pil = Image.open(img_path).convert('RGB')
        # GT mask & rendered SP regions at ref view (in image coords)
        gt_bbox = bbox_from_mask(gt_np > 0)
        correct_sp_mask_np = cache_masks_cpu[correct_idx].numpy()
        correct_bbox = bbox_from_mask(correct_sp_mask_np)
        top1_sp_mask_np = cache_masks_cpu[clip_top1_idx].numpy()
        top1_bbox = bbox_from_mask(top1_sp_mask_np)

        # Note: rendered cam is at scene-rendering resolution which may equal the train image resolution.
        # We crop the original image at the same H,W.
        if img_pil.size != (W, H):
            img_pil = img_pil.resize((W, H), Image.BICUBIC)

        gt_crop_score = crop_and_clip_score(vlm, img_pil, gt_bbox)
        correct_crop_score = crop_and_clip_score(vlm, img_pil, correct_bbox)
        top1_crop_score = crop_and_clip_score(vlm, img_pil, top1_bbox)

        del cache_masks
        torch.cuda.empty_cache()

        # Verdict
        # (a) CLIP-limit if GT crop fails too
        # (b) Mixing if GT crop OK but correct SP nag_feat got pulled toward top1 (high cos sim)
        if gt_crop_score < 0.6:
            verdict = "(a) CLIP-LIMIT"
        elif cos_correct_top1 > 0.85:
            verdict = "(b) MIXING (correct SP feature ~ wrong SP feature)"
        elif gt_crop_score > 0.7 and nag_score_correct < 0.5:
            verdict = "(b)-ish (raw OK but nag_feat weak)"
        else:
            verdict = "MIXED / inconclusive"

        print(f"  [{prompt}] ref={ref_frame}")
        print(f"     Correct SP=(lvl{correct_lvl},sp{correct_sp_id}, IoU={correct_iou:.3f})  CLIPtop1=(lvl{clip_top1_lvl},sp{clip_top1_sp_id}, IoU={clip_top1_iou:.3f})")
        print(f"     nag_feat scores: correct={nag_score_correct:.3f} rank {nag_rank_correct}/{nag_pool} | top1={pool_scores[clip_top1_idx]:.3f}")
        print(f"     cosine(correct_nag, top1_nag) = {cos_correct_top1:.3f}")
        print(f"     RAW CLIP crop scores: GT={gt_crop_score:.3f}  correct_SP_region={correct_crop_score:.3f}  top1_SP_region={top1_crop_score:.3f}")
        print(f"     verdict: {verdict}")

        out_rows.append({
            'scene': scene_name, 'prompt': prompt, 'ref_frame': ref_frame,
            'correct_sp_lvl': correct_lvl, 'correct_sp_id': correct_sp_id,
            'correct_single_iou': correct_iou,
            'clip_top1_lvl': clip_top1_lvl, 'clip_top1_sp_id': clip_top1_sp_id,
            'clip_top1_iou': clip_top1_iou,
            'nag_score_correct': nag_score_correct,
            'nag_rank_correct': nag_rank_correct, 'nag_pool': nag_pool,
            'nag_score_top1': float(pool_scores[clip_top1_idx]),
            'cos_correct_top1': cos_correct_top1,
            'gt_crop_score': gt_crop_score,
            'correct_sp_crop_score': correct_crop_score,
            'top1_sp_crop_score': top1_crop_score,
            'verdict': verdict,
        })

    if out_rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        new_file = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            if new_file:
                w.writeheader()
            for r in out_rows:
                w.writerow(r)
        print(f"\n  Wrote {len(out_rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--out_csv", type=str, default="output/diagnostics/raw_vs_nag_feature.csv")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--thresh", type=float, default=0.5)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("\nDone.")
