"""
A2 image-CLIP ceiling diagnostic.

Question: if we feed CLIP image encoder a clean crop of the GT object,
does the correct prompt come up as top-1 against the scene's prompt set?

4 crop policies × 67 prompts (one row per (prompt, ref_frame, policy)):
  (i) tight   — minimal bbox of GT polygon, raw RGB
  (ii) mask   — minimal bbox, background blacked out outside polygon (SAM-style proxy)
  (iii) context — 1.5× expanded bbox, raw RGB
  (iv) method — polygon-blackout + 1.2× expanded bbox (method-matched proxy;
        true method-matched requires SAM mask + replicated preprocessing — deferred)

For each crop:
  - CLIP image encode (ViT-B-16, same model as ClipSimMeasure)
  - Score against scene's prompts via canon-contrast (matches method)
  - Score against scene's prompts via raw cosine (no canon)
  - Record rank of true prompt, top-1 prompt, top-1 score

Output: output/diagnostics/a2_image_clip_ceiling.csv
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
from PIL import Image

import open_clip


CANON = ["object", "things", "stuff", "texture"]


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask > 0


def union_mask_for_prompt(img_shape, objects, prompt):
    h, w = img_shape
    mask = np.zeros((h, w), dtype=bool)
    for obj in objects:
        if obj['category'] != prompt:
            continue
        m = polygon_to_mask((h, w), obj['segmentation'])
        mask = mask | m
    return mask


def bbox_of_mask(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def expand_bbox(bbox, factor, w_max, h_max):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) / 2 * factor, (y1 - y0) / 2 * factor
    nx0 = max(0, int(cx - half_w))
    ny0 = max(0, int(cy - half_h))
    nx1 = min(w_max, int(cx + half_w))
    ny1 = min(h_max, int(cy + half_h))
    if nx1 <= nx0 or ny1 <= ny0:
        return None
    return nx0, ny0, nx1, ny1


def make_crop(image_rgb_np, mask, policy):
    """Return PIL Image cropped per policy, or None if no GT pixels."""
    h, w = image_rgb_np.shape[:2]
    bb = bbox_of_mask(mask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    if policy == "tight":
        crop = image_rgb_np[y0:y1, x0:x1].copy()
    elif policy == "mask":
        img2 = image_rgb_np.copy()
        img2[~mask] = 0
        crop = img2[y0:y1, x0:x1].copy()
    elif policy == "context":
        bb2 = expand_bbox(bb, 1.5, w, h)
        if bb2 is None:
            return None
        x0, y0, x1, y1 = bb2
        crop = image_rgb_np[y0:y1, x0:x1].copy()
    elif policy == "method":
        img2 = image_rgb_np.copy()
        img2[~mask] = 0
        bb2 = expand_bbox(bb, 1.2, w, h)
        if bb2 is None:
            return None
        x0, y0, x1, y1 = bb2
        crop = img2[y0:y1, x0:x1].copy()
    else:
        raise ValueError(f"unknown policy {policy}")
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return Image.fromarray(crop)


def canon_contrast_score(image_feat, prompt_text_feat, canon_text_feats):
    """Return canon-contrasted similarity (single scalar)."""
    # image_feat: (D,), prompt_text_feat: (D,), canon_text_feats: (4, D)
    pos = (image_feat @ prompt_text_feat).item()
    neg_list = [(image_feat @ c).item() for c in canon_text_feats]
    # softmax(10 * [pos, neg]) per canon, take pos prob, return min across canons (worst case)
    sims = []
    for neg in neg_list:
        s = torch.softmax(torch.tensor([10.0 * pos, 10.0 * neg]), dim=0)
        sims.append(float(s[0].item()))
    return min(sims)  # worst canon contrast = most conservative


CSV_HEADER = [
    'scene', 'prompt', 'ref_frame', 'policy',
    'crop_w', 'crop_h',
    # canon-contrast scores
    'canon_score_true', 'canon_score_top1',
    'canon_rank_true', 'canon_top1_prompt',
    # raw cosine scores
    'raw_score_true', 'raw_score_top1',
    'raw_rank_true', 'raw_top1_prompt',
    # full ranking (for analysis)
    'num_scene_prompts',
]


@torch.no_grad()
def run(args):
    device = torch.device("cuda")
    print(f"Loading CLIP ViT-B-16 laion2b_s34b_b88k ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    canon_tokens = tokenizer(CANON).to(device)
    canon_feats = model.encode_text(canon_tokens).float()
    canon_feats = canon_feats / canon_feats.norm(dim=-1, keepdim=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    mode = 'w' if args.append == 0 else 'a'
    new_header = (mode == 'w' or not os.path.exists(args.out_csv))
    f_csv = open(args.out_csv, mode, newline='')
    writer = csv.writer(f_csv)
    if new_header:
        writer.writerow(CSV_HEADER)

    scenes = args.scenes
    policies = ["tight", "mask", "context", "method"]

    for scene_name in scenes:
        src = os.path.join(args.data_root, scene_name)
        label_dir = os.path.join(args.data_root, 'label', scene_name)
        img_dir = os.path.join(src, 'images')
        if not os.path.isdir(label_dir) or not os.path.isdir(img_dir):
            print(f"  [skip] {scene_name}: missing label or images", flush=True)
            continue
        print(f"\n=== {scene_name} ===", flush=True)

        # collect prompts and ref_frames (ref = first GT frame for each prompt)
        img_list = sorted([f for f in os.listdir(label_dir) if f.endswith('.jpg')])
        frames_data = {}
        prompt_to_frames = defaultdict(list)
        for im in img_list:
            image_name = im.split('.')[0]
            js = os.path.join(label_dir, image_name + '.json')
            try:
                anno = json.load(open(js))
            except Exception:
                continue
            frames_data[image_name] = anno['objects']
            for p in set(o['category'] for o in anno['objects']):
                prompt_to_frames[p].append(image_name)
        for p in prompt_to_frames:
            prompt_to_frames[p] = sorted(prompt_to_frames[p])
        scene_prompts = sorted(prompt_to_frames.keys())
        print(f"  {len(scene_prompts)} unique prompts, {len(frames_data)} GT frames", flush=True)

        # Encode all scene prompts once (text features)
        prompt_tokens = tokenizer(scene_prompts).to(device)
        prompt_feats = model.encode_text(prompt_tokens).float()
        prompt_feats = prompt_feats / prompt_feats.norm(dim=-1, keepdim=True)
        # shape (N, 512)

        for prompt in scene_prompts:
            frames = prompt_to_frames[prompt]
            ref_frame = frames[0]
            img_path = os.path.join(img_dir, ref_frame + '.jpg')
            if not os.path.exists(img_path):
                print(f"  [skip] {prompt}: image {img_path} missing", flush=True)
                continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            image_rgb_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb_np.shape[:2]
            mask = union_mask_for_prompt((h, w), frames_data[ref_frame], prompt)
            if not mask.any():
                continue
            true_idx = scene_prompts.index(prompt)

            for policy in policies:
                crop_img = make_crop(image_rgb_np, mask, policy)
                if crop_img is None:
                    writer.writerow([
                        scene_name, prompt, ref_frame, policy, 0, 0,
                        '', '', '', '', '', '', '', '', len(scene_prompts)])
                    continue
                pre = preprocess(crop_img).unsqueeze(0).to(device).half()
                img_feat = model.encode_image(pre).float()
                img_feat = (img_feat / img_feat.norm(dim=-1, keepdim=True)).squeeze(0)

                # raw cosine vs every scene prompt
                raw_sims = (prompt_feats @ img_feat).cpu().numpy()
                raw_order = np.argsort(-raw_sims)
                raw_rank_true = int(np.where(raw_order == true_idx)[0][0] + 1)
                raw_top1_idx = int(raw_order[0])
                raw_score_true = float(raw_sims[true_idx])
                raw_score_top1 = float(raw_sims[raw_top1_idx])
                raw_top1_prompt = scene_prompts[raw_top1_idx]

                # canon-contrast vs every scene prompt
                canon_scores = np.empty(len(scene_prompts), dtype=np.float32)
                for j in range(len(scene_prompts)):
                    canon_scores[j] = canon_contrast_score(
                        img_feat, prompt_feats[j], canon_feats)
                canon_order = np.argsort(-canon_scores)
                canon_rank_true = int(np.where(canon_order == true_idx)[0][0] + 1)
                canon_top1_idx = int(canon_order[0])
                canon_score_true = float(canon_scores[true_idx])
                canon_score_top1 = float(canon_scores[canon_top1_idx])
                canon_top1_prompt = scene_prompts[canon_top1_idx]

                cw, ch = crop_img.size
                writer.writerow([
                    scene_name, prompt, ref_frame, policy, cw, ch,
                    f'{canon_score_true:.6f}', f'{canon_score_top1:.6f}',
                    canon_rank_true, canon_top1_prompt,
                    f'{raw_score_true:.6f}', f'{raw_score_top1:.6f}',
                    raw_rank_true, raw_top1_prompt,
                    len(scene_prompts),
                ])
            f_csv.flush()
            print(
                f"  {prompt}: raw_rank(tight/mask/ctx/method)= "
                f"{[scene_prompts.index(p)+1 if False else '-' for p in []]} "
                f"true={prompt}", flush=True)

    f_csv.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/lerf_ovs")
    parser.add_argument("--scenes", type=str, nargs="+",
                        default=["figurines", "ramen", "teatime", "waldo_kitchen"])
    parser.add_argument("--out_csv", type=str,
                        default="output/diagnostics/a2_image_clip_ceiling.csv")
    parser.add_argument("--append", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    run(args)
    print("\nDone.")
