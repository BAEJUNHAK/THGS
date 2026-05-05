"""
D arm encoder: FastSAM + OpenCLIP ConvNeXt-L (320px) bbox-crop encoder.

OUTPUT (drop-in compatible with image_encoding.py / encode_fastsam_clip.py):
    {feat_subdir}/{frame}_s.npy  (4, H, W) int32   FastSAM 4-scale mask ids (-1 = bg)
    {feat_subdir}/{frame}_f.npy  (M, 768) fp16     per-mask ConvNeXt-L feature

APPROXIMATION NOTE:
    True "Mask-Adapter" (referenced by EmbodiedSplat) injects a learnable mask token
    into a CLIP visual tower trained for mask-conditioned features. This script does
    the simpler bbox-crop variant: for each FastSAM mask we
        crop bbox -> square pad -> resize 320 -> ConvNeXt-L visual encode -> L2-norm.
    This is the higher-capacity analogue of B (FastSAM + ViT-B/16 224-crop), upgrading
    to a stronger 768-D backbone. Useful for measuring the *mask + 768D backbone*
    contribution without entangling the learned mask-token training.

Per-image timing CSV is written to <source_path>/timing_maskadapter.csv.
"""
import os
import csv
import time
import argparse

import numpy as np
import torch
import torchvision
from torch import nn
from tqdm import tqdm
import cv2

try:
    import open_clip
except ImportError as e:
    raise ImportError("open_clip is not installed: pip install open-clip-torch") from e

from encode_fastsam_clip import (  # noqa: E402
    fastsam_4scale, masks_update, LEVEL_NAMES, seed_everything,
    pad_img, get_seg_img, _HAS_FASTSAM,
)
try:
    from ultralytics import FastSAM
except ImportError:
    FastSAM = None


# ---- ConvNeXt-L visual encoder ----

CONVNEXT_NAME = 'convnext_large_d_320'
CONVNEXT_PRETRAIN = 'laion2b_s29b_b131k_ft_soup'
CONVNEXT_INPUT = 320
CONVNEXT_DIM = 768


class ConvNeXtCLIPVisual(nn.Module):
    """OpenCLIP ConvNeXt-L visual tower @ 320×320 → 768-D image features."""
    def __init__(self):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            CONVNEXT_NAME, pretrained=CONVNEXT_PRETRAIN, precision='fp16',
        )
        model.eval()
        self.model = model.to('cuda')
        self.process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((CONVNEXT_INPUT, CONVNEXT_INPUT)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def encode_image(self, x_bchw_float):
        x = self.process(x_bchw_float).half()
        return self.model.encode_image(x)


def mask2feat_convnext(masks, image_bgr_hwc, encoder: ConvNeXtCLIPVisual):
    """Crop -> pad -> resize 320 -> ConvNeXt-L -> L2-norm. Returns ((n, 768) fp16, segmap)."""
    seg_imgs = []
    seg_map = -np.ones(image_bgr_hwc.shape[:2], dtype=np.int32)
    for m in masks:
        crop = get_seg_img(m, image_bgr_hwc)
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        pad = cv2.resize(pad_img(crop), (CONVNEXT_INPUT, CONVNEXT_INPUT))
        seg_imgs.append(pad)
        seg_map[m['segmentation']] = len(seg_imgs) - 1

    if len(seg_imgs) == 0:
        return torch.zeros((0, CONVNEXT_DIM), dtype=torch.float16), seg_map

    arr = np.stack(seg_imgs, axis=0).astype('float32') / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).to('cuda')
    with torch.no_grad():
        emb = encoder.encode_image(t)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.detach().cpu().half(), seg_map


def encode_one_image(image_bgr, fastsam_model, encoder: ConvNeXtCLIPVisual):
    H, W = image_bgr.shape[:2]
    timing = {}

    torch.cuda.synchronize(); t0 = time.perf_counter()
    masks_per_level = fastsam_4scale(fastsam_model, image_bgr)
    torch.cuda.synchronize()
    timing['t_mask_ms'] = (time.perf_counter() - t0) * 1000

    masks_d, masks_s, masks_m, masks_l = (masks_per_level[k] for k in LEVEL_NAMES)
    masks_d, masks_s, masks_m, masks_l = masks_update(
        masks_d, masks_s, masks_m, masks_l,
        iou_thr=0.8, score_thr=0.7, inner_thr=0.5,
    )

    torch.cuda.synchronize(); t0 = time.perf_counter()
    feats, seg_maps = {}, {}
    for name, masks in zip(LEVEL_NAMES, [masks_d, masks_s, masks_m, masks_l]):
        if len(masks) == 0:
            feats[name] = torch.zeros((0, CONVNEXT_DIM), dtype=torch.float16)
            seg_maps[name] = -np.ones((H, W), dtype=np.int32)
            continue
        emb, sm = mask2feat_convnext(masks, image_bgr, encoder)
        feats[name] = emb
        seg_maps[name] = sm
    torch.cuda.synchronize()
    timing['t_clip_ms'] = (time.perf_counter() - t0) * 1000
    timing['t_total_ms'] = timing['t_mask_ms'] + timing['t_clip_ms']
    timing['n_masks'] = sum(f.shape[0] for f in feats.values())
    return feats, seg_maps, timing


def save_one(save_path, feats_per_level, seg_per_level, embed_size=CONVNEXT_DIM):
    H, W = next(iter(seg_per_level.values())).shape
    lengths = [feats_per_level[n].shape[0] for n in LEVEL_NAMES]
    total = sum(lengths)
    if total > 0:
        feat_concat = torch.zeros((total, embed_size), dtype=torch.float16)
    else:
        feat_concat = torch.zeros((1, embed_size), dtype=torch.float16)
    seg_stack = np.zeros((4, H, W), dtype=np.int32)
    cum = 0
    for j, name in enumerate(LEVEL_NAMES):
        sm = seg_per_level[name].copy()
        f = feats_per_level[name]
        n = lengths[j]
        if n > 0:
            feat_concat[cum:cum+n] = f
            if j > 0:
                sm[sm != -1] += cum
        seg_stack[j] = sm
        cum += n
    np.save(save_path + '_s.npy', seg_stack)
    np.save(save_path + '_f.npy', feat_concat[:max(total, 1)].numpy())


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--fastsam_ckpt', type=str, default='ckpts/FastSAM-x.pt')
    parser.add_argument('--feat_subdir', type=str, default='language_features_maskadapter')
    parser.add_argument('--timing_csv', type=str, default=None)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    if not _HAS_FASTSAM:
        raise RuntimeError("ultralytics not installed: pip install ultralytics")

    img_folder = os.path.join(args.source_path, 'images')
    save_folder = os.path.join(args.source_path, args.feat_subdir)
    os.makedirs(save_folder, exist_ok=True)

    if args.timing_csv is None:
        args.timing_csv = os.path.join(args.source_path, 'timing_maskadapter.csv')

    encoder = ConvNeXtCLIPVisual()
    fastsam = FastSAM(args.fastsam_ckpt)

    write_header = not os.path.exists(args.timing_csv)
    timing_fp = open(args.timing_csv, 'a', newline='')
    timing_writer = csv.writer(timing_fp)
    if write_header:
        timing_writer.writerow(['frame', 't_mask_ms', 't_clip_ms', 't_total_ms',
                                'n_masks', 'peak_mem_mb',
                                'resolution_w', 'resolution_h'])

    WARNED = False
    for data_path in tqdm(sorted(os.listdir(img_folder)),
                          desc="ConvNeXt-L+FastSAM encoding"):
        frame_name = os.path.splitext(data_path)[0]
        out_s = os.path.join(save_folder, frame_name + '_s.npy')
        out_f = os.path.join(save_folder, frame_name + '_f.npy')
        if os.path.exists(out_s) and os.path.exists(out_f):
            continue

        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"skip (cannot read): {image_path}"); continue
        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            global_down = orig_h / 1080 if orig_h > 1080 else 1
            if orig_h > 1080 and not WARNED:
                print("[INFO] >1080P, rescaling to 1080P."); WARNED = True
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)

        torch.cuda.reset_peak_memory_stats()
        feats, seg_maps, timing = encode_one_image(image, fastsam, encoder)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        save_one(os.path.join(save_folder, frame_name), feats, seg_maps)

        timing_writer.writerow([
            frame_name,
            f"{timing['t_mask_ms']:.2f}",
            f"{timing['t_clip_ms']:.2f}",
            f"{timing['t_total_ms']:.2f}",
            timing['n_masks'],
            f"{peak_mb:.1f}",
            resolution[0], resolution[1],
        ])
        timing_fp.flush()

    timing_fp.close()
    print(f"\nDone. Features: {save_folder}, timing: {args.timing_csv}")


if __name__ == '__main__':
    main()
