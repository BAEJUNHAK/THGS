"""
D arm encoder — EmbodiedSplat / Mask-Adapter style:
    full image → ConvNeXt-L dense visual feature map (Hp, Wp, 768)
    + FastSAM masks  →  average pool inside each mask
    →  instance feature (M, 768)

This faithfully matches the paper's two-branch design:
  branch A (encoder):  pixel-level CLIP-aligned dense feature
  branch B (segmenter): FastSAM masks
  merge:               mask-wise average pooling of pixel features

OUTPUT (drop-in compatible with image_encoding.py / encode_fastsam_clip.py):
    {feat_subdir}/{frame}_s.npy  (4, H, W) int32   FastSAM 4-scale mask ids (-1 = bg)
    {feat_subdir}/{frame}_f.npy  (M, 768) fp16     per-mask avg-pooled ConvNeXt-L feature

Optional pixel feature export (for diagnostic visualization):
    {feat_subdir}/{frame}_p.npy  (Hp, Wp, 768) fp16
    only saved for frames in --pixel_frames or with --pixel_all.

NOTE: This replaces the previous bbox-crop implementation (`mask2feat_convnext`),
which was an over-approximation that broke the paper's "global context preserved"
property. The previous version cropped each mask's bbox and re-encoded, losing the
full-image context that pixel-level CLIP-aligned encoders are supposed to preserve.

Per-image timing CSV is written to <source_path>/timing_maskadapter.csv.
"""
import os
import csv
import time
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2

try:
    import open_clip
except ImportError as e:
    raise ImportError("open_clip is not installed: pip install open-clip-torch") from e

from encode_fastsam_clip import (  # noqa: E402  - reuse FastSAM utilities
    fastsam_4scale, masks_update, LEVEL_NAMES, seed_everything,
    _HAS_FASTSAM,
)
from encode_openseg_fastsam import avg_pool_per_mask, downsize_pixel_feat  # noqa: E402

try:
    from ultralytics import FastSAM
except ImportError:
    FastSAM = None


CONVNEXT_NAME = 'convnext_large_d_320'
CONVNEXT_PRETRAIN = 'laion2b_s29b_b131k_ft_soup'
CONVNEXT_INPUT = 320
CONVNEXT_DIM = 768
PIXFEAT_MAX_DIM = 256


# ---- Dense ConvNeXt-L feature extractor ----

class ConvNeXtLDense:
    """Full-image dense ConvNeXt-L feature extractor.

    OpenCLIP's `convnext_large_d_320` has the timm trunk do internal global pooling
    (trunk forward returns (B, 1536) post-pool). To get spatial features we use the
    standard timm convention `trunk.forward_features(x)` which always returns the
    pre-pool spatial map (B, C, H, W). Then we apply OpenCLIP's head (Mlp/Linear)
    per-spatial-location to get (B, H, W, embed_dim).

    For 320 input → ConvNeXt-L stride 32 → 10×10 dense grid × 768-D.
    """
    def __init__(self, input_size=CONVNEXT_INPUT,
                 model_name=CONVNEXT_NAME, pretrained=CONVNEXT_PRETRAIN):
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, precision='fp16',
        )
        model.eval()
        self.visual = model.visual.to('cuda')
        self.input_size = input_size
        self.dim = CONVNEXT_DIM
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                 device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                device='cuda').view(1, 3, 1, 1)
        if not hasattr(self.visual, 'trunk') or not hasattr(self.visual.trunk, 'forward_features'):
            raise RuntimeError(
                "OpenCLIP ConvNeXt visual lacks `trunk.forward_features` — "
                "incompatible model or OpenCLIP version."
            )

    def _preprocess(self, rgb_uint8_HW3):
        x = torch.from_numpy(rgb_uint8_HW3).permute(2, 0, 1).unsqueeze(0)
        x = x.float().to('cuda') / 255.0
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode='bilinear', align_corners=False)
        return ((x - self.mean) / self.std).half()

    @torch.no_grad()
    def __call__(self, rgb_uint8_HW3):
        """Returns (Hp, Wp, 768) np.float32 dense feature map."""
        x = self._preprocess(rgb_uint8_HW3)
        # Spatial features: bypass trunk's internal global_pool.
        spatial = self.visual.trunk.forward_features(x)
        if spatial.dim() != 4:
            raise RuntimeError(
                f"trunk.forward_features returned shape {tuple(spatial.shape)}, "
                "expected 4D (B, C, H, W). Incompatible model."
            )
        B, C, H, W = spatial.shape

        # (B, C, H, W) → (B*H*W, C) for per-token head application
        flat = spatial.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C)

        # Apply OpenCLIP head to each spatial token.
        # Try the whole head first (works when head has only Linear/Mlp/Dropout).
        try:
            out = self.visual.head(flat)
        except Exception as e_full:
            # Iterate sub-modules, skipping anything that breaks on flat input
            # (pool / flatten / rearrange layers).
            out = flat
            applied = 0
            for name, mod in self.visual.head.named_children():
                cls = type(mod).__name__.lower()
                if any(k in cls for k in ['pool', 'flatten', 'rearrange']):
                    continue
                try:
                    out = mod(out); applied += 1
                except Exception:
                    pass
            if applied == 0:
                raise RuntimeError(
                    f"Could not apply OpenCLIP head per-token. "
                    f"Original error: {e_full}"
                )

        if out.dim() != 2 or out.shape[0] != B * H * W:
            raise RuntimeError(
                f"head output shape {tuple(out.shape)} unexpected. "
                f"Expected ({B * H * W}, embed_dim)."
            )

        out = out.reshape(B, H, W, -1)
        return out[0].float().cpu().numpy()


# ---- Per-image pipeline ----

def encode_one_image(image_bgr, fastsam_model, dense_encoder):
    """Returns feats_per_level, seg_per_level, pixel_feat (Hp,Wp,D), timing."""
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    timing = {}

    # ---- Mask proposer (FastSAM 4-scale) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    masks_per_level = fastsam_4scale(fastsam_model, image_bgr)
    torch.cuda.synchronize()
    timing['t_mask_ms'] = (time.perf_counter() - t0) * 1000

    masks_d, masks_s, masks_m, masks_l = (masks_per_level[k] for k in LEVEL_NAMES)
    masks_d, masks_s, masks_m, masks_l = masks_update(
        masks_d, masks_s, masks_m, masks_l,
        iou_thr=0.8, score_thr=0.7, inner_thr=0.5,
    )

    # ---- ConvNeXt-L dense feature ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pixel_feat = dense_encoder(image_rgb)        # (Hp, Wp, 768)
    torch.cuda.synchronize()
    timing['t_feat_ms'] = (time.perf_counter() - t0) * 1000

    # ---- Mask average pooling ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    feats, seg_maps = {}, {}
    D = pixel_feat.shape[-1]
    for name, masks in zip(LEVEL_NAMES, [masks_d, masks_s, masks_m, masks_l]):
        if len(masks) == 0:
            feats[name] = torch.zeros((0, D), dtype=torch.float16)
            seg_maps[name] = -np.ones((H, W), dtype=np.int32)
            continue
        emb, sm = avg_pool_per_mask(pixel_feat, masks, target_HW=(H, W))
        feats[name] = emb
        seg_maps[name] = sm
    torch.cuda.synchronize()
    timing['t_pool_ms'] = (time.perf_counter() - t0) * 1000
    timing['t_total_ms'] = (timing['t_mask_ms'] + timing['t_feat_ms']
                            + timing['t_pool_ms'])
    timing['n_masks'] = sum(f.shape[0] for f in feats.values())
    return feats, seg_maps, pixel_feat, timing


def save_one(save_path, feats_per_level, seg_per_level, pixel_feat=None,
             save_pixel=False, embed_size=CONVNEXT_DIM):
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
    if save_pixel and pixel_feat is not None:
        small = downsize_pixel_feat(pixel_feat).astype(np.float16)
        np.save(save_path + '_p.npy', small)


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--fastsam_ckpt', type=str, default='ckpts/FastSAM-x.pt')
    parser.add_argument('--feat_subdir', type=str, default='language_features_maskadapter')
    parser.add_argument('--timing_csv', type=str, default=None)
    parser.add_argument('--pixel_frames', nargs='*', default=None,
                        help='Save (downsized) pixel feature *_p.npy only for these frames.')
    parser.add_argument('--pixel_all', action='store_true')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    if not _HAS_FASTSAM:
        raise RuntimeError("ultralytics not installed: pip install ultralytics")

    img_folder = os.path.join(args.source_path, 'images')
    save_folder = os.path.join(args.source_path, args.feat_subdir)
    os.makedirs(save_folder, exist_ok=True)
    if args.timing_csv is None:
        args.timing_csv = os.path.join(args.source_path, 'timing_maskadapter.csv')

    pixel_frames_set = set(args.pixel_frames) if args.pixel_frames else set()

    fastsam = FastSAM(args.fastsam_ckpt)
    dense_encoder = ConvNeXtLDense()

    write_header = not os.path.exists(args.timing_csv)
    timing_fp = open(args.timing_csv, 'a', newline='')
    timing_writer = csv.writer(timing_fp)
    if write_header:
        timing_writer.writerow(['frame', 't_mask_ms', 't_feat_ms', 't_pool_ms',
                                't_total_ms', 'n_masks', 'peak_mem_mb',
                                'resolution_w', 'resolution_h', 'pixel_saved'])

    WARNED = False
    for data_path in tqdm(sorted(os.listdir(img_folder)),
                          desc="ConvNeXt-L dense + FastSAM pool"):
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
        feats, seg_maps, pixel_feat, timing = encode_one_image(
            image, fastsam, dense_encoder)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        save_pixel = args.pixel_all or (frame_name in pixel_frames_set)
        save_one(os.path.join(save_folder, frame_name),
                 feats, seg_maps, pixel_feat=pixel_feat,
                 save_pixel=save_pixel, embed_size=pixel_feat.shape[-1])

        timing_writer.writerow([
            frame_name,
            f"{timing['t_mask_ms']:.2f}",
            f"{timing['t_feat_ms']:.2f}",
            f"{timing['t_pool_ms']:.2f}",
            f"{timing['t_total_ms']:.2f}",
            timing['n_masks'],
            f"{peak_mb:.1f}",
            resolution[0], resolution[1],
            int(save_pixel),
        ])
        timing_fp.flush()

    timing_fp.close()
    print(f"\nDone. Features: {save_folder}, timing: {args.timing_csv}")


if __name__ == '__main__':
    main()
