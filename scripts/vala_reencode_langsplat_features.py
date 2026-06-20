#!/usr/bin/env python3
"""Re-encode cached LangSplat segmentation maps with VALA's OpenCLIP encoder.

The repository already stores per-view LangSplat-style segmentation maps as
`*_s.npy`. VALA's official `run_sam.py --use_langsplat` produces the same
segmentation maps for checked LERF-OVS frames, but with OpenCLIP ViT-B/16
language features. This script reuses the cached masks and recomputes only
the OpenCLIP features, avoiding an expensive SAM pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vala-root", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--cached-feature-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def pad_img(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    side = max(w, h)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    if h > w:
        out[:, (h - w) // 2 : (h - w) // 2 + w, :] = img
    else:
        out[(w - h) // 2 : (w - h) // 2 + h, :, :] = img
    return out


def crop_mask_tile(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    masked = image_rgb.copy()
    masked[~mask] = np.array([255, 255, 255], dtype=np.uint8)
    crop = masked[y0:y1, x0:x1, :]
    return cv2.resize(pad_img(crop), (224, 224))


def build_tiles(image_rgb: np.ndarray, seg_stack: np.ndarray) -> tuple[torch.Tensor, list[int], int]:
    ids = sorted(int(x) for x in np.unique(seg_stack) if int(x) >= 0)
    tiles = []
    for idx in ids:
        mask = np.any(seg_stack == idx, axis=0)
        tiles.append(crop_mask_tile(image_rgb, mask))
    if not tiles:
        raise ValueError("no masks found")
    arr = np.stack(tiles, axis=0).astype("float32")
    return torch.from_numpy(arr).permute(0, 3, 1, 2) / 255.0, ids, max(ids)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.vala_root))
    from run_sam import OpenCLIPNetwork, OpenCLIPNetworkConfig

    args.output_root.mkdir(parents=True, exist_ok=True)
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

    image_paths = sorted((args.scene_root / "images").glob("*.jpg"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    for image_path in image_paths:
        stem = image_path.stem
        out_f = args.output_root / f"{stem}_f.npy"
        out_s = args.output_root / f"{stem}_s.npy"
        if not args.force and out_f.exists() and out_s.exists():
            continue

        seg_path = args.cached_feature_root / f"{stem}_s.npy"
        if not seg_path.exists():
            raise FileNotFoundError(seg_path)

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        seg_stack = np.load(seg_path).astype(np.int32)

        tiles, ids, max_id = build_tiles(image_rgb, seg_stack)
        tiles = tiles.to("cuda")
        embeds = []
        with torch.no_grad():
            for start in range(0, tiles.shape[0], 64):
                batch = tiles[start : start + 64]
                feat = model.encode_image(batch)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embeds.append(feat.detach().cpu().half())
        compact_embed = torch.cat(embeds, dim=0).numpy()
        img_embed = np.zeros((max_id + 1, compact_embed.shape[1]), dtype=np.float16)
        img_embed[np.asarray(ids, dtype=np.int64)] = compact_embed

        np.save(out_f, img_embed)
        np.save(out_s, seg_stack)
        print(stem, img_embed.shape, seg_stack.shape)


if __name__ == "__main__":
    main()
