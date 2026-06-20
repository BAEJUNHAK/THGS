#!/usr/bin/env python3
"""Check whether a 3DGS checkpoint camera radius matches a COLMAP scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_colmap_readers(vala_root: Path):
    sys.path.insert(0, str(vala_root))
    from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text, qvec2rotmat
    from utils.graphics_utils import getWorld2View2

    return read_extrinsics_binary, read_extrinsics_text, qvec2rotmat, getWorld2View2


def camera_radius(scene_root: Path, vala_root: Path) -> float:
    read_extrinsics_binary, read_extrinsics_text, qvec2rotmat, getWorld2View2 = load_colmap_readers(vala_root)
    sparse = scene_root / "sparse" / "0"
    try:
        extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))

    centers = []
    for extr in extrinsics.values():
        r = np.transpose(qvec2rotmat(extr.qvec))
        t = np.array(extr.tvec)
        w2c = getWorld2View2(r, t)
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3:4])
    cam_centers = np.hstack(centers)
    center = np.mean(cam_centers, axis=1, keepdims=True)
    diagonal = np.max(np.linalg.norm(cam_centers - center, axis=0, keepdims=True))
    return float(diagonal * 1.1)


def checkpoint_radius(checkpoint: Path) -> tuple[int, float]:
    model_params, _ = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(model_params, tuple) or len(model_params) != 12:
        raise ValueError(f"Expected 12-tuple RGB checkpoint, got len={len(model_params)}")
    return int(model_params[1].shape[0]), float(model_params[11])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vala-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-ratio-delta", type=float, default=0.03)
    args = parser.parse_args()

    n_gaussians, ckpt_radius = checkpoint_radius(args.checkpoint)
    colmap_radius = camera_radius(args.scene_root, args.vala_root)
    ratio = colmap_radius / ckpt_radius
    delta = abs(ratio - 1.0)

    print(f"scene_root: {args.scene_root}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"gaussians: {n_gaussians}")
    print(f"checkpoint spatial_lr_scale: {ckpt_radius:.9f}")
    print(f"COLMAP camera radius: {colmap_radius:.9f}")
    print(f"ratio: {ratio:.9f}")

    if delta > args.max_ratio_delta:
        raise SystemExit(
            f"Checkpoint/COLMAP radius mismatch: |ratio - 1|={delta:.6f} > {args.max_ratio_delta}"
        )


if __name__ == "__main__":
    main()
