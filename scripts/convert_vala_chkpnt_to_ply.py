#!/usr/bin/env python3
"""Convert a VALA/3DGS 12-tuple RGB checkpoint to point_cloud.ply.

ReferSplat publishes the plain RGB 3DGS state as ``*chkpnt30000.pth``. VALA's
``Scene(load_iteration=...)`` expects a matching
``point_cloud/iteration_*/point_cloud.ply`` before it restores the checkpoint.
This CPU-only converter writes that PLY directly from the checkpoint tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().contiguous().numpy()


def load_model_params(path: Path) -> tuple[tuple, int]:
    model_params, iteration = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model_params, tuple) or len(model_params) != 12:
        raise ValueError(f"Expected 12-tuple model params, got {type(model_params)} len={len(model_params)}")
    return model_params, int(iteration)


def write_ply(model_params: tuple, out_path: Path) -> None:
    xyz = model_params[1]
    fdc = model_params[2]
    frest = model_params[3]
    scaling = model_params[4]
    rotation = model_params[5]
    opacity = model_params[6]

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Unexpected xyz shape: {tuple(xyz.shape)}")
    if fdc.shape[1:] != (1, 3):
        raise ValueError(f"Unexpected features_dc shape: {tuple(fdc.shape)}")
    if scaling.shape[1] != 3 or rotation.shape[1] != 4 or opacity.shape[1] != 1:
        raise ValueError(
            "Unexpected Gaussian parameter shapes: "
            f"scale={tuple(scaling.shape)} rot={tuple(rotation.shape)} opacity={tuple(opacity.shape)}"
        )

    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(fdc.shape[1] * fdc.shape[2])]
    fields += [f"f_rest_{i}" for i in range(frest.shape[1] * frest.shape[2])]
    fields += ["opacity"]
    fields += [f"scale_{i}" for i in range(scaling.shape[1])]
    fields += [f"rot_{i}" for i in range(rotation.shape[1])]

    xyz_np = tensor_to_numpy(xyz).astype(np.float32)
    normals = np.zeros_like(xyz_np, dtype=np.float32)
    fdc_np = tensor_to_numpy(fdc.transpose(1, 2).flatten(1)).astype(np.float32)
    frest_np = tensor_to_numpy(frest.transpose(1, 2).flatten(1)).astype(np.float32)
    attrs = np.concatenate(
        [
            xyz_np,
            normals,
            fdc_np,
            frest_np,
            tensor_to_numpy(opacity).astype(np.float32),
            tensor_to_numpy(scaling).astype(np.float32),
            tensor_to_numpy(rotation).astype(np.float32),
        ],
        axis=1,
    )
    if attrs.shape[1] != len(fields):
        raise ValueError(f"Field mismatch: attrs={attrs.shape[1]} fields={len(fields)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    elements = np.empty(xyz_np.shape[0], dtype=[(name, "f4") for name in fields])
    elements[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    model_params, iteration = load_model_params(args.checkpoint)
    write_ply(model_params, args.out)

    print(f"checkpoint: {args.checkpoint}")
    print(f"iteration: {iteration}")
    print(f"active_sh_degree: {model_params[0]}")
    print(f"gaussians: {model_params[1].shape[0]}")
    print(f"spatial_lr_scale: {float(model_params[11]):.9f}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
