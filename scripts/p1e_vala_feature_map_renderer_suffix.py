#!/usr/bin/env python3
"""Render VALA 2D feature maps from a selectable language checkpoint suffix.

The public VALA renderer hardcodes ``_stochastic_gate`` in both checkpoint and
output directory names. RF-V2 needs the same RGB/source/eval pipeline with only
the language aggregation checkpoint changed, so this script keeps the renderer
logic but exposes:

  - ``--checkpoint_suffix``: e.g. ``_stochastic_gate`` or empty string.
  - ``--output_tag``: e.g. ``robust`` or ``mean``.

Only ``renders_npy`` are written; diagnostic PCA images are intentionally
omitted because the official 2D evaluator consumes the npy feature maps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
VALA_ROOT = ROOT / "external_methods" / "VALA"
sys.path.insert(0, str(VALA_ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # type: ignore  # noqa: E402
from gaussian_renderer import GaussianModel, render  # type: ignore  # noqa: E402
from scene import Scene  # type: ignore  # noqa: E402
from utils.general_utils import safe_state  # type: ignore  # noqa: E402


def render_set(
    model_path: str,
    split_name: str,
    iteration: int,
    source_path: str,
    views,
    gaussians,
    pipeline,
    background: torch.Tensor,
    feature_level: int,
    output_tag: str,
) -> None:
    save_path = os.path.join(
        model_path,
        split_name,
        f"ours_{iteration}_langfeat_{feature_level}_{output_tag}",
    )
    render_npy_path = os.path.join(save_path, "renders_npy")
    os.makedirs(render_npy_path, exist_ok=True)

    for view in tqdm(views, desc=f"Rendering {split_name} level={feature_level} tag={output_tag}"):
        render_pkg = render(view, gaussians, pipeline, background, include_feature=True)
        rendering = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        stem = view.image_name.split(".")[0]
        np.save(os.path.join(render_npy_path, f"{stem}.npy"), rendering)


def render_sets(
    dataset: ModelParams,
    opt: OptimizationParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    feature_level: int,
    ablation_type: str,
    checkpoint_suffix: str,
    output_tag: str,
) -> None:
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, include_feature=True)

        model_path = os.path.join(dataset.model_path, ablation_type)
        checkpoint = os.path.join(model_path, f"chkpnt{iteration}_langfeat_{feature_level}{checkpoint_suffix}.pth")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        model_params, _ = torch.load(checkpoint)
        gaussians.restore_language_features(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, dataset.source_path, scene.getTrainCameras(), gaussians, pipeline, background, feature_level, output_tag)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, dataset.source_path, scene.getTestCameras(), gaussians, pipeline, background, feature_level, output_tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-V2 suffix-aware VALA feature-map renderer")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ablation_type", type=str, default="none")
    parser.add_argument("--checkpoint_suffix", type=str, default="_stochastic_gate")
    parser.add_argument("--output_tag", type=str, default="suffix")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    render_sets(
        model.extract(args),
        opt.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.feature_level,
        args.ablation_type,
        args.checkpoint_suffix,
        args.output_tag,
    )


if __name__ == "__main__":
    main()
