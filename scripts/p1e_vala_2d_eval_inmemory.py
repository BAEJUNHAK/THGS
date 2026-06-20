#!/usr/bin/env python3
"""Run VALA's LERF-OVS 2D evaluation without dumping 512D feature maps."""

from __future__ import annotations

import csv
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VALA_ROOT = REPO_ROOT / "external_methods" / "VALA"
sys.path.insert(0, str(VALA_ROOT))


from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: E402
from eval.openclip_encoder import OpenCLIPNetwork  # noqa: E402
from eval.utils import polygon_to_mask, smooth, stack_mask  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def load_lerf_gt(json_dir: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    frame_to_ann: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for js_path in sorted(json_dir.glob("frame_*.json")):
        with js_path.open("r") as f:
            gt_data = json.load(f)
        h, w = gt_data["info"]["height"], gt_data["info"]["width"]
        frame_name = Path(gt_data["info"]["name"]).stem
        img_ann: dict[str, dict[str, np.ndarray]] = {}
        for obj in gt_data["objects"]:
            label = obj["category"]
            box = np.asarray(obj["bbox"]).reshape(-1)
            mask = polygon_to_mask((h, w), obj["segmentation"])
            if label in img_ann:
                mask = stack_mask(img_ann[label]["mask"], mask)
                img_ann[label]["bboxes"] = np.concatenate(
                    [img_ann[label]["bboxes"].reshape(-1, 4), box.reshape(-1, 4)],
                    axis=0,
                )
            else:
                img_ann[label] = {"bboxes": box}
            img_ann[label]["mask"] = mask
        frame_to_ann[frame_name] = img_ann
    return frame_to_ann


def normalize_and_threshold(relevancy: np.ndarray, thresh: float, use_smooth: bool) -> np.ndarray:
    output = relevancy - np.min(relevancy)
    output = output / (np.max(output) + 1e-9)
    output = output * 2.0 - 1.0
    output = np.clip(output, 0.0, 1.0)
    mask_pred = (output > thresh).astype(np.uint8)
    if use_smooth:
        mask_pred = smooth(mask_pred)
    return mask_pred


def eval_frame(
    sem_map: torch.Tensor,
    img_ann: dict[str, dict[str, np.ndarray]],
    clip_model: OpenCLIPNetwork,
    thresholds: list[float],
    use_smooth: bool,
) -> list[dict[str, object]]:
    clip_model.set_positives(list(img_ann.keys()))
    valid_map = clip_model.get_max_across(sem_map)
    n_levels, n_prompt, _, _ = valid_map.shape
    kernel = np.ones((30, 30), dtype=np.float32) / float(30 * 30)
    rows: list[dict[str, object]] = []

    for prompt_idx in range(n_prompt):
        level_relevancies = []
        level_scores = []
        for level_idx in range(n_levels):
            relevancy = valid_map[level_idx, prompt_idx].float().detach().cpu().numpy()
            filtered = cv2.filter2D(relevancy, -1, kernel)
            mixed = 0.5 * (filtered + relevancy)
            level_relevancies.append(mixed)
            level_scores.append(float(mixed.max()))

        chosen_level = int(np.argmax(np.asarray(level_scores)))
        prompt = clip_model.positives[prompt_idx]
        mask_gt = img_ann[prompt]["mask"].astype(np.uint8)
        for thresh in thresholds:
            mask_pred = normalize_and_threshold(
                level_relevancies[chosen_level],
                thresh,
                use_smooth,
            )
            inter = np.logical_and(mask_gt, mask_pred).sum()
            union = np.logical_or(mask_gt, mask_pred).sum()
            iou = float(inter / max(union, 1))
            pred_area = int(mask_pred.sum())
            gt_area = int(mask_gt.sum())
            rows.append(
                {
                    "prompt": prompt,
                    "threshold": thresh,
                    "iou": iou,
                    "chosen_level": chosen_level + 1,
                    "gt_area": gt_area,
                    "pred_area": pred_area,
                    "area_ratio": float(pred_area / max(gt_area, 1)),
                    "level_scores": ";".join(f"{x:.6f}" for x in level_scores),
                }
            )
    return rows


def load_level_scene(args, dataset, opt, pipeline, level: int):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, include_feature=True)
    checkpoint = Path(args.model_path) / args.ablation_type / (
        f"chkpnt{args.iteration}_langfeat_{level}_stochastic_gate.pth"
    )
    model_params, _ = torch.load(str(checkpoint))
    gaussians.restore_language_features(model_params, opt)
    return scene, gaussians


def main() -> None:
    os.chdir(VALA_ROOT)
    parser = ArgumentParser(description="In-memory VALA LERF-OVS 2D evaluator")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ablation_type", type=str, default="none")
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--json_dir", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5])
    parser.add_argument("--no_smooth", action="store_true")
    args = get_combined_args(parser)
    missing = [k for k in ("scene_name", "json_dir", "out_csv", "out_json") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join('--' + k for k in missing)}")
    safe_state(args.quiet)

    dataset = model.extract(args)
    opt_params = opt.extract(args)
    pipe = pipeline.extract(args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[load] scene={args.scene_name} model={args.model_path}", flush=True)
    level_scenes = []
    level_gaussians = []
    for level in (1, 2, 3):
        scene, gaussians = load_level_scene(args, dataset, opt_params, pipe, level)
        level_scenes.append(scene)
        level_gaussians.append(gaussians)

    frame_to_ann = load_lerf_gt(Path(args.json_dir) / args.scene_name)
    cameras = {cam.image_name: cam for cam in level_scenes[0].getTestCameras()}
    clip_model = OpenCLIPNetwork(torch.device("cuda"))
    rows = []

    with torch.no_grad():
        for frame_name, img_ann in sorted(frame_to_ann.items()):
            if frame_name not in cameras:
                print(f"[warn] no test camera for {frame_name}; skip", flush=True)
                continue
            sem_maps = []
            for gaussians in level_gaussians:
                pkg = render(cameras[frame_name], gaussians, pipe, background, include_feature=True)
                sem_maps.append(pkg["render"].permute(1, 2, 0).detach())
            sem_map = torch.stack(sem_maps, dim=0)
            frame_rows = eval_frame(
                sem_map,
                img_ann,
                clip_model,
                thresholds=args.thresholds,
                use_smooth=not args.no_smooth,
            )
            for row in frame_rows:
                row["scene"] = args.scene_name
                row["frame"] = frame_name
            rows.extend(frame_rows)
            print(f"[frame] {frame_name}: {len(frame_rows)} rows", flush=True)
            del sem_map, sem_maps
            torch.cuda.empty_cache()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "frame",
        "prompt",
        "threshold",
        "iou",
        "chosen_level",
        "gt_area",
        "pred_area",
        "area_ratio",
        "level_scores",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for thresh in args.thresholds:
        vals = [float(row["iou"]) for row in rows if abs(float(row["threshold"]) - thresh) < 1e-9]
        summary[str(thresh)] = {
            "mIoU": float(np.mean(vals)) if vals else 0.0,
            "n": len(vals),
        }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
