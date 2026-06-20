#!/usr/bin/env python3
"""Build a VALA-native 2D level/threshold oracle table.

This is the mechanism precursor for P1-VALA. It does not transfer THGS
superpoint ranks into VALA. Instead, it asks whether VALA's own rendered 2D
feature maps contain a better level/threshold candidate than the native
max-score level selection.

Full runs should be launched through Slurm with ``--device cuda``. The default
device is CPU so this script does not accidentally occupy a GPU outside a
reservation.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VALA_ROOT = ROOT / "external_methods" / "VALA"
DEFAULT_SUMMARY = ROOT / "output/diagnostics/p1e_vala_official_2d_eval_all_84.csv"
DEFAULT_LABEL_ROOT = ROOT / "data/lerf_ovs/label"
DEFAULT_CLASS_TABLE = ROOT / "output/diagnostics/cross_method_d2_decomposition.csv"
DEFAULT_DETAIL = ROOT / "output/diagnostics/p1e_vala_native_2d_oracle_detail.csv"
DEFAULT_AGG = ROOT / "output/diagnostics/p1e_vala_native_2d_oracle_agg.csv"
DEFAULT_RUN_ROOT = VALA_ROOT / "output/official_2d_eval_all_84"


NEGATIVES = ("object", "things", "stuff", "texture")


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    variant: str
    scene: str
    feat_root: Path


def polygon_to_mask(shape: tuple[int, int], points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def read_gt(js_path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    data = json.loads(js_path.read_text())
    h = int(data["info"]["height"])
    w = int(data["info"]["width"])
    masks: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    for obj in data["objects"]:
        prompt = obj["category"]
        mask = polygon_to_mask((h, w), obj["segmentation"])
        if prompt in masks:
            masks[prompt] = np.maximum(masks[prompt], mask)
        else:
            masks[prompt] = mask
        counts[prompt] += 1
    return masks, dict(counts)


def read_classes(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["scene"], row["prompt"])] = {
                "thgs_class": row.get("thgs_class", ""),
                "relags_class": row.get("relags_class", ""),
                "thgs_rank": row.get("thgs_rank", ""),
                "relags_rank": row.get("relags_rank", ""),
            }
    return out


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return iou, precision, recall, float(pred_b.sum()), float(gt_b.sum())


def mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def frame_index(stem: str) -> int:
    return int(stem.split("_")[-1]) - 1


def discover_cases(summary_path: Path, run_root: Path, case_filter: str | None) -> list[EvalCase]:
    seen: set[str] = set()
    cases: list[EvalCase] = []
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            case_id = row["case_id"]
            if case_id in seen:
                continue
            if case_filter and case_filter not in case_id:
                continue
            feat_root = run_root / "feat_dir" / case_id
            if not feat_root.exists():
                print(f"skip missing feat_root: {feat_root}", file=sys.stderr)
                continue
            cases.append(EvalCase(
                case_id=case_id,
                variant=row["variant"],
                scene=row["scene"],
                feat_root=feat_root,
            ))
            seen.add(case_id)
    return cases


def load_text_embeds(prompts: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    sys.path.insert(0, str(VALA_ROOT))
    import open_clip  # type: ignore

    precision = "fp16" if device == "cuda" else "fp32"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        precision=precision,
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    with torch.no_grad():
        pos_tok = torch.cat([tokenizer(p) for p in prompts]).to(device)
        neg_tok = torch.cat([tokenizer(p) for p in NEGATIVES]).to(device)
        pos = model.encode_text(pos_tok).float()
        neg = model.encode_text(neg_tok).float()
    pos = torch.nn.functional.normalize(pos, dim=-1)
    neg = torch.nn.functional.normalize(neg, dim=-1)
    return pos, neg


def relevance_maps(
    feature_path: Path,
    pos_embeds: torch.Tensor,
    neg_embeds: torch.Tensor,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    arr = np.load(feature_path, mmap_mode="r")
    h, w, c = arr.shape
    n_prompts = pos_embeds.shape[0]
    out = np.empty((n_prompts, h, w), dtype=np.float32)
    flat = arr.reshape(-1, c)

    with torch.no_grad():
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            chunk = torch.from_numpy(np.asarray(flat[start:end]).copy()).float().to(device)
            pos_scores = chunk @ pos_embeds.T
            neg_scores = chunk @ neg_embeds.T
            rel = torch.sigmoid(10.0 * (pos_scores[:, :, None] - neg_scores[:, None, :]))
            rel = rel.min(dim=2).values
            out.reshape(n_prompts, -1)[:, start:end] = rel.T.cpu().numpy()
            del chunk, pos_scores, neg_scores, rel
    return out


def smooth_binary(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mask.astype(np.uint8)
    if mode == "median7":
        return cv2.medianBlur(mask.astype(np.uint8), 7)
    raise ValueError(f"unknown smooth mode: {mode}")


def relevance_to_mask(relev: np.ndarray, threshold: float, smooth_mode: str) -> np.ndarray:
    scale = 30
    kernel = np.ones((scale, scale), dtype=np.float32) / float(scale * scale)
    filtered = cv2.filter2D(relev.astype(np.float32), -1, kernel)
    output = 0.5 * (filtered + relev.astype(np.float32))
    output = output - float(output.min())
    output = output / (float(output.max()) + 1e-9)
    output = output * 2.0 - 1.0
    output = np.clip(output, 0.0, 1.0)
    mask = (output > threshold).astype(np.uint8)
    return smooth_binary(mask, smooth_mode)


def vala_class(actual: float, oracle_level: float, oracle_level_thresh: float) -> str:
    if actual >= 0.5:
        return "vala_actual_ok"
    if oracle_level >= 0.5:
        return "vala_selection_fail"
    if oracle_level_thresh >= 0.5:
        return "vala_calibration_fail"
    return "vala_representation_fail"


def process_case(
    case: EvalCase,
    label_root: Path,
    classes: dict[tuple[str, str], dict[str, str]],
    native_threshold: float,
    oracle_thresholds: list[float],
    device: str,
    chunk_size: int,
    smooth_mode: str,
    max_frames: int | None,
    max_prompts: int | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    scene_label = label_root / case.scene
    js_paths = sorted(scene_label.glob("frame_*.json"))
    if max_frames is not None:
        js_paths = js_paths[:max_frames]

    all_prompts = sorted({
        prompt
        for js_path in js_paths
        for prompt in read_gt(js_path)[0].keys()
    })
    if max_prompts is not None:
        all_prompts = all_prompts[:max_prompts]
    prompt_to_idx = {prompt: idx for idx, prompt in enumerate(all_prompts)}
    pos_embeds, neg_embeds = load_text_embeds(all_prompts, device)

    for js_path in js_paths:
        frame = js_path.stem
        idx = frame_index(frame)
        gt_masks, n_instances = read_gt(js_path)
        prompts = [prompt for prompt in sorted(gt_masks) if prompt in prompt_to_idx]
        prompt_indices = [prompt_to_idx[prompt] for prompt in prompts]

        level_native_iou: dict[int, list[float]] = {}
        level_native_precision: dict[int, list[float]] = {}
        level_native_recall: dict[int, list[float]] = {}
        level_native_area: dict[int, list[float]] = {}
        level_oracle_iou: dict[int, list[float]] = {}
        level_score: dict[int, list[float]] = {}

        for level in (1, 2, 3):
            feature_path = case.feat_root / f"{case.scene}_{level}" / "train/ours_None/renders_npy" / f"{idx}.npy"
            if not feature_path.exists():
                raise FileNotFoundError(feature_path)
            rel_maps = relevance_maps(feature_path, pos_embeds, neg_embeds, device, chunk_size)

            native_ious: list[float] = []
            native_precs: list[float] = []
            native_recs: list[float] = []
            native_areas: list[float] = []
            oracle_ious: list[float] = []
            scores: list[float] = []
            for p_idx, prompt in enumerate(prompts):
                relev = rel_maps[prompt_indices[p_idx]]
                gt = gt_masks[prompt]
                scores.append(float(relev.max()))

                mask = relevance_to_mask(relev, native_threshold, smooth_mode)
                iou, prec, rec, pred_area, _ = binary_iou(mask, gt)
                native_ious.append(iou)
                native_precs.append(prec)
                native_recs.append(rec)
                native_areas.append(pred_area)

                thresh_ious: list[float] = []
                for th in oracle_thresholds:
                    th_mask = relevance_to_mask(relev, th, smooth_mode)
                    th_iou, _, _, _, _ = binary_iou(th_mask, gt)
                    thresh_ious.append(th_iou)
                oracle_ious.append(max(thresh_ious))

            level_native_iou[level] = native_ious
            level_native_precision[level] = native_precs
            level_native_recall[level] = native_recs
            level_native_area[level] = native_areas
            level_oracle_iou[level] = oracle_ious
            level_score[level] = scores
            del rel_maps
            if device == "cuda":
                torch.cuda.empty_cache()

        for p_idx, prompt in enumerate(prompts):
            levels = (1, 2, 3)
            chosen_level = max(levels, key=lambda lvl: level_score[lvl][p_idx])
            oracle_level = max(levels, key=lambda lvl: level_native_iou[lvl][p_idx])
            oracle_thresh_level = max(levels, key=lambda lvl: level_oracle_iou[lvl][p_idx])

            actual = level_native_iou[chosen_level][p_idx]
            oracle_level_iou = level_native_iou[oracle_level][p_idx]
            oracle_level_thresh_iou = level_oracle_iou[oracle_thresh_level][p_idx]
            cls = classes.get((case.scene, prompt), {})
            _, _, _, _, gt_area = binary_iou(np.zeros_like(gt_masks[prompt]), gt_masks[prompt])
            rows.append({
                "case_id": case.case_id,
                "variant": case.variant,
                "scene": case.scene,
                "frame": frame,
                "prompt": prompt,
                "n_instances": str(n_instances[prompt]),
                "native_threshold": f"{native_threshold:.4f}",
                "oracle_thresholds": " ".join(f"{t:.4f}" for t in oracle_thresholds),
                "smooth_mode": smooth_mode,
                "chosen_level": str(chosen_level),
                "oracle_level": str(oracle_level),
                "oracle_thresh_level": str(oracle_thresh_level),
                "chosen_score": f"{level_score[chosen_level][p_idx]:.6f}",
                "actual_iou": f"{actual:.6f}",
                "actual_precision": f"{level_native_precision[chosen_level][p_idx]:.6f}",
                "actual_recall": f"{level_native_recall[chosen_level][p_idx]:.6f}",
                "actual_pred_area": f"{level_native_area[chosen_level][p_idx]:.6f}",
                "gt_area": f"{gt_area:.6f}",
                "oracle_level_iou": f"{oracle_level_iou:.6f}",
                "oracle_level_thresh_iou": f"{oracle_level_thresh_iou:.6f}",
                "level_selection_gap": f"{(oracle_level_iou - actual):.6f}",
                "threshold_gap": f"{(oracle_level_thresh_iou - oracle_level_iou):.6f}",
                "vala_class": vala_class(actual, oracle_level_iou, oracle_level_thresh_iou),
                "thgs_class": cls.get("thgs_class", ""),
                "relags_class": cls.get("relags_class", ""),
                "thgs_rank": cls.get("thgs_rank", ""),
                "relags_rank": cls.get("relags_rank", ""),
            })
    return rows


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for scope, value in (
            ("overall", "all"),
            ("vala_class", row["vala_class"]),
            ("thgs_class", row["thgs_class"]),
            ("relags_class", row["relags_class"]),
        ):
            if value:
                groups[(row["case_id"], row["scene"], scope, value)].append(row)

    for (case_id, scene, scope, value), group in sorted(groups.items()):
        actual = [float(r["actual_iou"]) for r in group]
        oracle_level = [float(r["oracle_level_iou"]) for r in group]
        oracle_thresh = [float(r["oracle_level_thresh_iou"]) for r in group]
        out.append({
            "case_id": case_id,
            "scene": scene,
            "scope": scope,
            "scope_value": value,
            "n_rows": str(len(group)),
            "n_prompts": str(len({r["prompt"] for r in group})),
            "n_frames": str(len({r["frame"] for r in group})),
            "actual_iou": f"{mean(actual):.6f}",
            "oracle_level_iou": f"{mean(oracle_level):.6f}",
            "oracle_level_thresh_iou": f"{mean(oracle_thresh):.6f}",
            "level_selection_gap": f"{mean([b - a for a, b in zip(actual, oracle_level)]):.6f}",
            "threshold_gap": f"{mean([c - b for b, c in zip(oracle_level, oracle_thresh)]):.6f}",
        })
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def parse_thresholds(spec: str) -> list[float]:
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        vals = []
        x = start
        while x <= stop + 1e-9:
            vals.append(round(x, 6))
            x += step
        return vals
    return [float(x) for x in spec.split(",") if x]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--label_root", type=Path, default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--class_table", type=Path, default=DEFAULT_CLASS_TABLE)
    ap.add_argument("--detail_csv", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--agg_csv", type=Path, default=DEFAULT_AGG)
    ap.add_argument("--case_filter", default="refersplat_3dgs_valafeat_full")
    ap.add_argument("--native_threshold", type=float, default=0.5)
    ap.add_argument("--oracle_thresholds", default="0.1:0.9:0.05")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--chunk_size", type=int, default=32768)
    ap.add_argument("--smooth_mode", default="median7", choices=("median7", "none"))
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--max_prompts", type=int, default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    cases = discover_cases(args.summary, args.run_root, args.case_filter)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    if not cases:
        raise RuntimeError("no cases matched")

    thresholds = parse_thresholds(args.oracle_thresholds)
    classes = read_classes(args.class_table)
    rows: list[dict[str, str]] = []
    for case in cases:
        print(f"case {case.case_id} scene={case.scene} feat_root={case.feat_root}")
        rows.extend(process_case(
            case=case,
            label_root=args.label_root,
            classes=classes,
            native_threshold=args.native_threshold,
            oracle_thresholds=thresholds,
            device=args.device,
            chunk_size=args.chunk_size,
            smooth_mode=args.smooth_mode,
            max_frames=args.max_frames,
            max_prompts=args.max_prompts,
        ))

    detail_fields = [
        "case_id", "variant", "scene", "frame", "prompt", "n_instances",
        "native_threshold", "oracle_thresholds", "smooth_mode",
        "chosen_level", "oracle_level", "oracle_thresh_level", "chosen_score",
        "actual_iou", "actual_precision", "actual_recall", "actual_pred_area",
        "gt_area", "oracle_level_iou", "oracle_level_thresh_iou",
        "level_selection_gap", "threshold_gap", "vala_class",
        "thgs_class", "relags_class", "thgs_rank", "relags_rank",
    ]
    write_csv(args.detail_csv, rows, detail_fields)

    agg = aggregate(rows)
    agg_fields = [
        "case_id", "scene", "scope", "scope_value", "n_rows", "n_prompts",
        "n_frames", "actual_iou", "oracle_level_iou",
        "oracle_level_thresh_iou", "level_selection_gap", "threshold_gap",
    ]
    write_csv(args.agg_csv, agg, agg_fields)


if __name__ == "__main__":
    main()
