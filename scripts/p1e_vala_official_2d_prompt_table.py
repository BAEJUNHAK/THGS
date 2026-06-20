#!/usr/bin/env python3
"""Build prompt-level tables from VALA official 2D evaluation outputs.

The official VALA evaluator reports only scene-level means in its log. For P1
we need the per-(frame,prompt) masks to split results by THGS/ReLaGS regimes.
This script reuses the official evaluator's saved `chosen_*.png` masks and
scores them with the LERF-OVS JSON polygons.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_SUMMARY = Path("output/diagnostics/p1e_vala_official_2d_eval_all_84.csv")
DEFAULT_LABEL_ROOT = Path("data/lerf_ovs/label")
DEFAULT_CLASS_TABLE = Path("output/diagnostics/cross_method_d2_decomposition.csv")
DEFAULT_DETAIL = Path("output/diagnostics/p1e_vala_official_2d_prompt_detail.csv")
DEFAULT_AGG = Path("output/diagnostics/p1e_vala_official_2d_prompt_agg.csv")


def polygon_to_mask(shape: tuple[int, int], points: Iterable[Iterable[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {
        "inter": tp,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pred_area": float(pred.sum()),
        "gt_area": float(gt.sum()),
    }


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


def gt_mask_from_json(js_path: Path, prompt: str) -> tuple[np.ndarray, int]:
    data = json.loads(js_path.read_text())
    h = int(data["info"]["height"])
    w = int(data["info"]["width"])
    gt = np.zeros((h, w), dtype=np.uint8)
    n_instances = 0
    for obj in data["objects"]:
        if obj["category"] != prompt:
            continue
        gt = np.maximum(gt, polygon_to_mask((h, w), obj["segmentation"]))
        n_instances += 1
    return gt, n_instances


def iter_prompts(js_path: Path) -> list[str]:
    data = json.loads(js_path.read_text())
    return sorted({obj["category"] for obj in data["objects"]})


def frame_dir_name(stem: str) -> str:
    return f"{int(stem.split('_')[-1]):05d}"


def build_detail(summary: Path, label_root: Path, class_table: Path) -> list[dict[str, str]]:
    classes = read_classes(class_table)
    rows: list[dict[str, str]] = []
    with summary.open() as f:
        for case in csv.DictReader(f):
            scene = case["scene"]
            eval_scene_dir = Path(case["log"]).parent
            scene_label = label_root / scene
            for js_path in sorted(scene_label.glob("frame_*.json")):
                frame = js_path.stem
                prompt_list = iter_prompts(js_path)
                pred_dir = eval_scene_dir / frame_dir_name(frame)
                for prompt in prompt_list:
                    gt, n_instances = gt_mask_from_json(js_path, prompt)
                    pred_path = pred_dir / f"chosen_{prompt}.png"
                    missing = not pred_path.exists()
                    if missing:
                        pred = np.zeros_like(gt, dtype=bool)
                    else:
                        pred_img = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                        if pred_img is None:
                            pred = np.zeros_like(gt, dtype=bool)
                            missing = True
                        else:
                            if pred_img.shape != gt.shape:
                                pred_img = cv2.resize(pred_img, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                            pred = pred_img > 127
                    met = binary_metrics(pred, gt > 0)
                    cls = classes.get((scene, prompt), {})
                    rows.append({
                        "case_id": case["case_id"],
                        "variant": case["variant"],
                        "scene": scene,
                        "threshold": case["threshold"],
                        "frame": frame,
                        "prompt": prompt,
                        "n_instances": str(n_instances),
                        "thgs_class": cls.get("thgs_class", ""),
                        "relags_class": cls.get("relags_class", ""),
                        "thgs_rank": cls.get("thgs_rank", ""),
                        "relags_rank": cls.get("relags_rank", ""),
                        "missing_pred": str(int(missing)),
                        "pred_path": str(pred_path),
                        **{k: f"{v:.6f}" for k, v in met.items()},
                    })
    return rows


def mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def add_agg(out: list[dict[str, str]], rows: list[dict[str, str]], scope: str, scope_value: str) -> None:
    if not rows:
        return
    ious = [float(r["iou"]) for r in rows]
    precs = [float(r["precision"]) for r in rows]
    recs = [float(r["recall"]) for r in rows]

    prompt_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    frame_groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        prompt_groups[(r["scene"], r["prompt"])].append(float(r["iou"]))
        frame_groups[r["frame"]].append(float(r["iou"]))

    common = {
        "case_id": rows[0]["case_id"],
        "variant": rows[0]["variant"],
        "scene": rows[0]["scene"],
        "threshold": rows[0]["threshold"],
        "scope": scope,
        "scope_value": scope_value,
        "n_rows": str(len(rows)),
        "n_prompts": str(len(prompt_groups)),
        "n_frames": str(len(frame_groups)),
    }
    out.append({
        **common,
        "agg": "flat",
        "mean_iou": f"{mean(ious):.6f}",
        "mean_precision": f"{mean(precs):.6f}",
        "mean_recall": f"{mean(recs):.6f}",
    })
    out.append({
        **common,
        "agg": "per_prompt",
        "mean_iou": f"{mean([mean(v) for v in prompt_groups.values()]):.6f}",
        "mean_precision": "",
        "mean_recall": "",
    })
    out.append({
        **common,
        "agg": "per_image",
        "mean_iou": f"{mean([mean(v) for v in frame_groups.values()]):.6f}",
        "mean_precision": "",
        "mean_recall": "",
    })


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    case_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        case_groups[(row["case_id"], row["scene"], row["threshold"])].append(row)
    for _, group in sorted(case_groups.items()):
        add_agg(out, group, "overall", "all")
        for cls_name in ("thgs_class", "relags_class"):
            vals = sorted({r[cls_name] for r in group if r[cls_name]})
            for val in vals:
                add_agg(out, [r for r in group if r[cls_name] == val], cls_name, val)
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--label_root", type=Path, default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--class_table", type=Path, default=DEFAULT_CLASS_TABLE)
    ap.add_argument("--detail_csv", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--agg_csv", type=Path, default=DEFAULT_AGG)
    args = ap.parse_args()

    detail = build_detail(args.summary, args.label_root, args.class_table)
    detail_fields = [
        "case_id", "variant", "scene", "threshold", "frame", "prompt",
        "n_instances", "thgs_class", "relags_class", "thgs_rank", "relags_rank",
        "missing_pred", "pred_path", "inter", "iou", "precision", "recall",
        "pred_area", "gt_area",
    ]
    write_csv(args.detail_csv, detail, detail_fields)

    agg = aggregate(detail)
    agg_fields = [
        "case_id", "variant", "scene", "threshold", "scope", "scope_value",
        "agg", "n_rows", "n_prompts", "n_frames", "mean_iou",
        "mean_precision", "mean_recall",
    ]
    write_csv(args.agg_csv, agg, agg_fields)


if __name__ == "__main__":
    main()
