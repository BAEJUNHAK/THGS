"""VALA source diagnostics for LERF-OVS language feature maps.

This is a CPU-only check: for each GT object mask, ask whether the existing
LangSplat/THGS language_features segmentation map contains any 2D mask that
overlaps the GT object. It separates 2D source-mask failure from later 3DGS
lifting / VALA relevance-thresholding failure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


LEVEL_NAMES = ["default", "s", "m", "l"]


def polygon_to_mask(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def union_gt_mask(json_path: Path, prompt: str) -> tuple[np.ndarray, int]:
    data = json.loads(json_path.read_text())
    h = int(data["info"]["height"])
    w = int(data["info"]["width"])
    mask = np.zeros((h, w), dtype=bool)
    n_inst = 0
    for obj in data["objects"]:
        if obj["category"] != prompt:
            continue
        mask |= polygon_to_mask((h, w), obj["segmentation"])
        n_inst += 1
    return mask, n_inst


def prompts_in_frame(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text())
    return sorted({obj["category"] for obj in data["objects"]})


def best_seg_iou(seg_level: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int]:
    ids = np.unique(seg_level[gt])
    ids = ids[ids >= 0]
    best_iou = 0.0
    best_recall = 0.0
    best_id = -1
    best_area = 0
    gt_area = int(gt.sum())
    if gt_area == 0:
        return 0.0, 0.0, -1, 0
    for mid in ids:
        pred = seg_level == mid
        inter = int((pred & gt).sum())
        if inter == 0:
            continue
        union = int((pred | gt).sum())
        iou = inter / max(union, 1)
        recall = inter / gt_area
        if iou > best_iou:
            best_iou = float(iou)
            best_recall = float(recall)
            best_id = int(mid)
            best_area = int(pred.sum())
    return best_iou, best_recall, best_id, best_area


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for scene in args.scenes:
        label_dir = data_root / "label" / scene
        feat_dir = data_root / scene / args.feature_dir
        for json_path in sorted(label_dir.glob("frame_*.json")):
            frame = json_path.stem
            seg_path = feat_dir / f"{frame}_s.npy"
            if not seg_path.exists():
                continue
            seg = np.load(seg_path)
            for prompt in prompts_in_frame(json_path):
                gt, n_inst = union_gt_mask(json_path, prompt)
                gt_area = int(gt.sum())
                best_all = (0.0, 0.0, "", -1, 0)
                per_level: dict[str, tuple[float, float, int, int]] = {}
                for li, lname in enumerate(LEVEL_NAMES):
                    iou, rec, mid, area = best_seg_iou(seg[li], gt)
                    per_level[lname] = (iou, rec, mid, area)
                    if iou > best_all[0]:
                        best_all = (iou, rec, lname, mid, area)
                row: dict[str, object] = {
                    "scene": scene,
                    "frame": frame,
                    "prompt": prompt,
                    "gt_area": gt_area,
                    "n_instances": n_inst,
                    "best_iou": f"{best_all[0]:.6f}",
                    "best_recall": f"{best_all[1]:.6f}",
                    "best_level": best_all[2],
                    "best_mask_id": best_all[3],
                    "best_mask_area": best_all[4],
                }
                for lname, vals in per_level.items():
                    row[f"{lname}_iou"] = f"{vals[0]:.6f}"
                    row[f"{lname}_recall"] = f"{vals[1]:.6f}"
                rows.append(row)

    fieldnames = [
        "scene",
        "frame",
        "prompt",
        "gt_area",
        "n_instances",
        "best_iou",
        "best_recall",
        "best_level",
        "best_mask_id",
        "best_mask_area",
    ]
    for lname in LEVEL_NAMES:
        fieldnames += [f"{lname}_iou", f"{lname}_recall"]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_scene: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene"]), []).append(row)
    for scene, scene_rows in sorted(by_scene.items()):
        vals = np.asarray([float(r["best_iou"]) for r in scene_rows], dtype=np.float32)
        recs = np.asarray([float(r["best_recall"]) for r in scene_rows], dtype=np.float32)
        print(
            f"{scene}: n={len(scene_rows)} best2d_iou={vals.mean():.4f} "
            f"best2d_recall={recs.mean():.4f} "
            f"iou>=0.5={(vals >= 0.5).mean():.3f} iou>=0.25={(vals >= 0.25).mean():.3f}"
        )
    print(f"wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/lerf_ovs")
    parser.add_argument("--feature_dir", default="language_features")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["ramen", "teatime"],
    )
    parser.add_argument(
        "--out_csv",
        default="output/diagnostics/p1e_vala_2d_source_oracle.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
