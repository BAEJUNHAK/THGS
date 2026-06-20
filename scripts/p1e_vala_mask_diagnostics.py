"""Per-frame VALA mask diagnostics.

Joins VALA rendered silhouettes with GT masks and optional 2D source oracle
statistics. This is CPU-only and is meant to localize whether a low VALA IoU is
caused by missing 2D source masks or by later 3D lifting / thresholding.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def polygon_to_mask(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def frame_gt_masks(json_path: Path) -> dict[str, tuple[np.ndarray, int]]:
    data = json.loads(json_path.read_text())
    h = int(data["info"]["height"])
    w = int(data["info"]["width"])
    out: dict[str, tuple[np.ndarray, int]] = {}
    for obj in data["objects"]:
        prompt = obj["category"]
        mask, n = out.get(prompt, (np.zeros((h, w), dtype=bool), 0))
        mask |= polygon_to_mask((h, w), obj["segmentation"])
        out[prompt] = (mask, n + 1)
    return out


def load_pred(path: Path, shape: tuple[int, int], threshold: int) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > threshold


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    oracle = {}
    if args.source_oracle_csv:
        with Path(args.source_oracle_csv).open() as f:
            for row in csv.DictReader(f):
                oracle[(row["scene"], row["frame"], row["prompt"])] = row

    rows = []
    for scene in args.scenes:
        label_dir = data_root / "label" / scene
        pred_dir = pred_root / scene / args.ablation / f"predictions_mask_{args.mask_thresh}" / "renders_silhouette"
        for json_path in sorted(label_dir.glob("frame_*.json")):
            frame = json_path.stem
            for prompt, (gt, n_inst) in frame_gt_masks(json_path).items():
                pred = load_pred(pred_dir / frame / f"{prompt}.png", gt.shape, args.pixel_threshold)
                inter = int((gt & pred).sum())
                gt_area = int(gt.sum())
                pred_area = int(pred.sum())
                union = int((gt | pred).sum())
                iou = inter / max(union, 1)
                precision = inter / max(pred_area, 1)
                recall = inter / max(gt_area, 1)
                area_ratio = pred_area / max(gt_area, 1)
                source = oracle.get((scene, frame, prompt), {})
                rows.append(
                    {
                        "scene": scene,
                        "frame": frame,
                        "prompt": prompt,
                        "n_instances": n_inst,
                        "gt_area": gt_area,
                        "pred_area": pred_area,
                        "area_ratio": f"{area_ratio:.6f}",
                        "inter": inter,
                        "iou": f"{iou:.6f}",
                        "precision": f"{precision:.6f}",
                        "recall": f"{recall:.6f}",
                        "f1": f"{f1(precision, recall):.6f}",
                        "source_best_iou": source.get("best_iou", ""),
                        "source_best_recall": source.get("best_recall", ""),
                        "source_best_level": source.get("best_level", ""),
                    }
                )

    fieldnames = [
        "scene",
        "frame",
        "prompt",
        "n_instances",
        "gt_area",
        "pred_area",
        "area_ratio",
        "inter",
        "iou",
        "precision",
        "recall",
        "f1",
        "source_best_iou",
        "source_best_recall",
        "source_best_level",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for scene in args.scenes:
        scene_rows = [r for r in rows if r["scene"] == scene]
        ious = np.asarray([float(r["iou"]) for r in scene_rows], dtype=np.float32)
        precs = np.asarray([float(r["precision"]) for r in scene_rows], dtype=np.float32)
        recs = np.asarray([float(r["recall"]) for r in scene_rows], dtype=np.float32)
        ratios = np.asarray([float(r["area_ratio"]) for r in scene_rows], dtype=np.float32)
        src_vals = [
            float(r["source_best_iou"])
            for r in scene_rows
            if str(r["source_best_iou"]) != ""
        ]
        src = np.asarray(src_vals, dtype=np.float32)
        print(
            f"{scene}: n={len(scene_rows)} iou={ious.mean():.4f} "
            f"precision={precs.mean():.4f} recall={recs.mean():.4f} "
            f"area_ratio_median={np.median(ratios):.3f} "
            f"source_iou={src.mean():.4f}"
        )
        worst = sorted(scene_rows, key=lambda r: float(r["iou"]))[:8]
        for r in worst:
            print(
                f"  low {r['prompt']:<16s} {r['frame']} "
                f"iou={float(r['iou']):.3f} p={float(r['precision']):.3f} "
                f"r={float(r['recall']):.3f} area={float(r['area_ratio']):.2f} "
                f"src={r['source_best_iou']}"
            )
    print(f"wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/lerf_ovs")
    parser.add_argument("--pred_root", default="external_methods/VALA/output/3dgs/lerf_ovs")
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--mask_thresh", default="0.6")
    parser.add_argument("--pixel_threshold", type=int, default=10)
    parser.add_argument("--scenes", nargs="+", default=["ramen", "teatime"])
    parser.add_argument(
        "--source_oracle_csv",
        default="output/diagnostics/p1e_vala_2d_source_oracle.csv",
    )
    parser.add_argument(
        "--out_csv",
        default="output/diagnostics/p1e_vala_mask_diagnostics.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
