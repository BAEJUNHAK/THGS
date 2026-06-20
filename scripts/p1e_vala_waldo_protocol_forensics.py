#!/usr/bin/env python3
"""Forensic comparison of plausible VALA Waldo 2D evaluation protocols.

This script separates fixed public outputs from tuned/adaptive/oracle-like
diagnostics. It does not claim that the paper used any oracle; it only reports
which protocol family can numerically explain the reported Waldo 2D result.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from p1e_vala_native_2d_oracle import (
    DEFAULT_LABEL_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SUMMARY,
    EvalCase,
    binary_iou,
    discover_cases,
    frame_index,
    load_text_embeds,
    read_gt,
    relevance_maps,
    relevance_to_mask,
)


ROOT = Path(__file__).resolve().parents[1]
VALA_ROOT = ROOT / "external_methods" / "VALA"
DEFAULT_OFFICIAL_DETAIL = ROOT / "output/diagnostics/p1e_vala_official_2d_prompt_detail.csv"
DEFAULT_P1_DETAIL = ROOT / "output/diagnostics/p1e_vala_native_2d_oracle_official_actual_detail.csv"
DEFAULT_OUT = ROOT / "output/diagnostics/p1e_vala_waldo_protocol_forensics.csv"
PAPER_WALDO_2D = 0.651


def load_compute_dynamic_threshold():
    path = VALA_ROOT / "eval/eval_utils.py"
    spec = importlib.util.spec_from_file_location("vala_eval_utils_forensic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_dynamic_threshold


def mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def fmt(x: float) -> str:
    return f"{x:.6f}" if np.isfinite(x) else "nan"


def read_official_actuals(path: Path) -> dict[tuple[str, float], list[float]]:
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["scene"] != "waldo_kitchen":
                continue
            groups[(row["case_id"], float(row["threshold"]))].append(float(row["iou"]))
    return groups


def read_existing_p1_oracles(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["scene"] == "waldo_kitchen":
                groups[row["case_id"]].append(row)

    out: dict[str, dict[str, float]] = {}
    for case_id, rows in groups.items():
        out[case_id] = {
            "official_actual_0.5": mean([float(r["actual_iou"]) for r in rows]),
            "level_oracle_0.5": mean([float(r["oracle_level_iou"]) for r in rows]),
            "per_row_threshold_oracle": mean([float(r["oracle_level_thresh_iou"]) for r in rows]),
        }
    return out


def smooth_binary(mask: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(mask.astype(np.uint8), 7)


def dynamic_mask(raw_relev: np.ndarray, threshold: float) -> np.ndarray:
    output = raw_relev.astype(np.float32)
    output = output - float(output.min())
    output = output / (float(output.max()) - float(output.min()) + 1e-9)
    return smooth_binary((output > threshold).astype(np.uint8))


def official_score(raw_relev: np.ndarray) -> float:
    scale = 30
    kernel = np.ones((scale, scale), dtype=np.float32) / float(scale * scale)
    filtered = cv2.filter2D(raw_relev.astype(np.float32), -1, kernel)
    return float((0.5 * (filtered + raw_relev.astype(np.float32))).max())


def protocol_rows(
    case: EvalCase,
    label_root: Path,
    thresholds: list[float],
    device: str,
    chunk_size: int,
    max_frames: int | None,
) -> tuple[list[dict[str, str]], dict[str, float]]:
    compute_dynamic_threshold = load_compute_dynamic_threshold()
    eval_params = {
        "stability_thresh": 0.3,
        "min_mask_size": 0.001,
        "max_mask_size": 0.95,
    }

    js_paths = sorted((label_root / case.scene).glob("frame_*.json"))
    if max_frames is not None:
        js_paths = js_paths[:max_frames]
    all_prompts = sorted({
        prompt
        for js_path in js_paths
        for prompt in read_gt(js_path)[0].keys()
    })
    prompt_to_idx = {prompt: idx for idx, prompt in enumerate(all_prompts)}
    pos_embeds, neg_embeds = load_text_embeds(all_prompts, device)

    score_level_by_threshold: dict[float, list[float]] = {t: [] for t in thresholds}
    dynamic_ious: list[float] = []
    level_oracle_ious: list[float] = []
    per_row_oracle_ious: list[float] = []
    recomputed_actual_05: list[float] = []
    n_prompt_rows = 0

    for js_path in js_paths:
        frame = js_path.stem
        idx = frame_index(frame)
        gt_masks, _ = read_gt(js_path)
        prompts = sorted(gt_masks)
        prompt_indices = [prompt_to_idx[p] for p in prompts]

        raw_by_level: dict[int, np.ndarray] = {}
        for level in (1, 2, 3):
            feature_path = case.feat_root / f"{case.scene}_{level}" / "train/ours_None/renders_npy" / f"{idx}.npy"
            if not feature_path.exists():
                raise FileNotFoundError(feature_path)
            raw_by_level[level] = relevance_maps(feature_path, pos_embeds, neg_embeds, device, chunk_size)

        for local_prompt_idx, prompt in enumerate(prompts):
            p_idx = prompt_indices[local_prompt_idx]
            gt = gt_masks[prompt]
            n_prompt_rows += 1

            level_scores = {
                level: official_score(raw_by_level[level][p_idx])
                for level in (1, 2, 3)
            }
            chosen_level = max((1, 2, 3), key=lambda lvl: level_scores[lvl])

            fixed05_by_level: dict[int, float] = {}
            for level in (1, 2, 3):
                mask05 = relevance_to_mask(raw_by_level[level][p_idx], 0.5, "median7")
                fixed05_by_level[level] = binary_iou(mask05, gt)[0]
            recomputed_actual_05.append(fixed05_by_level[chosen_level])
            level_oracle_ious.append(max(fixed05_by_level.values()))

            best_any = 0.0
            for th in thresholds:
                chosen_mask = relevance_to_mask(raw_by_level[chosen_level][p_idx], th, "median7")
                chosen_iou = binary_iou(chosen_mask, gt)[0]
                score_level_by_threshold[th].append(chosen_iou)
                for level in (1, 2, 3):
                    th_mask = relevance_to_mask(raw_by_level[level][p_idx], th, "median7")
                    best_any = max(best_any, binary_iou(th_mask, gt)[0])
            per_row_oracle_ious.append(best_any)

            stack = torch.from_numpy(np.stack([raw_by_level[level][p_idx] for level in (1, 2, 3)], axis=0)).cpu()
            dyn_lvl, dyn_thresh = compute_dynamic_threshold(
                stack,
                prompt,
                eval_params=eval_params,
            )
            dyn_mask = dynamic_mask(raw_by_level[dyn_lvl + 1][p_idx], float(dyn_thresh))
            dynamic_ious.append(binary_iou(dyn_mask, gt)[0])

        del raw_by_level
        if device == "cuda":
            torch.cuda.empty_cache()

    threshold_means = {th: mean(vals) for th, vals in score_level_by_threshold.items()}
    best_threshold, best_score = max(threshold_means.items(), key=lambda kv: kv[1])
    metrics = {
        "n_prompt_rows": float(n_prompt_rows),
        "recomputed_score_level_0.5": mean(recomputed_actual_05),
        "level_oracle_0.5": mean(level_oracle_ious),
        "scene_best_fixed_threshold": best_score,
        "scene_best_threshold": float(best_threshold),
        "dynamic_threshold_stability": mean(dynamic_ious),
        "per_row_threshold_oracle": mean(per_row_oracle_ious),
    }
    rows = []
    for th, val in sorted(threshold_means.items()):
        rows.append({
            "case_id": case.case_id,
            "scene": case.scene,
            "protocol": "score_level_fixed_threshold_sweep",
            "threshold_policy": f"fixed_{th:.2f}",
            "level_policy": "score_selected_level",
            "uses_gt_for_threshold": "no",
            "uses_gt_for_level": "no",
            "mean_iou": fmt(val),
            "paper_waldo_2d": fmt(PAPER_WALDO_2D),
            "delta_to_paper_2d": fmt(val - PAPER_WALDO_2D),
            "n_rows": str(n_prompt_rows),
            "notes": "recomputed from feature maps; fixed threshold set before eval",
        })
    return rows, metrics


def add_summary_row(
    rows: list[dict[str, str]],
    case_id: str,
    protocol: str,
    threshold_policy: str,
    level_policy: str,
    uses_gt_for_threshold: bool,
    uses_gt_for_level: bool,
    mean_iou: float,
    n_rows: int,
    notes: str,
    scene: str = "waldo_kitchen",
) -> None:
    rows.append({
        "case_id": case_id,
        "scene": scene,
        "protocol": protocol,
        "threshold_policy": threshold_policy,
        "level_policy": level_policy,
        "uses_gt_for_threshold": "yes" if uses_gt_for_threshold else "no",
        "uses_gt_for_level": "yes" if uses_gt_for_level else "no",
        "mean_iou": fmt(mean_iou),
        "paper_waldo_2d": fmt(PAPER_WALDO_2D),
        "delta_to_paper_2d": fmt(mean_iou - PAPER_WALDO_2D),
        "n_rows": str(n_rows),
        "notes": notes,
    })


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "scene", "protocol", "threshold_policy", "level_policy",
        "uses_gt_for_threshold", "uses_gt_for_level", "mean_iou",
        "paper_waldo_2d", "delta_to_paper_2d", "n_rows", "notes",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def parse_thresholds(spec: str) -> list[float]:
    start, stop, step = (float(x) for x in spec.split(":"))
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--official_detail", type=Path, default=DEFAULT_OFFICIAL_DETAIL)
    ap.add_argument("--p1_detail", type=Path, default=DEFAULT_P1_DETAIL)
    ap.add_argument("--label_root", type=Path, default=DEFAULT_LABEL_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cases", default="official_train_officialsam_waldo__waldo_kitchen,refersplat_3dgs_valafeat_full__waldo_kitchen")
    ap.add_argument("--thresholds", default="0.1:0.9:0.05")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--chunk_size", type=int, default=32768)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    requested = {c.strip() for c in args.cases.split(",") if c.strip()}
    all_cases = discover_cases(args.summary, args.run_root, None)
    cases = [case for case in all_cases if case.case_id in requested]
    missing = requested - {case.case_id for case in cases}
    if missing:
        raise RuntimeError(f"missing cases: {sorted(missing)}")

    official_actuals = read_official_actuals(args.official_detail)
    existing_p1 = read_existing_p1_oracles(args.p1_detail)
    thresholds = parse_thresholds(args.thresholds)

    rows: list[dict[str, str]] = []
    for case in cases:
        print(f"forensic case={case.case_id}")
        n_rows = len(next(iter([v for (cid, _), v in official_actuals.items() if cid == case.case_id]), []))
        for threshold in (0.5, 0.4):
            vals = official_actuals.get((case.case_id, threshold), [])
            if vals:
                add_summary_row(
                    rows, case.case_id, f"official_saved_actual_{threshold:.1f}",
                    f"fixed_{threshold:.1f}", "public_score_selected_saved_png",
                    False, False, mean(vals), len(vals),
                    "actual output from saved chosen_*.png; valid for leaderboard",
                )

        sweep_rows, metrics = protocol_rows(
            case=case,
            label_root=args.label_root,
            thresholds=thresholds,
            device=args.device,
            chunk_size=args.chunk_size,
            max_frames=args.max_frames,
        )
        rows.extend(sweep_rows)

        n_prompt_rows = int(metrics["n_prompt_rows"])
        add_summary_row(
            rows, case.case_id, "recomputed_score_level_0.5", "fixed_0.5",
            "score_selected_level", False, False,
            metrics["recomputed_score_level_0.5"], n_prompt_rows,
            "sanity recomputation from feature maps; official saved actual remains authoritative",
        )
        add_summary_row(
            rows, case.case_id, "scene_best_fixed_threshold", f"best_fixed_{metrics['scene_best_threshold']:.2f}",
            "score_selected_level", True, False,
            metrics["scene_best_fixed_threshold"], n_prompt_rows,
            "GT-tuned once over Waldo scene; diagnostic only, not leaderboard",
        )
        add_summary_row(
            rows, case.case_id, "dynamic_threshold_stability", "VALA_compute_dynamic_threshold",
            "stability_selected_level", False, False,
            metrics["dynamic_threshold_stability"], n_prompt_rows,
            "GT-free VALA dynamic threshold function from eval_utils.py, applied to LERF 2D feature maps",
        )
        add_summary_row(
            rows, case.case_id, "per_row_level_oracle_fixed_0.5", "fixed_0.5",
            "GT_best_level_per_row", False, True,
            metrics["level_oracle_0.5"], n_prompt_rows,
            "diagnostic level oracle; uses GT to select among levels",
        )
        add_summary_row(
            rows, case.case_id, "per_row_threshold_oracle", "GT_best_threshold_per_row",
            "GT_best_level_per_row", True, True,
            metrics["per_row_threshold_oracle"], n_prompt_rows,
            "diagnostic upper bound; uses GT per frame/prompt and must not be reported as actual performance",
        )

        if case.case_id in existing_p1:
            vals = existing_p1[case.case_id]
            add_summary_row(
                rows, case.case_id, "existing_p1_official_actual_0.5", "fixed_0.5",
                "public_score_selected_saved_png", False, False,
                vals["official_actual_0.5"], n_prompt_rows,
                "same as reconciled P1 detail table, included for traceability",
            )
            add_summary_row(
                rows, case.case_id, "existing_p1_per_row_threshold_oracle", "GT_best_threshold_per_row",
                "GT_best_level_per_row", True, True,
                vals["per_row_threshold_oracle"], n_prompt_rows,
                "same as reconciled P1 detail table, included for traceability",
            )

    write_csv(args.out, rows)


if __name__ == "__main__":
    main()
