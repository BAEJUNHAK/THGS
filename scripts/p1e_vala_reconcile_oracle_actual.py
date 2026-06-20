#!/usr/bin/env python3
"""Use official saved VALA chosen masks as actuals for oracle analysis.

`p1e_vala_native_2d_oracle.py` recomputes VALA relevance maps to measure level
and threshold oracle ceilings. For the actual score, the fairest source is the
official evaluator's saved `chosen_*.png` masks, already scored by
`p1e_vala_official_2d_prompt_table.py`. This script joins those official
actuals into the oracle table and recomputes the VALA-native failure class.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_ORACLE = Path("output/diagnostics/p1e_vala_native_2d_oracle_detail.csv")
DEFAULT_OFFICIAL = Path("output/diagnostics/p1e_vala_official_2d_prompt_detail.csv")
DEFAULT_DETAIL = Path("output/diagnostics/p1e_vala_native_2d_oracle_official_actual_detail.csv")
DEFAULT_AGG = Path("output/diagnostics/p1e_vala_native_2d_oracle_official_actual_agg.csv")


def vala_class(row: pd.Series) -> str:
    if row.actual_iou >= 0.5:
        return "vala_actual_ok"
    if row.oracle_level_iou >= 0.5:
        return "vala_selection_fail"
    if row.oracle_level_thresh_iou >= 0.5:
        return "vala_calibration_fail"
    return "vala_representation_fail"


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_group(scope: str, value: str, group: pd.DataFrame) -> None:
        rows.append({
            "case_id": group.case_id.iloc[0],
            "scene": group.scene.iloc[0],
            "scope": scope,
            "scope_value": value,
            "n_rows": len(group),
            "n_prompts": group.prompt.nunique(),
            "n_frames": group.frame.nunique(),
            "actual_iou": group.actual_iou.mean(),
            "oracle_level_iou": group.oracle_level_iou.mean(),
            "oracle_level_thresh_iou": group.oracle_level_thresh_iou.mean(),
            "level_selection_gap": group.level_selection_gap.mean(),
            "threshold_gap": group.threshold_gap.mean(),
        })

    for _, group in detail.groupby(["case_id", "scene"], sort=True):
        add_group("overall", "all", group)
        for scope in ("vala_class", "thgs_class", "relags_class"):
            for value, subgroup in group.groupby(scope, sort=True):
                if pd.isna(value) or value == "":
                    continue
                add_group(scope, str(value), subgroup)

    out = pd.DataFrame(rows)
    for col in (
        "actual_iou",
        "oracle_level_iou",
        "oracle_level_thresh_iou",
        "level_selection_gap",
        "threshold_gap",
    ):
        out[col] = out[col].map(lambda x: f"{x:.6f}")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_detail", type=Path, default=DEFAULT_ORACLE)
    ap.add_argument("--official_detail", type=Path, default=DEFAULT_OFFICIAL)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--detail_csv", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--agg_csv", type=Path, default=DEFAULT_AGG)
    args = ap.parse_args()

    oracle = pd.read_csv(args.oracle_detail)
    official = pd.read_csv(args.official_detail)
    official = official[
        official.case_id.isin(oracle.case_id.unique())
        & (official.threshold.astype(float) == args.threshold)
    ]

    keys = ["case_id", "scene", "frame", "prompt"]
    merged = oracle.merge(
        official[keys + ["iou", "precision", "recall", "pred_area"]],
        on=keys,
        how="left",
        suffixes=("", "_official"),
    )
    missing = int(merged.iou.isna().sum())
    if missing:
        raise RuntimeError(f"missing official actual rows: {missing}")

    merged["recomputed_actual_iou"] = merged["actual_iou"]
    merged["recomputed_actual_precision"] = merged["actual_precision"]
    merged["recomputed_actual_recall"] = merged["actual_recall"]
    merged["recomputed_actual_pred_area"] = merged["actual_pred_area"]
    merged["actual_iou"] = merged["iou"]
    merged["actual_precision"] = merged["precision"]
    merged["actual_recall"] = merged["recall"]
    merged["actual_pred_area"] = merged["pred_area"]
    merged["actual_source"] = "official_saved_chosen_png"
    merged["level_selection_gap"] = (merged["oracle_level_iou"] - merged["actual_iou"]).clip(lower=0)
    merged["threshold_gap"] = (merged["oracle_level_thresh_iou"] - merged["oracle_level_iou"]).clip(lower=0)
    merged["vala_class"] = merged.apply(vala_class, axis=1)
    merged = merged.drop(columns=["iou", "precision", "recall", "pred_area"])

    front = list(oracle.columns)
    extra = [col for col in merged.columns if col not in front]
    merged = merged[front + extra]

    args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.detail_csv, index=False)
    agg = aggregate(merged)
    agg.to_csv(args.agg_csv, index=False)
    print(f"wrote {args.detail_csv} ({len(merged)} rows)")
    print(f"wrote {args.agg_csv} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
