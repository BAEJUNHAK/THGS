#!/usr/bin/env python3
"""Add condition/delta summaries to RF-V2 reconciled VALA detail and aggregate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETAIL = ROOT / "output/diagnostics/p1e_vala_rf_v2_robust_vs_mean_detail.csv"
DEFAULT_AGG = ROOT / "output/diagnostics/p1e_vala_rf_v2_robust_vs_mean_agg.csv"


def condition(case_id: str) -> str:
    if case_id.startswith("rfv2_robust__"):
        return "robust_gate"
    if case_id.startswith("rfv2_mean__"):
        return "mean_non_gated"
    return ""


def add_columns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(1, "condition", df.case_id.map(condition))
    df.insert(2, "base_variant", "refersplat_3dgs_valafeat_full")
    df.to_csv(path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail_csv", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--agg_csv", type=Path, default=DEFAULT_AGG)
    args = ap.parse_args()

    detail = add_columns(args.detail_csv)
    agg = add_columns(args.agg_csv)

    print(f"updated {args.detail_csv} ({len(detail)} rows)")
    print(f"updated {args.agg_csv} ({len(agg)} rows)")

    overall = agg[(agg.scope == "overall") & (agg.scope_value == "all")].copy()
    if not overall.empty:
        print(overall[["scene", "condition", "actual_iou", "oracle_level_thresh_iou", "level_selection_gap", "threshold_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
