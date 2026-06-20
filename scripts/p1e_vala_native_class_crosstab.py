#!/usr/bin/env python3
"""Cross-tab THGS/ReLaGS failure classes against VALA-native failure classes.

The input detail table is produced by ``p1e_vala_reconcile_oracle_actual.py``.
This script is deliberately descriptive: THGS/ReLaGS classes are used only as
an interpretive bridge, while VALA's own actual/oracle fields determine the
VALA-native class.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETAIL = ROOT / "output/diagnostics/p1e_vala_native_2d_oracle_official_actual_detail.csv"
DEFAULT_ROW_OUT = ROOT / "output/diagnostics/p1e_vala_native_class_crosstab_rows.csv"
DEFAULT_PROMPT_OUT = ROOT / "output/diagnostics/p1e_vala_native_class_crosstab_prompts.csv"


def f(row: dict[str, str], key: str) -> float:
    val = row.get(key, "")
    return float(val) if val not in ("", "nan", "None") else float("nan")


def mean(vals: list[float]) -> float:
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def vala_class(actual: float, oracle_level: float, oracle_level_thresh: float) -> str:
    if actual >= 0.5:
        return "vala_actual_ok"
    if oracle_level >= 0.5:
        return "vala_selection_fail"
    if oracle_level_thresh >= 0.5:
        return "vala_calibration_fail"
    return "vala_representation_fail"


def majority(values: list[str]) -> str:
    clean = [v for v in values if v]
    if not clean:
        return ""
    return Counter(clean).most_common(1)[0][0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def prompt_aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["scene"], row["prompt"])].append(row)

    out: list[dict[str, str]] = []
    for (case_id, scene, prompt), group in sorted(groups.items()):
        actual = mean([f(r, "actual_iou") for r in group])
        oracle_level = mean([f(r, "oracle_level_iou") for r in group])
        oracle_thresh = mean([f(r, "oracle_level_thresh_iou") for r in group])
        out.append({
            "case_id": case_id,
            "scene": scene,
            "prompt": prompt,
            "n_rows_source": str(len(group)),
            "n_frames_source": str(len({r["frame"] for r in group})),
            "thgs_class": majority([r.get("thgs_class", "") for r in group]),
            "relags_class": majority([r.get("relags_class", "") for r in group]),
            "vala_class": vala_class(actual, oracle_level, oracle_thresh),
            "actual_iou": f"{actual:.6f}",
            "oracle_level_iou": f"{oracle_level:.6f}",
            "oracle_level_thresh_iou": f"{oracle_thresh:.6f}",
            "oracle_gap": f"{(oracle_thresh - actual):.6f}",
            "level_selection_gap": f"{(oracle_level - actual):.6f}",
            "threshold_gap": f"{(oracle_thresh - oracle_level):.6f}",
        })
    return out


def crosstab(rows: list[dict[str, str]], unit_name: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    all_rows = rows + [{**row, "scene": "ALL"} for row in rows]

    for taxonomy, class_col in (("THGS", "thgs_class"), ("ReLaGS", "relags_class")):
        denominators: dict[tuple[str, str], int] = defaultdict(int)
        groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

        for row in all_rows:
            source_class = row.get(class_col, "")
            vala = row.get("vala_class", "")
            if not source_class or not vala:
                continue
            denominators[(row["scene"], source_class)] += 1
            groups[(row["scene"], source_class, vala)].append(row)

        for (scene, source_class, vala), group in sorted(groups.items()):
            denom = denominators[(scene, source_class)]
            actual = [f(r, "actual_iou") for r in group]
            oracle_level = [f(r, "oracle_level_iou") for r in group]
            oracle_thresh = [f(r, "oracle_level_thresh_iou") for r in group]
            frame_values = {r["frame"] for r in group if r.get("frame")}
            n_frames = len(frame_values)
            if not frame_values and any(r.get("n_frames_source") for r in group):
                n_frames = sum(int(r.get("n_frames_source", "0") or "0") for r in group)
            out.append({
                "unit": unit_name,
                "taxonomy": taxonomy,
                "scene": scene,
                "source_class": source_class,
                "vala_class": vala,
                "n_units": str(len(group)),
                "n_prompts": str(len({r["prompt"] for r in group if r.get("prompt")})),
                "n_frames": str(n_frames),
                "fraction_within_source_class": f"{(len(group) / denom):.6f}",
                "mean_actual_iou": f"{mean(actual):.6f}",
                "mean_oracle_level_iou": f"{mean(oracle_level):.6f}",
                "mean_oracle_level_thresh_iou": f"{mean(oracle_thresh):.6f}",
                "mean_oracle_gap": f"{mean([b - a for a, b in zip(actual, oracle_thresh)]):.6f}",
                "mean_level_selection_gap": f"{mean([b - a for a, b in zip(actual, oracle_level)]):.6f}",
                "mean_threshold_gap": f"{mean([c - b for b, c in zip(oracle_level, oracle_thresh)]):.6f}",
            })
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "unit", "taxonomy", "scene", "source_class", "vala_class",
        "n_units", "n_prompts", "n_frames", "fraction_within_source_class",
        "mean_actual_iou", "mean_oracle_level_iou",
        "mean_oracle_level_thresh_iou", "mean_oracle_gap",
        "mean_level_selection_gap", "mean_threshold_gap",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    ap.add_argument("--row_out", type=Path, default=DEFAULT_ROW_OUT)
    ap.add_argument("--prompt_out", type=Path, default=DEFAULT_PROMPT_OUT)
    args = ap.parse_args()

    rows = read_rows(args.detail)
    write_csv(args.row_out, crosstab(rows, "row"))
    prompt_rows = prompt_aggregate(rows)
    write_csv(args.prompt_out, crosstab(prompt_rows, "prompt"))


if __name__ == "__main__":
    main()
