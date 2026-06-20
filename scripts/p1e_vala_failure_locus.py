#!/usr/bin/env python3
"""Classify where VALA failures arise for P1-E diagnostics.

The script joins three already-computed signals for each LERF-OVS query:

* best available 2D SAM mask overlap with GT (source oracle)
* best CLIP-selected 2D mask overlap (semantic top)
* final VALA rendered mask overlap

It intentionally stays CPU-only and dependency-light so we can run it while GPU
jobs are unavailable.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


KEYS = ("scene", "frame", "prompt")


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def read_csv(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = tuple(row[k] for k in KEYS)
            rows[key] = row
    return rows


def classify(row: dict[str, str]) -> str:
    final_iou = f(row, "final_iou")
    precision = f(row, "precision")
    recall = f(row, "recall")
    area_ratio = f(row, "area_ratio")
    source_iou = f(row, "source_best_iou")
    sem_top_iou = f(row, "sem_top_iou")
    oracle_iou = f(row, "oracle_iou")

    if final_iou >= 0.5:
        return "ok_final"
    if source_iou < 0.5:
        return "2d_source_missing"
    if sem_top_iou < 0.25 and oracle_iou >= 0.5:
        return "2d_semantic_selection"
    if area_ratio > 3.0 and recall > 0.5 and precision < 0.35:
        return "3d_overgrowth"
    if recall < 0.5 or area_ratio < 0.75:
        return "3d_undercoverage"
    if sem_top_iou >= 0.5 and sem_top_iou - final_iou > 0.2:
        return "3d_lifting_degradation"
    return "mixed_uncertain"


def aggregate(rows: list[dict[str, str]], group_key: str) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row[group_key])].append(row)

    out: list[dict[str, str]] = []
    for (scene, group), items in sorted(groups.items()):
        counts: dict[str, int] = defaultdict(int)
        for item in items:
            counts[item["failure_locus"]] += 1
        n = len(items)
        out.append(
            {
                "scene": scene,
                group_key: group,
                "n": str(n),
                "final_iou": f"{mean(f(x, 'final_iou') for x in items):.4f}",
                "precision": f"{mean(f(x, 'precision') for x in items):.4f}",
                "recall": f"{mean(f(x, 'recall') for x in items):.4f}",
                "area_ratio": f"{mean(f(x, 'area_ratio') for x in items):.4f}",
                "source_best_iou": f"{mean(f(x, 'source_best_iou') for x in items):.4f}",
                "sem_top_iou": f"{mean(f(x, 'sem_top_iou') for x in items):.4f}",
                "oracle_iou": f"{mean(f(x, 'oracle_iou') for x in items):.4f}",
                "ok_final": str(counts["ok_final"]),
                "2d_source_missing": str(counts["2d_source_missing"]),
                "2d_semantic_selection": str(counts["2d_semantic_selection"]),
                "3d_overgrowth": str(counts["3d_overgrowth"]),
                "3d_undercoverage": str(counts["3d_undercoverage"]),
                "3d_lifting_degradation": str(counts["3d_lifting_degradation"]),
                "mixed_uncertain": str(counts["mixed_uncertain"]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask_csv",
        type=Path,
        default=Path("output/diagnostics/p1e_vala_mask_diagnostics.csv"),
    )
    parser.add_argument(
        "--semantic_csv",
        type=Path,
        default=Path("output/diagnostics/p1e_vala_2d_semantic_selection.csv"),
    )
    parser.add_argument(
        "--out_prefix",
        type=Path,
        default=Path("output/diagnostics/p1e_vala_failure_locus"),
    )
    args = parser.parse_args()

    mask_rows = read_csv(args.mask_csv)
    semantic_rows = read_csv(args.semantic_csv)

    joined: list[dict[str, str]] = []
    for key, mask_row in sorted(mask_rows.items()):
        sem_row = semantic_rows.get(key)
        if sem_row is None:
            continue
        row = {
            "scene": key[0],
            "frame": key[1],
            "prompt": key[2],
            "final_iou": f"{f(mask_row, 'iou'):.6f}",
            "precision": f"{f(mask_row, 'precision'):.6f}",
            "recall": f"{f(mask_row, 'recall'):.6f}",
            "area_ratio": f"{f(mask_row, 'area_ratio'):.6f}",
            "source_best_iou": f"{f(mask_row, 'source_best_iou'):.6f}",
            "source_best_recall": f"{f(mask_row, 'source_best_recall'):.6f}",
            "sem_top_iou": f"{f(sem_row, 'sem_top_iou'):.6f}",
            "sem_top_recall": f"{f(sem_row, 'sem_top_recall'):.6f}",
            "sem_top_area_ratio": f"{f(sem_row, 'sem_top_area_ratio'):.6f}",
            "oracle_iou": f"{f(sem_row, 'oracle_iou'):.6f}",
            "oracle_rank": f"{f(sem_row, 'oracle_score_rank_in_level'):.1f}",
        }
        row["final_minus_sem_top"] = f"{f(row, 'final_iou') - f(row, 'sem_top_iou'):.6f}"
        row["sem_gap_to_oracle"] = f"{f(row, 'oracle_iou') - f(row, 'sem_top_iou'):.6f}"
        row["final_gap_to_source"] = f"{f(row, 'source_best_iou') - f(row, 'final_iou'):.6f}"
        row["failure_locus"] = classify(row)
        joined.append(row)

    write_csv(args.out_prefix.with_suffix(".csv"), joined)
    write_csv(Path(f"{args.out_prefix}_by_prompt.csv"), aggregate(joined, "prompt"))
    write_csv(Path(f"{args.out_prefix}_by_scene.csv"), aggregate(joined, "scene"))

    for row in aggregate(joined, "scene"):
        print(
            row["scene"],
            "n=", row["n"],
            "final=", row["final_iou"],
            "sem_top=", row["sem_top_iou"],
            "source=", row["source_best_iou"],
            "ok=", row["ok_final"],
            "sem_fail=", row["2d_semantic_selection"],
            "over=", row["3d_overgrowth"],
            "under=", row["3d_undercoverage"],
            "mixed=", row["mixed_uncertain"],
        )


if __name__ == "__main__":
    main()
