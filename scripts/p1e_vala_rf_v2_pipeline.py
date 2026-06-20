#!/usr/bin/env python3
"""Prepare RF-V2 robust-gate vs mean/non-gated VALA 2D evaluation.

This GPU-side driver keeps the RGB 3DGS, source language_features, frames,
prompts, GT, and evaluator fixed. The only changed factor is the language
aggregation checkpoint:

  robust: chkpnt30000_langfeat_{1,2,3}_stochastic_gate.pth
  mean:   chkpnt30000_langfeat_{1,2,3}.pth

It renders 512D feature maps with a suffix-aware renderer, adapts them to the
public VALA 2D evaluator layout, runs the public evaluator at threshold 0.5,
and writes a summary CSV consumed by the existing prompt/oracle scripts.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALA = ROOT / "external_methods" / "VALA"
DEFAULT_RUN_ROOT = VALA / "output/rf_v2_robust_vs_mean"
DEFAULT_SUMMARY = ROOT / "output/diagnostics/p1e_vala_rf_v2_official_2d_summary.csv"
MODEL_VARIANT = "refersplat_3dgs_valafeat_full"
THRESHOLD = 0.5


@dataclass(frozen=True)
class Case:
    condition: str
    case_id: str
    variant: str
    scene: str
    model: Path
    source: Path
    checkpoint_suffix: str
    output_tag: str


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def read_source_path(cfg_args: Path) -> Path:
    text = cfg_args.read_text(errors="replace")
    match = re.search(r"source_path='([^']+)'", text)
    if not match:
        raise RuntimeError(f"no source_path in {cfg_args}")
    source = Path(match.group(1))
    if not source.exists():
        raise FileNotFoundError(source)
    return source


def label_stems(scene: str) -> list[str]:
    stems = [path.stem for path in sorted((ROOT / "data/lerf_ovs/label" / scene).glob("frame_*.json"))]
    if not stems:
        raise RuntimeError(f"no labels for scene={scene}")
    return stems


def frame_index(stem: str) -> int:
    return int(stem.split("_")[-1]) - 1


def ensure_test_split(source: Path, scene: str) -> None:
    sparse = source / "sparse/0"
    sparse.mkdir(parents=True, exist_ok=True)
    expected = "\n".join(label_stems(scene)) + "\n"
    test_txt = sparse / "test.txt"
    if not test_txt.exists() or test_txt.read_text(errors="replace") != expected:
        test_txt.write_text(expected)
        print(f"wrote eval split: {test_txt}", flush=True)


def cases_for_scenes(scenes: list[str]) -> list[Case]:
    cases: list[Case] = []
    for scene in scenes:
        model = VALA / "output" / MODEL_VARIANT / "lerf_ovs" / scene
        if not model.exists():
            raise FileNotFoundError(model)
        source = read_source_path(model / "cfg_args")
        for condition, suffix, tag in (
            ("robust", "_stochastic_gate", "robust"),
            ("mean", "", "mean"),
        ):
            variant = f"rfv2_{condition}"
            cases.append(Case(
                condition=condition,
                case_id=f"{variant}__{scene}",
                variant=variant,
                scene=scene,
                model=model,
                source=source,
                checkpoint_suffix=suffix,
                output_tag=tag,
            ))
    return cases


def ensure_mean_checkpoint(case: Case, level: int, env: dict[str, str], force: bool) -> None:
    if case.condition != "mean":
        return
    ckpt = case.model / "none" / f"chkpnt30000_langfeat_{level}.pth"
    if ckpt.exists() and not force:
        print(f"mean checkpoint skip: {ckpt}", flush=True)
        return
    run(
        [
            sys.executable,
            "gaussian_feature_extractor.py",
            "-s",
            str(case.source),
            "-m",
            str(case.model),
            "--iteration",
            "30000",
            "--feature_level",
            str(level),
            "--eval",
        ],
        cwd=VALA,
        env=env,
    )
    if not ckpt.exists():
        raise RuntimeError(f"mean checkpoint not produced: {ckpt}")


def render_dir(case: Case, level: int) -> Path:
    return case.model / "test" / f"ours_30000_langfeat_{level}_{case.output_tag}" / "renders_npy"


def ensure_render(case: Case, level: int, env: dict[str, str], force: bool) -> Path:
    out = render_dir(case, level)
    stems = label_stems(case.scene)
    if all((out / f"{stem}.npy").exists() for stem in stems) and not force:
        print(f"render skip: {case.case_id} level={level} {out}", flush=True)
        return out
    run(
        [
            sys.executable,
            str(ROOT / "scripts/p1e_vala_feature_map_renderer_suffix.py"),
            "-s",
            str(case.source),
            "-m",
            str(case.model),
            "--iteration",
            "30000",
            "--feature_level",
            str(level),
            "--eval",
            "--skip_train",
            "--checkpoint_suffix",
            case.checkpoint_suffix,
            "--output_tag",
            case.output_tag,
        ],
        cwd=ROOT,
        env=env,
    )
    missing = [stem for stem in stems if not (out / f"{stem}.npy").exists()]
    if missing:
        raise RuntimeError(f"missing rendered feature maps for {case.case_id} level={level}: {missing}")
    return out


def adapt_feature_dir(run_root: Path, case: Case, render_dirs: dict[int, Path]) -> Path:
    feat_root = run_root / "feat_dir" / case.case_id
    stems = label_stems(case.scene)
    indices = [frame_index(stem) for stem in stems]
    max_idx = max(indices)
    for level, src_dir in render_dirs.items():
        dst = feat_root / f"{case.scene}_{level}" / "train/ours_None/renders_npy"
        dst.mkdir(parents=True, exist_ok=True)
        first = src_dir / f"{stems[0]}.npy"
        for idx in range(max_idx + 1):
            link = dst / f"{idx}.npy"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(first)
        for stem, idx in zip(stems, indices):
            src = src_dir / f"{stem}.npy"
            link = dst / f"{idx}.npy"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src)
    return feat_root


def run_official_eval(run_root: Path, case: Case, feat_root: Path, env: dict[str, str]) -> Path:
    out_dir = run_root / "eval" / case.case_id / f"thresh_{THRESHOLD}"
    run(
        [
            sys.executable,
            "eval/evaluate_iou_loc.py",
            "--dataset_name",
            case.scene,
            "--feat_dir",
            str(feat_root),
            "--ae_ckpt_dir",
            str(run_root / "dummy_ae"),
            "--output_dir",
            str(out_dir),
            "--json_folder",
            str(ROOT / "data/lerf_ovs/label"),
            "--mask_thresh",
            str(THRESHOLD),
            "--direct_512",
        ],
        cwd=VALA,
        env=env,
    )
    logs = sorted((out_dir / case.scene).glob("*.log"))
    if not logs:
        raise RuntimeError(f"no official eval log produced in {out_dir}")
    return logs[-1]


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(errors="replace")
    out = {"log": str(log_path)}
    for key, pattern in {
        "miou": r"iou chosen:\s*([0-9.]+)",
        "localization": r"Localization accuracy:\s*([0-9.]+)",
    }.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else ""
    return out


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "variant", "scene", "threshold", "miou", "localization", "model", "source", "log"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="ramen,waldo_kitchen")
    ap.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--force_features", action="store_true")
    ap.add_argument("--force_render", action="store_true")
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{VALA}:{env.get('PYTHONPATH', '')}"

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    rows: list[dict[str, str]] = []
    for case in cases_for_scenes(scenes):
        print(f"\n=== RF-V2 {case.case_id} ===", flush=True)
        ensure_test_split(case.source, case.scene)
        render_dirs: dict[int, Path] = {}
        for level in (1, 2, 3):
            ensure_mean_checkpoint(case, level, env, args.force_features)
            render_dirs[level] = ensure_render(case, level, env, args.force_render)
        feat_root = adapt_feature_dir(args.run_root, case, render_dirs)
        log = run_official_eval(args.run_root, case, feat_root, env)
        parsed = parse_log(log)
        rows.append({
            "case_id": case.case_id,
            "variant": case.variant,
            "scene": case.scene,
            "threshold": str(THRESHOLD),
            "miou": parsed["miou"],
            "localization": parsed["localization"],
            "model": str(case.model),
            "source": str(case.source),
            "log": parsed["log"],
        })
    write_summary(args.summary, rows)


if __name__ == "__main__":
    main()
