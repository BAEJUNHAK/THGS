#!/usr/bin/env python3
"""Run VALA public 2D evaluation for every existing LERF-OVS language model.

This intentionally keeps VALA's feature renderer and evaluator as the core
implementation. The only compatibility work here is:
  1. ensure the eval split contains the LERF-OVS label frames;
  2. adapt renderer frame_*.npy outputs to evaluate_iou_loc.py's numeric input;
  3. parse the resulting official logs into one CSV.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
THRESHOLDS = (0.5, 0.4)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    variant: str
    scene: str
    model: Path
    source: Path


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def read_source_path(cfg_args: Path) -> Path | None:
    if not cfg_args.exists():
        return None
    text = cfg_args.read_text(errors="replace")
    match = re.search(r"source_path='([^']+)'", text)
    if not match:
        return None
    return Path(match.group(1))


def discover_cases(root: Path) -> list[EvalCase]:
    output_root = root / "external_methods" / "VALA" / "output"
    cases: list[EvalCase] = []

    for model in sorted(output_root.glob("*/lerf_ovs/*")):
        if not model.is_dir() or model.name not in SCENES:
            continue
        if not all((model / "none" / f"chkpnt30000_langfeat_{level}_stochastic_gate.pth").exists() for level in (1, 2, 3)):
            continue
        source = read_source_path(model / "cfg_args")
        if source is None:
            print(f"SKIP no source_path in cfg_args: {model}", flush=True)
            continue
        if not source.exists():
            print(f"SKIP missing source_path: {model} -> {source}", flush=True)
            continue
        if not (source / "langsplat" / "language_features").exists():
            print(f"SKIP missing langsplat/language_features: {source}", flush=True)
            continue

        rel = model.relative_to(output_root)
        variant = rel.parts[0]
        case_id = f"{variant}__{model.name}"
        cases.append(EvalCase(case_id=case_id, variant=variant, scene=model.name, model=model, source=source))

    return cases


def label_stems(label_root: Path, scene: str) -> list[str]:
    scene_label = label_root / scene
    stems = [path.stem for path in sorted(scene_label.glob("frame_*.json"))]
    if not stems:
        raise RuntimeError(f"No LERF-OVS labels found for scene={scene}: {scene_label}")
    return stems


def frame_index(stem: str) -> int:
    return int(stem.split("_")[-1]) - 1


def ensure_test_split(source: Path, stems: list[str]) -> None:
    sparse = source / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    test_txt = sparse / "test.txt"
    expected = "\n".join(stems) + "\n"
    if test_txt.exists() and test_txt.read_text(errors="replace") == expected:
        return
    print(f"write eval split: {test_txt}", flush=True)
    test_txt.write_text(expected)


def render_level(root: Path, vala: Path, case: EvalCase, level: int, force_render: bool, env: dict[str, str]) -> Path:
    render_dir = case.model / "test" / f"ours_30000_langfeat_{level}_stochastic_gate" / "renders_npy"
    stems = label_stems(root / "data" / "lerf_ovs" / "label", case.scene)
    have_all = all((render_dir / f"{stem}.npy").exists() for stem in stems)
    if have_all and not force_render:
        print(f"render skip: {case.case_id} level={level} {render_dir}", flush=True)
        return render_dir

    run(
        [
            sys.executable,
            str(root / "scripts" / "p1e_run_vala_feature_map_renderer_safe.py"),
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
        ],
        cwd=vala,
        env=env | {"VALA_ROOT": str(vala)},
    )
    missing = [stem for stem in stems if not (render_dir / f"{stem}.npy").exists()]
    if missing:
        raise RuntimeError(f"Renderer did not produce expected frames for {case.case_id} level={level}: {missing}")
    return render_dir


def adapt_feature_dir(root: Path, run_root: Path, case: EvalCase, render_dirs: dict[int, Path]) -> Path:
    feat_root = run_root / "feat_dir" / case.case_id
    stems = label_stems(root / "data" / "lerf_ovs" / "label", case.scene)
    indices = [frame_index(stem) for stem in stems]
    max_idx = max(indices)

    for level, render_dir in render_dirs.items():
        dst_dir = feat_root / f"{case.scene}_{level}" / "train" / "ours_None" / "renders_npy"
        dst_dir.mkdir(parents=True, exist_ok=True)
        first = render_dir / f"{stems[0]}.npy"
        if not first.exists():
            raise RuntimeError(f"Missing first render file: {first}")

        for idx in range(max_idx + 1):
            link = dst_dir / f"{idx}.npy"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(first)

        for stem, idx in zip(stems, indices):
            src = render_dir / f"{stem}.npy"
            link = dst_dir / f"{idx}.npy"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src)

    return feat_root


def run_official_eval(root: Path, vala: Path, run_root: Path, case: EvalCase, feat_root: Path, threshold: float, env: dict[str, str]) -> Path:
    out_dir = run_root / "eval" / case.case_id / f"thresh_{threshold}"
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
            str(root / "data" / "lerf_ovs" / "label"),
            "--mask_thresh",
            str(threshold),
            "--direct_512",
        ],
        cwd=vala,
        env=env,
    )
    logs = sorted((out_dir / case.scene).glob("*.log"))
    if not logs:
        raise RuntimeError(f"No official eval log produced: {out_dir}")
    return logs[-1]


def parse_log(log_path: Path) -> dict[str, str]:
    text = log_path.read_text(errors="replace")
    values: dict[str, str] = {"log": str(log_path)}
    for key, pattern in {
        "miou": r"iou chosen:\s*([0-9.]+)",
        "localization": r"Localization accuracy:\s*([0-9.]+)",
        "trunc_thresh": r"trunc thresh:\s*([0-9.]+)",
    }.items():
        match = re.search(pattern, text)
        values[key] = match.group(1) if match else ""
    return values


def write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "variant",
        "scene",
        "threshold",
        "miou",
        "localization",
        "model",
        "source",
        "log",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {summary_path}", flush=True)


def main() -> None:
    root = Path(os.environ.get("THGS_ROOT", "/mnt/pilab_nas/projects/THGS")).resolve()
    vala = root / "external_methods" / "VALA"
    run_tag = os.environ.get("RUN_TAG", os.environ.get("SLURM_JOB_ID", "manual"))
    run_root = vala / "output" / f"official_2d_eval_all_{run_tag}"
    force_render = os.environ.get("FORCE_RENDER", "0") == "1"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{vala}:{env.get('PYTHONPATH', '')}"

    cases = discover_cases(root)
    if not cases:
        raise RuntimeError("No complete VALA LERF-OVS language-feature cases found.")

    print(f"run_root={run_root}", flush=True)
    print(f"cases={len(cases)}", flush=True)
    for case in cases:
        print(f"case {case.case_id}: model={case.model} source={case.source}", flush=True)

    rows: list[dict[str, str]] = []
    label_root = root / "data" / "lerf_ovs" / "label"
    for case in cases:
        print(f"\n=== {case.case_id} ===", flush=True)
        stems = label_stems(label_root, case.scene)
        ensure_test_split(case.source, stems)
        render_dirs = {level: render_level(root, vala, case, level, force_render, env) for level in (1, 2, 3)}
        feat_root = adapt_feature_dir(root, run_root, case, render_dirs)

        for threshold in THRESHOLDS:
            log = run_official_eval(root, vala, run_root, case, feat_root, threshold, env)
            parsed = parse_log(log)
            rows.append(
                {
                    "case_id": case.case_id,
                    "variant": case.variant,
                    "scene": case.scene,
                    "threshold": str(threshold),
                    "miou": parsed["miou"],
                    "localization": parsed["localization"],
                    "model": str(case.model),
                    "source": str(case.source),
                    "log": parsed["log"],
                }
            )
            write_summary(run_root / "summary.csv", rows)
            write_summary(root / "output" / "diagnostics" / f"p1e_vala_official_2d_eval_all_{run_tag}.csv", rows)


if __name__ == "__main__":
    main()
