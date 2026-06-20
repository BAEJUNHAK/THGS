#!/usr/bin/env python3
"""Download ReferSplat RGB 3DGS checkpoints used as VALA-compatible inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


FILES = [
    "figurineschkpnt30000.pth",
    "ramenchkpnt30000.pth",
    "teatimechkpnt30000.pth",
    "kitchenchkpnt30000.pth",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="FudanCVL/RefSplat")
    parser.add_argument("--output-root", type=Path, default=Path("external_methods/_refersplat_hf"))
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="model",
            filename=filename,
            local_dir=args.output_root,
        )
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"{filename}: {path} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
