#!/usr/bin/env python3
"""Run VALA's official feature_map_renderer.py with a filename-only I/O guard.

The official renderer saves evaluation PNGs to `view.image_name`. In LERF-OVS,
camera names may be extensionless (for example `frame_00053`), which makes PIL
raise `ValueError: unknown file extension`. The `.npy` feature maps used by the
official evaluator are unchanged; this wrapper only appends `.png` for image
side-products so the unmodified official renderer can finish.
"""

from __future__ import annotations

import os
import runpy
import sys

import torchvision.utils


def main() -> None:
    vala_root = os.environ.get(
        "VALA_ROOT",
        "/mnt/pilab_nas/projects/THGS/external_methods/VALA",
    )
    renderer = os.path.join(vala_root, "feature_map_renderer.py")

    save_image = torchvision.utils.save_image

    def save_image_with_extension(tensor, fp, *args, **kwargs):
        path = os.fspath(fp)
        if not os.path.splitext(path)[1]:
            path = f"{path}.png"
        return save_image(tensor, path, *args, **kwargs)

    torchvision.utils.save_image = save_image_with_extension
    sys.argv = [renderer, *sys.argv[1:]]
    runpy.run_path(renderer, run_name="__main__")


if __name__ == "__main__":
    main()
