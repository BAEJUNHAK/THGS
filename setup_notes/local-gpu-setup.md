---
name: local-gpu-setup
description: "How the THGS conda env is built on this local RTX 4060 Ti machine, and its setup quirks"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3dc409c0-3955-46c9-93b9-b33c5c495196
---

THGS runs locally on an **RTX 4060 Ti (Ada, compute capability 8.9)**. There is NO system CUDA toolkit and NO system conda; everything lives in conda.

- Conda: **Miniforge** at `~/miniforge3`. Env name **`thgs`** (`~/miniforge3/envs/thgs`), Python 3.10.13, PyTorch 2.2.0 + cu118.
- The CUDA **build** toolchain is inside the env (system gcc is 15.x, too new for CUDA 11.8): `cuda-nvcc 11.8`, `cuda-cccl 11.8.89`, `gcc/gxx_linux-64 11.4` — all from `nvidia/label/cuda-11.8.0` + conda-forge.
- **Build env vars** required for every CUDA extension build:
  `CUDA_HOME=$CONDA_PREFIX`, `TORCH_CUDA_ARCH_LIST=8.9`, `FORCE_CUDA=1`,
  `CC/CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-{gcc,g++}`,
  `NVCC_PREPEND_FLAGS="-ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"`.
- Always `pip install --no-build-isolation` for the local/git extensions (setup.py imports torch).

Setup gotchas already hit and fixed:
- Generic `cuda-cccl`/`cuda-profiler-api` resolve to v13.2 and put thrust/cub under `include/cccl/` → nvcc 11.8 can't find them. Must pin `cuda-cccl=11.8.89`.
- setuptools 82 removed `pkg_resources`, which FRNN's setup.py needs → pinned `setuptools<81` (80.10.2).

Built OK locally: diff-surfel-rasterization, simple-knn, FRNN, prefix_sum (CUDA); grid_graph + parallel cut-pursuit via `scripts/setup_dependencies.py build_ext` (CPU).

The 3 external-GitHub deps in environment.yml (pytorch3d, point_geometric_features, segment-anything-langsplat) **cannot be pip-installed by Claude** (sandbox blocks git+ installs); user runs **`scripts/install_git_deps.sh`** for those. BUT for the **eval path you only need `pgeof`**, which is on **PyPI** (`pip install pgeof` → 0.3.4, prebuilt wheel, NOT blocked). pytorch3d + SAM are only needed to *re-run the pipeline* (sp_partition/graph_weight/merge_proj), not to evaluate the ready-to-use scenes. See [[thgs-data-layout]].

**NVML/`nvidia-smi`**: a pending driver update left a kernel(595.58)/userspace(595.71) mismatch, so `nvidia-smi` failed (torch warned "Can't initialize NVML") while `torch.cuda` + CUDA kernels still ran fine. The on-disk nvidia module matched the running kernel (7.0.0-15-generic), so a reboot fixes it. `modprobe -r/modprobe` would also work but nvidia_drm is held by the display (gdm/gnome-shell/Xwayland) and modprobe needs real root (no sudo password available). **Reboot was the only privileged action Claude could do without a password** — the shell sits in the user's active local Wayland session (loginctl session 517), so polkit grants `CanReboot=yes` and `systemctl reboot` works without sudo. On 2026-05-27 Claude triggered `systemctl reboot` to reconcile the driver; after it, `nvidia-smi` should work.

Watch-item: `transformers 5.9.0` got installed and disables PyTorch (<2.4 → refused). `utils/vlm_utils.py` uses **open_clip** so CLIP query is fine, but anything importing transformers models (e.g. `image_encoding.py`) will break until transformers is downgraded to a torch-2.2-compatible 4.x.
