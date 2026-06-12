# Remote-side setup instructions for THGS on the pilab GPU server

This project was prepared on a local machine (RTX 4060 Ti, 8 GB) and rsync'd
here. Code, the LERF-OVS dataset, the author's ready-to-use 2DGS+semantic
scenes, and one local ramen evaluation result are already in place. Only the
Python environment and CUDA extensions need to be (re)built on this machine.

> **Status (2026-05-28):** the procedure below has been executed end-to-end on
> pilab and ramen reproduces (mIoU 0.42141, mAcc 0.96094 vs. local 0.42073,
> 0.95946 — 0.16% drift, within Blackwell numerical noise). The text below
> reflects the *fixed* procedure; the gotchas that were hit during the first
> run are catalogued in section 6 so the next person doesn't re-discover them.

---

## 0) Layout already in place

- `~/THGS` is a symlink → `/mnt/pilab_nas/projects/THGS/` (lab NAS, 16 TB free).
  Do *not* keep large things in `/home/meaabebe` — root FS only has ~5 GB free.
- `data/lerf-ovs/` — LERF-OVS dataset (images + COLMAP + GT labels). Symlinks
  `data/lerf-ovs → data/lerf_ovs` and `data/lerf → data/lerf_ovs` already set.
- `output/lerf/{figurines,ramen,teatime,waldo_kitchen}/` — author's ready-to-use
  scenes including `sai_nag.pt`. Eval can run with these directly.
- `output/render/lerf/ramen/` — masks from the local ramen run; can be overwritten.
- `scripts/install_git_deps.sh` — for the three external GitHub deps that the
  Claude Code sandbox blocks. **pgeof is not in this script** because pgeof has
  a PyPI wheel; use that instead (see below).
- `setup_notes/local-gpu-setup.md` and `thgs-data-layout.md` — companion notes
  from the local build (which gotchas were hit, in what order).

## 1) This machine is Blackwell — the local recipe needs version bumps

`nvidia-smi --query-gpu=compute_cap` reports **`12.0`** (Blackwell, sm_120) on
3× RTX PRO 5000 Blackwell, 48 GB each. CUDA 11.8 (which the local build used)
does NOT support sm_120 — it tops out at sm_90. Building or running the THGS
CUDA extensions against cu118 here will either fail or PTX-JIT with severe perf
loss. The system already has `/usr/local/cuda-12.4` installed and driver
`580.95.05`, but sm_120 was first added in CUDA 12.8 / PyTorch 2.6.

**Use PyTorch 2.6+ with cu128 wheels and a conda-provided `cuda-nvcc=12.8` (or
12.9) instead of the local recipe's PyTorch 2.2 + cu118.** Set
`TORCH_CUDA_ARCH_LIST="12.0"` when building extensions. Everything else from
the local recipe applies as-is.

## 2) Where to put Miniforge + the conda env

Root `/` only has ~5 GB free; conda envs are ~5–10 GB. **Put Miniforge on the
NAS** to avoid `No space left on device`. Recommended:

```
~/miniforge3 → /mnt/pilab_nas/projects/THGS/miniforge3   # symlink
```

or install directly at `/mnt/pilab_nas/projects/THGS/miniforge3` and source
its activate script from your shell.

NFS does mean conda solves and pip installs run slower than on local disk;
that is acceptable but expect each step to take 1.5–3× longer than the timings
in the local notes.

## 3) Build steps (mirror of local, with version bumps)

```bash
# install Miniforge to NAS
curl -L -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p /mnt/pilab_nas/projects/THGS/miniforge3
ln -s /mnt/pilab_nas/projects/THGS/miniforge3 ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh

# create env: Python + base deps only (NO pytorch-cuda here — see Gotcha #1)
mamba create -n thgs -y -c conda-forge \
  python=3.10.13 pip "plyfile=0.8.1" "numpy=1.26.4" einops tqdm ninja cmake git

conda activate thgs

# PyTorch cu128 via pip (the pytorch conda channel tops out at pytorch-cuda=12.4)
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
# This currently installs torch 2.11.0+cu128. The doc was originally written
# for 2.6, but 2.6+ is the actual requirement and 2.11 works fine on sm_120.
# Pin the version (`torch==2.6.0`) if you want exact reproducibility.

# CUDA 12.8 dev toolchain (nvcc, cccl, cudart-dev, gcc 11)
mamba install -n thgs -y -c "nvidia/label/cuda-12.8.0" -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvtx \
  "cuda-cccl=12.8" cuda-profiler-api \
  gxx_linux-64=11 gcc_linux-64=11

# pure-python deps (same list as environment.yml, minus things we handle below)
pip install \
  trimesh kiui pymeshlab open3d scipy dearpygui omegaconf open_clip_torch \
  transformations transformers yapf pycocotools mediapy lpips scikit-image \
  "opencv-python==4.7.0.72" h5py colorhash seaborn pyrootutils \
  hydra-core hydra-colorlog hydra-submitit-launcher numba \
  "torch_geometric==2.3.0" pytorch-lightning rich ipyfilechooser natsort

# PyG CUDA wheels — must match the actually-installed torch (check `python -c
# "import torch; print(torch.__version__)"` first, then plug that into the URL).
# For torch 2.11.0+cu128:
pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
# For other torch versions, swap the URL accordingly. If wheels for that exact
# combo don't exist, pyg falls back to a source build — still fine, since
# nvcc + arch are set up.

# downgrade setuptools so FRNN's setup.py (which uses pkg_resources) builds
pip install "setuptools<81"

# pgeof from PyPI (NOT git+) — wheel covers the eval path entirely
pip install pgeof

# build env vars for the CUDA extensions
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1 MAX_JOBS=8
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

cd ~/THGS

# CRITICAL: nuke any leftover build dirs first. The rsync'd project tree may
# contain stale build/ from a different user (different CUDA version, different
# RPATH). pip will reuse those .o files and produce a .so linked against
# libcudart.so.11.0, which then fails to import. See Gotcha #4.
rm -rf ./submodules/diff-surfel-rasterization/build ./submodules/diff-surfel-rasterization/*.egg-info
rm -rf ./submodules/simple-knn/build ./submodules/simple-knn/*.egg-info
rm -rf ./ext/spt/dependencies/FRNN/build ./ext/spt/dependencies/FRNN/*.egg-info
rm -rf ./ext/spt/dependencies/FRNN/external/prefix_sum/build ./ext/spt/dependencies/FRNN/external/prefix_sum/*.egg-info

# Patch simple_knn.cu for CUDA 12+ (FLT_MAX no longer transitively included)
# See Gotcha #3. Idempotent.
grep -q '#include <float.h>' ./submodules/simple-knn/simple_knn.cu || \
  sed -i '/^#include "simple_knn.h"/a #include <float.h>' ./submodules/simple-knn/simple_knn.cu

pip install --no-build-isolation --no-cache-dir \
  ./submodules/diff-surfel-rasterization \
  ./submodules/simple-knn \
  ./ext/spt/dependencies/FRNN \
  ./ext/spt/dependencies/FRNN/external/prefix_sum

# CPU C++ extensions (grid_graph + parallel cut-pursuit)
python scripts/setup_dependencies.py build_ext

# Verify the built extensions link against libcudart.so.12 (not 11)
python -c "
import subprocess, glob, os
env = os.environ['CONDA_PREFIX']
for name in ['frnn', 'simple_knn', 'diff_surfel_rasterization', 'prefix_sum']:
    so = glob.glob(f'{env}/lib/python3.10/site-packages/{name}/_C*.so')
    if not so: so = glob.glob(f'{env}/lib/python3.10/site-packages/{name}.cpython-*.so')
    if not so: print(name, 'NOT FOUND'); continue
    out = subprocess.check_output(['readelf', '-d', so[0]]).decode()
    cu = [l for l in out.splitlines() if 'libcudart' in l]
    print(name, '->', cu[0].strip() if cu else 'no cudart needed')
"
```

If the Claude Code sandbox here also blocks `git+https://` installs (it does
on the local machine), the three external repos in environment.yml
(`pytorch3d`, `point_geometric_features`, `segment-anything-langsplat`) cannot
be installed by Claude. **They are not needed to evaluate ramen — only pgeof
is, and that came from PyPI.** Run `scripts/install_git_deps.sh` only if you
later want to re-run the full pipeline (graph_weight / sp_partition /
merge_proj) from scratch.

## 4) ramen sanity check (should reproduce the local number)

```bash
cd ~/THGS
python test_lerf.py -s data/lerf-ovs/ramen -m output/lerf/ramen \
                    --path_pred output/render/lerf
python scripts/eval_seg.py --dataset lerf --scene_list ramen \
                           --path_pred output/render/lerf \
                           --path_gt data/lerf-ovs/label
```

Expected: `ramen mIoU, mAcc: 0.42073…, 0.95946…` (local) or
`0.42141…, 0.96094…` (pilab Blackwell run on 2026-05-28 — ~0.16% drift, OK).
If it diverges much more, suspect: (a) different open_clip weights downloaded,
(b) Blackwell numerical differences, (c) something rebuilt against a different
arch, (d) an extension still linked against libcudart.so.11 (the readelf check
at the end of section 3 catches this).

Note: `test_lerf.py` calls `safe_state(True)` (utils/general_utils.py:112)
which **silences stdout entirely**. A successful run produces no console
output other than HF Hub warnings on stderr — confirm success by checking that
`output/render/lerf/ramen/frame_*/` PNGs have fresh timestamps, then run
`eval_seg.py` to see the numbers.

## 5) Going beyond ramen

With 3× 48 GB Blackwell, the THGS pipeline that the local 8 GB GPU couldn't
host (`graph_weight.py`, `sp_partition.py`, `merge_proj.py`) is comfortable.
For full 4-scene eval:

```bash
for sc in figurines ramen teatime waldo_kitchen; do
  python test_lerf.py -s data/lerf-ovs/$sc -m output/lerf/$sc \
                      --path_pred output/render/lerf
done
python scripts/eval_seg.py --dataset lerf \
  --scene_list figurines ramen teatime waldo_kitchen \
  --path_pred output/render/lerf --path_gt data/lerf-ovs/label
```

The `configs/lerf.yml` default memory parameters were tuned for 24 GB GPUs;
on Blackwell 48 GB you can comfortably leave them as is or increase the
clip/feature batch sizes for speed.

**GPU pinning**: the pilab box is shared. As of 2026-05-28, user `dorong` was
running long jobs on GPUs 0 and 1. Pin THGS work to GPU 2 with
`export CUDA_VISIBLE_DEVICES=2` before invoking python. Check `nvidia-smi`
before bigger jobs and pick whichever GPU is free.

## 6) Gotchas encountered on the first Blackwell run (2026-05-28)

These are documented so the next person setting this up — or migrating to
another Blackwell box — doesn't re-discover them. The commands in sections
2–4 above already incorporate the fixes; this section is *why*.

### Gotcha #1 — `pytorch-cuda=12.8` does not exist on the pytorch conda channel

The doc originally said
```
mamba create ... -c pytorch -c nvidia ... pytorch ... pytorch-cuda=12.8
```
This fails with `pytorch-cuda =12.8 * does not exist`. The `pytorch-cuda`
metapackage on the official `pytorch` channel currently tops out at 12.4
(versions: 11.6, 11.7, 11.8, 12.1, 12.4). For cu128 you have to go via the
pytorch.org wheel index:
```
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
```
That index currently serves `torch 2.11.0+cu128`. The doc was originally
written for 2.6, but 2.6+ is the actual requirement and 2.11 imports/works
on sm_120 (verified — `torch.cuda.get_arch_list()` includes `sm_120`).

### Gotcha #2 — PyG wheel URL must match the actually-installed torch version

The doc had `https://data.pyg.org/whl/torch-2.6.0+cu128.html` as an example.
If you installed torch 2.11 (per Gotcha #1) and use the 2.6.0 URL, pip will
either silently install incompatible binaries or fall back to a slow source
build. Always derive the URL from the installed torch:
```
TV=$(python -c "import torch; print(torch.__version__.replace('+','%2B'))")
pip install pyg_lib torch_scatter torch_cluster \
  -f "https://data.pyg.org/whl/torch-${TV/\%2B/+}.html"
```
For torch `2.11.0+cu128`, wheels exist (`pyg_lib-0.6.0+pt211cu128`,
`torch_scatter-2.1.2+pt211cu128`, `torch_cluster-1.6.3+pt211cu128`).

### Gotcha #3 — `simple_knn.cu` fails to compile under CUDA 12+ (`FLT_MAX undefined`)

`submodules/simple-knn/simple_knn.cu` uses `FLT_MAX` but doesn't include
`<float.h>`. Under CUDA 11 some other header transitively pulled it in;
under CUDA 12 that's gone. nvcc errors out:
```
simple_knn.cu(90): error: identifier "FLT_MAX" is undefined
simple_knn.cu(154): error: identifier "FLT_MAX" is undefined
```
Fix: add `#include <float.h>` after `#include "simple_knn.h"`. The build
command block in section 3 patches this automatically (idempotent sed).

### Gotcha #4 — Stale `build/` dirs from previous users cause `ImportError: libcudart.so.11.0`

If the project tree was rsync'd from another user's machine, the
`ext/spt/dependencies/FRNN/build/temp.linux-x86_64-cpython-310/` directory
may contain pre-built `.o` files compiled against CUDA 11 (with paths under
that user's home, e.g. `/home/baejunhak/...`). When you re-run
`pip install --no-build-isolation ./ext/spt/dependencies/FRNN`, setuptools'
build system sees these existing object files, **skips recompilation**, and
just re-links them into a new wheel. The resulting `_C.so` requires
`libcudart.so.11.0`, which doesn't exist in a cu128 env:
```
ImportError: libcudart.so.11.0: cannot open shared object file
```
Symptoms to spot it:
- `readelf -d .../_C.so | grep RPATH` shows another user's home path
- `nm -D .../_C.so | grep cudart` shows symbols tagged `@libcudart.so.11.0`

Fix: nuke `build/` and `*.egg-info` before every `pip install`, and pass
`--no-cache-dir` so pip can't pull a previously-built wheel either. The
build block in section 3 does both. Same applies to
`submodules/diff-surfel-rasterization` and `submodules/simple-knn` — clean
them too even if they currently look fine on this machine.

### Gotcha #5 — `test_lerf.py` silences stdout (looks like a no-op)

`utils/general_utils.py:112 safe_state(True)` replaces `sys.stdout` with a
class whose `write()` does nothing when `silent=True`. test_lerf.py calls
`safe_state(True)` near the top of `__main__`, so a successful run produces
no stdout — only HF Hub warnings on stderr and an `exit 0`. Don't conclude
that the script didn't do anything. Verify success by:
- checking `output/render/lerf/<scene>/frame_*/*.png` have fresh mtimes, or
- running `scripts/eval_seg.py` and reading the printed metric (that script
  doesn't silence stdout).
