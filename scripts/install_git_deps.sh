#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Build the 3 external-GitHub dependencies that THGS needs (declared in
# environment.yml). The Claude Code sandbox is NOT allowed to install code
# from arbitrary GitHub repos, so you must run this yourself, e.g.:
#
#     ! bash scripts/install_git_deps.sh
#
# It is safe to run while the scene download is still going (CPU vs network).
# GPU is NOT required for these builds (only nvcc + gcc-11, already installed).
# -------------------------------------------------------------------------
set -u

source ~/miniforge3/etc/profile.d/conda.sh
conda activate thgs
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
echo "repo: $(pwd)   python: $(which python)"

export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.9"        # RTX 4060 Ti (Ada)
export FORCE_CUDA=1
export MAX_JOBS=8
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

run() { echo; echo "=============== $1 ==============="; shift; "$@"; echo "EXIT=$?"; }

run "1/3 segment-anything-langsplat" \
    pip install --no-build-isolation "git+https://github.com/minghanqin/segment-anything-langsplat.git"

run "2/3 point_geometric_features (pgeof)" \
    pip install --no-build-isolation "git+https://github.com/drprojects/point_geometric_features.git"

# pytorch3d is the slow one (~10-20 min of nvcc compilation).
# If main fails to build against torch 2.2 / CUDA 11.8, retry pinned to a tag:
#   pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.5"
run "3/3 pytorch3d (slow)" \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

echo; echo "All git-dependency builds attempted. Check EXIT codes above."
