#!/usr/bin/env bash
#SBATCH --job-name=vala-2d-all
#SBATCH --output=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/vala_2d_all_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/vala_2d_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA

source "$ROOT/envs/vala-port/bin/activate"
export THGS_ROOT="$ROOT"
export HOME="$ROOT/.valahome"
export XDG_CACHE_HOME="$HOME/cache"
export HF_HOME="$HOME/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$HOME/torch"
export MPLCONFIGDIR="$ROOT/output/diagnostics/matplotlib_cache"
export PYTHONPATH="$VALA:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/output/diagnostics/logs" "$MPLCONFIGDIR"

echo "START $(date)"
echo "HOST $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "FORCE_RENDER=${FORCE_RENDER:-0}"

cd "$ROOT"
python scripts/p1e_vala_official_2d_eval_all.py

echo "DONE $(date)"
