#!/usr/bin/env bash
#SBATCH --job-name=vala-waldo-thresh
#SBATCH --output=output/diagnostics/logs/vala_waldo_thresh_%j.log
#SBATCH --error=output/diagnostics/logs/vala_waldo_thresh_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA="$ROOT/external_methods/VALA"
SRC="$ROOT/external_methods/VALA_officialsam_waldo_runs/76/dataset/3dgs/lerf_ovs/waldo_kitchen"
MODEL_ROOT="$VALA/output/official_train_officialsam_waldo/lerf_ovs"
MODEL="$MODEL_ROOT/waldo_kitchen"

source "$ROOT/envs/vala-port/bin/activate"
export HOME="$ROOT/.cache_home"
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$ROOT/.cache/huggingface/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export PYTHONPATH="$VALA:$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$VALA"

echo "START $(date)"
echo "HOST $(hostname)"
echo "MODEL=$MODEL"

for thresh in 0.35 0.4 0.45 0.5 0.55 0.6; do
  echo "[threshold=$thresh] render"
  python -m eval.render_lerf_by_text_langsplat \
    -s "$SRC" \
    -m "$MODEL" \
    --iteration 30000 \
    --mask_thresh "$thresh" \
    --scene_name waldo_kitchen \
    --dataset_name lerf_ovs \
    --ae_ckpt_dir "$MODEL_ROOT" \
    --base_dir "$MODEL" \
    --output_dir "$MODEL" \
    --eval \
    --skip_train

  echo "[threshold=$thresh] iou"
  python -m eval.compute_lerf_iou \
    --scene_name waldo_kitchen \
    --gt_dir "$MODEL_ROOT" \
    --pred_dir "$MODEL_ROOT" \
    --output_dir "$MODEL_ROOT" \
    --iteration 30000 \
    --mask_thresh "$thresh" \
    --ablation_type none \
    --json_dir "$ROOT/data/lerf_ovs/label"

  result="$MODEL_ROOT/all_metrics_30000_${thresh}.json"
  echo "[threshold=$thresh] result=$result"
  cat "$result"
done

echo "DONE $(date)"
