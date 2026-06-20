#!/usr/bin/env bash
# Decisive Waldo reproduction test:
#   official VALA RGB training -> official VALA feature assignment -> 3D eval -> in-memory 2D eval.
# Submit with:
#   sbatch -p gpu --reservation="$RID" --gres=gpu:1 scripts/run_vala_official_train_waldo_decision.sh

#SBATCH --job-name=vala-waldo-official
#SBATCH --output=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/vala_waldo_official_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/vala_waldo_official_%j.err
#SBATCH --time=12:00:00

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
SRC=$ROOT/external_methods/VALA_officialsam_waldo_runs/76/dataset/3dgs/lerf_ovs/waldo_kitchen
MODEL_ROOT=$VALA/output/official_train_officialsam_waldo/lerf_ovs
MODEL=$MODEL_ROOT/waldo_kitchen
OUT_JSON=$ROOT/output/diagnostics/vala_2d_eval/officialtrain_officialsam_waldo_kitchen.json
OUT_CSV=$ROOT/output/diagnostics/vala_2d_eval/officialtrain_officialsam_waldo_kitchen.csv
MASK_THRESH=${MASK_THRESH:-0.6}

cd "$ROOT"
source "$ROOT/envs/vala-port/bin/activate"

export HOME="$ROOT/.valahome"
export XDG_CACHE_HOME="$HOME/cache"
export HF_HOME="$HOME/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$HOME/torch"
export PYTHONPATH="$VALA:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$ROOT/output/diagnostics/matplotlib_cache"
export PYTHONUNBUFFERED=1

mkdir -p "$MODEL_ROOT" "$MODEL" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"

echo "START $(date)"
echo "HOST $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<slurm-managed>}"
echo "SRC=$SRC"
echo "MODEL=$MODEL"
echo "MASK_THRESH=$MASK_THRESH"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

if [[ ! -d "$SRC/langsplat/language_features" ]]; then
  echo "Missing official SAM/LangSplat features: $SRC/langsplat/language_features" >&2
  exit 2
fi

cd "$VALA"

if [[ ! -f "$MODEL/chkpnt30000.pth" ]]; then
  echo "[1] Official VALA RGB training from scratch"
  python train.py \
    -s "$SRC" \
    -m "$MODEL" \
    --iterations 30000
else
  echo "[1] Reusing existing RGB checkpoint: $MODEL/chkpnt30000.pth"
fi

if [[ ! -f "$MODEL/point_cloud/iteration_30000/point_cloud.ply" ]]; then
  echo "Missing trained point cloud after RGB training" >&2
  exit 3
fi

echo "[2] Official VALA language feature assignment"
for level in 1 2 3; do
  ckpt="$MODEL/none/chkpnt30000_langfeat_${level}_stochastic_gate.pth"
  if [[ -f "$ckpt" && "${FORCE_FEATURES:-0}" != "1" ]]; then
    echo "  level $level already exists: $ckpt"
  else
    python gaussian_feature_extractor.py \
      -s "$SRC" \
      -m "$MODEL" \
      --iteration 30000 \
      --feature_level "$level" \
      --use_efficient \
      --eval
  fi
done

echo "[3] VALA official 3D selection/render/eval"
python -m eval.render_lerf_by_text_langsplat \
  -s "$SRC" \
  -m "$MODEL" \
  --iteration 30000 \
  --mask_thresh "$MASK_THRESH" \
  --scene_name waldo_kitchen \
  --dataset_name lerf_ovs \
  --ae_ckpt_dir "$MODEL_ROOT" \
  --base_dir "$MODEL" \
  --output_dir "$MODEL" \
  --eval \
  --skip_train

python -m eval.compute_lerf_iou \
  --scene_name waldo_kitchen \
  --gt_dir "$MODEL_ROOT" \
  --pred_dir "$MODEL_ROOT" \
  --output_dir "$MODEL_ROOT" \
  --iteration 30000 \
  --mask_thresh "$MASK_THRESH" \
  --ablation_type none \
  --json_dir "$ROOT/data/lerf_ovs/label"

echo "[4] VALA 2D in-memory eval on official-trained checkpoint"
python "$ROOT/scripts/p1e_vala_2d_eval_inmemory.py" \
  -s "$SRC" \
  -m "$MODEL" \
  --iteration 30000 \
  --eval \
  --quiet \
  --scene_name waldo_kitchen \
  --json_dir "$ROOT/data/lerf_ovs/label" \
  --thresholds 0.4 0.5 \
  --out_csv "$OUT_CSV" \
  --out_json "$OUT_JSON"

echo "[5] Results"
find "$MODEL_ROOT" -maxdepth 4 -name all_metrics_30000_${MASK_THRESH}.json -print -exec cat {} \;
cat "$OUT_JSON"
echo "DONE $(date)"
