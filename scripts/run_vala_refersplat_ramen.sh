#!/usr/bin/env bash
#SBATCH --job-name=vala-ref-ramen
#SBATCH --output=/mnt/pilab_nas/projects/THGS/external_methods/_vala_refersplat_ramen_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/external_methods/_vala_refersplat_ramen_%j.err
#SBATCH --time=03:00:00

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
VP=$ROOT/envs/vala-port/bin/python
S=$ROOT/data/lerf_ovs/ramen
M=$VALA/output/refersplat_3dgs/lerf_ovs/ramen

cd "$VALA"

export HOME=$ROOT/.valahome
export HF_HOME=$ROOT/.valahome/hf
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1

echo "START $(date)"
echo "HOST $(hostname)"
echo "ROOT $ROOT"
echo "S $S"
echo "M $M"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<slurm-managed>}"

echo "[0] torch/cuda sanity"
"$VP" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

echo "[1] RGB sanity render"
"$VP" render.py \
  -s "$S" \
  -m "$M" \
  --iteration 30000 \
  --eval \
  --skip_train

echo "[2] feature extraction levels 1/2/3"
for L in 1 2 3; do
  echo "[2.$L] feature level $L"
  "$VP" gaussian_feature_extractor.py \
    -s "$S" \
    -m "$M" \
    --iteration 30000 \
    --feature_level "$L" \
    --use_efficient \
    --eval
done

echo "[3] render language masks"
"$VP" -m eval.render_lerf_by_text_langsplat \
  -s "$S" \
  -m "$M" \
  --iteration 30000 \
  --mask_thresh 0.6 \
  --scene_name ramen \
  --dataset_name lerf_ovs \
  --ae_ckpt_dir output/refersplat_3dgs \
  --base_dir "$M" \
  --output_dir "$M" \
  --eval \
  --skip_train

echo "[4] compute IoU"
"$VP" -m eval.compute_lerf_iou \
  --scene_name ramen \
  --gt_dir output/refersplat_3dgs/lerf_ovs \
  --pred_dir output/refersplat_3dgs/lerf_ovs \
  --output_dir output/refersplat_3dgs/lerf_ovs \
  --ablation_type none \
  --mask_thresh 0.6 \
  --iteration 30000 \
  --json_dir "$S/../label"

echo "RESULT"
cat "$M/none/predictions_mask_0.6/result.txt"
echo "DONE $(date)"
