#!/usr/bin/env bash
#SBATCH --job-name=vala-ref-sweep
#SBATCH --output=/mnt/pilab_nas/projects/THGS/external_methods/_vala_refersplat_ramen_sweep_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/external_methods/_vala_refersplat_ramen_sweep_%j.err
#SBATCH --time=01:00:00

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

for T in 0.4 0.5 0.6 0.7 0.8; do
  echo "[threshold $T] render"
  "$VP" -m eval.render_lerf_by_text_langsplat \
    -s "$S" \
    -m "$M" \
    --iteration 30000 \
    --mask_thresh "$T" \
    --scene_name ramen \
    --dataset_name lerf_ovs \
    --ae_ckpt_dir output/refersplat_3dgs \
    --base_dir "$M" \
    --output_dir "$M" \
    --eval \
    --skip_train

  echo "[threshold $T] iou"
  "$VP" -m eval.compute_lerf_iou \
    --scene_name ramen \
    --gt_dir output/refersplat_3dgs/lerf_ovs \
    --pred_dir output/refersplat_3dgs/lerf_ovs \
    --output_dir output/refersplat_3dgs/lerf_ovs \
    --ablation_type none \
    --mask_thresh "$T" \
    --iteration 30000 \
    --json_dir "$S/../label"

  echo "[threshold $T] result"
  cat "$M/none/predictions_mask_$T/result.txt"
done

echo "DONE $(date)"
