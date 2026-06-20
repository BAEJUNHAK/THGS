#!/usr/bin/env bash
#SBATCH --job-name=vala-offfeat-ramen
#SBATCH --output=/mnt/pilab_nas/projects/THGS/external_methods/_vala_official_feature_refersplat_ramen_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/external_methods/_vala_official_feature_refersplat_ramen_%j.err
#SBATCH --time=08:00:00

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
VP=$ROOT/envs/vala-port/bin/python
REPRO_ROOT=$ROOT/external_methods/VALA_repro_root
S=$REPRO_ROOT/dataset/3dgs/lerf_ovs/ramen
M=$VALA/output/refersplat_3dgs_valafeat/lerf_ovs/ramen
SAM=$ROOT/ckpts/sam_vit_h_4b8939.pth

export HOME=$ROOT/.valahome
export HF_HOME=$ROOT/.valahome/hf
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1

cd "$VALA"

echo "START $(date)"
echo "HOST $(hostname)"
echo "ROOT $ROOT"
echo "REPRO_ROOT $REPRO_ROOT"
echo "S $S"
echo "M $M"
echo "SAM $SAM"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<slurm-managed>}"

echo "[0] torch/cuda/import sanity"
"$VP" - <<'PY'
import segment_anything, torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
print("segment_anything", segment_anything.__file__)
PY

echo "[1] source/model path sanity"
test -d "$S/images"
test -d "$S/sparse/0"
test -f "$S/sparse/0/test.txt"
test -f "$M/chkpnt30000.pth"
test -f "$M/point_cloud/iteration_30000/point_cloud.ply"
test -f "$SAM"
find -L "$S/images" -maxdepth 1 -name '*.jpg' | wc -l
find -L "$S/langsplat/language_features" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l || true

echo "[2] regenerate VALA/LangSplat language_features"
"$VP" run_sam.py \
  --root_dir "$REPRO_ROOT" \
  --dataset_name lerf_ovs \
  --rep 3dgs \
  --scene ramen \
  --sam_checkpoint "$SAM" \
  --get_semantic \
  --use_langsplat

echo "[3] generated feature count"
find -L "$S/langsplat/language_features" -maxdepth 1 -name '*.npy' | wc -l
"$VP" - <<'PY'
from pathlib import Path
import numpy as np
root = Path("/mnt/pilab_nas/projects/THGS/external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs/ramen/langsplat/language_features")
for stem in ["frame_00006", "frame_00024", "frame_00128"]:
    f = np.load(root / f"{stem}_f.npy")
    s = np.load(root / f"{stem}_s.npy")
    print(stem, "feat", f.shape, f.dtype, "seg", s.shape, s.dtype, "levels", s.shape[0], "minmax", int(s.min()), int(s.max()))
PY

echo "[4] feature extraction levels 1/2/3 using official regenerated features"
for L in 1 2 3; do
  echo "[4.$L] feature level $L"
  "$VP" gaussian_feature_extractor.py \
    -s "$S" \
    -m "$M" \
    --iteration 30000 \
    --feature_level "$L" \
    --use_efficient \
    --eval
done

echo "[5] render language masks"
"$VP" -m eval.render_lerf_by_text_langsplat \
  -s "$S" \
  -m "$M" \
  --iteration 30000 \
  --mask_thresh 0.6 \
  --scene_name ramen \
  --dataset_name lerf_ovs \
  --ae_ckpt_dir output/refersplat_3dgs_valafeat \
  --base_dir "$M" \
  --output_dir "$M" \
  --eval \
  --skip_train

echo "[6] compute IoU"
"$VP" -m eval.compute_lerf_iou \
  --scene_name ramen \
  --gt_dir output/refersplat_3dgs_valafeat/lerf_ovs \
  --pred_dir output/refersplat_3dgs_valafeat/lerf_ovs \
  --output_dir output/refersplat_3dgs_valafeat/lerf_ovs \
  --ablation_type none \
  --mask_thresh 0.6 \
  --iteration 30000 \
  --json_dir "$ROOT/data/lerf_ovs/label"

echo "RESULT"
cat "$M/none/predictions_mask_0.6/result.txt"
echo "DONE $(date)"
