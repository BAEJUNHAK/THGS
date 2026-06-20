#!/usr/bin/env bash
# Waldo-only VALA reproduction with official run_sam.py LangSplat features.
# Submit with:
#   sbatch -p gpu --reservation="$RID" --gres=gpu:1 scripts/run_vala_official_sam_waldo.sh

#SBATCH --job-name=vala-sam-waldo
#SBATCH --output=/mnt/pilab_nas/projects/THGS/external_methods/_vala_official_sam_waldo_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/external_methods/_vala_official_sam_waldo_%j.err
#SBATCH --time=06:00:00

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
VP=$ROOT/envs/vala-port/bin/python
SRC_DATA=$ROOT/data/lerf_ovs
CKPT_ROOT=$ROOT/external_methods/_refersplat_hf
SCENE=waldo_kitchen
MASK_THRESH=${MASK_THRESH:-0.6}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}

RUN_ROOT=$ROOT/external_methods/VALA_officialsam_waldo_runs/$RUN_TAG
DATA_ROOT=$RUN_ROOT/dataset/3dgs/lerf_ovs
S=$DATA_ROOT/$SCENE
OUT_ROOT=$VALA/output/refersplat_3dgs_officialsam_waldo_$RUN_TAG/lerf_ovs
M=$OUT_ROOT/$SCENE

export HOME=$ROOT/.valahome
export HF_HOME=$ROOT/.valahome/hf
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
export PYTHONPATH=$VALA:${PYTHONPATH:-}

mkdir -p "$HOME" "$HF_HOME" "$MPLCONFIGDIR" "$S" "$OUT_ROOT" "$M"

if [[ ! -e "$DATA_ROOT/label" ]]; then
  ln -s "$SRC_DATA/label" "$DATA_ROOT/label"
fi
if [[ ! -e "$S/images" ]]; then
  ln -s "$SRC_DATA/$SCENE/images" "$S/images"
fi
if [[ ! -e "$S/sparse" ]]; then
  ln -s "$SRC_DATA/$SCENE/sparse" "$S/sparse"
fi

echo "START $(date)"
echo "HOST $(hostname)"
echo "RUN_TAG $RUN_TAG"
echo "RUN_ROOT $RUN_ROOT"
echo "DATA_ROOT $DATA_ROOT"
echo "OUT_ROOT $OUT_ROOT"
echo "MASK_THRESH $MASK_THRESH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<slurm-managed>}"

cd "$VALA"

echo "[0] Python/CUDA/SAM sanity"
"$VP" - <<'PY'
import inspect
import torch
from segment_anything import SamAutomaticMaskGenerator

src = inspect.getsource(SamAutomaticMaskGenerator.generate)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
print("langsplat_tuple_generate", "curr_anns_s" in src and "return curr_anns" in src)
PY

echo "[1] Official VALA run_sam.py for Waldo LangSplat features"
"$VP" run_sam.py \
  --root_dir "$RUN_ROOT" \
  --dataset_name lerf_ovs \
  --rep 3dgs \
  --scene "$SCENE" \
  --sam_checkpoint "$ROOT/ckpts/sam_vit_h_4b8939.pth" \
  --get_semantic \
  --use_langsplat

echo "[2] Validate official language feature cache"
"$VP" - "$S/langsplat/language_features" <<'PY'
from pathlib import Path
import sys
import numpy as np

feat_dir = Path(sys.argv[1])
seg_paths = sorted(feat_dir.glob("*_s.npy"))
feat_paths = sorted(feat_dir.glob("*_f.npy"))
print("seg_count", len(seg_paths), "feat_count", len(feat_paths))
if not seg_paths:
    raise SystemExit("no semantic files generated")
for seg_path in seg_paths:
    feat_path = feat_dir / f"{seg_path.name[:-6]}_f.npy"
    if not feat_path.exists():
        raise SystemExit(f"missing {feat_path}")
    seg = np.load(seg_path)
    feat = np.load(feat_path, mmap_mode="r")
    valid = seg[seg >= 0]
    if valid.size and int(valid.max()) >= int(feat.shape[0]):
        raise SystemExit(
            f"invalid {seg_path.name}: max_seg={int(valid.max())} rows={int(feat.shape[0])}"
        )
print("feature_cache_valid", feat_dir)
PY

echo "[3] Prepare ReferSplat RGB checkpoint"
ln -sf "$CKPT_ROOT/kitchenchkpnt30000.pth" "$M/chkpnt30000.pth"

"$VP" - "$S" "$M" <<'PY'
from pathlib import Path
import sys

source_path = sys.argv[1]
model_path = sys.argv[2]
cfg = (
    "Namespace("
    "sh_degree=3, "
    f"source_path='{source_path}', "
    f"model_path='{model_path}', "
    "images='images', "
    "depths='', "
    "resolution=-1, "
    "white_background=False, "
    "train_test_exp=False, "
    "data_device='cuda', "
    "eval=False, "
    "language_features_name='language_features', "
    "feature_level=2, "
    f"lf_path='{source_path}/langsplat/language_features'"
    ")"
)
out = Path(model_path) / "cfg_args"
out.write_text(cfg)
print("wrote", out)
PY

if [[ ! -f "$M/point_cloud/iteration_30000/point_cloud.ply" ]]; then
  "$VP" "$ROOT/scripts/convert_vala_chkpnt_to_ply.py" \
    --checkpoint "$M/chkpnt30000.pth" \
    --out "$M/point_cloud/iteration_30000/point_cloud.ply"
fi

"$VP" "$ROOT/scripts/check_vala_chkpnt_alignment.py" \
  --vala-root "$VALA" \
  --scene-root "$S" \
  --checkpoint "$M/chkpnt30000.pth" \
  --max-ratio-delta 0.03

echo "[4] VALA feature assignment from official SAM features"
for level in 1 2 3; do
  "$VP" gaussian_feature_extractor.py \
    -s "$S" \
    -m "$M" \
    --iteration 30000 \
    --feature_level "$level" \
    --use_efficient \
    --eval
done

echo "[5] Render Waldo language masks"
"$VP" -m eval.render_lerf_by_text_langsplat \
  -s "$S" \
  -m "$M" \
  --iteration 30000 \
  --mask_thresh "$MASK_THRESH" \
  --scene_name "$SCENE" \
  --dataset_name lerf_ovs \
  --ae_ckpt_dir "$OUT_ROOT" \
  --base_dir "$M" \
  --output_dir "$M" \
  --eval \
  --skip_train

echo "[6] Compute Waldo IoU"
"$VP" -m eval.compute_lerf_iou \
  --scene_name "$SCENE" \
  --gt_dir "$OUT_ROOT" \
  --pred_dir "$OUT_ROOT" \
  --output_dir "$OUT_ROOT" \
  --iteration 30000 \
  --mask_thresh "$MASK_THRESH" \
  --ablation_type none \
  --json_dir "$DATA_ROOT/label"

echo "[7] Result"
find "$OUT_ROOT" -name result.txt -print -exec cat {} \;
echo "DONE $(date)"
