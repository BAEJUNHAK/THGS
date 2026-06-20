#!/usr/bin/env bash
# VALA LERF-OVS 3D reproduction using ReferSplat RGB 3DGS checkpoints.
# Submit with:
#   sbatch -p gpu --reservation="$RID" --gres=gpu:1 scripts/run_vala_refersplat_lerf_ovs_full.sh

#SBATCH --job-name=vala-ref-full
#SBATCH --output=/mnt/pilab_nas/projects/THGS/external_methods/_vala_refersplat_lerf_full_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/external_methods/_vala_refersplat_lerf_full_%j.err
#SBATCH --time=05:00:00

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
VP=$ROOT/envs/vala-port/bin/python
REPRO_ROOT=$ROOT/external_methods/VALA_repro_root
DATA_ROOT=$REPRO_ROOT/dataset/3dgs/lerf_ovs
SRC_DATA=$ROOT/data/lerf_ovs
CKPT_ROOT=$ROOT/external_methods/_refersplat_hf
OUT_ROOT=$VALA/output/refersplat_3dgs_valafeat_full/lerf_ovs
MASK_THRESH=${MASK_THRESH:-0.6}

SCENES=(figurines ramen teatime waldo_kitchen)
declare -A CKPT_FILES=(
  [figurines]=figurineschkpnt30000.pth
  [ramen]=ramenchkpnt30000.pth
  [teatime]=teatimechkpnt30000.pth
  [waldo_kitchen]=kitchenchkpnt30000.pth
)
declare -A FEATURE_REENCODED=()

export HOME=$ROOT/.valahome
export HF_HOME=$ROOT/.valahome/hf
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
export PYTHONPATH=$VALA:${PYTHONPATH:-}

mkdir -p "$HOME" "$HF_HOME" "$MPLCONFIGDIR" "$DATA_ROOT" "$OUT_ROOT"

count_files() {
  local dir=$1
  local pattern=$2
  if [[ -d "$dir" ]]; then
    find -L "$dir" -maxdepth 1 -type f -name "$pattern" | wc -l
  else
    echo 0
  fi
}

feature_cache_valid() {
  local feat_dir=$1
  "$VP" - "$feat_dir" <<'PY'
from pathlib import Path
import sys
import numpy as np

feat_dir = Path(sys.argv[1])
for seg_path in sorted(feat_dir.glob("*_s.npy")):
    feat_path = feat_dir / f"{seg_path.name[:-6]}_f.npy"
    if not feat_path.exists():
        print(f"missing {feat_path}")
        raise SystemExit(1)
    seg = np.load(seg_path)
    feat = np.load(feat_path, mmap_mode="r")
    valid = seg[seg >= 0]
    if valid.size and int(valid.max()) >= int(feat.shape[0]):
        print(
            f"invalid {seg_path.name}: max_seg={int(valid.max())} "
            f"feature_rows={int(feat.shape[0])}"
        )
        raise SystemExit(1)
print(f"valid {feat_dir}")
PY
}

setup_scene_links() {
  local scene=$1
  local target=$DATA_ROOT/$scene
  mkdir -p "$target"
  if [[ ! -e "$target/images" ]]; then
    ln -s "$SRC_DATA/$scene/images" "$target/images"
  fi
  if [[ ! -e "$target/sparse" ]]; then
    ln -s "$SRC_DATA/$scene/sparse" "$target/sparse"
  fi
}

write_cfg_args() {
  local source_path=$1
  local model_path=$2
  "$VP" - "$source_path" "$model_path" <<'PY'
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
print(f"wrote {out}")
PY
}

echo "START $(date)"
echo "HOST $(hostname)"
echo "ROOT $ROOT"
echo "DATA_ROOT $DATA_ROOT"
echo "CKPT_ROOT $CKPT_ROOT"
echo "OUT_ROOT $OUT_ROOT"
echo "MASK_THRESH $MASK_THRESH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<slurm-managed>}"

cd "$VALA"

echo "[0] Python/CUDA sanity"
"$VP" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

echo "[1] Dataset links and VALA/OpenCLIP feature cache"
if [[ ! -e "$DATA_ROOT/label" ]]; then
  ln -s "$SRC_DATA/label" "$DATA_ROOT/label"
fi

for scene in "${SCENES[@]}"; do
  setup_scene_links "$scene"
  test -d "$DATA_ROOT/$scene/images"
  test -d "$DATA_ROOT/$scene/sparse/0"
  test -f "$DATA_ROOT/$scene/sparse/0/test.txt"

  img_count=$(count_files "$DATA_ROOT/$scene/images" '*.jpg')
  feat_dir=$DATA_ROOT/$scene/langsplat/language_features
  npy_count=$(count_files "$feat_dir" '*.npy')
  expected=$((img_count * 2))
  FEATURE_REENCODED[$scene]=0
  echo "scene=$scene images=$img_count feature_npy=$npy_count expected=$expected"
  if [[ "$npy_count" -lt "$expected" ]]; then
    "$VP" "$ROOT/scripts/vala_reencode_langsplat_features.py" \
      --vala-root "$VALA" \
      --scene-root "$DATA_ROOT/$scene" \
      --cached-feature-root "$SRC_DATA/$scene/language_features" \
      --output-root "$feat_dir"
    FEATURE_REENCODED[$scene]=1
  elif ! feature_cache_valid "$feat_dir"; then
    echo "feature cache has non-addressable seg ids; re-encoding with preserved ids for $scene"
    "$VP" "$ROOT/scripts/vala_reencode_langsplat_features.py" \
      --vala-root "$VALA" \
      --scene-root "$DATA_ROOT/$scene" \
      --cached-feature-root "$SRC_DATA/$scene/language_features" \
      --output-root "$feat_dir" \
      --force
    FEATURE_REENCODED[$scene]=1
  fi
done

echo "[2] Prepare ReferSplat RGB checkpoints for VALA"
for scene in "${SCENES[@]}"; do
  S=$DATA_ROOT/$scene
  M=$OUT_ROOT/$scene
  ckpt_file=${CKPT_FILES[$scene]}
  ckpt_src=$CKPT_ROOT/$ckpt_file
  ckpt_dst=$M/chkpnt30000.pth
  ply=$M/point_cloud/iteration_30000/point_cloud.ply

  test -f "$ckpt_src"
  mkdir -p "$M"
  ln -sf "$ckpt_src" "$ckpt_dst"
  write_cfg_args "$S" "$M"

  if [[ ! -f "$ply" ]]; then
    "$VP" "$ROOT/scripts/convert_vala_chkpnt_to_ply.py" \
      --checkpoint "$ckpt_dst" \
      --out "$ply"
  fi

  "$VP" "$ROOT/scripts/check_vala_chkpnt_alignment.py" \
    --vala-root "$VALA" \
    --scene-root "$S" \
    --checkpoint "$ckpt_dst" \
    --max-ratio-delta 0.03
done

echo "[3] VALA feature assignment"
for scene in "${SCENES[@]}"; do
  S=$DATA_ROOT/$scene
  M=$OUT_ROOT/$scene
  for level in 1 2 3; do
    lang_ckpt=$M/none/chkpnt30000_langfeat_${level}_stochastic_gate.pth
    if [[ -f "$lang_ckpt" && "${FEATURE_REENCODED[$scene]:-0}" != "1" ]]; then
      echo "skip gaussian_feature_extractor scene=$scene level=$level"
    else
      echo "gaussian_feature_extractor scene=$scene level=$level"
      "$VP" gaussian_feature_extractor.py \
        -s "$S" \
        -m "$M" \
        --iteration 30000 \
        --feature_level "$level" \
        --use_efficient \
        --eval
    fi
  done
done

echo "[4] Render language masks"
for scene in "${SCENES[@]}"; do
  S=$DATA_ROOT/$scene
  M=$OUT_ROOT/$scene
  echo "render scene=$scene"
  "$VP" -m eval.render_lerf_by_text_langsplat \
    -s "$S" \
    -m "$M" \
    --iteration 30000 \
    --mask_thresh "$MASK_THRESH" \
    --scene_name "$scene" \
    --dataset_name lerf_ovs \
    --ae_ckpt_dir "$OUT_ROOT" \
    --base_dir "$M" \
    --output_dir "$M" \
    --eval \
    --skip_train
done

echo "[5] Compute IoU"
"$VP" -m eval.compute_lerf_iou \
  --gt_dir "$OUT_ROOT" \
  --pred_dir "$OUT_ROOT" \
  --output_dir "$OUT_ROOT" \
  --iteration 30000 \
  --mask_thresh "$MASK_THRESH" \
  --ablation_type none \
  --json_dir "$DATA_ROOT/label"

echo "[6] Paper comparison CSV"
"$VP" - <<'PY'
import csv
import json
from pathlib import Path

mask_thresh = "0.6"
out_root = Path("/mnt/pilab_nas/projects/THGS/external_methods/VALA/output/refersplat_3dgs_valafeat_full/lerf_ovs")
metrics_path = out_root / f"all_metrics_30000_{mask_thresh}.json"
paper = {
    "figurines": {"mIoU": 0.6038, "mAcc": 0.8929},
    "ramen": {"mIoU": 0.4541, "mAcc": 0.6761},
    "teatime": {"mIoU": 0.7061, "mAcc": 0.8814},
    "waldo_kitchen": {"mIoU": 0.5571, "mAcc": 0.8636},
    "mean": {"mIoU": 0.5802, "mAcc": 0.8285},
}
data = json.loads(metrics_path.read_text())
rows = []
for scene in ["figurines", "ramen", "teatime", "waldo_kitchen"]:
    m = data["per_scene"][scene]
    rows.append({
        "scene": scene,
        "repro_mIoU": m["mIoU"],
        "paper_mIoU": paper[scene]["mIoU"],
        "delta_mIoU": m["mIoU"] - paper[scene]["mIoU"],
        "repro_Acc@0.25": m["Acc@0.25"],
        "repro_Acc@0.5": m["Acc@0.5"],
        "paper_mAcc": paper[scene]["mAcc"],
    })
mean = data["mean"]
rows.append({
    "scene": "mean",
    "repro_mIoU": mean["mIoU"],
    "paper_mIoU": paper["mean"]["mIoU"],
    "delta_mIoU": mean["mIoU"] - paper["mean"]["mIoU"],
    "repro_Acc@0.25": mean["Acc@0.25"],
    "repro_Acc@0.5": mean["Acc@0.5"],
    "paper_mAcc": paper["mean"]["mAcc"],
})
csv_path = out_root / "paper_table1_3d_comparison.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(csv_path)
for row in rows:
    print(row)
PY

echo "RESULT FILES"
find "$OUT_ROOT" -name result.txt -print -exec cat {} \;
echo "DONE $(date)"
