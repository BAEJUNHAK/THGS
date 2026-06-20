#!/usr/bin/env bash
# Reproduce VALA LERF-OVS 3D evaluation for all four scenes.
# Submit with:
#   sbatch -p gpu --reservation="$RID" --gres=gpu:1 scripts/run_vala_lerf_ovs_full_repro.sh

#SBATCH --job-name=vala-lerf-full
#SBATCH --output=/mnt/pilab_nas/projects/THGS/external_methods/_vala_lerf_full_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/external_methods/_vala_lerf_full_%j.err
#SBATCH --time=07:30:00

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
VP=$ROOT/envs/vala-port/bin/python
REPRO_ROOT=$ROOT/external_methods/VALA_repro_root
DATA_ROOT=$REPRO_ROOT/dataset/3dgs/lerf_ovs
SRC_DATA=$ROOT/data/lerf_ovs
OUT_ROOT=$VALA/output/official_lerf_ovs_repro
SAM=$ROOT/ckpts/sam_vit_h_4b8939.pth
MASK_THRESH=${MASK_THRESH:-0.6}

SCENES=(figurines teatime ramen waldo_kitchen)

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

echo "START $(date)"
echo "HOST $(hostname)"
echo "ROOT $ROOT"
echo "VALA $VALA"
echo "REPRO_ROOT $REPRO_ROOT"
echo "DATA_ROOT $DATA_ROOT"
echo "OUT_ROOT $OUT_ROOT"
echo "MASK_THRESH $MASK_THRESH"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<slurm-managed>}"

cd "$VALA"

echo "[0] Python/CUDA sanity"
"$VP" - <<'PY'
import torch, segment_anything
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
print("segment_anything", segment_anything.__file__)
PY

echo "[1] Dataset symlink setup"
if [[ ! -e "$DATA_ROOT/label" ]]; then
  ln -s "$SRC_DATA/label" "$DATA_ROOT/label"
fi
for scene in "${SCENES[@]}"; do
  setup_scene_links "$scene"
  test -d "$DATA_ROOT/$scene/images"
  test -d "$DATA_ROOT/$scene/sparse/0"
  test -f "$DATA_ROOT/$scene/sparse/0/test.txt"
  echo "$scene images $(count_files "$DATA_ROOT/$scene/images" '*.jpg')"
done

echo "[2] VALA/LangSplat feature generation"
for scene in "${SCENES[@]}"; do
  img_count=$(count_files "$DATA_ROOT/$scene/images" '*.jpg')
  feat_dir=$DATA_ROOT/$scene/langsplat/language_features
  npy_count=$(count_files "$feat_dir" '*.npy')
  expected=$((img_count * 2))
  echo "scene=$scene image_count=$img_count feature_npy=$npy_count expected=$expected"
  if [[ "$npy_count" -ge "$expected" ]]; then
    echo "skip VALA OpenCLIP re-encoding for $scene"
  else
    "$VP" "$ROOT/scripts/vala_reencode_langsplat_features.py" \
      --vala-root "$VALA" \
      --scene-root "$DATA_ROOT/$scene" \
      --cached-feature-root "$SRC_DATA/$scene/language_features" \
      --output-root "$feat_dir"
  fi
done

echo "[3] 3DGS training and VALA feature assignment"
for scene in "${SCENES[@]}"; do
  S=$DATA_ROOT/$scene
  M=$OUT_ROOT/$scene
  mkdir -p "$M"

  if [[ -f "$M/chkpnt30000.pth" && -f "$M/point_cloud/iteration_30000/point_cloud.ply" ]]; then
    echo "skip train.py for $scene"
  else
    echo "train.py for $scene"
    "$VP" train.py -s "$S" -m "$M" --iterations 30000
  fi

  for level in 1 2 3; do
    ckpt=$M/none/chkpnt30000_langfeat_${level}_stochastic_gate.pth
    if [[ -f "$ckpt" ]]; then
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

echo "[4] Render masks"
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

out_root = Path("/mnt/pilab_nas/projects/THGS/external_methods/VALA/output/official_lerf_ovs_repro")
metrics_path = out_root / "all_metrics_30000_0.6.json"
paper = {
    "figurines": {"mIoU": 0.6038, "mAcc": 0.8929},
    "ramen": {"mIoU": 0.4541, "mAcc": 0.6761},
    "teatime": {"mIoU": 0.7061, "mAcc": 0.8814},
    "waldo_kitchen": {"mIoU": 0.5571, "mAcc": 0.8636},
    "mean": {"mIoU": 0.5802, "mAcc": 0.8285},
}
data = json.loads(metrics_path.read_text())
rows = []
for scene, m in data["per_scene"].items():
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
