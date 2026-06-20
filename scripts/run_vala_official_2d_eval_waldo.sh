#!/usr/bin/env bash
#SBATCH --job-name=vala-waldo-2d-official
#SBATCH --output=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/vala_waldo_2d_official_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/vala_waldo_2d_official_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
SRC=$ROOT/external_methods/VALA_officialsam_waldo_runs/76/dataset/3dgs/lerf_ovs/waldo_kitchen
MODEL=$VALA/output/official_train_officialsam_waldo/lerf_ovs/waldo_kitchen
RUN_ROOT=$VALA/output/official_2d_eval_waldo_${SLURM_JOB_ID:-manual}
FEAT_ROOT=$RUN_ROOT/feat_dir
EVAL_ROOT=$RUN_ROOT/eval

source "$ROOT/envs/vala-port/bin/activate"
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

mkdir -p "$RUN_ROOT" "$FEAT_ROOT" "$EVAL_ROOT" "$MPLCONFIGDIR"

echo "START $(date)"
echo "HOST $(hostname)"
echo "SRC=$SRC"
echo "MODEL=$MODEL"
echo "RUN_ROOT=$RUN_ROOT"

cd "$VALA"

echo "[1] Render official 2D feature maps with VALA feature_map_renderer.py"
for level in 1 2 3; do
  out_dir="$MODEL/test/ours_30000_langfeat_${level}_stochastic_gate/renders_npy"
  if [[ -f "$out_dir/frame_00053.npy" && -f "$out_dir/frame_00154.npy" && "${FORCE_RENDER:-0}" != "1" ]]; then
    echo "  level $level already rendered: $out_dir"
  else
    VALA_ROOT="$VALA" python "$ROOT/scripts/p1e_run_vala_feature_map_renderer_safe.py" \
      -s "$SRC" \
      -m "$MODEL" \
      --iteration 30000 \
      --feature_level "$level" \
      --eval \
      --skip_train
  fi
done

echo "[2] Adapt official renderer output path to official evaluate_iou_loc.py input convention"
for level in 1 2 3; do
  src_dir="$MODEL/test/ours_30000_langfeat_${level}_stochastic_gate/renders_npy"
  dst_dir="$FEAT_ROOT/waldo_kitchen_${level}/train/ours_None/renders_npy"
  mkdir -p "$dst_dir"

  first="$src_dir/frame_00053.npy"
  test -f "$first"

  # evaluate_iou_loc.py sorts numeric filenames and loads frame_number - 1.
  # Unused indices only need valid placeholders; GT indices below point to actual rendered frames.
  for idx in $(seq 0 186); do
    ln -sfn "$first" "$dst_dir/${idx}.npy"
  done
  ln -sfn "$src_dir/frame_00053.npy" "$dst_dir/52.npy"
  ln -sfn "$src_dir/frame_00066.npy" "$dst_dir/65.npy"
  ln -sfn "$src_dir/frame_00089.npy" "$dst_dir/88.npy"
  ln -sfn "$src_dir/frame_00140.npy" "$dst_dir/139.npy"
  ln -sfn "$src_dir/frame_00154.npy" "$dst_dir/153.npy"

  echo "  level $level numeric files: $(find "$dst_dir" -maxdepth 1 -name '*.npy' | wc -l)"
done

echo "[3] Run official eval/evaluate_iou_loc.py with paper threshold 0.5"
python eval/evaluate_iou_loc.py \
  --dataset_name waldo_kitchen \
  --feat_dir "$FEAT_ROOT" \
  --ae_ckpt_dir "$RUN_ROOT/dummy_ae" \
  --output_dir "$EVAL_ROOT/thresh_0.5" \
  --json_folder "$ROOT/data/lerf_ovs/label" \
  --mask_thresh 0.5 \
  --direct_512

echo "[4] Run official eval/evaluate_iou_loc.py with public-script default threshold 0.4"
python eval/evaluate_iou_loc.py \
  --dataset_name waldo_kitchen \
  --feat_dir "$FEAT_ROOT" \
  --ae_ckpt_dir "$RUN_ROOT/dummy_ae" \
  --output_dir "$EVAL_ROOT/thresh_0.4" \
  --json_folder "$ROOT/data/lerf_ovs/label" \
  --mask_thresh 0.4 \
  --direct_512

echo "[5] Extract logs"
find "$EVAL_ROOT" -type f -name '*.log' -print -exec tail -20 {} \;
echo "DONE $(date)"
