#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
cd "$ROOT"
source "$ROOT/envs/vala-port/bin/activate"

export HOME="$ROOT/.valahome"
export XDG_CACHE_HOME="$HOME/cache"
export HF_HOME="$HOME/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$HOME/torch"
export PYTHONPATH="$ROOT/external_methods/VALA:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$ROOT/output/diagnostics/matplotlib_cache"

OUT_DIR="$ROOT/output/diagnostics/vala_2d_eval"
mkdir -p "$OUT_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME"

run_scene() {
  local scene="$1"
  local source_path="$2"
  local model_path="$3"
  local tag="$4"

  echo "=== VALA 2D eval: ${tag}/${scene} ==="
  python "$ROOT/scripts/p1e_vala_2d_eval_inmemory.py" \
    -s "$source_path" \
    -m "$model_path" \
    --iteration 30000 \
    --eval \
    --quiet \
    --scene_name "$scene" \
    --json_dir "$ROOT/data/lerf_ovs/label" \
    --thresholds 0.4 0.5 \
    --out_csv "$OUT_DIR/${tag}_${scene}.csv" \
    --out_json "$OUT_DIR/${tag}_${scene}.json"
}

run_scene \
  waldo_kitchen \
  "$ROOT/external_methods/VALA_officialsam_waldo_runs/76/dataset/3dgs/lerf_ovs/waldo_kitchen" \
  "$ROOT/external_methods/VALA/output/refersplat_3dgs_officialsam_waldo_76/lerf_ovs/waldo_kitchen" \
  "officialsam_refersplat"

run_scene \
  ramen \
  "$ROOT/external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs/ramen" \
  "$ROOT/external_methods/VALA/output/refersplat_3dgs_valafeat/lerf_ovs/ramen" \
  "officialfeat_refersplat"
