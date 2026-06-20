#!/usr/bin/env bash

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VP=$ROOT/envs/vala-port/bin/python
DATA_ROOT=$ROOT/external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs
PRED_ROOT=$ROOT/external_methods/VALA/output/refersplat_3dgs_valafeat/lerf_ovs
SRC_CSV=$ROOT/output/diagnostics/p1e_vala_official_feature_2d_source_oracle.csv
SEM_CSV=$ROOT/output/diagnostics/p1e_vala_official_feature_2d_semantic_selection.csv
MASK_CSV=$ROOT/output/diagnostics/p1e_vala_official_feature_refersplat_mask_diagnostics.csv
LOC_PREFIX=$ROOT/output/diagnostics/p1e_vala_official_feature_refersplat_failure_locus

cd "$ROOT"

"$VP" scripts/p1e_vala_source_diagnostics.py \
  --data_root "$DATA_ROOT" \
  --feature_dir langsplat/language_features \
  --scenes ramen \
  --out_csv "$SRC_CSV"

"$VP" scripts/p1e_vala_2d_semantic_diagnostics.py \
  --data_root "$DATA_ROOT" \
  --feature_dir langsplat/language_features \
  --scenes ramen \
  --out_csv "$SEM_CSV"

"$VP" scripts/p1e_vala_mask_diagnostics.py \
  --data_root "$ROOT/data/lerf_ovs" \
  --pred_root "$PRED_ROOT" \
  --scenes ramen \
  --source_oracle_csv "$SRC_CSV" \
  --out_csv "$MASK_CSV"

"$VP" scripts/p1e_vala_failure_locus.py \
  --mask_csv "$MASK_CSV" \
  --semantic_csv "$SEM_CSV" \
  --out_prefix "$LOC_PREFIX"
