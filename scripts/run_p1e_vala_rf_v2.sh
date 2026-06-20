#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pilab_nas/projects/THGS
source /mnt/pilab_nas/projects/THGS/envs/vala-port/bin/activate

export HOME=/mnt/pilab_nas/projects/THGS/.valahome
export HF_HOME=/mnt/pilab_nas/projects/THGS/.valahome/hf
export MPLCONFIGDIR=/tmp/mpl

python scripts/p1e_vala_rf_v2_pipeline.py

python scripts/p1e_vala_official_2d_prompt_table.py \
  --summary output/diagnostics/p1e_vala_rf_v2_official_2d_summary.csv \
  --detail_csv output/diagnostics/p1e_vala_rf_v2_official_2d_prompt_detail.csv \
  --agg_csv output/diagnostics/p1e_vala_rf_v2_official_2d_prompt_agg.csv

python scripts/p1e_vala_native_2d_oracle.py \
  --summary output/diagnostics/p1e_vala_rf_v2_official_2d_summary.csv \
  --run_root external_methods/VALA/output/rf_v2_robust_vs_mean \
  --case_filter rfv2_ \
  --detail_csv output/diagnostics/p1e_vala_rf_v2_oracle_detail.csv \
  --agg_csv output/diagnostics/p1e_vala_rf_v2_oracle_agg.csv \
  --device cuda \
  --chunk_size 65536

python scripts/p1e_vala_reconcile_oracle_actual.py \
  --oracle_detail output/diagnostics/p1e_vala_rf_v2_oracle_detail.csv \
  --official_detail output/diagnostics/p1e_vala_rf_v2_official_2d_prompt_detail.csv \
  --threshold 0.5 \
  --detail_csv output/diagnostics/p1e_vala_rf_v2_robust_vs_mean_detail.csv \
  --agg_csv output/diagnostics/p1e_vala_rf_v2_robust_vs_mean_agg.csv

python scripts/p1e_vala_rf_v2_finalize.py
