#!/usr/bin/env bash
#SBATCH --job-name=vala-offfeat-sweep
#SBATCH --output=/mnt/pilab_nas/projects/THGS/external_methods/_vala_official_feature_refersplat_ramen_sweep_%j.log
#SBATCH --error=/mnt/pilab_nas/projects/THGS/external_methods/_vala_official_feature_refersplat_ramen_sweep_%j.err
#SBATCH --time=01:30:00

set -euo pipefail

ROOT=/mnt/pilab_nas/projects/THGS
VALA=$ROOT/external_methods/VALA
VP=$ROOT/envs/vala-port/bin/python
REPRO_ROOT=$ROOT/external_methods/VALA_repro_root
S=$REPRO_ROOT/dataset/3dgs/lerf_ovs/ramen
M=$VALA/output/refersplat_3dgs_valafeat/lerf_ovs/ramen
OUTCSV=$ROOT/output/diagnostics/p1e_vala_official_feature_refersplat_threshold_sweep.csv

export HOME=$ROOT/.valahome
export HF_HOME=$ROOT/.valahome/hf
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1

cd "$VALA"

echo "START $(date)"
echo "HOST $(hostname)"
echo "S $S"
echo "M $M"
echo "mask_thresh,mIoU,Acc@0.25,Acc@0.5" > "$OUTCSV"

for T in 0.4 0.5 0.6 0.7 0.8; do
  echo "[sweep] threshold $T"
  "$VP" -m eval.render_lerf_by_text_langsplat \
    -s "$S" \
    -m "$M" \
    --iteration 30000 \
    --mask_thresh "$T" \
    --scene_name ramen \
    --dataset_name lerf_ovs \
    --ae_ckpt_dir output/refersplat_3dgs_valafeat \
    --base_dir "$M" \
    --output_dir "$M" \
    --eval \
    --skip_train

  "$VP" -m eval.compute_lerf_iou \
    --scene_name ramen \
    --gt_dir output/refersplat_3dgs_valafeat/lerf_ovs \
    --pred_dir output/refersplat_3dgs_valafeat/lerf_ovs \
    --output_dir output/refersplat_3dgs_valafeat/lerf_ovs \
    --ablation_type none \
    --mask_thresh "$T" \
    --iteration 30000 \
    --json_dir "$ROOT/data/lerf_ovs/label"

  RESULT="$M/none/predictions_mask_${T}/result.txt"
  "$VP" - "$T" "$RESULT" "$OUTCSV" <<'PY'
import re
import sys
from pathlib import Path

t, result_path, out_csv = sys.argv[1:]
text = Path(result_path).read_text()
miou = re.search(r"mean iou:\s*([0-9.]+)", text)
acc25 = re.search(r"Acc@0\.25:\s*([0-9.]+)", text)
acc50 = re.search(r"Acc@0\.5:\s*([0-9.]+)", text)
row = [t, miou.group(1), acc25.group(1), acc50.group(1)]
with Path(out_csv).open("a") as f:
    f.write(",".join(row) + "\n")
print(",".join(row))
PY
done

cat "$OUTCSV"
echo "DONE $(date)"
