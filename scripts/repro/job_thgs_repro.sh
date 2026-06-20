#!/bin/bash
#SBATCH --job-name=repro-thgs
#SBATCH --output=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/repro_thgs_%j.log
#SBATCH --time=02:00:00
set -e
cd /mnt/pilab_nas/projects/THGS
source /mnt/pilab_nas/projects/THGS/envs/thgs/bin/activate
export PYTHONPATH=/mnt/pilab_nas/projects/THGS:$PYTHONPATH
export HOME=/mnt/pilab_nas/projects/THGS/.valahome
export HF_HOME=$HOME/hf; export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=$HOME/cache; export MPLCONFIGDIR=$HOME/mpl; export HF_HUB_OFFLINE=1
echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PRED=output/render/repro/thgs
for sc in figurines ramen teatime waldo_kitchen; do
  echo "=== render THGS ablate: $sc ==="
  python scripts/repro/render_thgs_ablate.py \
      -s data/lerf_ovs/$sc -m output/lerf/$sc --path_pred $PRED
done

echo "=== CPU eval (threshold x aggregation sweep) ==="
python scripts/repro/eval_thgs_ablate.py --pred $PRED --gt data/lerf_ovs/label \
    --csv output/diagnostics/repro_thgs_ablation.csv
echo "=== repro-thgs DONE $(date) ==="
