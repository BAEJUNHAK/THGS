#!/bin/bash
#SBATCH --job-name=repro-relags
#SBATCH --output=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/repro_relags_%j.log
#SBATCH --time=02:00:00
set -e
ROOT=/mnt/pilab_nas/projects/THGS
cd $ROOT
source $ROOT/envs/thgs/bin/activate
export HOME=$ROOT/.valahome
export HF_HOME=$HOME/hf; export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=$HOME/cache; export MPLCONFIGDIR=$HOME/mpl; export HF_HUB_OFFLINE=1
echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PRED=$ROOT/output/render/repro/relags/RELAGS
cd $ROOT/ReLaGS
for sc in figurines ramen teatime waldo_kitchen; do
  echo "=== render ReLaGS (faithful Algorithm-1, level_until=1 topk=5): $sc ==="
  PYTHONPATH=. python scripts/render_relags_eval.py \
      -s $ROOT/data/lerf_ovs/$sc \
      -m output/lerf_hf/scenes/LeRF/$sc \
      --label_root $ROOT/data/lerf_ovs/label \
      --path_pred $PRED
done

cd $ROOT
echo "=== single-harness eval: ReLaGS (vs paper 64.4) ==="
python scripts/repro/eval_soft.py --pred $ROOT/output/render/repro/relags \
    --gt $ROOT/data/lerf_ovs/label --paper relags \
    --csv $ROOT/output/diagnostics/repro_relags_singleharness.csv

echo "=== single-harness eval: THGS released-default L23_k3 (vs paper 54.94), same evaluator ==="
python scripts/repro/eval_soft.py --pred $ROOT/output/render/repro/thgs \
    --gt $ROOT/data/lerf_ovs/label --paper thgs \
    --csv $ROOT/output/diagnostics/repro_thgs_singleharness.csv
echo "=== repro-relags DONE $(date) ==="
