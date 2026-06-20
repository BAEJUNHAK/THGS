#!/bin/bash
#SBATCH --job-name=fair-relags
#SBATCH --output=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/fair_relags_%j.log
#SBATCH --time=06:00:00
# FAIR COMPARISON (A): run the ReLaGS pipeline on THGS's trained 2DGS, so THGS and
# ReLaGS are compared on the SAME geometry substrate. THGS arm = released sai_nag
# (already = THGS-pipeline-on-THGS-2DGS = 58.87). This job computes ReLaGS-pipeline-
# on-THGS-2DGS, then evaluates with the single harness.
# Self-contained: all dir setup runs here (compute node), THGS release is NOT touched.
set -e
ROOT=/mnt/pilab_nas/projects/THGS
source $ROOT/envs/thgs/bin/activate
export HOME=$ROOT/.valahome HF_HOME=$ROOT/.valahome/hf HUGGINGFACE_HUB_CACHE=$ROOT/.valahome/hf/hub
export XDG_CACHE_HOME=$ROOT/.valahome/cache MPLCONFIGDIR=$ROOT/.valahome/mpl HF_HUB_OFFLINE=1
export PYTHONPATH=$ROOT/ReLaGS
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

SCENES="figurines ramen teatime waldo_kitchen"
PRED=$ROOT/output/render/repro/relags_on_thgs2dgs/RELAGS_THGS2DGS

for SC in $SCENES; do
  DST=$ROOT/output/repro_fair/relags_on_thgs/$SC
  SRC=$ROOT/output/lerf/$SC                       # THGS released 2DGS substrate (read-only via symlink)
  echo "=========== $SC : setup shared 2DGS dir ==========="
  mkdir -p $DST/point_cloud
  ln -sfn $SRC/cameras.json              $DST/cameras.json
  ln -sfn $SRC/input.ply                 $DST/input.ply
  ln -sfn $SRC/point_cloud/iteration_30000 $DST/point_cloud/iteration_30000
  rm -f $DST/cfg_args
  cat > $DST/cfg_args <<EOF
Namespace(sh_degree=3, source_path='$ROOT/data/lerf_ovs/$SC', model_path='$DST', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
EOF

  cd $ROOT/ReLaGS
  if [ ! -f $DST/sai_nag_3.0.pt ]; then
    echo "--- [1/5] max_weight_pruning (prune THGS 2DGS, contribution 0.0005) ---"
    python max_weight_pruning.py -s $ROOT/data/lerf_ovs/$SC -m $DST --contribution_threshold 0.0005
    echo "--- [2/5] sp_partition (build graph) ---"
    python sp_partition.py -s $ROOT/data/lerf_ovs/$SC -m $DST --iteration 0 -a
    echo "--- [3/5] graph_weight (SAM edge reweight) ---"
    python graph_weight.py --iteration 0 -s $ROOT/data/lerf_ovs/$SC -m $DST --config ./configs/lerf.yml --level 1
    echo "--- [4/5] sp_partition -k (re-partition) ---"
    python sp_partition.py -s $ROOT/data/lerf_ovs/$SC -m $DST --iteration 0 -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a
    echo "--- [5/5] merge_proj (ROFA merge + CLIP reproject, tau=3 -> sai_nag_3.0.pt) ---"
    python merge_proj.py -s $ROOT/data/lerf_ovs/$SC -m $DST --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0
  else
    echo "--- pipeline already done (sai_nag_3.0.pt exists), skipping steps 1-5 ---"
  fi
  ln -sfn $DST/sai_nag_3.0.pt $DST/sai_nag.pt   # render_relags_eval expects sai_nag.pt
  echo "--- render (faithful ReLaGS Alg-1) for single-harness eval ---"
  python scripts/render_relags_eval.py -s $ROOT/data/lerf_ovs/$SC -m $DST \
      --label_root $ROOT/data/lerf_ovs/label --path_pred $PRED
done

cd $ROOT
echo "=== single-harness eval: ReLaGS-on-THGS-2DGS (vs paper 64.4) ==="
python scripts/repro/eval_soft.py --pred $ROOT/output/render/repro/relags_on_thgs2dgs \
    --gt $ROOT/data/lerf_ovs/label --paper relags \
    --csv $ROOT/output/diagnostics/repro_fair_relags_on_thgs2dgs.csv
echo "=== FAIR-RELAGS DONE $(date) ==="
