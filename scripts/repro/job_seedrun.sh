#!/bin/bash
#SBATCH --job-name=seedrun-fair
#SBATCH --output=/mnt/pilab_nas/projects/THGS/output/diagnostics/logs/seedrun_%j.log
#SBATCH --time=08:00:00
# RIGOROUS FAIR COMPARISON: same THGS-2DGS substrate, same seed(s), re-run BOTH pipelines.
# THGS-pipe (no prune, no ROFA) and ReLaGS-pipe (prune+ROFA) on THGS's iteration_30000.
# graph_weight is seeded via PIPELINE_SEED (both codebases patched). -> mean±std over seeds.
set -e
ROOT=/mnt/pilab_nas/projects/THGS
source $ROOT/envs/thgs/bin/activate
export HOME=$ROOT/.valahome HF_HOME=$ROOT/.valahome/hf HUGGINGFACE_HUB_CACHE=$ROOT/.valahome/hf/hub
export XDG_CACHE_HOME=$ROOT/.valahome/cache MPLCONFIGDIR=$ROOT/.valahome/mpl HF_HUB_OFFLINE=1
echo "=== node $(hostname) gpu $CUDA_VISIBLE_DEVICES $(date) ==="
SCENES="figurines ramen teatime waldo_kitchen"
SEEDS="0 1 2"

setup_dir () {  # $1=dst  ; symlink THGS 2DGS (iteration_30000 only) + write cfg_args
  local DST=$1 SC=$2
  mkdir -p $DST/point_cloud
  ln -sfn $ROOT/output/lerf/$SC/cameras.json $DST/cameras.json
  ln -sfn $ROOT/output/lerf/$SC/input.ply $DST/input.ply
  ln -sfn $ROOT/output/lerf/$SC/point_cloud/iteration_30000 $DST/point_cloud/iteration_30000
  rm -f $DST/cfg_args
  cat > $DST/cfg_args <<EOF
Namespace(sh_degree=3, source_path='$ROOT/data/lerf_ovs/$SC', model_path='$DST', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
EOF
}

for SEED in $SEEDS; do
  export PIPELINE_SEED=$SEED
  echo "############## SEED $SEED ##############"
  for SC in $SCENES; do
    # ---------- THGS pipeline (launcher-faithful: no prune, no ROFA) ----------
    TDST=$ROOT/output/repro_fair/seedrun/thgs_s$SEED/$SC
    if [ ! -f $TDST/sai_nag.pt ]; then
      setup_dir $TDST $SC
      export PYTHONPATH=$ROOT
      cd $ROOT
      echo "--- THGS s$SEED $SC: sp_partition build ---"
      python sp_partition.py -s $ROOT/data/lerf_ovs/$SC -m $TDST -a
      echo "--- THGS graph_weight ---"
      python graph_weight.py -s $ROOT/data/lerf_ovs/$SC -m $TDST --config configs/lerf.yml --level 1
      echo "--- THGS sp_partition -k ---"
      python sp_partition.py -s $ROOT/data/lerf_ovs/$SC -m $TDST -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1
      echo "--- THGS merge_proj ---"
      python merge_proj.py -s $ROOT/data/lerf_ovs/$SC -m $TDST --thres_connect 0.9,0.7,0.7 --thres_merge 20 --feat_assign 2
    else echo "--- THGS s$SEED $SC exists, skip ---"; fi
    export PYTHONPATH=$ROOT
    cd $ROOT
    python scripts/repro/render_thgs_single.py -s $ROOT/data/lerf_ovs/$SC -m $TDST \
        --path_pred $ROOT/output/render/repro/seedrun/THGS_s$SEED/RUN

    # ---------- ReLaGS pipeline (run_lerf-faithful: prune + ROFA tau=3) ----------
    RDST=$ROOT/output/repro_fair/seedrun/relags_s$SEED/$SC
    if [ ! -f $RDST/sai_nag_3.0.pt ]; then
      setup_dir $RDST $SC
      export PYTHONPATH=$ROOT/ReLaGS
      cd $ROOT/ReLaGS
      echo "--- ReLaGS s$SEED $SC: prune ---"
      python max_weight_pruning.py -s $ROOT/data/lerf_ovs/$SC -m $RDST --contribution_threshold 0.0005
      echo "--- ReLaGS sp_partition build ---"
      python sp_partition.py -s $ROOT/data/lerf_ovs/$SC -m $RDST --iteration 0 -a
      echo "--- ReLaGS graph_weight ---"
      python graph_weight.py --iteration 0 -s $ROOT/data/lerf_ovs/$SC -m $RDST --config ./configs/lerf.yml --level 1
      echo "--- ReLaGS sp_partition -k ---"
      python sp_partition.py -s $ROOT/data/lerf_ovs/$SC -m $RDST --iteration 0 -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a
      echo "--- ReLaGS merge_proj (tau=3) ---"
      python merge_proj.py -s $ROOT/data/lerf_ovs/$SC -m $RDST --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0
    else echo "--- ReLaGS s$SEED $SC exists, skip ---"; fi
    ln -sfn $RDST/sai_nag_3.0.pt $RDST/sai_nag.pt
    export PYTHONPATH=$ROOT/ReLaGS
    cd $ROOT/ReLaGS
    python scripts/render_relags_eval.py -s $ROOT/data/lerf_ovs/$SC -m $RDST \
        --label_root $ROOT/data/lerf_ovs/label --path_pred $ROOT/output/render/repro/seedrun/RELAGS_s$SEED/RUN
  done
done

cd $ROOT
echo "=== AGGREGATE mean±std ==="
python scripts/repro/aggregate_seedrun.py --base $ROOT/output/render/repro/seedrun --gt $ROOT/data/lerf_ovs/label --seeds 0,1,2
echo "=== SEEDRUN DONE $(date) ==="
