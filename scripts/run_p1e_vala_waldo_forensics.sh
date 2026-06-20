#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pilab_nas/projects/THGS
source /mnt/pilab_nas/projects/THGS/envs/vala-port/bin/activate

export HOME=/mnt/pilab_nas/projects/THGS/.valahome
export HF_HOME=/mnt/pilab_nas/projects/THGS/.valahome/hf
export MPLCONFIGDIR=/tmp/mpl

python scripts/p1e_vala_waldo_protocol_forensics.py \
  --device cuda \
  --chunk_size 65536
