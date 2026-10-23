#!/bin/bash
# Bidirectional checkpoint interchange between the even and edge-light gpt-oss partitions (8 GPUs).
set -euo pipefail
WORK=${WORK:-/tmp/gptoss-partition-ckpt-$$}
rm -rf $WORK; mkdir -p $WORK
trap 'rm -rf $WORK' EXIT
T=tests/test_gpt_oss_partition_checkpoint.py
R=(torchrun --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=localhost:15219)
python3 $T model $WORK
# even -> edge
LAYOUT=even "${R[@]}" $T train $WORK
cp -r $WORK/ckpt-even-train $WORK/ckpt-load-src
LAYOUT=edge "${R[@]}" $T load-src $WORK
python3 $T compare ckpt-even-train ckpt-edge-load-src $WORK
# edge -> even
rm -rf $WORK/ckpt-load-src
LAYOUT=edge "${R[@]}" $T train $WORK
cp -r $WORK/ckpt-edge-train $WORK/ckpt-load-src
LAYOUT=even "${R[@]}" $T load-src $WORK
python3 $T compare ckpt-edge-train ckpt-even-load-src $WORK
echo "PARTITION CHECKPOINT INTERCHANGE PASS"
