#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
echo "Slurm job ${SLURM_JOB_ID:?run through srun} on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
cd "$E2EFP8_ROOT/pith-train"
exec .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    "$E2EFP8_ROOT/bench/bf16.py" "$@"
