#!/bin/bash
set -euo pipefail
bundle=$1
output=$2
shift 2

export RANK="$SLURM_PROCID" LOCAL_RANK="$SLURM_LOCALID" WORLD_SIZE="$SLURM_NTASKS"
export LOCAL_WORLD_SIZE="$SLURM_NTASKS" MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export DG_JIT_CACHE_DIR="$bundle/deepgemm" DJ_JIT_CACHE_DIR="$bundle/deepgemm"
export DG_JIT_NVCC_COMPILER="$PWD/tools/fp8_preflight/deepgemm-nvcc"
export DG_JIT_PRINT_LOAD_TIME=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_ALLOC_CONF CUDA_LAUNCH_BLOCKING PYTORCH_NO_CUDA_MEMORY_CACHING

if (( RANK == 0 )); then
    printf '%s\n' "$SLURM_JOB_ID" > "$output/slurm-job"
    printf '%s\n' "$(hostname -s)" > "$output/node"
    TZ=UTC nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,power.draw,memory.used \
        --format=csv,nounits --loop-ms=200 > "$output/gpu.csv" &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
    "${PITHTRAIN_PYTHON:-python}" -B -m tools.fp8_preflight.worker --bundle "$bundle" "$@"
else
    exec "${PITHTRAIN_PYTHON:-python}" -B -m tools.fp8_preflight.worker --bundle "$bundle" "$@"
fi
