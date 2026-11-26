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

python=${PITHTRAIN_PYTHON:-python}
stage=''
runtime_lock=''
if (( RANK == 0 )); then
    printf '%s\n' "$SLURM_JOB_ID" > "$output/slurm-job"
    printf '%s\n' "$(hostname -s)" > "$output/node"
    TZ=UTC nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,power.draw,memory.used \
        --format=csv,nounits --loop-ms=200 > "$output/gpu.csv" &
    sampler=$!
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        wait "$sampler" 2>/dev/null || true
        if [[ -n $runtime_lock ]]; then
            exec {runtime_lock}>&-
        fi
        if [[ -n $stage ]]; then
            flock -x -w 5 "$output/runtime.lock" rm -rf -- "$stage"
        fi
    }
    trap cleanup EXIT
fi

if [[ -s $bundle/runtime/sha256 ]]; then
    if (( RANK == 0 )); then
        source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
        printf 'STARTUP rank=0 time=%s phase=node_runtime state=begin\n' "$(date -u -Iseconds)"
        started=$SECONDS
        stage=$(mktemp -d "${PITHTRAIN_NODE_STORAGE:-/dev/shm}/pithtrain-runtime.${SLURM_JOB_ID}.XXXXXXXX")
        stage_runtime "$bundle/runtime" "$stage"
        printf '%s\n' "$stage" > "$output/runtime-pending"
        mv "$output/runtime-pending" "$output/runtime-ready"
        printf 'STARTUP rank=0 time=%s phase=node_runtime state=end seconds=%s\n' \
            "$(date -u -Iseconds)" "$((SECONDS - started))"
    else
        until read -r stage 2>/dev/null < "$output/runtime-ready"; do
            if (( SECONDS >= 30 )); then
                echo "Rank $RANK timed out waiting for the node runtime" >&2
                exit 1
            fi
            sleep 0.1
        done
    fi
    exec {runtime_lock}>"$output/runtime.lock"
    flock -s "$runtime_lock"
    python="$stage/bin/python"
    export PITHTRAIN_STAGED_RUNTIME=1
    unset PYTHONPYCACHEPREFIX
fi

if (( RANK == 0 )); then
    "$python" -B -m tools.fp8_preflight.worker --bundle "$bundle" "$@"
else
    exec "$python" -B -m tools.fp8_preflight.worker --bundle "$bundle" "$@"
fi
