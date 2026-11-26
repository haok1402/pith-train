#!/bin/bash
set -euo pipefail

if (( $# < 4 || $# > 6 )); then
    echo 'Usage: examples/pretrain_lm/launch.sh --fp8-preflight MODEL WEIGHTS DATASET OUTPUT [STEPS] [--audit-losses]' >&2
    exit 2
fi
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    echo 'Run FP8 preflight from outside any Slurm allocation.' >&2
    exit 2
fi
model=$(realpath -e -- "$1")
weights=$(realpath -e -- "$2")
dataset=$(realpath -e -- "$3")
output=$(realpath -m -- "$4")
steps=${5:-5}
audit=()
if [[ ${6:-} == --audit-losses ]]; then
    audit=(--audit-losses)
elif [[ -n ${6:-} ]]; then
    echo "Unknown argument: $6" >&2
    exit 2
fi
[[ $steps =~ ^[1-9][0-9]*$ ]]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
: "${CUDA_HOME:?Set CUDA_HOME to CUDA 13.2}"
: "${DEEP_JIT_INCLUDE:?Set DEEP_JIT_INCLUDE to the installed DeepGEMM deep_jit include path}"
export CUDA_HOME DEEP_JIT_INCLUDE
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUTE_DSL_ARCH=sm_90a
export CUTE_DSL_DISABLE_FILE_CACHING=1 TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
mkdir -p "$(dirname -- "$output")"
mkdir -- "$output"
export TMPDIR="$output/tmp" XDG_CACHE_HOME="$output/cache"
export CUDA_CACHE_PATH="$output/cache/cuda" PITHTRAIN_TUNING_DIR="$output/cache/tuning"
export TORCHINDUCTOR_CACHE_DIR="$output/cache/inductor" TRITON_CACHE_DIR="$output/cache/triton"
export TORCH_EXTENSIONS_DIR="$output/cache/extensions" WANDB_DIR="$output"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
cache=${PITHTRAIN_PREFLIGHT_CACHE:-"$output/bundles"}
seed=()
if [[ -n ${PITHTRAIN_WEIGHT_SEED:-} ]]; then
    seed=(--weight-seed "$PITHTRAIN_WEIGHT_SEED")
fi
model_args=(--model "$model" --weights "$weights" --cache "$cache")
gemm_seed=()
if [[ -n ${PITHTRAIN_DEEPGEMM_SEED:-} ]]; then
    gemm_seed=(--deepgemm-seed "$PITHTRAIN_DEEPGEMM_SEED")
fi
python=${PITHTRAIN_PYTHON:-python}
# Kernel workers cold-import the stack, export CuTe libraries, then run the offline DeepGEMM build.
srun -A nvr_elm_llm -p cpu -q cpu-short -J deltamatch-fp8-kernels \
    -N1 -n1 -c4 --mem=24G --time=00:10:00 \
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 CUTE_DSL_SHOW_STACKTRACE=1 DG_JIT_NVCC_COMPILER="$CUDA_HOME/bin/nvcc" \
        "$python" -B -m tools.fp8_preflight.prepare kernels "${model_args[@]}" \
        --ready "$output/kernels-ready" "${gemm_seed[@]}" > "$output/preflight.log" 2>&1
# Four rank pack workers read the 50 GiB checkpoint and verify the resulting shards before sealing.
srun -A nvr_elm_llm -p cpu -q cpu-short -J deltamatch-fp8-weights \
    -N1 -n1 -c8 --mem=48G --time=00:08:00 \
    env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
        "$python" -B -m tools.fp8_preflight.prepare weights "${model_args[@]}" \
        --kernels-ready "$output/kernels-ready" --dataset "$dataset" --steps "$steps" \
        --staged-dataset "$output/dataset" --ready "$output/ready" "${seed[@]}" \
        > "$output/weights.log" 2>&1
read -r bundle < "$output/ready"
test -s "$bundle/seal.json"
if [[ ${PITHTRAIN_PREFLIGHT_ONLY:-0} == 1 ]]; then
    exit 0
fi
recipe=(--model "$model" --weights "$weights" --dataset "$output/dataset")
srun -A nvr_elm_llm -p batch -q short -J deltamatch-fp8-preflight-train \
    --constraint=nvidia_h200 -N1 -n8 --gpus=8 --gpu-bind=none -c4 --mem=128G \
    --time="${PITHTRAIN_GPU_TIME:-00:08:00}" --kill-on-bad-exit=1 -W0 \
    bash tools/fp8_preflight/slurm-worker.sh "$bundle" "$output" "${recipe[@]}" \
        --steps "$steps" "${audit[@]}" > "$output/run.log" 2>&1
