#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 || $# > 7 )); then
    echo 'Usage: run.sh {bf16|fp8|matched|tuned} MODEL_DIR CORPUS_BIN OUTPUT_DIR STEPS ARM [FP8_ARM]' >&2
    echo 'ARM = microbatch:MLP-only-tail:kept-tail:mixers:FSDP-group[:FP8-cache-MiB[:replay]]' >&2
    exit 2
fi
mode=$1
model=$(realpath -e "$2")
corpus=$(realpath -e "$3")
out=$(realpath -m "$4")
steps=$5
recipe=$6
fp8_recipe=${7:-}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
revision=$(git -C "$source_dir" rev-parse --verify "${QWEN35_REVISION:-HEAD}^{commit}")

: "${E2EFP8_GPU_RUN_ID:?launch this script through the shared GPU gate}"
: "${SLURM_JOB_ID:?the GPU gate must run this script inside Slurm}"
: "${QWEN35_TORCHRUN:?set QWEN35_TORCHRUN to the verified virtualenv launcher}"
[[ -x $QWEN35_TORCHRUN && -f $model/config.json && -f $corpus && $corpus == *.bin ]]
[[ ${CUDA_HOME:?}/bin/ptxas -ef $(command -v ptxas) ]]
[[ $steps =~ ^[1-9][0-9]*$ ]]
case $mode in
    bf16|fp8|matched) [[ -z $fp8_recipe ]] ;;
    tuned) [[ -n $fp8_recipe ]] ;;
    *) echo "Unknown mode $mode" >&2; exit 2 ;;
esac

mkdir -p -- "$(dirname -- "$out")"
mkdir -- "$out"
mkdir -- "$out/source" "$out/tmp"
git -C "$source_dir" archive "$revision" | tar -xf - -C "$out/source"
cp -- "$model/config.json" "$out/config.json"
printf 'node=%s\njob=%s\nticket=%s\nrevision=%s\nmode=%s\nsteps=%s\n' \
    "$(hostname)" "$SLURM_JOB_ID" "$E2EFP8_GPU_RUN_ID" "$revision" "$mode" "$steps" > "$out/run.txt"

export TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp"
export PYTHONPATH="$out/source" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=2
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
cd "$out/source"
benchmark=benchmarks/end2end/qwen3.5-27b-seq4096-bf16/h200-1n8g/fsdp2/script.py

run_arm() {
    local label=$1 precision=$2 spec=$3
    local mb mlp keep mixers group cache replay extra rc=0
    local -a flags=()
    IFS=: read -r mb mlp keep mixers group cache replay extra <<< "$spec"
    if [[ ! $mb =~ ^(1|2|4|8)$ || ! $mlp =~ ^[0-9]+$ || ! $keep =~ ^[0-9]+$ ||
          ! $group =~ ^[1-9][0-9]*$ || ! ${cache:-0} =~ ^[0-9]+$ ||
          ! ${replay:-off} =~ ^(off|replay)$ || -n $extra ]]; then
        echo "Invalid $precision arm $spec" >&2
        return 2
    fi
    if [[ $precision == fp8 ]]; then
        flags+=(--fp8 --fp8-attention --fp8-shard-cache-mib "${cache:-0}")
        if [[ $replay == replay ]]; then flags+=(--fp8-mlp-recompute); fi
    elif [[ ${cache:-0} != 0 || ${replay:-off} != off ]]; then
        echo "FP8 cache and replay require an FP8 arm" >&2
        return 2
    fi

    local arm="$out/$label"
    mkdir -p -- "$arm/corpus"
    ln -s -- "$corpus" "$arm/corpus/tokens.bin"
    printf '%s\n' "$spec" > "$arm/recipe.txt"
    printf '%s %s %s %s\n' "$(date -Is)" "$label" "$precision" "$spec"
    timeout -k 30s "${QWEN35_ARM_TIMEOUT:-12m}" "$QWEN35_TORCHRUN" \
        --standalone --nnodes=1 --nproc-per-node=8 "$benchmark" \
        --model "$model" --weights "$model" --dataset "$arm/corpus" \
        --seq 4096 --batch 64 --steps "$steps" --micro-batch "$mb" \
        --checkpoint-mlp-tail "$mlp" --checkpoint-keep-tail "$keep" \
        --checkpoint-replay-mixers "$mixers" --fsdp-group-size "$group" \
        "${flags[@]}" > "$arm/train.log" 2>&1 || rc=$?
    if (( rc != 0 )); then
        tail -n 30 "$arm/train.log" >&2
        echo "$label failed with exit $rc" >&2
        return "$rc"
    fi
    (cd "$arm/corpus" && sha256sum indices.npy offsets.npy) > "$arm/corpus.sha256"
    printf '%s\n' "$(date -Is) completed $label"
}

case $mode in
    bf16|fp8) run_arm "01-$mode" "$mode" "$recipe" ;;
    matched|tuned)
        fp8_recipe=${fp8_recipe:-$recipe}
        run_arm 01-bf16 bf16 "$recipe"
        run_arm 02-fp8 fp8 "$fp8_recipe"
        run_arm 03-fp8 fp8 "$fp8_recipe"
        run_arm 04-bf16 bf16 "$recipe"
        for arm in 02-fp8 03-fp8 04-bf16; do
            cmp "$out/01-bf16/corpus.sha256" "$out/$arm/corpus.sha256"
        done
        ;;
esac
