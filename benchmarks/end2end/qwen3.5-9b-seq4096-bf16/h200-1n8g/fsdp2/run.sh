#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo 'Usage: run.sh {bf16|fp8|matched|tuned} MODEL_DIR CORPUS_BIN OUTPUT_DIR' >&2
    exit 2
fi
mode=$1
shift
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
end2end_dir=$(cd -- "$recipe_dir/../../.." && pwd)
runner="$end2end_dir/qwen35-family/run.sh"

unset PYTORCH_ALLOC_CONF PYTORCH_NO_CUDA_MEMORY_CACHING
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bf16=4:0:32:gated-delta:8
fp8=4:0:32:gated-delta:4:0
case $mode in
    bf16|matched) exec bash "$runner" "$mode" "$@" 12 "$bf16" ;;
    fp8) exec bash "$runner" fp8 "$@" 12 "$fp8" ;;
    tuned) exec bash "$runner" tuned "$@" 12 "$bf16" "$fp8" ;;
    *) echo "Unknown mode $mode" >&2; exit 2 ;;
esac
