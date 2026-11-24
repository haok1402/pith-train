#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo 'Usage: run-qwen35-0p8b.sh {bf16|fp8|matched|tuned} MODEL_DIR CORPUS_BIN OUTPUT_DIR' >&2
    exit 2
fi

# The independent FP8 screen selected the same configuration as the BF16 screen.
arm=8:0:24:none:8
case "$1" in
    bf16|fp8|matched) set -- "$@" 12 "$arm" ;;
    tuned) set -- "$@" 12 "$arm" "$arm" ;;
    *) echo "Unknown benchmark mode $1" >&2; exit 2 ;;
esac

# This commit has the same source tree as the pinned family harness at 5c67803.
export QWEN35_REVISION="${QWEN35_REVISION:-bc1c3862eaadf6b9f8e04d6f54ff6ecfe6578300}"
unset PYTORCH_ALLOC_CONF PYTORCH_NO_CUDA_MEMORY_CACHING
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

runner=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run.sh
exec bash "$runner" "$@"
