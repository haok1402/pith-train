#!/usr/bin/env bash
# Launch the Qwen3.5-4B same-node BF16/FP8 comparisons on eight H200s.
set -euo pipefail

if (( $# != 4 )) || [[ $1 != matched && $1 != tuned ]]; then
    printf 'Usage: %s {matched|tuned} MODEL_DIR CORPUS_BIN OUTPUT_DIR\n' "${0##*/}" >&2
    exit 2
fi

runner="$(dirname -- "${BASH_SOURCE[0]}")/run.sh"
export QWEN35_REVISION=5c6780386876a119535a012d1434d3992d2d2a47
unset PYTORCH_ALLOC_CONF PYTORCH_NO_CUDA_MEMORY_CACHING
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bf16_recipe=4:0:32:none:16
if [[ $1 == matched ]]; then
    exec bash "$runner" "$@" 12 "$bf16_recipe"
fi

fp8_recipe=4:0:32:none:16:0
exec bash "$runner" "$@" 12 "$bf16_recipe" "$fp8_recipe"
