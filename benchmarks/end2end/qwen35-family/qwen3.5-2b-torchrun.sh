#!/usr/bin/env bash
set -euo pipefail

precision=bf16
for arg; do
    if [[ $arg == --fp8 ]]; then
        precision=fp8
        break
    fi
done
allocator=${QWEN35_2B_BF16_ALLOCATOR:?}
if [[ $precision == fp8 ]]; then
    allocator=${QWEN35_2B_FP8_ALLOCATOR:?}
fi

unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF PYTORCH_NO_CUDA_MEMORY_CACHING
case "$allocator" in
    default) ;;
    expandable) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ;;
    *) printf 'Unsupported allocator %s\n' "$allocator" >&2; exit 2 ;;
esac
printf 'Qwen3.5-2B precision=%s allocator=%s\n' "$precision" "$allocator"
exec "${QWEN35_2B_TORCHRUN:?}" "$@"
