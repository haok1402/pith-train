#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
tree="$E2EFP8_ROOT/${E2EFP8_INTEGRATION_TREE:-wt-integration}"
export PYTHONPATH="$tree${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=2
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+ $LD_PRELOAD}"
model="$E2EFP8_ROOT/inputs/Qwen3.5-27B"

mode=${1:?usage: bash bench/run-integration.sh train|loss [flags]}
shift
case "$mode" in
    train) options=(--dataset "$E2EFP8_ROOT/inputs/integration-wikitext-train" --batch 64 --steps 6) ;;
    loss) options=(--dataset "$E2EFP8_ROOT/inputs/fsdp2-wikitext-test" --batch 8 --steps 1) ;;
    *) printf 'Unknown benchmark mode %s\n' "$mode" >&2; exit 2 ;;
esac

printf 'Slurm job %s on %s, PithTrain %s\n' "${SLURM_JOB_ID:?run through Slurm}" "$(hostname)" "$(git -C "$tree" rev-parse HEAD)"
cd "$tree"
exec "$E2EFP8_ROOT/pith-train/.venv/bin/torchrun" --standalone --nnodes=1 --nproc-per-node=8 \
    benchmarks/end2end/qwen3.5-27b-seq4096-bf16/h200-1n8g/fsdp2/script.py \
    --model "$model" --weights "$model" --seq 4096 --compile-blocks "${options[@]}" "$@"
