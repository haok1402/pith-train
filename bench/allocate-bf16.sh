#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p logs
exec bench/gpu-run.sh --gpus 8 --time "${E2EFP8_WALL_TIME:-00:25:00}" -- bash -c '
    exec bash bench/run-bf16.sh "$@" > "logs/bf16-$SLURM_JOB_ID.log" 2>&1
' bash "$@"
