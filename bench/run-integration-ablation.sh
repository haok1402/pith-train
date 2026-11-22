#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

control_tree=${1:?usage: bash bench/run-integration-ablation.sh CONTROL_TREE NAME TREE [FLAGS] [-- NAME TREE [FLAGS] ...]}
shift
run_id=${E2EFP8_GPU_RUN_ID:-${SLURM_JOB_ID:?run through gpu-run.sh}}
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_arm() {
    local name=$1 tree=$2
    shift 2
    local log="logs/fsdp2-ablation-$name-$run_id.log"
    local status=0
    printf 'Training log %s\n' "$log"
    E2EFP8_INTEGRATION_TREE="$tree" bash bench/run-integration.sh train \
        --micro-batch 2 --checkpoint selective --reduce-dtype bf16 --bf16-sharded-gradients \
        --skip-embedding-backward-gather --blocks-per-group 2 "$@" > "$log" 2>&1 || status=$?
    if (( status != 0 )); then
        printf 'Arm %s exited %d. See %s\n' "$name" "$status" "$log" >&2
        return "$status"
    fi
    python3 -B bench/summarize.py --first 3 "$log"
}

run_arm control "$control_tree"
status=0
while (( $# > 0 )); do
    name=$1
    tree=${2:?missing candidate worktree}
    shift 2
    flags=()
    while (( $# > 0 )) && [[ $1 != -- ]]; do
        flags+=("$1")
        shift
    done
    if (( $# > 0 )); then shift; fi
    run_arm "$name" "$tree" "${flags[@]}" || status=$?
done
exit "$status"
