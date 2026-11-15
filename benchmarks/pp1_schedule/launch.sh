#!/bin/bash
# Run one bench.py (or check_pretrain.py) process on this node, optionally under nsys.
# usage: launch.sh [--nsys <report-prefix>] <script.py> <args...>
set -euo pipefail
export OMP_NUM_THREADS=4
# Keep CPU use small: other agents benchmark on these nodes.
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export PYTHONUNBUFFERED=1
source .venv/bin/activate

NSYS=()
if [[ "${1:-}" == "--nsys" ]]; then
    mkdir -p "$(dirname "$2")"
    # nsys streams each capture to $TMPDIR and converts it only at exit; on shared storage the raw
    # .qdstrm of a preempted step survives and converts offline with QdstrmImporter.
    # Only nsys itself gets it: the training processes keep a node-local /tmp, since the inductor
    # cache there holds compiled kernels that are mmapped and must not live on weka (SIGBUS when a
    # process on another node rewrites a mapped file); the Triton cache likewise.
    NSYS_TMP=$PWD/workspace/pp1/nsys-tmp/j${SLURM_JOB_ID:-0}.${SLURM_STEP_ID:-0}
    mkdir -p "$NSYS_TMP"
    NSYS=(env TMPDIR=$NSYS_TMP nsys profile --stats=false --trace=cuda,nvtx --force-overwrite=true --output="$2"
          --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat --delay=0)
    shift 2
fi
PORT=$((15000 + RANDOM % 5000))
TORCHRUN_ARGS=(--nnodes=1 --nproc-per-node=gpu --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT)
exec "${NSYS[@]}" env TMPDIR=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER TRITON_CACHE_DIR=/tmp/triton_$USER torchrun "${TORCHRUN_ARGS[@]}" "$@"
