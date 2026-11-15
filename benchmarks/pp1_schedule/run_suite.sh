#!/bin/bash
# Run one bench.py suite under nsys until it finishes, resuming after an OOM or a crash. Reports and
# logs are named by job, step and attempt, so a later step never overwrites an earlier one.
# usage: run_suite.sh <name> <bench.py args...>
set -uo pipefail
NAME=$1; shift
mkdir -p workspace/pp1/nsys
TAG=j${SLURM_JOB_ID:-0}.${SLURM_STEP_ID:-0}
oom_only=0
for attempt in $(seq 1 ${PP1_MAX_ATTEMPTS:-80}); do
    PREFIX=workspace/pp1/nsys/$NAME-$TAG-a$attempt
    LOG=workspace/pp1/$NAME-$TAG-a$attempt.log
    export PP1_NSYS_PREFIX=$PREFIX
    echo "=== $NAME attempt $attempt $(date -u +%FT%TZ) on $(hostname)"
    bash benchmarks/pp1_schedule/launch.sh --nsys $PREFIX benchmarks/pp1_schedule/bench.py "$@" > $LOG 2>&1
    rc=$?
    grep -E "RESULT|OOM in|Traceback|SUITE DONE" $LOG | cut -c1-300
    if grep -q "SUITE DONE" $LOG; then echo "=== $NAME SUITE DONE"; exit 0; fi
    if grep -q '"workload": "setup", "status": "oom"' $LOG; then echo "=== $NAME OOM AT SETUP"; exit 0; fi
    if grep -q "OOM in" $LOG && ! grep -q "RESULT" $LOG; then
        oom_only=$((oom_only + 1))
        # Every OOM is recorded and skipped next time, so a run of them still makes progress.
        if [ $oom_only -ge ${PP1_OOM_STOP:-40} ]; then echo "=== $NAME $oom_only attempts in a row OOMed with no result; stopping"; exit 0; fi
    else
        oom_only=0
    fi
    if ! grep -qE "RESULT|OOM in" $LOG; then
        echo "=== $NAME attempt $attempt made no progress (rc=$rc)"; tail -30 $LOG
        exit 1
    fi
done
echo "=== $NAME gave up"; exit 1
