#!/bin/bash
# Run one bench.py suite under nsys until it finishes, resuming after an OOM or a crash (each
# attempt writes its own report prefix, since capture numbering restarts per process).
# usage: run_suite.sh <name> <bench.py args...>
set -uo pipefail
NAME=$1; shift
mkdir -p workspace/pp1/nsys
oom_only=0
for attempt in $(seq 1 ${PP1_MAX_ATTEMPTS:-14}); do
    PREFIX=workspace/pp1/nsys/$NAME-a$attempt
    export PP1_NSYS_PREFIX=$PREFIX
    echo "=== $NAME attempt $attempt $(date -u +%FT%TZ) on $(hostname)"
    bash benchmarks/pp1_schedule/launch.sh --nsys $PREFIX benchmarks/pp1_schedule/bench.py "$@" > workspace/pp1/$NAME-a$attempt.log 2>&1
    rc=$?
    grep -E "RESULT|OOM in|Traceback|SUITE DONE" workspace/pp1/$NAME-a$attempt.log | cut -c1-300
    if grep -q "SUITE DONE" workspace/pp1/$NAME-a$attempt.log; then echo "=== $NAME SUITE DONE"; exit 0; fi
    if grep -q '"workload": "setup", "status": "oom"' workspace/pp1/$NAME-a$attempt.log; then echo "=== $NAME OOM AT SETUP"; exit 0; fi
    if grep -q "OOM in" workspace/pp1/$NAME-a$attempt.log && ! grep -q "RESULT" workspace/pp1/$NAME-a$attempt.log; then
        oom_only=$((oom_only + 1))
        if [ $oom_only -ge 3 ]; then echo "=== $NAME three attempts in a row OOMed with no result; stopping"; exit 0; fi
    else
        oom_only=0
    fi
    if ! grep -qE "RESULT|OOM in" workspace/pp1/$NAME-a$attempt.log; then
        echo "=== $NAME attempt $attempt made no progress (rc=$rc)"; tail -30 workspace/pp1/$NAME-a$attempt.log
        exit 1
    fi
done
echo "=== $NAME gave up"; exit 1
