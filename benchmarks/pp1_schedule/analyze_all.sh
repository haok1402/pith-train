#!/bin/bash
# Convert leftover raw nsys streams, export every report to SQLite and run analyze.py on each,
# appending one JSON line per report to workspace/pp1/analysis.jsonl (reports already there are
# skipped). Run as a CPU-only srun step.
set -uo pipefail
cd ~/pith-train-pp1-schedule
source .venv/bin/activate
NSYS_ROOT=$(dirname $(readlink -f $(which nsys)))/..
IMPORTER=$NSYS_ROOT/host-linux-x64/QdstrmImporter
OUT=workspace/pp1/analysis.jsonl
mkdir -p workspace/pp1/nsys-recovered workspace/pp1/sqlite
touch $OUT

# A .qdstrm left by a killed process; a live step's streams are skipped (still being written).
for q in $(find workspace/pp1/nsys-tmp -name "*.qdstrm" -mmin +10); do
    base=workspace/pp1/nsys-recovered/$(basename $(dirname $q))-$(basename $q .qdstrm)
    [ -f $base.nsys-rep ] && continue
    echo "import $q"
    $IMPORTER -i $q -o $base.nsys-rep > /dev/null 2>&1 || echo "import failed: $q"
done

for rep in workspace/pp1/nsys/*.nsys-rep workspace/pp1/nsys-recovered/*.nsys-rep; do
    [ -f "$rep" ] || continue
    grep -q "\"report\": \"$rep\"" $OUT && continue
    db=workspace/pp1/sqlite/$(basename $rep .nsys-rep).sqlite
    [ -f $db ] || nsys export --type=sqlite --force-overwrite=true --output=$db $rep > /dev/null 2>&1 || { echo "export failed: $rep"; continue; }
    python benchmarks/pp1_schedule/analyze.py $db --json | python -c "
import json, sys
r = json.loads(sys.stdin.read()); r['report'] = '$rep'; r['sqlite'] = '$db'
print(json.dumps(r))" >> $OUT && echo "analyzed $rep" || echo "analyze failed: $rep"
done
echo "ANALYZE DONE $(date -u +%T)"
