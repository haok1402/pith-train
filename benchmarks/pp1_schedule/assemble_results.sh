#!/bin/bash
# RESULTS.md = the written sections plus the tables report.py generates from the records.
cd ~/pith-train-pp1-schedule
D=benchmarks/pp1_schedule
{ sed '$d' $D/RESULTS_HEAD.md; cat $D/RESULTS_ANALYSIS.md; echo; echo "## Results"; python3 $D/report.py; cat $D/RESULTS_TAIL.md; } > RESULTS.md
