
## Pending

Interrupted by the shared quota filling at about 02:40 UTC on 2026-11-15, and resumed after:

- dsv2-lite EP8 recapture (`main-dsv2-ep8-r2`): 12 of the 24 traces of the measured pass were lost
  to a preemption; this pass repeats all 13 workloads with allocator statistics.
- Allocator attribution of the near-limit slowdowns: dsv2 DP8 reruns (uniform M=32, M=4 even, M=8 even,
  M=8 intra) and the dsv2 EP8 2-pack probe at 12k-24k, now recording per-step `num_alloc_retries`,
  cudaMalloc calls, and peak allocated and reserved memory.
- OOM probes: dsv2 DP8 2 packs (unfinished above 2,048), qwen3 EP8 1 pack (unfinished above 8,192 ovl
  / 6,144 seq), qwen3 EP8 2 packs (not run).
- A kernel-level breakdown of DP8 vs EP8 (no DP8 uniform trace survived; M = 128 is never profiled).

## Reproduce

```bash
# one 8-GPU step per suite, inside the allocation (no --overlap)
srun --jobid=<job> -N1 -n1 --cpus-per-task=32 --gres=gpu:8 -W 0 -o logging/pp1/main-<name>.log \
  bash benchmarks/pp1_schedule/run_suite.sh main-<name> --model deepseek-v2-lite --ep 8 --suite main \
  --nsys --out workspace/pp1/main-<name>.jsonl
# OOM probe: --suite oom --oom-M 2 --oom-lengths 8192,12288,16384,...
# analysis (CPU step): bash benchmarks/pp1_schedule/analyze_all.sh
# tables: python benchmarks/pp1_schedule/report.py
```

Records: `workspace/pp1/{main,probe}-*.jsonl` and `*.oom.rank*.jsonl`; nsys analysis:
`workspace/pp1/analysis.jsonl`; reports under `workspace/pp1/nsys/` and `workspace/pp1/nsys-recovered/`
(`workspace` is the shared-space symlink, `/mnt/weka/shrd/ci/haok/pith-train-pp1-schedule/workspace`).
