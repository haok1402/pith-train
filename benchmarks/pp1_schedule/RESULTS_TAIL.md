
## Not measured

- A kernel-level breakdown of where DP8's step time goes (the traces give compute, exposed
  communication and idle time per rank, not per kernel family); M = 128 is never profiled (nsys is
  limited to M <= 32 to keep reports small).
- qwen3-30b-a3b DP8 probes at lengths below 2,048: skipped on the user's static-memory estimate above.
- Timed steps are 3 per cell; differences under about 2% are within the step-to-step spread seen.
- 10 raw nsys streams were truncated by killed processes and could not be converted; the cells they
  would have filled show `-`.

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
`workspace/pp1/analysis.jsonl`; allocator statistics in the `-r2` records; reports under `workspace/pp1/nsys/` and `workspace/pp1/nsys-recovered/`
(`workspace` is the shared-space symlink, `/mnt/weka/shrd/ci/haok/pith-train-pp1-schedule/workspace`).
