# pp=1 schedule: overlapped DualPipeV vs sequential, and EP8 vs full DP8

Branch `haok/pp1-sequential-prof` of haok1402/pith-train, from upstream main `2bcbb82`. 8x H200, one node,
pp1, cp1, MoE parallel folding (attention always DP8; experts EP8, or EP1 = experts replicated
over 8 DP ranks). ZeRO-1 FSDP as today in both schedules.

Everything below is measured; no number in this file is an estimate unless it says so. What was not
measured is listed under [Not measured](#not-measured).

## What was changed

- `TrainingCfg.pp1_sequential_schedule` (default off) -> `DualPipeV.sequential_at_pp1`: at
  pipeline_parallel_size 1, each micro-batch runs `forward_chunk(0)`, `forward_chunk(1)`,
  `backward_chunk(1)`, `backward_chunk(0)` with no forward/backward pairing. The end-of-step
  ZeRO-1 reduce-scatter is unchanged.
- At pp=1 the existing schedule already runs micro-batch 0 unpaired (F0 F1), then pairs
  F(k+1) with B(k) on alternating chunks (`2M - 3` paired ranges per step: 1 at M=2, 5 at M=4,
  13 at M=8, 61 at M=32, seen in the traces), and still defers the last chunk-0 backward's
  weight gradients to a W pass (zero bubble). With M=1 there is no pairing at all; the only
  difference left is that deferral.
- `benchmarks/pp1_schedule/`: the driver (`bench.py`), runners, nsys whole-step analysis
  (`analyze.py`, built on the analyze-nsys-profile skill's primitives) and the table generator
  (`report.py`) that produced every table below.

## Correctness

Same benchmark config (random init, seed 1234, force-balanced routing, Muon, lr 1e-6), deepseek-v2-lite,
pp1 ep8, gbs 64, seq 2048, 5 steps through `pretrain_lm.launch`:

| step | loss overlapped | loss sequential | grad norm overlapped | grad norm sequential |
|---|---|---|---|---|
| 1 | 11.9176 | 11.9175 | 19.9192 | 19.9200 |
| 2 | 11.9361 | 11.9358 | 20.3748 | 20.3746 |
| 3 | 11.9172 | 11.9169 | 20.2374 | 20.2372 |
| 4 | 11.9122 | 11.9120 | 21.1317 | 21.1312 |
| 5 | 11.8792 | 11.8791 | 19.1835 | 19.1849 |

Losses agree to 4e-4 and grad norms to 0.002 (bf16, different backward order). A no-update check on
identical weights (`bench.py --suite check`, uniform M=4 and an uneven packed M=4 batch) agrees to
about 1e-4 in loss.

## Method

- Workloads, per step, on every rank:
  - (a) the pretraining benchmark: uniform seq 2048, M = 32 (gbs 256) and M = 128 (gbs 1024, the existing
    benchmark's global batch) micro-batches per rank.
  - (b) RL-like packs: fixed budget of 131,072 tokens per step (16,384 per rank on average), split into
    M = 1, 2, 4, 8 packed micro-batches (cu_seqlens, lognormal documents about 1.5k long).
    `even`: every pack the same length. `intra`: lengths differ across a rank's micro-batches (so paired
    micro-batches differ), same list on every rank. `full`: the per-rank budget also differs across ranks
    (6k-37k tokens) and lengths differ within each rank.
  - (c) OOM probes: M = 1 or 2 equal packs of growing length, until each schedule OOMs.
- Step time: wall time of the full step (model.step, grad scale, clip, Muon step), max over ranks, 3
  timed steps after 2 warmup steps. The tables show the fastest of the 3 and all 3; comparisons use the
  fastest. A timed step can carry a recompilation (each step's packs have a different document count,
  so the cu_seqlens shape changes; one dsv2 step took 21.6 s against 0.99 s) or, near the memory
  limit, allocator retries; the latter are reported on their own (allocator table), the former are
  not a cost of either schedule. tokens/s/GPU = real tokens per step /
  fastest step time / 8. Peak memory: max over ranks of `max_memory_allocated` / `max_memory_reserved`, cache
  emptied before each workload.
- Both schedules of a workload run back to back in one process on one node (column `host`), so
  node-to-node differences do not enter a comparison.
- nsys: one extra step per (workload, schedule) under `cudaProfilerStart/Stop`, M <= 32. Exposed
  all-to-all = time of NCCL kernels launched inside `stage2_*`/`stage4_*` NVTX ranges (EP dispatch and
  combine) not covered by any compute kernel on that rank, averaged over ranks, whole `model.step`
  window. A NCCL kernel's time includes waiting for its slowest peer, so exposed all-to-all includes the
  cost of imbalance across ranks. One profiled step per cell, so these are noisier than the step times.
- Imbalance cost: step time of `intra` or `full` minus `even` at the same M and token budget, same
  schedule.

## Findings beyond the schedule question

- **The SM90 fused cross-entropy rejects micro-batches above (2^31 - 1) / vocab tokens**: 20,971 for
  deepseek-v2-lite, 14,134 for qwen3 (`ValueError: Out of bound mX.shape[0] ... int32 range`). It
  indexes the flattened logits with int32. The harness runs the same kernel on row slices, in place
  (`ChunkedCrossEntropy`, no extra memory); PithTrain itself would need this for packs of RL size.
- Upstream main does not install fresh: it pins `apache-tvm-tirx==0.1.0a6`, which PyPI does not have
  (0.1.0a5, 0.1.0 only; 0.1.0 needs a newer tvm-ffi). Ran with 0.1.0a5 (local, uncommitted).
- `benchmarks/pretraining/<model>/setup.py` tokenization crashes in a GPU-less step: its spawned
  workers re-import the script, which imports `pithtrain.modules.training`, which queries the device.
- nsys `--capture-range-end=repeat` keeps every capture in `$TMPDIR` until the process exits, so a
  preempted step can lose traces (on SIGTERM it converts some). With `$TMPDIR` on weka, the profiled
  processes died with SIGBUS (every SIGBUS seen came from runs started that way), so it stays
  node-local; the Triton and inductor caches are kept node-local as well.

## Results
