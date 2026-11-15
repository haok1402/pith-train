# pp=1 schedule: overlapped DualPipeV vs sequential, and EP8 vs full DP8

Branch `haok/pp1-sequential-prof` of haok1402/pith-train, from upstream main `2bcbb82`. 8x H200, one node,
pp1, cp1, MoE parallel folding (attention always DP8; experts EP8, or EP1 = experts replicated
over 8 DP ranks). ZeRO-1 FSDP as today in both schedules.

**Status: partial.** Everything below is measured; cells marked `pending` were not run yet (see
[Pending](#pending)). No number in this file is an estimate unless it says so.

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
- Step time: wall time of the full step (model.step, grad scale, clip, Muon step), max over ranks, median
  of 3 timed steps after 2 warmup steps; min-max of the 3 shown. tokens/s/GPU = real tokens per step /
  step time / 8. Peak memory: max over ranks of `max_memory_allocated` / `max_memory_reserved`, cache
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
- nsys `--capture-range-end=repeat` keeps every capture in `$TMPDIR` until the process exits; a
  preempted step lost its traces until the harness put nsys's (only nsys's) temp dir on shared storage.

## Analysis

All numbers are from the tables in [Results](#results). "ovl" is the overlapped DualPipeV schedule,
"seq" the sequential one; percentages are seq step time relative to ovl (positive: seq slower).

**Where the overlap pays off.** On qwen3-30b-a3b EP8, ovl is faster in every workload measured:
uniform pretraining by 22.8% at M=32 (6.62 vs 8.13 s) and 10.5% at M=128; packs by 1.6-3.7% at M=2,
0.7-7.2% at M=4 and 7.5-12.6% at M=8. The traces show why: ovl hides 17-47% of the EP all-to-all
(uniform M=32: 1.34 s exposed of 2.53 s), seq hides none of it (0.0-0.3%). On deepseek-v2-lite EP8 the
gain is smaller and mixed: uniform M=32 is 10.2% faster with ovl (exposed all-to-all 0.70 vs 1.19 s),
uniform M=128 is 3.0% faster with seq (not profiled), and packs are within -3.4% to +4.3% except the
fully uneven M=8 case (+10.4%).

**Where sequential wins.**
- *Single micro-batch*: no pairing exists, and the two schedules run at the same speed (dsv2 EP8 M=1:
  1.016 vs 1.017 s; 1.222 vs 1.238 s uneven). The largest single pack is the same on dsv2 EP8 (24k
  fits, 32k OOMs); on dsv2 DP8, seq fits one step larger (10k vs 8k), probably the deferred zero-bubble
  weight gradients the ovl schedule still keeps at M=1, which is not verified.
- *Memory*: with M >= 2, seq has the lower reserved peak in every pair, and a lower or equal
  allocated peak: dsv2 EP8 M=2 even 64.7 vs 70.3 GiB
  allocated (68.2 vs 79.9 reserved), M=2 full 85.2 vs 95.5 (91.5 vs 112.5); qwen3 EP8 M=2 even
  106.3 vs 112.1 (114.5 vs 128.5). ovl holds two micro-batches' activations while seq holds one.
- *Bigger packs*: with 2 packs per step on dsv2 EP8, seq fits 24,576-token packs and ovl 16,384 (first
  OOM 32,768 vs 20,480).
- *Near the memory limit*: at equal lengths the two are equal (8k: 0.996 vs 0.994 s; 12k: 1.346 vs
  1.350 s), but at 16k ovl, at 133 GiB reserved, took 2.73 s (1.83-3.34) against seq's 1.66 s
  (1.61-1.67). seq shows a similar erratic step at its own limit (24k: 2.39-3.14 s). Attributing this
  to allocator retries is pending the rerun that records them.

**Imbalance.** Uneven budgets across ranks (`full`) cost the most: dsv2 EP8 +0.21 to +0.39 s with ovl
and +0.22 to +0.67 s with seq over the even step (the all-to-all waits for the rank with the most
tokens; per-rank compute in the profiled step spreads 322-648 ms for dsv2 M=2 full vs 449-486 even;
qwen3 M=4 full 756-1397 vs 834-893). Uneven lengths within a rank (`intra`, so paired micro-batches
differ) cost little on dsv2 (+0.00 to +0.13 s) and more on qwen3 (M=4: +0.34 s ovl vs +0.23 s seq;
M=8: +0.42 vs +0.34): ovl loses more of its gain to mismatched pairs, but stays faster in absolute
terms (M=4 intra 1.987 vs 2.000 s; M=8 intra 2.601 vs 2.796 s).

**Crossover.** At a fixed 131k tokens per step, fewer and larger packs are faster under both schedules
(dsv2 EP8 even: 1.02, 1.04, 1.15, 1.66 s at M = 1, 2, 4, 8; qwen3 EP8 ovl: 1.57, 1.65, 2.18 s at M = 2,
4, 8), and that effect is larger than anything the overlap adds. The overlap's advantage grows with M:
ties at M=1, within about 4% at M=2 (where seq also has the memory headroom), and clearly ahead on
qwen3 from M=4 (7.2%) and M=8 (12.6%). For the RL case of few large packs, the measured trade is: at
M=1 the schedules are equivalent, at M=2 seq costs 0.9-3.7% of speed and buys 9-21 GiB of peak
reserved memory, or about 1.5x the pack length; with many micro-batches ovl is the faster choice.

**Full DP8 (EP1) vs EP8.** On deepseek-v2-lite, DP8 is faster only for uniform pretraining with M=128
(16.36 vs 17.28 s, the best of each, -5.3%). It is slower for uniform M=32 (+13.3%) and for every
packed workload that fits (+27% to +53%), and it OOMs for every packed workload with M <= 2 and every
workload uneven across ranks, at 111-120 GiB allocated where EP8 needs 41-49 for the same workloads. The DP8 trace (M=4 even) shows
no all-to-all at all (0 ms), yet the step is still slower. Which kernels make up the difference, and why
DP8 wins at M=128, is not profiled yet (pending). On qwen3-30b-a3b, DP8 does not fit at any workload
tested: every one OOMs in the first step under both schedules, including 2,048-token micro-batches, at
133-135 GiB allocated (the at-rest, sharded footprint after setup is 14.2 GiB). The probes at shorter
lengths were skipped: by the user's estimate (not measured), replicated experts under ZeRO-1 need about
30B x (2 B bf16 weights + 4 B fp32 main_grad), about 180 GB per GPU, before activations.

## Results

### deepseek-v2-lite EP8

| workload | M | max pack | step s, median (min-max of 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 32 | 2048 | 4.271 (4.27-4.39) / 4.707 (4.67-4.78) | +10.2% | 15344 / 13923 | 40.7 / 40.2 | 58.3 / 55.6 | 704 / 1190 | 42 | 2374-2433 | 518 |
| uniform-seq2048-mb128 | 128 | 2048 | 17.804 (17.61-17.83) / 17.278 (17.17-17.36) | -3.0% | 14724 / 15172 | 40.8 / 40.2 | 58.7 / 55.7 | - / - | - | - | 518 |
| pack16k-M1-even | 1 | 16384 | 1.016 (1.01-22.34) / 1.017 (1.00-1.03) | +0.1% | 16133 / 16113 | 86.8 / 86.8 | 88.8 / 88.8 | 480 / 111 | 0 | 398-430 | 518 |
| pack16k-M1-full | 1 | 27648 | 1.222 (1.21-1.31) / 1.238 (1.22-1.31) | +1.3% | 13405 / 13229 | 115.7 / 115.7 | 122.6 / 122.7 | 267 / 292 | 0 | 225-502 | 518 |
| pack16k-M2-even | 2 | 8192 | 1.035 (1.03-1.05) / 1.043 (1.04-1.05) | +0.9% | 15837 / 15702 | 70.3 / 64.7 | 79.9 / 68.2 | 117 / 373 | 27 | 449-486 | 518 |
| pack16k-M2-intra | 2 | 9216 | 1.038 (1.03-1.04) / 1.053 (1.05-1.06) | +1.5% | 15788 / 15558 | 69.8 / 68.9 | 83.7 / 74.5 | 156 / 169 | 21 | 442-475 | 518 |
| pack16k-M2-full | 2 | 15616 | 1.305 (1.30-1.34) / 1.336 (1.32-1.36) | +2.4% | 12558 / 12262 | 95.5 / 85.2 | 112.5 / 91.5 | 268 / 339 | 19 | 322-648 | 518 |
| pack16k-M4-even | 4 | 4096 | 1.149 (1.13-1.15) / 1.175 (1.17-1.19) | +2.3% | 14264 / 13944 | 49.4 / 48.4 | 60.8 / 57.2 | - / - | - | - | 518 |
| pack16k-M4-intra | 4 | 9856 | 1.262 (1.24-1.29) / 1.261 (1.26-1.29) | -0.1% | 12983 / 12997 | 71.8 / 71.6 | 83.8 / 77.8 | - / - | - | - | 518 |
| pack16k-M4-full | 4 | 11648 | 1.456 (1.45-1.46) / 1.519 (1.51-1.54) | +4.3% | 11255 / 10788 | 71.2 / 70.5 | 88.4 / 81.3 | - / - | - | - | 518 |
| pack16k-M8-even | 8 | 2048 | 1.662 (1.64-1.75) / 1.606 (1.60-1.61) | -3.4% | 9860 / 10203 | 40.9 / 40.4 | 58.8 / 55.8 | - / - | - | - | 518 |
| pack16k-M8-intra | 8 | 5120 | 1.715 (1.69-1.73) / 1.734 (1.72-1.76) | +1.1% | 9552 / 9446 | 49.1 / 49.1 | 64.8 / 59.1 | - / - | - | - | 518 |
| pack16k-M8-full | 8 | 20224 | 2.056 (2.04-2.06) / 2.270 (2.23-2.28) | +10.4% | 7969 / 7217 | 89.5 / 89.2 | 114.3 / 107.9 | - / - | - | - | 518 |

Imbalance cost, deepseek-v2-lite EP8: step time of uneven packs minus even packs, same tokens and M.

| M | schedule | even s | intra s (cost) | full s (cost) |
|---|---|---|---|---|
| 1 | overlapped | 1.016 | - | 1.222 (+0.207) |
| 1 | sequential | 1.017 | - | 1.238 (+0.222) |
| 2 | overlapped | 1.035 | 1.038 (+0.003) | 1.305 (+0.270) |
| 2 | sequential | 1.043 | 1.053 (+0.010) | 1.336 (+0.293) |
| 4 | overlapped | 1.149 | 1.262 (+0.113) | 1.456 (+0.307) |
| 4 | sequential | 1.175 | 1.261 (+0.086) | 1.519 (+0.344) |
| 8 | overlapped | 1.662 | 1.715 (+0.054) | 2.056 (+0.394) |
| 8 | sequential | 1.606 | 1.734 (+0.129) | 2.270 (+0.665) |

### deepseek-v2-lite DP8 (EP1)

| workload | M | max pack | step s, median (min-max of 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 32 | 2048 | 4.840 (4.84-4.94) / 4.843 (4.84-4.84) | +0.1% | 13541 / 13532 | 111.0 / 110.5 | 130.6 / 132.6 | - / - | - | - | 441 |
| uniform-seq2048-mb128 | 128 | 2048 | 16.356 (16.33-16.40) / 16.402 (16.40-16.41) | +0.3% | 16027 / 15982 | 111.0 / 110.6 | 130.6 / 132.6 | - / - | - | - | 411 |
| pack16k-M4-even | 4 | 4096 | 2.086 (1.88-2.42) / 1.759 (1.75-1.80) | -15.7% | 7855 / 9315 | 119.5 / 118.7 | 132.6 / 130.0 | 0 / 0 | - | 668-688 | 028 |
| pack16k-M8-even | 8 | 2048 | 2.035 (2.00-2.57) / 2.135 (2.13-2.61) | +4.9% | 8052 / 7674 | 111.0 / 110.5 | 130.6 / 132.6 | - / - | - | - | 028 |
| pack16k-M8-intra | 8 | 5120 | 2.419 (2.22-2.56) / 2.483 (2.32-2.72) | +2.6% | 6773 / 6599 | 119.2 / 119.2 | 135.2 / 136.7 | - / - | - | - | 028 |
| pack16k-M1-even | 1 | 16384 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-even | 2 | 8192 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-intra | 2 | 9216 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-full | 2 | 15616 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M4-intra | 4 | 9856 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M8-full | 8 | 20224 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M1-full | 1 | 27648 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M4-full | 4 | 11648 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |

Imbalance cost, deepseek-v2-lite DP8: step time of uneven packs minus even packs, same tokens and M.

| M | schedule | even s | intra s (cost) | full s (cost) |
|---|---|---|---|---|
| 1 | overlapped | OOM | - | OOM |
| 1 | sequential | OOM | - | OOM |
| 2 | overlapped | OOM | OOM | OOM |
| 2 | sequential | OOM | OOM | OOM |
| 4 | overlapped | 2.086 | OOM | OOM |
| 4 | sequential | 1.759 | OOM | OOM |
| 8 | overlapped | 2.035 | 2.419 (+0.384) | OOM |
| 8 | sequential | 2.135 | 2.483 (+0.348) | OOM |

### qwen3-30b-a3b EP8

| workload | M | max pack | step s, median (min-max of 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 32 | 2048 | 6.621 (6.57-6.63) / 8.130 (7.83-8.26) | +22.8% | 9898 / 8061 | 74.6 / 74.6 | 103.1 / 95.6 | 1335 / - | 47 | 3742-3827 | 108 |
| uniform-seq2048-mb128 | 128 | 2048 | 24.628 (24.61-25.45) / 27.213 (27.20-27.37) | +10.5% | 10644 / 9633 | 74.6 / 74.6 | 103.1 / 95.6 | - / - | - | - | 108 |
| pack16k-M2-even | 2 | 8192 | 1.572 (1.54-1.81) / 1.597 (1.60-2.48) | +1.6% | 10423 / 10262 | 112.1 / 106.3 | 128.5 / 114.5 | 481 / 485 | 17 | 703-806 | 108 |
| pack16k-M2-intra | 2 | 9216 | 1.573 (1.56-1.58) / 1.631 (1.61-1.63) | +3.7% | 10415 / 10044 | 113.7 / 112.5 | 127.2 / 117.5 | 273 / - | 27 | 714-805 | 108 |
| pack16k-M4-even | 4 | 4096 | 1.648 (1.64-1.65) / 1.767 (1.76-1.78) | +7.2% | 9940 / 9275 | 82.4 / 81.7 | 104.7 / 95.3 | 305 / 457 | 47 | 834-893 | 108 |
| pack16k-M4-intra | 4 | 9856 | 1.987 (1.95-2.03) / 2.000 (2.00-2.08) | +0.7% | 8245 / 8191 | 116.7 / 116.4 | 133.8 / 128.1 | 336 / 582 | 31 | 774-888 | 108 |
| pack16k-M4-full | 4 | 11648 | 2.325 (2.29-2.43) / 2.457 (2.45-2.48) | +5.7% | 7048 / 6669 | 113.0 / 112.2 | 134.0 / 128.5 | 729 / 971 | 33 | 756-1397 | 108 |
| pack16k-M8-even | 8 | 2048 | 2.177 (2.17-2.18) / 2.452 (2.45-2.97) | +12.6% | 7525 / 6681 | 74.6 / 74.6 | 103.1 / 95.6 | 451 / 643 | 32 | 909-933 | 108 |
| pack16k-M8-intra | 8 | 5120 | 2.601 (2.58-2.61) / 2.796 (2.77-2.82) | +7.5% | 6300 / 5860 | 82.7 / 82.6 | 109.5 / 100.0 | 521 / 1039 | 31 | 969-1006 | 108 |
| pack16k-M1-even | 1 | 16384 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-full | 2 | 15616 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M8-full | 8 | 20224 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M1-full | 1 | 27648 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |

Imbalance cost, qwen3-30b-a3b EP8: step time of uneven packs minus even packs, same tokens and M.

| M | schedule | even s | intra s (cost) | full s (cost) |
|---|---|---|---|---|
| 1 | overlapped | OOM | - | OOM |
| 1 | sequential | OOM | - | OOM |
| 2 | overlapped | 1.572 | 1.573 (+0.001) | OOM |
| 2 | sequential | 1.597 | 1.631 (+0.035) | OOM |
| 4 | overlapped | 1.648 | 1.987 (+0.339) | 2.325 (+0.677) |
| 4 | sequential | 1.767 | 2.000 (+0.234) | 2.457 (+0.690) |
| 8 | overlapped | 2.177 | 2.601 (+0.424) | OOM |
| 8 | sequential | 2.452 | 2.796 (+0.343) | OOM |

### qwen3-30b-a3b DP8 (EP1)

| workload | M | max pack | step s, median (min-max of 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pack16k-M1-even | 1 | 16384 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-even | 2 | 8192 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-intra | 2 | 9216 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M4-intra | 4 | 9856 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M8-even | 8 | 2048 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M8-intra | 8 | 5120 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M8-full | 8 | 20224 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| uniform-seq2048-mb32 | 32 | 2048 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-full | 2 | 15616 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M4-full | 4 | 11648 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| uniform-seq2048-mb128 | 128 | 2048 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M4-even | 4 | 4096 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M1-full | 1 | 27648 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |

Imbalance cost, qwen3-30b-a3b DP8: step time of uneven packs minus even packs, same tokens and M.

| M | schedule | even s | intra s (cost) | full s (cost) |
|---|---|---|---|---|
| 1 | overlapped | OOM | - | OOM |
| 1 | sequential | OOM | - | OOM |
| 2 | overlapped | OOM | OOM | OOM |
| 2 | sequential | OOM | OOM | OOM |
| 4 | overlapped | OOM | OOM | OOM |
| 4 | sequential | OOM | OOM | OOM |
| 8 | overlapped | OOM | OOM | OOM |
| 8 | sequential | OOM | OOM | OOM |

### DP8 (EP1) vs EP8, deepseek-v2-lite

| workload | EP8 overlapped s | EP8 sequential s | DP8 overlapped s | DP8 sequential s | best DP8 vs best EP8 | peak alloc GiB EP8 / DP8 (overlapped) |
|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 4.271 | 4.707 | 4.840 | 4.843 | +13.3% | 40.7 / 111.0 |
| uniform-seq2048-mb128 | 17.804 | 17.278 | 16.356 | 16.402 | -5.3% | 40.8 / 111.0 |
| pack16k-M1-even | 1.016 | 1.017 | OOM | OOM | - | 86.8 / OOM |
| pack16k-M1-full | 1.222 | 1.238 | OOM | OOM | - | 115.7 / OOM |
| pack16k-M2-even | 1.035 | 1.043 | OOM | OOM | - | 70.3 / OOM |
| pack16k-M2-intra | 1.038 | 1.053 | OOM | OOM | - | 69.8 / OOM |
| pack16k-M2-full | 1.305 | 1.336 | OOM | OOM | - | 95.5 / OOM |
| pack16k-M4-even | 1.149 | 1.175 | 2.086 | 1.759 | +53.1% | 49.4 / 119.5 |
| pack16k-M4-intra | 1.262 | 1.261 | OOM | OOM | - | 71.8 / OOM |
| pack16k-M4-full | 1.456 | 1.519 | OOM | OOM | - | 71.2 / OOM |
| pack16k-M8-even | 1.662 | 1.606 | 2.035 | 2.135 | +26.7% | 40.9 / 111.0 |
| pack16k-M8-intra | 1.715 | 1.734 | 2.419 | 2.483 | +41.0% | 49.1 / 119.2 |
| pack16k-M8-full | 2.056 | 2.270 | OOM | OOM | - | 89.5 / OOM |

### DP8 (EP1) vs EP8, qwen3-30b-a3b

| workload | EP8 overlapped s | EP8 sequential s | DP8 overlapped s | DP8 sequential s | best DP8 vs best EP8 | peak alloc GiB EP8 / DP8 (overlapped) |
|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 6.621 | 8.130 | OOM | OOM | - | 74.6 / OOM |
| uniform-seq2048-mb128 | 24.628 | 27.213 | OOM | OOM | - | 74.6 / OOM |
| pack16k-M2-even | 1.572 | 1.597 | OOM | OOM | - | 112.1 / OOM |
| pack16k-M2-intra | 1.573 | 1.631 | OOM | OOM | - | 113.7 / OOM |
| pack16k-M4-even | 1.648 | 1.767 | OOM | OOM | - | 82.4 / OOM |
| pack16k-M4-intra | 1.987 | 2.000 | OOM | OOM | - | 116.7 / OOM |
| pack16k-M4-full | 2.325 | 2.457 | OOM | OOM | - | 113.0 / OOM |
| pack16k-M8-even | 2.177 | 2.452 | OOM | OOM | - | 74.6 / OOM |
| pack16k-M8-intra | 2.601 | 2.796 | OOM | OOM | - | 82.7 / OOM |
| pack16k-M1-even | OOM | OOM | OOM | OOM | - | OOM / OOM |
| pack16k-M2-full | OOM | OOM | OOM | OOM | - | OOM / OOM |
| pack16k-M8-full | OOM | OOM | OOM | OOM | - | OOM / OOM |
| pack16k-M1-full | OOM | OOM | OOM | OOM | - | OOM / OOM |

### deepseek-v2-lite EP8, 2 packs per step, both schedules at the same lengths

| pack tokens | overlapped s (3 steps) | sequential s (3 steps) | peak alloc / reserved GiB (ovl) | (seq) |
|---|---|---|---|---|
| 8192 | 0.996 (0.99, 1.00, 1.01) | 0.994 (0.99, 0.99, 1.01) | 70.2 / 78.6 | 64.7 / 67.8 |
| 12288 | 1.346 (1.35, 1.84, 1.35) | 1.350 (1.35, 1.30, 1.36) | 94.4 / 108.5 | 81.0 / 85.7 |
| 16384 | 2.734 (2.73, 3.34, 1.83) | 1.663 (1.66, 1.61, 1.67) | 118.7 / 133.3 | 97.3 / 103.6 |
| 20480 | OOM | 1.952 (1.95, 1.93, 1.97) | - | 113.6 / 117.1 |
| 24576 | not run (shorter length OOMed) | 2.477 (2.48, 3.14, 2.39) | - | 130.0 / 134.0 |
| 32768 | not run (shorter length OOMed) | OOM | - | - |

### Largest pack before OOM

| model | mesh | M | schedule | largest fitting pack (tokens/rank/micro-batch) | its step s | its peak alloc / reserved GiB | first OOM |
|---|---|---|---|---|---|---|---|
| deepseek-v2-lite | DP8 (EP1) | 1 | overlapped | 8192 | 1.540 | 113.8 / 136.5 | 10240 |
| deepseek-v2-lite | DP8 (EP1) | 1 | sequential | 10240 | 3.035 | 105.7 / 136.1 | 12288 |
| deepseek-v2-lite | DP8 (EP1) | 2 | overlapped | >= 2048 (probe unfinished) | 1.288 | 110.7 / 125.5 | pending |
| deepseek-v2-lite | DP8 (EP1) | 2 | sequential | >= 1024 (probe unfinished) | 1.261 | 106.4 / 125.0 | pending |
| deepseek-v2-lite | EP8 | 1 | overlapped | 24576 | 1.311 | 119.5 / 125.6 | 32768 |
| deepseek-v2-lite | EP8 | 1 | sequential | 24576 | 1.316 | 119.5 / 121.6 | 32768 |
| deepseek-v2-lite | EP8 | 2 | overlapped | 16384 | 2.734 | 118.7 / 133.3 | 20480 |
| deepseek-v2-lite | EP8 | 2 | sequential | 24576 | 2.477 | 130.0 / 134.0 | 32768 |
| qwen3-30b-a3b | EP8 | 1 | overlapped | >= 8192 (probe unfinished) | 1.020 | 88.8 / 104.0 | pending |
| qwen3-30b-a3b | EP8 | 1 | sequential | >= 6144 (probe unfinished) | 0.903 | 76.5 / 93.5 | pending |

### nsys reports

| suite | workload | schedule | report | a2a ms | exposed a2a ms | overlap % | GPU idle ms |
|---|---|---|---|---|---|---|---|
| dsv2-ep1 | pack16k-M4-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep1-j2410103.16-a4.1.nsys-rep` | 0 | 0 | - | 111 |
| dsv2-ep1 | pack16k-M4-even | sequential | `workspace/pp1/nsys/main-dsv2-ep1-j2410103.16-a4.2.nsys-rep` | 0 | 0 | - | 598 |
| dsv2-ep8 | pack16k-M1-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-a1.5.nsys-rep` | 480 | 480 | 0 | 601 |
| dsv2-ep8 | pack16k-M1-even | sequential | `workspace/pp1/nsys/main-dsv2-ep8-a1.6.nsys-rep` | 112 | 111 | 0 | 373 |
| dsv2-ep8 | pack16k-M1-full | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-a1.7.nsys-rep` | 267 | 267 | 0 | 384 |
| dsv2-ep8 | pack16k-M1-full | sequential | `workspace/pp1/nsys/main-dsv2-ep8-a1.8.nsys-rep` | 292 | 292 | 0 | 859 |
| dsv2-ep8 | pack16k-M2-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-a1.9.nsys-rep` | 160 | 117 | 27 | 98 |
| dsv2-ep8 | pack16k-M2-even | sequential | `workspace/pp1/nsys/main-dsv2-ep8-a1.10.nsys-rep` | 374 | 373 | 0 | 631 |
| dsv2-ep8 | pack16k-M2-full | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-a1.13.nsys-rep` | 330 | 268 | 19 | 780 |
| dsv2-ep8 | pack16k-M2-full | sequential | `workspace/pp1/nsys/main-dsv2-ep8-a1.14.nsys-rep` | 339 | 339 | 0 | 298 |
| dsv2-ep8 | pack16k-M2-intra | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-a1.11.nsys-rep` | 197 | 156 | 21 | 913 |
| dsv2-ep8 | pack16k-M2-intra | sequential | `workspace/pp1/nsys/main-dsv2-ep8-a1.12.nsys-rep` | 170 | 169 | 0 | 1128 |
| dsv2-ep8 | uniform-seq2048-mb32 | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-a1.3.nsys-rep` | 1204 | 704 | 42 | 1904 |
| dsv2-ep8 | uniform-seq2048-mb32 | sequential | `workspace/pp1/nsys/main-dsv2-ep8-a1.4.nsys-rep` | 1191 | 1190 | 0 | 1486 |
| qwen3-ep8 | pack16k-M2-even | overlapped | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-d90e.nsys-rep` | 579 | 481 | 17 | 788 |
| qwen3-ep8 | pack16k-M2-even | sequential | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-058f.nsys-rep` | 486 | 485 | 0 | 967 |
| qwen3-ep8 | pack16k-M2-intra | overlapped | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-d6c0.nsys-rep` | 372 | 273 | 27 | 129 |
| qwen3-ep8 | pack16k-M4-even | overlapped | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-c88a.nsys-rep` | 577 | 305 | 47 | 635 |
| qwen3-ep8 | pack16k-M4-even | sequential | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-92b3.nsys-rep` | 458 | 457 | 0 | 1191 |
| qwen3-ep8 | pack16k-M4-full | overlapped | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-7b56.nsys-rep` | 1077 | 729 | 33 | 1091 |
| qwen3-ep8 | pack16k-M4-full | sequential | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-77f5.nsys-rep` | 972 | 971 | 0 | 587 |
| qwen3-ep8 | pack16k-M4-intra | overlapped | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-3686.nsys-rep` | 485 | 336 | 31 | 1422 |
| qwen3-ep8 | pack16k-M4-intra | sequential | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-1c95.nsys-rep` | 582 | 582 | 0 | 1274 |
| qwen3-ep8 | pack16k-M8-even | overlapped | `workspace/pp1/nsys/main-qwen3-ep8-j2410103.11-a16.1.nsys-rep` | 660 | 451 | 32 | 1667 |
| qwen3-ep8 | pack16k-M8-even | sequential | `workspace/pp1/nsys/main-qwen3-ep8-j2410103.11-a16.2.nsys-rep` | 643 | 643 | 0 | 1680 |
| qwen3-ep8 | pack16k-M8-intra | overlapped | `workspace/pp1/nsys/main-qwen3-ep8-j2410103.11-a16.3.nsys-rep` | 759 | 521 | 31 | 1805 |
| qwen3-ep8 | pack16k-M8-intra | sequential | `workspace/pp1/nsys/main-qwen3-ep8-j2410103.11-a16.4.nsys-rep` | 1039 | 1039 | 0 | 1393 |
| qwen3-ep8 | uniform-seq2048-mb32 | overlapped | `workspace/pp1/nsys-recovered/j2410103.11-nsys-report-818a.nsys-rep` | 2533 | 1335 | 47 | 2845 |

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
