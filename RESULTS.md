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

## Analysis

All numbers are from the tables in [Results](#results). "ovl" is the overlapped DualPipeV schedule,
"seq" the sequential one; percentages are seq step time relative to ovl (positive: seq slower), from
the fastest of the 3 timed steps (see [Method](#method)).

**Where the overlap pays off.** On qwen3-30b-a3b EP8, ovl is faster in every workload measured:
uniform pretraining by 19.2% at M=32 (6.57 vs 7.83 s) and 10.5% at M=128 (24.61 vs 27.20 s), packs by
3.1-3.4% at M=2, 2.2-7.4% at M=4 and 7.6-12.6% at M=8. The traces show why: ovl hides 17-47% of the
EP all-to-all, seq hides almost none (uniform M=32: 1.34 s of 2.53 s exposed with ovl; M=8 even:
0.45 vs 0.64 s exposed). On deepseek-v2-lite EP8 the gain is smaller: uniform pretraining 11.9% at
M=32 (4.05 vs 4.53 s; exposed all-to-all 0.74 vs 0.99 s) and 3.3% at M=128; packs range from -5.2%
(M=8 even, seq faster) to +8.5% (M=8 full), and are within 1.3% at M=1 and M=2.

**Where sequential wins.**
- *Single micro-batch*: there is no pairing, and the schedules run at the same speed (dsv2 EP8: 0.994
  vs 0.991 s even, 1.218 vs 1.218 s uneven). The largest single pack is the same on dsv2 EP8 (24k
  fits, 32k OOMs) and qwen3 EP8 (14,336 fits, 16,384 OOMs). On dsv2 DP8, seq fits one probe step more
  (10,240 vs 8,192). ovl still defers the last chunk-0 backward's weight gradients at M=1; whether
  that explains the DP8 difference is not verified.
- *Memory*: with M >= 2, seq has the lower reserved peak in every pair measured and a lower or equal
  allocated peak: dsv2 EP8 M=2 even 64.7 vs 70.2 GiB allocated (68.2 vs 80.0 reserved), M=2 full
  85.1 vs 95.4 (91.9 vs 113.0); qwen3 EP8 M=2 even 106.3 vs 112.1 (114.5 vs 128.5). ovl keeps two
  micro-batches' activations alive, seq one.
- *Bigger packs*: with 2 packs per step, seq fits about 1.5x the pack: dsv2 EP8 24,576 vs 16,384
  tokens; dsv2 DP8 6,144 vs 4,096; qwen3 EP8 at least 12,288 (the longest probed) vs 10,240.
- *Near the memory limit*: at equal lengths below it the schedules are equal (dsv2 EP8 2 packs: 8k
  0.991 vs 0.992 s in the first probe, 12k 1.31 vs 1.32 s in the rerun). From about 130 GiB reserved,
  steps slow down and vary. In the dsv2 EP8 2-pack rerun every slow step recorded 1-2 allocator
  retries and 90-302 cudaMalloc calls and every normal-speed step none; in the DP8 reruns (125-137
  GiB reserved) retries and cudaMalloc bursts (up to 609) appear in most steps, and one slow step
  (2.43 s) had 60 cudaMalloc calls without a retry
  ([Allocator pressure](#allocator-pressure-reruns-that-record-it)).
  It hits whichever schedule gets there: ovl at 16k (1.81-2.80 s, against seq's 1.63-1.69 s at 16k),
  seq at 24k (3.05-3.15 s). So near the limit seq is faster because it has more headroom, not because
  the schedule is cheaper.

**Imbalance.** Uneven budgets across ranks (`full`) cost the most: on dsv2 EP8, +0.22 to +0.40 s over
the even step with ovl and +0.23 to +0.65 s with seq (22-42% of the even step). The all-to-all waits
for the rank with the most tokens: per-rank compute in the profiled step spreads 325-653 ms for dsv2
M=2 full (448-480 even), 442-986 ms for M=8 full (592-604 even), and 756-1397 ms for qwen3 M=4 full
(834-893 even). Uneven lengths within a rank (`intra`, so paired micro-batches differ) cost dsv2 at
most +0.16 s, and qwen3 more: M=4 +0.31 s with ovl vs +0.23 s with seq, M=8 +0.41 vs +0.33. ovl loses
more of its gain to mismatched pairs, but stays faster in absolute terms (qwen3 M=4 intra 1.952 vs
1.995 s; M=8 intra 2.577 vs 2.773 s).

**Crossover.** At a fixed 131k tokens per step, fewer and larger packs are faster under both schedules
(dsv2 EP8 even: 0.99, 1.02, 1.11, 1.54-1.62 s at M = 1, 2, 4, 8; qwen3 EP8 ovl even: 1.54, 1.64, 2.17 s
at M = 2, 4, 8), and going from M=8 to M=2 saves more than the overlap ever gains. The overlap's gain
grows with M: none at M=1, within 3.4% at M=2 (where seq has 8-21 GiB more reserved headroom), and
clear on qwen3 from M=4 (2.2-7.4%) and M=8 (7.6-12.6%). For few large packs the measured trade is: at
M=1 the schedules are equivalent; at M=2 seq costs 0.1-3.4% of speed and buys about 1.5x the pack
length; with many micro-batches ovl is the faster choice.

**Full DP8 (EP1) vs EP8.** Measured back to back on one node (fs-mbz-gpu-028), deepseek-v2-lite
uniform pretraining: at M=32 DP8 is 18.0% slower than EP8 (4.82 vs 4.08 s); at M=128 DP8 is 2.2%
faster than EP8's best step (16.25 vs 16.62 s), but EP8's three steps there spread 16.6-18.0 s, wider
than the gap, so at M=128 the two are even within noise. From M=32 to M=128 on that node, each extra
micro-batch costs DP8 0.119 s and EP8 0.126 s (seq) or 0.134 s (ovl), while DP8's M=32 step is 0.7 s
slower: DP8 can only catch up over long, uniform steps, and with EP8's spread at M=128 that crossover
is not resolved. The same-node traces at M=32 show where the time
goes (one profiled step each): DP8 has no all-to-all but 3.62-3.70 s of compute per rank, against
2.15-2.44 s on EP8, which exposes 0.80 s (ovl) or 1.17 s (seq) of all-to-all; other exposed
communication (the end-of-step ZeRO-1 reduce-scatter and the small collectives) is 0.47-1.03 s on DP8
and 0.80-0.86 s on EP8. So DP8 trades the exposed all-to-all for about 1.2-1.5 s more compute per
step. That the extra compute is the 64 replicated experts' grouped GEMMs on smaller per-expert token
groups is plausible but not broken down per kernel. On packs, DP8 is 30-58% slower wherever it fits
(M=4 even, M=8 even and intra), at 125-136 GiB reserved under allocator pressure (retries recorded),
and it OOMs for every packed workload with M <= 2 and every workload uneven across ranks. These
packed comparisons are across nodes, but node-to-node differences seen for the same configuration
(dsv2 EP8 uniform M=128: 15.4-16.0 s on one node, 16.6-18.0 s and 17.2-17.8 s on two others) are far
smaller than 30-58%. On qwen3-30b-a3b, DP8 does not fit at any workload tested: every one OOMs in
the first step under both schedules, including 2,048-token micro-batches, at 133-135 GiB allocated
(the at-rest, sharded footprint after setup is 14.2 GiB). Shorter probes were skipped: by the user's
estimate (not measured), replicated experts under ZeRO-1 need about 30B x (2 B bf16 weights + 4 B
fp32 main_grad), about 180 GB per GPU, before activations.

**Is the sequential schedule worth adding?** By these measurements, as an opt-in for few, large
packs: it never makes M=1 faster (same code path but for the deferred weight gradients), and at M=2
it is within 3.4% of ovl while fitting about 1.5x longer packs, and staying faster near the memory
limit where ovl runs into allocator retries. For pretraining or many micro-batches, the overlap is
worth 3-19% and should stay the default.

## Results

### deepseek-v2-lite EP8

| workload | M | max pack | step s, fastest of 3 (all 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 32 | 2048 | 4.050 (4.11, 4.05, 4.11) / 4.532 (4.53, 4.64, 4.62) | +11.9% | 16180 / 14461 | 40.7 / 40.2 | 58.3 / 55.6 | 735 / 994 | 41 | 2382-2429 | 108 |
| uniform-seq2048-mb128 | 128 | 2048 | 15.448 (15.45, 15.51, 15.51) / 15.950 (15.95, 16.10, 15.96) | +3.3% | 16970 / 16435 | 40.8 / 40.2 | 58.3 / 55.7 | - / - | - | - | 108 |
| pack16k-M1-even | 1 | 16384 | 0.994 (21.56, 1.58, 0.99) / 0.991 (1.02, 0.99, 1.01) | -0.3% | 16482 / 16526 | 86.8 / 86.8 | 88.8 / 88.8 | 194 / 181 | 0 | 488-534 | 108 |
| pack16k-M1-full | 1 | 27648 | 1.218 (1.22, 1.23, 1.31) / 1.218 (1.22, 1.24, 1.31) | -0.0% | 13448 / 13449 | 115.7 / 115.7 | 122.6 / 122.7 | 251 / 264 | 0 | 224-507 | 108 |
| pack16k-M2-even | 2 | 8192 | 1.022 (1.02, 1.03, 1.04) / 1.023 (1.03, 1.02, 1.04) | +0.1% | 16026 / 16009 | 70.2 / 64.7 | 80.0 / 68.2 | 350 / 168 | 11 | 448-480 | 108 |
| pack16k-M2-intra | 2 | 9216 | 1.017 (1.02, 1.02, 1.02) / 1.029 (1.03, 1.04, 1.05) | +1.1% | 16103 / 15921 | 69.7 / 68.8 | 83.1 / 75.0 | 186 / 371 | 18 | 438-466 | 108 |
| pack16k-M2-full | 2 | 15616 | 1.305 (1.31, 1.31, 1.35) / 1.322 (1.32, 1.32, 1.36) | +1.3% | 12553 / 12396 | 95.4 / 85.1 | 113.0 / 91.9 | 389 / 349 | 15 | 325-653 | 108 |
| pack16k-M4-even | 4 | 4096 | 1.107 (1.19, 1.11, 1.13) / 1.144 (1.14, 1.15, 1.15) | +3.4% | 14799 / 14316 | 49.3 / 48.3 | 64.6 / 59.8 | 178 / 247 | 42 | 543-574 | 108 |
| pack16k-M4-intra | 4 | 9856 | 1.264 (1.29, 1.26, 1.30) / 1.263 (1.26, 1.26, 1.29) | -0.1% | 12965 / 12973 | 71.7 / 71.5 | 83.5 / 78.0 | 180 / 215 | 30 | 467-506 | 108 |
| pack16k-M4-full | 4 | 11648 | 1.452 (1.45, 1.51, 1.46) / 1.508 (1.52, 1.53, 1.51) | +3.9% | 11282 / 10861 | 71.1 / 70.4 | 89.1 / 81.8 | 327 / 435 | 36 | 487-911 | 108 |
| pack16k-M8-even | 8 | 2048 | 1.622 (1.62, 1.68, 1.69) / 1.538 (1.61, 1.54, 1.64) | -5.2% | 10098 / 10650 | 40.8 / 40.3 | 60.0 / 56.9 | 189 / 253 | 35 | 592-604 | 108 |
| pack16k-M8-intra | 8 | 5120 | 1.644 (1.66, 1.65, 1.64) / 1.649 (1.67, 1.65, 1.71) | +0.3% | 9965 / 9933 | 49.1 / 49.0 | 68.6 / 62.5 | 179 / 346 | 38 | 631-653 | 108 |
| pack16k-M8-full | 8 | 20224 | 2.018 (2.03, 2.02, 2.02) / 2.190 (2.78, 2.28, 2.19) | +8.5% | 8118 / 7481 | 89.4 / 89.2 | 111.4 / 107.8 | 588 / 800 | 27 | 442-986 | 108 |

Imbalance cost, deepseek-v2-lite EP8: step time of uneven packs minus even packs, same tokens and M.

| M | schedule | even s | intra s (cost) | full s (cost) |
|---|---|---|---|---|
| 1 | overlapped | 0.994 | - | 1.218 (+0.224) |
| 1 | sequential | 0.991 | - | 1.218 (+0.227) |
| 2 | overlapped | 1.022 | 1.017 (-0.005) | 1.305 (+0.283) |
| 2 | sequential | 1.023 | 1.029 (+0.006) | 1.322 (+0.298) |
| 4 | overlapped | 1.107 | 1.264 (+0.157) | 1.452 (+0.345) |
| 4 | sequential | 1.144 | 1.263 (+0.118) | 1.508 (+0.364) |
| 8 | overlapped | 1.622 | 1.644 (+0.022) | 2.018 (+0.396) |
| 8 | sequential | 1.538 | 1.649 (+0.111) | 2.190 (+0.652) |

### deepseek-v2-lite DP8 (EP1)

| workload | M | max pack | step s, fastest of 3 (all 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 32 | 2048 | 4.843 (4.85, 4.84, 4.85) / 4.846 (4.85, 4.85, 4.85) | +0.0% | 13531 / 13525 | 111.0 / 110.5 | 130.6 / 132.6 | 0 / 0 | - | 3612-3696 | 052 |
| uniform-seq2048-mb128 | 128 | 2048 | 16.332 (16.40, 16.33, 16.36) / 16.396 (16.40, 16.40, 16.41) | +0.4% | 16051 / 15989 | 111.0 / 110.6 | 130.6 / 132.6 | - / - | - | - | 411 |
| pack16k-M4-even | 4 | 4096 | 1.771 (1.93, 1.81, 1.77) / 1.752 (2.43, 1.75, 1.77) | -1.0% | 9254 / 9350 | 119.5 / 118.6 | 133.3 / 136.1 | 0 / 0 | - | 677-695 | 052 |
| pack16k-M8-even | 8 | 2048 | 2.008 (2.01, 2.07, 2.01) / 2.163 (2.17, 2.17, 2.16) | +7.7% | 8158 / 7576 | 111.0 / 110.5 | 125.6 / 132.6 | 0 / 0 | - | 934-946 | 052 |
| pack16k-M8-intra | 8 | 5120 | 2.198 (4.38, 2.90, 2.20) / 2.270 (2.29, 2.33, 2.27) | +3.3% | 7453 / 7218 | 119.2 / 119.2 | 136.1 / 136.5 | 0 / 0 | - | 1024-1049 | 052 |
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
| 4 | overlapped | 1.771 | OOM | OOM |
| 4 | sequential | 1.752 | OOM | OOM |
| 8 | overlapped | 2.008 | 2.198 (+0.190) | OOM |
| 8 | sequential | 2.163 | 2.270 (+0.107) | OOM |

### qwen3-30b-a3b EP8

| workload | M | max pack | step s, fastest of 3 (all 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 32 | 2048 | 6.567 (6.63, 6.57, 6.62) / 7.825 (8.13, 8.26, 7.83) | +19.2% | 9980 / 8375 | 74.6 / 74.6 | 103.1 / 95.6 | 1335 / - | 47 | 3742-3827 | 108 |
| uniform-seq2048-mb128 | 128 | 2048 | 24.610 (24.63, 24.61, 25.45) / 27.196 (27.21, 27.37, 27.20) | +10.5% | 10652 / 9639 | 74.6 / 74.6 | 103.1 / 95.6 | - / - | - | - | 108 |
| pack16k-M2-even | 2 | 8192 | 1.543 (1.81, 1.54, 1.57) / 1.596 (1.60, 1.60, 2.48) | +3.4% | 10618 / 10267 | 112.1 / 106.3 | 128.5 / 114.5 | 481 / 485 | 17 | 703-806 | 108 |
| pack16k-M2-intra | 2 | 9216 | 1.560 (1.56, 1.57, 1.58) / 1.608 (1.61, 1.63, 1.63) | +3.1% | 10503 / 10192 | 113.7 / 112.5 | 127.2 / 117.5 | 273 / - | 27 | 714-805 | 108 |
| pack16k-M4-even | 4 | 4096 | 1.643 (1.65, 1.65, 1.64) / 1.765 (1.78, 1.76, 1.77) | +7.4% | 9970 / 9285 | 82.4 / 81.7 | 104.7 / 95.3 | 305 / 457 | 47 | 834-893 | 108 |
| pack16k-M4-intra | 4 | 9856 | 1.952 (1.99, 1.95, 2.03) / 1.995 (2.00, 2.00, 2.08) | +2.2% | 8392 / 8210 | 116.7 / 116.4 | 133.8 / 128.1 | 336 / 582 | 31 | 774-888 | 108 |
| pack16k-M4-full | 4 | 11648 | 2.286 (2.29, 2.43, 2.32) / 2.453 (2.45, 2.48, 2.46) | +7.3% | 7167 / 6679 | 113.0 / 112.2 | 134.0 / 128.5 | 729 / 971 | 33 | 756-1397 | 108 |
| pack16k-M8-even | 8 | 2048 | 2.171 (2.18, 2.17, 2.18) / 2.446 (2.97, 2.45, 2.45) | +12.6% | 7546 / 6699 | 74.6 / 74.6 | 103.1 / 95.6 | 451 / 643 | 32 | 909-933 | 108 |
| pack16k-M8-intra | 8 | 5120 | 2.577 (2.61, 2.60, 2.58) / 2.773 (2.77, 2.82, 2.80) | +7.6% | 6357 / 5909 | 82.7 / 82.6 | 109.5 / 100.0 | 521 / 1039 | 31 | 969-1006 | 108 |
| pack16k-M1-even | 1 | 16384 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M2-full | 2 | 15616 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M8-full | 8 | 20224 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |
| pack16k-M1-full | 1 | 27648 | OOM / OOM | - | OOM / OOM | OOM / OOM | OOM / OOM | - / - | - | - | - |

Imbalance cost, qwen3-30b-a3b EP8: step time of uneven packs minus even packs, same tokens and M.

| M | schedule | even s | intra s (cost) | full s (cost) |
|---|---|---|---|---|
| 1 | overlapped | OOM | - | OOM |
| 1 | sequential | OOM | - | OOM |
| 2 | overlapped | 1.543 | 1.560 (+0.017) | OOM |
| 2 | sequential | 1.596 | 1.608 (+0.012) | OOM |
| 4 | overlapped | 1.643 | 1.952 (+0.309) | 2.286 (+0.643) |
| 4 | sequential | 1.765 | 1.995 (+0.231) | 2.453 (+0.689) |
| 8 | overlapped | 2.171 | 2.577 (+0.406) | OOM |
| 8 | sequential | 2.446 | 2.773 (+0.327) | OOM |

### qwen3-30b-a3b DP8 (EP1)

| workload | M | max pack | step s, fastest of 3 (all 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |
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

### DP8 (EP1) vs EP8 on one node, deepseek-v2-lite uniform pretraining

| workload | host | EP8 ovl s | EP8 seq s | DP8 ovl s | DP8 seq s | best DP8 vs best EP8 | peak alloc / reserved GiB EP8 ovl | DP8 ovl | exposed a2a ms EP8 ovl / seq |
|---|---|---|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 028 | 4.082 (4.17, 4.08, 4.10) | 4.513 (4.55, 4.51, 4.52) | 4.818 (4.83, 4.82, 4.82) | 4.829 (4.83, 4.83, 4.83) | +18.0% | 40.7 / 58.2 | 111.0 / 125.4 | 798 / 1174 |
| uniform-seq2048-mb128 | 028 | 16.972 (17.48, 17.96, 16.97) | 16.621 (16.62, 16.93, 17.51) | 16.252 (16.25, 16.28, 16.70) | 16.308 (16.31, 16.31, 16.31) | -2.2% | 40.7 / 58.2 | 111.0 / 125.4 | - / - |

### DP8 (EP1) vs EP8 across nodes, deepseek-v2-lite

| workload | EP8 overlapped s | EP8 sequential s | DP8 overlapped s | DP8 sequential s | best DP8 vs best EP8 | peak alloc GiB EP8 / DP8 (overlapped) |
|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 4.050 | 4.532 | 4.843 | 4.846 | +19.6% | 40.7 / 111.0 |
| uniform-seq2048-mb128 | 15.448 | 15.950 | 16.332 | 16.396 | +5.7% | 40.8 / 111.0 |
| pack16k-M1-even | 0.994 | 0.991 | OOM | OOM | - | 86.8 / OOM |
| pack16k-M1-full | 1.218 | 1.218 | OOM | OOM | - | 115.7 / OOM |
| pack16k-M2-even | 1.022 | 1.023 | OOM | OOM | - | 70.2 / OOM |
| pack16k-M2-intra | 1.017 | 1.029 | OOM | OOM | - | 69.7 / OOM |
| pack16k-M2-full | 1.305 | 1.322 | OOM | OOM | - | 95.4 / OOM |
| pack16k-M4-even | 1.107 | 1.144 | 1.771 | 1.752 | +58.3% | 49.3 / 119.5 |
| pack16k-M4-intra | 1.264 | 1.263 | OOM | OOM | - | 71.7 / OOM |
| pack16k-M4-full | 1.452 | 1.508 | OOM | OOM | - | 71.1 / OOM |
| pack16k-M8-even | 1.622 | 1.538 | 2.008 | 2.163 | +30.6% | 40.8 / 111.0 |
| pack16k-M8-intra | 1.644 | 1.649 | 2.198 | 2.270 | +33.7% | 49.1 / 119.2 |
| pack16k-M8-full | 2.018 | 2.190 | OOM | OOM | - | 89.4 / OOM |

### DP8 (EP1) vs EP8 across nodes, qwen3-30b-a3b

| workload | EP8 overlapped s | EP8 sequential s | DP8 overlapped s | DP8 sequential s | best DP8 vs best EP8 | peak alloc GiB EP8 / DP8 (overlapped) |
|---|---|---|---|---|---|---|
| uniform-seq2048-mb32 | 6.567 | 7.825 | OOM | OOM | - | 74.6 / OOM |
| uniform-seq2048-mb128 | 24.610 | 27.196 | OOM | OOM | - | 74.6 / OOM |
| pack16k-M2-even | 1.543 | 1.596 | OOM | OOM | - | 112.1 / OOM |
| pack16k-M2-intra | 1.560 | 1.608 | OOM | OOM | - | 113.7 / OOM |
| pack16k-M4-even | 1.643 | 1.765 | OOM | OOM | - | 82.4 / OOM |
| pack16k-M4-intra | 1.952 | 1.995 | OOM | OOM | - | 116.7 / OOM |
| pack16k-M4-full | 2.286 | 2.453 | OOM | OOM | - | 113.0 / OOM |
| pack16k-M8-even | 2.171 | 2.446 | OOM | OOM | - | 74.6 / OOM |
| pack16k-M8-intra | 2.577 | 2.773 | OOM | OOM | - | 82.7 / OOM |
| pack16k-M1-even | OOM | OOM | OOM | OOM | - | OOM / OOM |
| pack16k-M2-full | OOM | OOM | OOM | OOM | - | OOM / OOM |
| pack16k-M8-full | OOM | OOM | OOM | OOM | - | OOM / OOM |
| pack16k-M1-full | OOM | OOM | OOM | OOM | - | OOM / OOM |

### deepseek-v2-lite EP8, 2 packs per step, both schedules at the same lengths

| pack tokens | overlapped s, fastest (all 3) | sequential s, fastest (all 3) | peak alloc / reserved GiB (ovl) | (seq) |
|---|---|---|---|---|
| 8192 | 0.991 (0.99, 1.00, 1.01) | 0.992 (0.99, 0.99, 1.01) | 70.2 / 78.6 | 64.7 / 67.8 |
| 12288 | 1.345 (1.35, 1.84, 1.35) | 1.301 (1.35, 1.30, 1.36) | 94.4 / 108.5 | 81.0 / 85.7 |
| 16384 | 1.831 (2.73, 3.34, 1.83) | 1.613 (1.66, 1.61, 1.67) | 118.7 / 133.3 | 97.3 / 103.6 |
| 20480 | OOM | 1.931 (1.95, 1.93, 1.97) | - | 113.6 / 117.1 |
| 24576 | not run (shorter length OOMed) | 2.387 (2.48, 3.14, 2.39) | - | 130.0 / 134.0 |
| 32768 | not run (shorter length OOMed) | OOM | - | - |

### deepseek-v2-lite EP8, rerun with allocator stats, 2 packs per step, both schedules at the same lengths

| pack tokens | overlapped s, fastest (all 3) | sequential s, fastest (all 3) | peak alloc / reserved GiB (ovl) | (seq) |
|---|---|---|---|---|
| 12288 | 1.308 (1.34, 1.31, 1.34; retries [0, 0, 0], mallocs [0, 0, 0]) | 1.324 (1.37, 1.32, 1.35; retries [0, 0, 0], mallocs [0, 0, 0]) | 94.4 / 106.6 | 81.0 / 85.6 |
| 16384 | 1.814 (2.80, 1.81, 1.85; retries [1, 1, 1], mallocs [123, 117, 90]) | 1.630 (1.69, 1.63, 1.68; retries [0, 0, 0], mallocs [0, 0, 0]) | 118.7 / 133.3 | 97.3 / 103.6 |
| 20480 | OOM | 1.959 (1.99, 1.96, 1.98; retries [0, 0, 0], mallocs [4, 0, 0]) | - | 113.6 / 120.8 |
| 24576 | not run (shorter length OOMed) | 3.049 (3.15, 3.05, 3.13; retries [2, 2, 2], mallocs [300, 302, 302]) | - | 130.0 / 133.9 |

### deepseek-v2-lite DP8 (EP1), 2 packs per step, both schedules at the same lengths

| pack tokens | overlapped s, fastest (all 3) | sequential s, fastest (all 3) | peak alloc / reserved GiB (ovl) | (seq) |
|---|---|---|---|---|
| 1024 | 1.254 (1.29, 1.25, 1.68) | 1.260 (1.26, 1.58, 1.26) | 106.5 / 130.3 | 106.4 / 125.0 |
| 2048 | 1.287 (1.32, 1.29, 1.29; retries [0, 0, 0], mallocs [46, 0, 0]) | 1.288 (1.29, 1.29, 1.29; retries [0, 0, 0], mallocs [0, 1, 0]) | 110.7 / 125.4 | 110.5 / 132.2 |
| 3072 | 1.353 (1.36, 1.36, 1.35; retries [0, 0, 0], mallocs [34, 1, 0]) | 1.350 (1.35, 1.35, 1.35; retries [0, 0, 0], mallocs [1, 0, 0]) | 114.9 / 136.2 | 114.6 / 136.5 |
| 4096 | 1.404 (1.49, 1.40, 1.49; retries [1, 0, 1], mallocs [40, 30, 3]) | 1.388 (1.43, 1.39, 1.39; retries [0, 0, 0], mallocs [65, 0, 1]) | 119.1 / 129.0 | 118.6 / 134.3 |
| 6144 | OOM | 1.548 (1.91, 1.81, 1.55; retries [1, 1, 0], mallocs [110, 84, 65]) | - | 126.8 / 133.7 |
| 8192 | not run (shorter length OOMed) | OOM | - | - |

### qwen3-30b-a3b EP8, 2 packs per step, both schedules at the same lengths

| pack tokens | overlapped s, fastest (all 3) | sequential s, fastest (all 3) | peak alloc / reserved GiB (ovl) | (seq) |
|---|---|---|---|---|
| 4096 | 1.088 (1.12, 10.26, 1.09; retries [0, 0, 0], mallocs [5, 1, 0]) | 1.108 (1.12, 1.12, 1.11; retries [0, 0, 0], mallocs [0, 1, 0]) | 82.4 / 100.2 | 81.7 / 95.9 |
| 6144 | 1.300 (1.30, 1.31, 1.32; retries [0, 0, 0], mallocs [0, 0, 0]) | 1.307 (1.31, 1.34, 1.34; retries [0, 0, 0], mallocs [0, 0, 0]) | 95.0 / 117.5 | 94.0 / 105.9 |
| 8192 | 1.529 (1.53, 1.54, 1.57; retries [0, 0, 0], mallocs [0, 0, 0]) | 1.566 (1.57, 1.57, 1.61; retries [0, 0, 0], mallocs [0, 0, 0]) | 112.2 / 128.6 | 106.3 / 116.6 |
| 10240 | 2.195 (2.37, 2.20, 2.24; retries [3, 3, 3], mallocs [168, 164, 160]) | 1.809 (1.88, 1.81, 1.89; retries [0, 0, 0], mallocs [0, 0, 0]) | 130.3 / 133.9 | 118.6 / 121.3 |
| 12288 | OOM | 2.026 (2.17, 2.03, 2.14; retries [0, 0, 0], mallocs [1, 0, 0]) | - | 130.9 / 132.8 |

### Allocator pressure (reruns that record it)

| suite | workload | schedule | host | step s | alloc retries | cudaMalloc calls | peak reserved GiB |
|---|---|---|---|---|---|---|---|
| main-dsv2-ep8-r2 | uniform-seq2048-mb32 | overlapped | 028 | 4.20, 4.15, 4.28 | [0, 0, 0] | [3, 2, 2] | 57.4, 57.8, 58.3 |
| main-dsv2-ep8-r2 | uniform-seq2048-mb32 | overlapped | 108 | 3.98, 4.09, 4.25 | [0, 0, 0] | [3, 2, 2] | 57.4, 57.8, 58.3 |
| main-dsv2-ep8-r2 | uniform-seq2048-mb32 | overlapped | 108 | 4.11, 4.05, 4.11 | [0, 0, 0] | [3, 2, 2] | 57.4, 57.8, 58.3 |
| main-dsv2-ep8-r2 | uniform-seq2048-mb32 | sequential | 108 | 4.53, 4.64, 4.62 | [0, 0, 0] | [1, 1, 1] | 54.8, 55.3, 55.6 |
| main-dsv2-ep8-r2 | uniform-seq2048-mb128 | overlapped | 108 | 15.45, 15.51, 15.51 | [0, 0, 0] | [4, 4, 2] | 57.4, 57.9, 58.3 |
| main-dsv2-ep8-r2 | uniform-seq2048-mb128 | sequential | 108 | 15.95, 16.10, 15.96 | [0, 0, 0] | [2, 1, 2] | 54.8, 55.3, 55.7 |
| main-dsv2-ep8-r2 | pack16k-M1-even | overlapped | 108 | 21.56, 1.58, 0.99 | [0, 0, 0] | [0, 0, 0] | 88.8, 88.8, 88.8 |
| main-dsv2-ep8-r2 | pack16k-M1-even | sequential | 108 | 1.02, 0.99, 1.01 | [0, 0, 0] | [0, 0, 0] | 88.8, 88.8, 88.8 |
| main-dsv2-ep8-r2 | pack16k-M1-full | overlapped | 108 | 1.22, 1.23, 1.31 | [0, 0, 0] | [0, 0, 0] | 122.6, 122.6, 122.6 |
| main-dsv2-ep8-r2 | pack16k-M1-full | sequential | 108 | 1.22, 1.24, 1.31 | [0, 0, 0] | [0, 0, 0] | 122.7, 122.7, 122.7 |
| main-dsv2-ep8-r2 | pack16k-M2-even | overlapped | 108 | 1.02, 1.03, 1.04 | [0, 0, 0] | [0, 0, 0] | 80.0, 80.0, 80.0 |
| main-dsv2-ep8-r2 | pack16k-M2-even | sequential | 108 | 1.03, 1.02, 1.04 | [0, 0, 0] | [0, 0, 0] | 68.2, 68.2, 68.2 |
| main-dsv2-ep8-r2 | pack16k-M2-intra | overlapped | 108 | 1.02, 1.02, 1.02 | [0, 0, 0] | [4, 0, 1] | 83.1, 83.1, 83.1 |
| main-dsv2-ep8-r2 | pack16k-M2-intra | sequential | 108 | 1.03, 1.04, 1.05 | [0, 0, 0] | [1, 0, 0] | 75.0, 75.0, 75.0 |
| main-dsv2-ep8-r2 | pack16k-M2-full | overlapped | 108 | 1.31, 1.31, 1.35 | [0, 0, 0] | [1, 0, 1] | 113.0, 113.0, 113.0 |
| main-dsv2-ep8-r2 | pack16k-M2-full | sequential | 108 | 1.32, 1.32, 1.36 | [0, 0, 0] | [0, 0, 0] | 91.9, 91.9, 91.9 |
| main-dsv2-ep8-r2 | pack16k-M4-even | overlapped | 108 | 1.19, 1.11, 1.13 | [0, 0, 0] | [2, 1, 1] | 63.7, 64.2, 64.6 |
| main-dsv2-ep8-r2 | pack16k-M4-even | sequential | 108 | 1.14, 1.15, 1.15 | [0, 0, 0] | [2, 1, 0] | 59.4, 59.8, 59.8 |
| main-dsv2-ep8-r2 | pack16k-M4-intra | overlapped | 108 | 1.29, 1.26, 1.30 | [0, 0, 0] | [1, 0, 1] | 83.5, 83.5, 83.5 |
| main-dsv2-ep8-r2 | pack16k-M4-intra | sequential | 108 | 1.26, 1.26, 1.29 | [0, 0, 0] | [1, 1, 1] | 78.0, 78.0, 78.0 |
| main-dsv2-ep8-r2 | pack16k-M4-full | overlapped | 108 | 1.45, 1.51, 1.46 | [0, 0, 0] | [4, 5, 1] | 89.1, 89.1, 89.1 |
| main-dsv2-ep8-r2 | pack16k-M4-full | sequential | 108 | 1.52, 1.53, 1.51 | [0, 0, 0] | [2, 1, 1] | 81.8, 81.8, 81.8 |
| main-dsv2-ep8-r2 | pack16k-M8-even | overlapped | 108 | 1.62, 1.68, 1.69 | [0, 0, 0] | [2, 2, 1] | 59.6, 60.0, 60.0 |
| main-dsv2-ep8-r2 | pack16k-M8-even | sequential | 108 | 1.61, 1.54, 1.64 | [0, 0, 0] | [1, 1, 1] | 56.4, 56.9, 56.9 |
| main-dsv2-ep8-r2 | pack16k-M8-intra | overlapped | 108 | 1.66, 1.65, 1.64 | [0, 0, 0] | [4, 4, 4] | 67.7, 68.2, 68.6 |
| main-dsv2-ep8-r2 | pack16k-M8-intra | sequential | 108 | 1.67, 1.65, 1.71 | [0, 0, 0] | [2, 1, 1] | 61.6, 62.1, 62.5 |
| main-dsv2-ep8-r2 | pack16k-M8-full | overlapped | 108 | 2.03, 2.02, 2.02 | [0, 0, 0] | [4, 3, 4] | 111.4, 111.4, 111.4 |
| main-dsv2-ep8-r2 | pack16k-M8-full | sequential | 108 | 2.78, 2.28, 2.19 | [0, 0, 0] | [2, 1, 1] | 107.8, 107.8, 107.8 |
| main-dsv2-ep1-r2 | uniform-seq2048-mb32 | overlapped | 267 | 4.87, 4.83, 4.83 | [0, 0, 0] | [49, 2, 1] | 130.6, 130.6, 130.6 |
| main-dsv2-ep1-r2 | uniform-seq2048-mb32 | overlapped | 052 | 4.85, 4.84, 4.85 | [0, 0, 0] | [49, 2, 1] | 130.6, 130.6, 130.6 |
| main-dsv2-ep1-r2 | uniform-seq2048-mb32 | sequential | 052 | 4.85, 4.85, 4.85 | [0, 0, 0] | [0, 0, 0] | 132.6, 132.6, 132.6 |
| main-dsv2-ep1-r2 | pack16k-M4-even | overlapped | 052 | 1.93, 1.81, 1.77 | [1, 1, 0] | [36, 25, 16] | 133.3, 130.3, 131.2 |
| main-dsv2-ep1-r2 | pack16k-M4-even | sequential | 052 | 2.43, 1.75, 1.77 | [0, 0, 0] | [60, 1, 0] | 136.1, 136.1, 136.1 |
| main-dsv2-ep1-r2 | pack16k-M8-even | overlapped | 052 | 2.01, 2.07, 2.01 | [0, 0, 0] | [0, 0, 0] | 125.6, 125.6, 125.6 |
| main-dsv2-ep1-r2 | pack16k-M8-even | sequential | 052 | 2.17, 2.17, 2.16 | [0, 0, 0] | [0, 0, 0] | 132.6, 132.6, 132.6 |
| main-dsv2-ep1-r2 | pack16k-M8-intra | overlapped | 052 | 4.38, 2.90, 2.20 | [1, 2, 1] | [140, 609, 145] | 135.5, 136.1, 130.5 |
| main-dsv2-ep1-r2 | pack16k-M8-intra | sequential | 052 | 2.29, 2.33, 2.27 | [0, 1, 0] | [126, 2, 111] | 136.5, 136.5, 136.5 |
| probe-dsv2-ep8-M2-r2 | oom-M2-L12288 | overlapped | 267 | 1.34, 1.31, 1.34 | [0, 0, 0] | [0, 0, 0] | 106.6, 106.6, 106.6 |
| probe-dsv2-ep8-M2-r2 | oom-M2-L12288 | sequential | 267 | 1.37, 1.32, 1.35 | [0, 0, 0] | [0, 0, 0] | 85.6, 85.6, 85.6 |
| probe-dsv2-ep8-M2-r2 | oom-M2-L16384 | overlapped | 267 | 2.80, 1.81, 1.85 | [1, 1, 1] | [123, 117, 90] | 131.9, 132.6, 133.3 |
| probe-dsv2-ep8-M2-r2 | oom-M2-L16384 | sequential | 267 | 1.69, 1.63, 1.68 | [0, 0, 0] | [0, 0, 0] | 103.6, 103.6, 103.6 |
| probe-dsv2-ep8-M2-r2 | oom-M2-L20480 | sequential | 267 | 1.99, 1.96, 1.98 | [0, 0, 0] | [4, 0, 0] | 120.8, 120.8, 120.8 |
| probe-dsv2-ep8-M2-r2 | oom-M2-L24576 | sequential | 267 | 3.15, 3.05, 3.13 | [2, 2, 2] | [300, 302, 302] | 133.9, 133.8, 133.8 |

### Largest pack before OOM

| model | mesh | M | schedule | largest fitting pack (tokens/rank/micro-batch) | its step s (fastest of 3) | its peak alloc / reserved GiB | first OOM |
|---|---|---|---|---|---|---|---|
| deepseek-v2-lite | DP8 (EP1) | 1 | overlapped | 8192 | 1.440 | 113.8 / 136.5 | 10240 |
| deepseek-v2-lite | DP8 (EP1) | 1 | sequential | 10240 | 1.504 | 105.7 / 136.1 | 12288 |
| deepseek-v2-lite | DP8 (EP1) | 2 | overlapped | 4096 | 1.404 | 119.1 / 129.0 | 6144 |
| deepseek-v2-lite | DP8 (EP1) | 2 | sequential | 6144 | 1.548 | 126.8 / 133.7 | 8192 |
| deepseek-v2-lite | EP8 | 1 | overlapped | 24576 | 1.284 | 119.5 / 125.6 | 32768 |
| deepseek-v2-lite | EP8 | 1 | sequential | 24576 | 1.278 | 119.5 / 121.6 | 32768 |
| deepseek-v2-lite | EP8 | 2 | overlapped | 16384 | 1.831 | 118.7 / 133.3 | 20480 |
| deepseek-v2-lite | EP8 | 2 | sequential | 24576 | 2.387 | 130.0 / 134.0 | 32768 |
| qwen3-30b-a3b | EP8 | 1 | overlapped | 14336 | 1.408 | 125.7 / 131.4 | - |
| qwen3-30b-a3b | EP8 | 1 | sequential | 14336 | 1.401 | 125.7 / 131.4 | - |
| qwen3-30b-a3b | EP8 | 2 | overlapped | 10240 | 2.195 | 130.3 / 133.9 | 12288 |
| qwen3-30b-a3b | EP8 | 2 | sequential | 12288 | 2.026 | 130.9 / 132.8 | - |

### nsys reports

| suite | workload | schedule | report | a2a ms | exposed a2a ms | overlap % | GPU idle ms |
|---|---|---|---|---|---|---|---|
| dsv2-ep1 | pack16k-M4-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep1-j2410103.16-a4.1.nsys-rep` | 0 | 0 | - | 111 |
| dsv2-ep1 | pack16k-M4-even | sequential | `workspace/pp1/nsys/main-dsv2-ep1-j2410103.16-a4.2.nsys-rep` | 0 | 0 | - | 598 |
| dsv2-ep1 | pack16k-M8-even | overlapped | `workspace/pp1/nsys-recovered/j2410103.16-nsys-report-0225.nsys-rep` | 0 | 0 | - | 393 |
| dsv2-ep1 | pack16k-M8-intra | overlapped | `workspace/pp1/nsys-recovered/j2410103.16-nsys-report-3e5f.nsys-rep` | 0 | 0 | - | 607 |
| dsv2-ep1-r2 | pack16k-M4-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.3.nsys-rep` | 0 | 0 | - | 108 |
| dsv2-ep1-r2 | pack16k-M4-even | sequential | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.4.nsys-rep` | 0 | 0 | - | 461 |
| dsv2-ep1-r2 | pack16k-M8-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.5.nsys-rep` | 0 | 0 | - | 822 |
| dsv2-ep1-r2 | pack16k-M8-even | sequential | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.6.nsys-rep` | 0 | 0 | - | 520 |
| dsv2-ep1-r2 | pack16k-M8-intra | overlapped | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.7.nsys-rep` | 0 | 0 | - | 112 |
| dsv2-ep1-r2 | pack16k-M8-intra | sequential | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.8.nsys-rep` | 0 | 0 | - | 162 |
| dsv2-ep1-r2 | uniform-seq2048-mb32 | overlapped | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.1.nsys-rep` | 0 | 0 | - | 320 |
| dsv2-ep1-r2 | uniform-seq2048-mb32 | sequential | `workspace/pp1/nsys/main-dsv2-ep1-r2-j2410103.40-a1.2.nsys-rep` | 0 | 0 | - | 176 |
| dsv2-ep1-same | uniform-seq2048-mb32 | overlapped | `workspace/pp1/nsys/main-dsv2-ep1-same-j2410103.48-a1.1.nsys-rep` | 0 | 0 | - | 267 |
| dsv2-ep1-same | uniform-seq2048-mb32 | sequential | `workspace/pp1/nsys/main-dsv2-ep1-same-j2410103.48-a1.2.nsys-rep` | 0 | 0 | - | 163 |
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
| dsv2-ep8-r2 | pack16k-M1-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a3.1.nsys-rep` | 194 | 194 | 0 | 511 |
| dsv2-ep8-r2 | pack16k-M1-even | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a3.2.nsys-rep` | 181 | 181 | 0 | 1108 |
| dsv2-ep8-r2 | pack16k-M1-full | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a3.3.nsys-rep` | 252 | 251 | 0 | 1049 |
| dsv2-ep8-r2 | pack16k-M1-full | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a3.4.nsys-rep` | 265 | 264 | 0 | 984 |
| dsv2-ep8-r2 | pack16k-M2-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a3.5.nsys-rep` | 392 | 350 | 11 | 822 |
| dsv2-ep8-r2 | pack16k-M2-even | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a3.6.nsys-rep` | 169 | 168 | 0 | 1212 |
| dsv2-ep8-r2 | pack16k-M2-full | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.3.nsys-rep` | 451 | 389 | 15 | 1096 |
| dsv2-ep8-r2 | pack16k-M2-full | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.4.nsys-rep` | 349 | 349 | 0 | 710 |
| dsv2-ep8-r2 | pack16k-M2-intra | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.1.nsys-rep` | 227 | 186 | 18 | 443 |
| dsv2-ep8-r2 | pack16k-M2-intra | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.2.nsys-rep` | 372 | 371 | 0 | 787 |
| dsv2-ep8-r2 | pack16k-M4-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.5.nsys-rep` | 301 | 178 | 42 | 1210 |
| dsv2-ep8-r2 | pack16k-M4-even | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.6.nsys-rep` | 247 | 247 | 0 | 1320 |
| dsv2-ep8-r2 | pack16k-M4-full | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.9.nsys-rep` | 499 | 327 | 36 | 288 |
| dsv2-ep8-r2 | pack16k-M4-full | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.10.nsys-rep` | 435 | 435 | 0 | 184 |
| dsv2-ep8-r2 | pack16k-M4-intra | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.7.nsys-rep` | 251 | 180 | 30 | 626 |
| dsv2-ep8-r2 | pack16k-M4-intra | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.8.nsys-rep` | 216 | 215 | 0 | 314 |
| dsv2-ep8-r2 | pack16k-M8-even | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.11.nsys-rep` | 288 | 189 | 35 | 882 |
| dsv2-ep8-r2 | pack16k-M8-even | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.12.nsys-rep` | 253 | 253 | 0 | 782 |
| dsv2-ep8-r2 | pack16k-M8-full | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.15.nsys-rep` | 799 | 588 | 27 | 911 |
| dsv2-ep8-r2 | pack16k-M8-full | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.16.nsys-rep` | 800 | 800 | 0 | 730 |
| dsv2-ep8-r2 | pack16k-M8-intra | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.13.nsys-rep` | 290 | 179 | 38 | 581 |
| dsv2-ep8-r2 | pack16k-M8-intra | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a4.14.nsys-rep` | 347 | 346 | 0 | 935 |
| dsv2-ep8-r2 | uniform-seq2048-mb32 | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a2.1.nsys-rep` | 1244 | 735 | 41 | 2381 |
| dsv2-ep8-r2 | uniform-seq2048-mb32 | sequential | `workspace/pp1/nsys/main-dsv2-ep8-r2-j2410103.34-a2.2.nsys-rep` | 994 | 994 | 0 | 1474 |
| dsv2-ep8-same | uniform-seq2048-mb32 | overlapped | `workspace/pp1/nsys/main-dsv2-ep8-same-j2410103.48-a1.1.nsys-rep` | 1321 | 798 | 40 | 2021 |
| dsv2-ep8-same | uniform-seq2048-mb32 | sequential | `workspace/pp1/nsys/main-dsv2-ep8-same-j2410103.48-a1.2.nsys-rep` | 1174 | 1174 | 0 | 1442 |
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
