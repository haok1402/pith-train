# HybridEP in PithTrain: results

Recorded on the fork (`haok1402/pith-train`, branch `haok/hybrid-ep`); not proposed upstream.

**Verdict.** HybridEP stays an opt-in backend (`DistributedCfg.ep_backend = "hybrid-ep"`), with
`hybrid_ep_mode = "permute"` as its default. Dispatch and combine are about 2x faster at op level,
but end to end it is only about 1-2% faster than the all-to-all path, and permute mode uses 3.4 GiB
more peak allocated memory (98.7 vs 95.3) and 2.6 GiB more device memory. Dedup mode roughly matches
all-to-all's memory but is 5-6% slower than all-to-all (about 7% slower than permute). That is not
worth upstreaming.

## Setup

- 1 node, 8x H200, PP1 / EP8 / CP1, Qwen3-30B-A3B-Base (HF import), PithSim's Nemotron-CC subset,
  seq 4096, micro-batch 1, global batch 256, seed 1234, global-batch LB loss 1e-3, constant lr
  1e-6, BF16 (fused CuTe experts), DualPipeV. The scenario script is
  `workspace/hybrid-ep/scripts/lb_scenario.py`; runs log to wandb `PithTrain/qwen3-30b-a3b`.
- DeepEP `hybrid-ep` at `10d4dd7`, installed by `uv sync` (see `pyproject.toml`), built with CUDA
  13.2 for SM90 and SM100. The nodes have no 13.0 toolkit, and torch is cu130 (same major
  version). The JIT kernel cache is `HYBRID_EP_CACHE_DIR`.
- Blocking and **dropless by construction**. Permute fusion is off, so HybridEP's token-drop path
  is not compiled (`jit/compiler.cu:305`), and every output is sized from the scan's exact counts.
  The step log reports rows received against T*K and, in dedup mode, the slot-order flag, which is
  the one live check. Every run had 0 dropped rows, and every run since the flag exists (609254e)
  had 0 slot-order violations.

## Design (`pithtrain/operators/hybrid_ep.py`, `pipeline/execution.py`, `models/qwen3_moe.py`)

- One dedicated HybridEP stream and one shared buffer. Calls go in the same order on every rank.
  The router weights travel with the tokens and are applied in stage 3, so stage 5 only adds the
  residual. The backward replays the cached handle with no all-gather, no scan and no host wait.
- `hybrid_ep_mode`: `permute` uses HybridEP's permute (one row per (token, expert)). `dedup` uses
  the non-permuting dispatch (one row per (token, rank)): the fused BF16 experts read the rows
  through `expand_idx`, and `bf16_expert_gather_sum` replaces the unpermute.
- `hybrid_ep_sync_late`: stage 1 issues the routing all-gather and scan right after the router,
  with an async copy of the counts into a pinned ring. Stage 2 waits for that copy and launches
  the transfer ahead of the other micro-batch's stage-2 backward.
- Only Qwen3MoE supports the backend; `setup_training` rejects it for the other models.

## Op level (`benchmarks/operators/bench_ep_dispatch.py`)

Uniform routing (5.34 rows per token to ranks, 8 to experts), T=4096 per rank, H=2048, 128
experts, top-8. Latency is wall time per call, host syncs included, max over ranks. GB/s is the
useful payload: (token, rank) rows x H x 2 bytes, the same for both backends. Log:
`workspace/hybrid-ep/logs/bench_1105-0037.log`.

| op | all-to-all (NCCL, 32 blocks) | HybridEP, 24 SMs |
|---|---|---|
| dispatch fwd (incl. metadata, syncs, permute) | 1121 us, 80 GB/s | 587 us, 153 GB/s |
| dispatch fwd, NCCL transfer only | 282 us, 317 GB/s | - |
| combine fwd (incl. unpermute + weighted sum) | 1000 us, 90 GB/s | 356 us, 251 GB/s |
| combine fwd, NCCL transfer only | 411 us, 218 GB/s | - |
| dispatch bwd | 271 us (transfer only) | 392 us (incl. reduction) |
| combine bwd | 972 us | 353 us |

- 16 SMs is enough for combine; dispatch fwd gains up to 24 SMs.
- The all-to-all combine rows run stage 5's weighting eagerly, which overstates them; in training
  that runs compiled.
- Concurrent grouped-GEMM slowdown is 20-50% for both backends. Trace analysis put the
  permute/unpermute share of that at about +9% on the quack grouped GEMM, about 25-30 ms/step.

## End to end

Steady-state mean, steps 6-25 for the 25-step runs and steps 4-11 for the 12-step runs. Memory is
the peak over steps 2-N, in GiB (bytes / 1024^3). Δloss is the max over all steps against the
all-to-all run on this branch. Run-to-run noise on identical code is 3e-4 (5e-4 against main's
k32mob1i). Per-rank receive imbalance (max/mean) is 1.06 in every run that logs it (from 1bfcf08
on). An H200 has 140.4 GiB (143,771 MiB).

| run (wandb id, commit, node) | step | tok/s | alloc / reserved / device used | Δloss |
|---|---|---|---|---|
| all-to-all (m3agasfg, 1bfcf08, gpu-148) | 9.57 s | 109.6k | 95.3 / 118.6 / 124.4 | - |
| HybridEP permute, first cut (2r1646n0, 8c0b6c4) | 11.53 s | 90.9k | 101.2 / 120.3 / 123.6 | 6e-4 |
| + fused router-weight multiply (1947duf1, 6f5671f) | 9.48 s | 110.6k | 100.2 / 125.0 / 128.2 | 4e-4 |
| + memory fixes (qsmk1i6c, f6f6c48, gpu-886) | 9.40 s | 111.5k | 98.7 / 123.7 / 127.0 | 4e-4 |
| permute + sync_late, 12 steps (45e9x9fe, 6498fbd, gpu-727) | 9.51 s | 110.3k | 98.6 / 122.9 / 126.2 | 5e-4 |
| dedup, 12 steps (i4fro1ol, 6498fbd, gpu-727) | 10.12 s | 103.6k | 96.0 / 119.9 / 123.2 | 5e-4 |
| dedup + sync_late, 12 steps (4hcma653, 6498fbd, gpu-727) | 10.05 s | 104.3k | 96.0 / 118.7 / 121.9 | 3e-4 |

Gradient norms agree within the ±1% run-to-run spread.

**Same-node sync-late A/B** (gpu-886, 6498fbd, 12 steps each, back to back): blocking permute
(a8gfvmks) against permute + sync_late (iqwr4sug), over steps 5-12. The mean is 9.452 vs 9.430 s and
the median 9.452 vs 9.398 s, so -0.2 to -0.6%, about 0.02-0.05 s/step. (Over steps 4-11 the means
are 9.48 vs 9.44 s, but that window holds one outlier step on each side.) Memory is identical.

Logs are in `workspace/hybrid-ep/logs/` (`e2e_*`, `e2e-hybrid-*`, `e2e12-*`, `ab-permute_{A,B}`).

## Why about 2x at op level gives only about 1% end to end

Per-step torch.profiler traces (rank 0, 6f5671f; `workspace/hybrid-ep/out/prof_*`, analysis in
`workspace/hybrid-ep/analysis/trace_*.txt`):

| | all-to-all | HybridEP permute |
|---|---|---|
| step window | 10.42 s | 10.24 s |
| compute stream busy / idle | 8.11 / 2.31 s | 7.35 / 2.89 s |
| host blocked in the per-layer sync | 2.04 s (cudaEventSynchronize x1536), compute idle 891 ms | 0.83 s (cudaStreamSynchronize x1553), compute idle 414 ms |

- DualPipeV already overlaps one micro-batch's all-to-all with another's compute. HybridEP
  removes 0.76 s of compute-stream work (stage-5 weighting, dispatch index kernels) but adds
  0.58 s of compute idle.
- The idle comes from waiting on the serialized HybridEP stream: the routing all-gather (1,536 calls,
  343.5 ms, 328.8 ms exposed, 0.224 ms each for about 64 KB, excluding the 56 FSDP all-gathers in the
  same NCCL class) and HybridEP's in-kernel barriers. "Exposed" here is the trace script's upper
  bound: time with no compute-stream kernel running.
- **Barrier map.** Every dispatch-kernel and combine-kernel call issues a `device_sync_kernel`
  before and after its data kernel (`backend/hybrid_ep_backend.cuh:5961-5978` and `:6082-6099`;
  the kernel is at `:4266`). That makes 8 per layer–micro-batch (fwd dispatch, fwd combine, combine
  bwd, dispatch bwd) and 12,288 per step: 1.9-2.0 s on every rank, about 0.6 s of it exposed.
  - The before-barrier protects the shared receive buffer from peers still reading the previous
    call; the after-barrier makes sure all writes have landed.
  - `device_side_sync_{dispatch,combine}_api` are hard-set true (`config.cuh:447,456`), and setting
    them false leaves no replacement synchronization.
  - Ranks 0, 1 and 4 show the same barrier and all-gather wait, so there is no single straggler.
    That the wait is per-layer timing jitter rotating between ranks is an inference from those
    equal per-rank waits, not shown directly. No safe patch was found.
- Moving the host wait off the critical path (sync_late) recovers only 0.05 s/step.

## Memory attribution

Step-3 memory snapshots at 6f5671f (`workspace/hybrid-ep/out/memsnap_*`). Both hit the 65,536-event
history cap, so they show about 48 of the about 72 live layer–micro-batches. The two backends also
peak at different moments (the all-to-all peak includes 1.245 GB of lm_head logits). The window
totals (+3.7 GB for HybridEP permute) are therefore approximate; the per-site items below are solid:

- Stage 3's saved input, 6.80 vs 4.60 GB (+2.2 GB). Permute gives one row per (token, expert), 8
  per token; the all-to-all path keeps the 5.34 dedup rows per token and indexes them. Dedup mode
  removes this: end to end it is -2.6 GB.
- `dispatch_tokens` kept alive (+0.8 GB). The compiled stage-1 graph does not save it (checked in
  the generated Inductor code), so it is now freed after the dispatch. Fixed in e675335.
- The combined output held by stage 5 (+0.8 GB); the residual add saves nothing, so it is now
  freed after stage 5's forward. Fixed in e675335.

Separately, not from the snapshots: the first cut's eager fp32 router-weight multiply cost about
1 GiB of peak (e2e peak allocated 101.2 to 100.2 between 8c0b6c4 and 6f5671f) and about 1.1 ms per
layer–micro-batch. It is now one fused Inductor kernel each way (6f5671f), and it was the 2 s/step
regression of the first cut.

## Tried and rejected

- **lanes=2** (separate forward/backward buffer and stream): never measured end to end. It was
  dropped on the ordering analysis: the combine backward ahead of a forward dispatch on the HybridEP
  stream waits on an earlier compute event than the dispatch does, so it does not delay it. The
  traces then showed equal waits on every rank. Removed in 609254e.
- **Permute-block cap** (`hybrid_ep_permute_blocks`): 32/64 blocks made combine and backward +7% to
  +179% slower at op level (64-block dispatch bwd 1.07 vs 0.384 ms); 128 blocks is neutral. The
  64-block e2e run (dqufoeyc, gpu-659) was 9.55 s against 9.40 s at the default grid (gpu-886). That
  gap crosses nodes (noise about 0.1 s), so it is suggestive, not clean. The knob remains, default 0.
- **Custom all-gather** (`hybrid_ep_custom_allgather`, ywhwtr5c): 9.39 s against 9.40 s, within
  noise. The knob remains, default off.
- **Non-blocking mode with a worst-case bound:** dropless needs T x EP x min(K, local experts) =
  262,144 rows per rank, 8x the mean. That would make about 77 GB of saved stage-3 input. Not
  viable.

## Open leads

- **Dedup stream costs** (6% slower than permute). On the HybridEP stream there is the dense copy,
  the slot-index build and `group_slots` before stage 3; the gather-sum before stage 5; the P x H
  `index_select` in the combine backward; and the zero-filled [D, 128] probs. The dedup + sync_late
  profile is captured (`out/prof_dedup-b`, `out/memsnap_dedup-b`) but not analyzed. Candidate
  fixes: fold the gather into the compiled weighting backward, reuse the experts' slot maps, build
  the indices at scan time, copy only this rank's 16 prob columns.
- **Skewed routing:** the op benchmark at 1.5x/2x hot rank and on the dumped real routing
  (`out/routing_base/layer{0,23,47}`) was never run. Its first attempt crashed on an unset
  `distributed.ep_rank` (fixed in the benchmark with this doc). Only the base-checkpoint layer-0
  real routing ran, on the earlier benchmark (`out/bench_base_layer0.json`).
- **Global-batch LB all-reduce** (both backends, independent of HybridEP; unverified):
  `GlobalBatchLoadBalanceLoss.__call__` at `pithtrain/modules/load_balance.py:125` runs a 128-float
  `funcol.all_reduce` per layer–micro-batch (1,536 per step). It is presumably inside the compiled
  stage-1 graph, which is not checked. It shows 0.40 s/step (HybridEP) and 0.57 s (all-to-all) of
  "exposed" time, the trace script's upper bound. It could be issued in the forward and waited in
  the injector's backward without changing the math. Not started.

## Test coverage

- `tests/test_hybrid_ep.py` (8 GPUs), all passing on the tree committed as c0ebc9e
  (`workspace/hybrid-ep/logs/test-review2.log`):
  - Both modes x sync_late against a dense fp32 single-rank reference with skewed routing:
    outputs, input, router-weight and expert-weight gradients, max error 3.4-3.9e-3.
  - A two-state DualPipeV-order interleave, and a depth-12 interleave (all forwards before the
    first backward).
  - Pinned-ring reuse: with 1 slot the overwrite is detected; with 2 slots the count stays intact.
  - Dedup vs permute on the real fused BF16 experts: bitwise equal (`torch.equal`) on outputs and
    all gradients.
  - A negative slot-order test that fires.
- DualPipeV against its single-device reference (`tests/test_dualpipev.sh`, Qwen3 with 8 layers;
  loss, every gradient and every canonical parameter must match), at 6498fbd:
  - **Passed:** pp2/ep2 for all-to-all, permute, permute + sync_late, dedup and dedup + sync_late;
    pp1/ep8 for dedup + sync_late. Logs: `workspace/hybrid-ep/logs/smoke_6498fbd/`.
  - **Skipped (not run):** pp1/ep8 for all-to-all, permute, permute + sync_late and dedup. These
    are covered only by the pp1/ep8 e2e loss-parity runs, which validate the forward but barely the
    backward. Recorded in `workspace/hybrid-ep/skipped/smoke-6498fbd/`.
- The matrix tested 6498fbd. The later commits only add tests, a setup-time guard and this doc.
