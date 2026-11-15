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
