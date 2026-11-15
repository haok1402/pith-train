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
