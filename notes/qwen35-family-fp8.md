# Qwen3.5 family FP8 vs BF16 on eight H200s

The comparison covers the four dense Qwen3.5 text towers below 27B and includes the 27B reference.
Parameters count trainable text weights, including the vocabulary projection and any shared
embedding once. The corresponding `-Base` checkpoints share the layer layouts, dimensions and
embedding policy of the four smaller towers.
[0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/main/config.json) ·
[2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/main/config.json) ·
[4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base/blob/main/config.json) ·
[9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/main/config.json)

The smallest MoE release, Qwen3.5-35B-A3B, has 35B total and 3B active parameters. Its expert layers
require PithTrain's MoE path. [Official Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35) ·
[35B-A3B config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)

| Model | Text params | BF16 tok/s/GPU | BF16 MFU | FP8 matched tok/s/GPU | FP8 tuned tok/s/GPU | FP8 MFU | Matched speedup | Tuned speedup | Peak GiB BF16 / matched / tuned | Loss sanity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json) | 752,393,024 | 73,402 | 36.57% | 88,794 | 88,365 | — | 1.210× | 1.192× | 77.25 / 72.52 / 72.52 | Max loss Δ 0.0280, stable norms |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) | 1,881,825,088 | 40,245 | 47.61% | 51,643 | 51,594 | — | 1.283× | 1.280× | 98.41 / 91.20 / 91.52 | Max loss Δ 0.0085, stable norms |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) | 4,205,751,296 | 17,640 | 46.96% | 23,367 | 23,345 | — | 1.325× | 1.325× | 99.80 / 91.79 / 91.79 | Max loss Δ 0.0077, stable norms |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json) | 8,953,803,264 | 10,383 | 51.13% | 14,653 | 14,594 | — | 1.411× | 1.409× | 123.71 / 113.96 / 112.00 | Max loss Δ 0.0257, stable norms |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) | 26,895,998,464 | 2,851 | 45.26% | 3,568 | 4,143 | — | 1.252× | 1.453× | 125.27 / 83.62 / 113.31 | BF16 loss 1.9209 → 1.7365–1.7383. FP8 loss ended at 1.7406–1.7409. |

All comparisons use eight H200s, sequence length 4,096, global batch 64, and the same token shard
and sample order. Each model starts from its own released BF16 checkpoint. The smaller models
use the family harness at `5c67803` and time updates 7–12. The 27B reference uses its existing
stack comparison at `5b29d9f` and times updates 3–10. Each comparison ran in both orders on a
single node. Throughput is the median of the per-arm medians, speedup is the median of the two
paired ratios, and peak memory is the maximum allocated GiB across timed updates and ranks. The
BF16 columns use the matched control. Megatron estimates 4.929, 11.706, 26.341, 48.726 and 157.065
GFLOP/token in ascending size order. [Runner and measurement protocol](../benchmarks/end2end/qwen35-family/README.md) ·
[Megatron estimator source](https://github.com/NVIDIA/Megatron-LM/blob/685fb8b/megatron/training/training.py)

The 0.8B and 2B matched pairs used microbatch 8, group size 8, and no recompute or FP8 shard cache.
Both orders had zero allocator retries. The speedups were 1.219× and 1.201× for 0.8B (job `1324050`)
and 1.286× and 1.280× for 2B (job `1323565`). 0.8B's independent FP8 screen chose the same recipe
and expandable segments. Its tuned-pair BF16 control ran at 74,141 tok/s/GPU (36.94% MFU). The lower
tuned speedup reflects repeat-to-repeat throughput variation. A GDN-replay FP8 candidate reduced
allocated memory to 33.30 GiB and reached 77,896 tok/s/GPU with a matching FP8 loss and norm trace.
[0.8B recipe and run details](https://github.com/haok1402/pith-train/commit/6e3c6de)

The 2B matched pair used the default allocator. Independent FP8 tuning chose group 12 and a
256 MiB FP8 shard cache with expandable segments. Its tuned-pair BF16 control ran at 40,293
tok/s/GPU (47.67% MFU). The tuned pair gave 1.284× and 1.277× with zero retries (job `1324478`).
The matched and tuned FP8 rates differed by 0.10%, within the observed repeat spread.
[2B recipe and run details](https://github.com/haok1402/pith-train/commit/0f8847845d6cd7b40c09e4d10771a87969610687)

The 4B sweeps both selected microbatch 4, group size 16, expandable segments, and no recompute or
shard cache. The tuned-pair BF16 control reached 17,620 tok/s/GPU (46.91% MFU). The matched pair
gave 1.322× and 1.327× in the two orders, and the tuned pair gave 1.324× and 1.326×, all with zero
retries (job `1324606`). Microbatch 8 without recompute ran out of memory in both formats, and the
FP8 replay candidate with 24 kept blocks retried allocations during timing.
[4B recipe and run details](https://github.com/haok1402/pith-train/commit/7ad2508)

The 9B matched pair used microbatch 4, group size 8, expandable segments, and no recompute or
shard cache. Tuned FP8 selected four-block groups. The tuned-pair BF16 control ran at 10,356
tok/s/GPU (51.00% MFU). The matched ratios were 1.409× and 1.414×, and the tuned ratios were 1.409×
and 1.410×. All arms had zero retries (job `1323631`).
[9B recipe and run details](https://github.com/haok1402/pith-train/commit/95b117f3f65e36ebdfab3cc3dbad2931c699ca82)

The 27B reference comes from job `1314203`. Both matched arms used microbatch 2, a 12-block
MLP-only tail, expandable segments, and no FP8 shard cache. Tuned FP8 used microbatch 8, a
four-block MLP-only tail, Gated DeltaNet and full-attention replay, specialized MLP replay, and a
1 GiB shard cache. Matched ratios were 1.258× and 1.245×, and tuned ratios were 1.460× and 1.446×.
All six arms had zero retries. The BF16 rate averages the controls from this job. The separate
2,859 tok/s/GPU BF16 reference and 4,145 tok/s/GPU FP8 confirmation come from earlier timings.
[27B stack recipe](fp8-recompute-tune.md)

Loss sanity compares every update's cross-entropy and pre-clip gradient norm with the model's
no-recompute BF16 control, including warmup. Two 9B BF16 microbatch-8 checkpoint candidates were
excluded. The first logged a gradient norm of 710,021 against 59.67 at microbatch 4 with recompute
disabled, and its loss rose from 2.1672 to 2.2166 by update 3. The candidate keeping the final eight
blocks intact opened at a gradient norm of 801,710.
The issue also reproduced in 2B with the outer checkpoint. Forward loss can match despite the
gradient error. Checkpointed candidates use Gated DeltaNet replay to omit the affected op from
the outer save list and require a gradient-norm check against the no-recompute BF16 control.
The 9B microbatch-4 audit reproduced it at identical step-one loss (2.16718459). The control norm
was 59.692, the full-selective norm was 698,220, and replaying Gated DeltaNet alone restored the
norm to 59.688. Its all-parameter relative gradient L2 was 0.00170 against 0.00185 for the two
control repeats. Replaying only full attention and removing only the matrix multiply save entries
both left the error in place (audit job `1323335`).

The ownership audit found that Inductor reused the BF16 Gated DeltaNet core-output buffer for the
output projection. All 48 cached rank-zero entries changed bytes while their pointers and version
counters stayed fixed. Fresh forward and replay results matched bit for bit. Cloning the core
output alone restored the 9B gradient norm to 59.686 versus 59.691 for the control (audit job
`1323554`). The benchmark recipe replays Gated DeltaNet in the outer checkpoint and keeps the
inner mixer checkpoint policy. The 2B and 9B configurations have GDN value width equal to model
width, the case that let the output projection reuse the buffer in the observed schedule.
The follow-up scheduler fix marks the GDN core output ineligible for reuse until it can be freed.
Its two-layer regression test reproduced the gradient mismatch on the unpatched stack and passed
with the fix. A separate full-size 9B audit covered all 427 gradients. BF16 fixed safe-save agreed
with the no-recompute control at the repeat-to-repeat noise level, and FP8 safe-save and replay
agreed exactly. [Scheduler fix and regression](https://github.com/haok1402/pith-train/commit/6a43bc6)

In a separate 9B BF16 microbatch-8 comparison, saving GDN with the scheduler fix gave 9,813
tok/s/GPU at 111.60 GiB allocated. Replaying GDN gave 9,718 tok/s/GPU at 102.51 GiB. The two
order-specific throughput gains for safe saving were 1.00% and 0.94%. All arms had zero allocator
retries and maximum loss delta was 0.0016 (job `1323736`).

Matched FP8 speedup rose from 1.210× at 0.8B to 1.411× at 9B. The vocabulary projection stays BF16
and falls from 31% to 13% of estimated FLOPs across those sizes, while decoder projections grow
from 61% to 85%. In the 0.8B profile, decoder GEMMs fell from 145.1 to 88.5 ms, and full attention
fell from 39.8 to 24.0 ms with Delta-Matching.
Gated DeltaNet stayed near 81 ms and the tied vocabulary projection near 68 ms. Together they took
40% of FP8 GPU timeline time, and standalone quantization and casts added about 31 ms. The profile
averages disjoint step-7 GPU timelines across eight ranks, and BF16 capture overhead inflated its
idle time. The table uses uninstrumented timings. The four smaller models reached their best
screened throughput with explicit recompute disabled.

## 27B matched profile

The `matched-mb2-nsys-v1` capture (job `1324107`, analysis `comparison-v2.json`) measured 11.446 s
BF16 and 9.192 s FP8 at microbatch 2 and MLP-only tail 12. Kernel categories below use the disjoint
timeline attribution. The 27B matched gain is smaller than 9B's even though decoder projections
now account for 93% of estimated FLOPs. Replay and quantization give back part of the GEMM saving.

- Forward GEMMs fell from 2.650 to 1.513 s, input-gradient GEMMs from 2.762 to 1.540 s, and
  weight-gradient GEMMs from 2.770 to 1.604 s. Combined non-replay GEMMs sped up 1.76×, saving
  3.524 s.
- Replay GEMMs grew from 0.678 to 2.040 s, and non-GEMM replay grew from 0.220 to 0.255 s.
  Quantize and cast kernels grew from 0.001 to 0.514 s. Those three categories added 1.910 s.
- Gated DeltaNet stayed at about 0.59 s, and full attention fell from 0.355 to 0.212 s. Exposed
  FSDP communication fell from 0.687 to 0.394 s, and norms and elementwise kernels fell from
  0.598 to 0.399 s.
- Loss and optimizer time went from 0.037 to 0.040 s, other memory work from 0.021 to 0.005 s,
  and idle time from 0.074 to 0.088 s.

At 27B, the tuned replay policy made microbatch 8 fit and raised the measured gain to 1.453×.
