# Qwen3.5-0.8B on eight H200s

## Confirmed throughput

Slurm job `1324050` ran both BF16→FP8 and FP8→BF16 pairs for each comparison
on the same eight H200s on `pool1-00073`. Each arm ran 12 updates. Throughput
is the median of steps 7–12 per arm, then the median of the two arms per format.
Speedup is the median of the two paired ratios.

| Comparison | BF16 tok/s/GPU | BF16 MFU | FP8 tok/s/GPU | FP8 MFU | FP8 speedup |
|---|---:|---:|---:|---:|---:|
| Matched | 73,402 | 36.57% | 88,794 | — | 1.210× |
| Tuned | 74,141 | 36.94% | 88,365 | — | 1.192× |

The order-specific speedups were **1.219× and 1.201×** for the matched confirmation
and **1.187× and 1.197×** for the tuned confirmation. Both screens selected
microbatch 8, eight-block FSDP groups, expandable segments and no explicit checkpoints.
FP8 shard caching was off. The tuned confirmation repeats that recipe independently.

| Format | Peak allocated GiB | Peak reserved GiB | Warmup / timed allocator retries |
|---|---:|---:|---:|
| BF16 | 77.25 | 78.34 | 0 / 0 |
| FP8 | 72.52 | 73.58 | 0 / 0 |

Loss and gradient norms were finite in all eight arms. Loss moved from 2.7905 to
2.4863–2.4864 in BF16 and from 2.8185 to 2.4966 in FP8. The maximum paired loss
difference was 0.0280. The corpus order hashes matched across all arms.

Raw logs and both `summary.json` files are under `.cache/qwen35-0p8b/confirm-1324050/`
in the benchmark workspace. The recipe launcher is
`benchmarks/end2end/qwen35-family/run-qwen35-0p8b.sh`.

## Workload

The pinned official checkpoint is `Qwen/Qwen3.5-0.8B` at
`2fc06364715b967f1860aea9cf38778875588b17`. The dense text tower has 752,393,024 parameters,
18 Gated DeltaNet layers and six full-attention layers. The input embedding and BF16 vocabulary
projection share one weight owned by root FSDP2. The FP8 stack uses blockwise FP8 decoder linears,
Delta-Matching full attention, and the CuTe FP32 wgrad kernel. The optimizer and gradient reductions
use FP32 in both formats. [Official configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)

The family harness is `5c67803` on verified stack `dda6dbb`. The first screen used `bc1c386`,
which has the same source tree as `5c67803`. Each arm starts fresh from the official checkpoint
and uses the pinned WikiText training shard, seed 1234, sequence length 4,096,
global batch 64, and fused AdamW at `1e-5`. The mesh is PP1/DP8/CP1/EP1. BF16 MFU uses
4,929,257,472 Megatron FLOPs per token and 989.4 dense BF16 TFLOP/s per H200.

## BF16 screen

The first screen used two decoder blocks per FSDP group. Job `1322947` on `pool1-00101`
timed steps 3–6. All runs completed with zero allocator retries during warmup and timing.

| Microbatch | Checkpoint policy | Allocator | Tok/s/GPU | Allocated / reserved GiB |
|---:|---|---|---:|---:|
| 2 | No explicit checkpoint | Default | 64,541 | 21.56 / 23.34 |
| 4 | No explicit checkpoint | Default | 71,520 | 40.17 / 41.97 |
| 8 | No explicit checkpoint | Default | 73,231 | 77.04 / 79.12 |
| 8 | No explicit checkpoint | Expandable | 74,787 | 77.01 / 78.03 |
| 8 | First 12 MLP-only, last 12 kept | Expandable | 74,285 | 71.76 / 72.77 |
| 8 | First 12 selective, last 12 kept | Expandable | 70,002 | 56.39 / 57.41 |

The MLP-tail and selective arms passed the per-step gradient-norm comparison with the expandable
no-checkpoint control. Their maximum relative deviations were 0.247% and 0.301% across six updates.
The family subsequently found severe gradient corruption from saving the Gated DeltaNet output in
the outer selective checkpoint at other model sizes. Subsequent checkpoint experiments replayed
Gated DeltaNet in the outer checkpoint.

The grouping screen used microbatch 8 with expandable segments and no explicit checkpoint.
Job `1323322` on `pool1-00142` timed steps 3–8. The group-2 BF16 arm had two slower steps
(0.619 s and 0.654 s) and zero allocator retries. The FP8 group-2 arm was the first stack smoke test.

| Format | Decoder blocks per FSDP group | Tok/s/GPU | Allocated / reserved GiB |
|---|---:|---:|---:|
| BF16 | 1 | 72,835 | 76.97 / 77.91 |
| BF16 | 2 | 69,536 | 77.01 / 78.03 |
| BF16 | 4 | 73,052 | 77.09 / 78.16 |
| BF16 | 8 | 73,491 | 77.25 / 78.34 |
| BF16 | 24 | 73,510 | 77.89 / 79.48 |
| FP8 | 2 | 84,880 | 72.39 / 73.50 |

## FP8 tuning and BF16 group retest

Job `1323575` on `pool1-00697` timed steps 7–12. In the reversed BF16 group retest,
group 8 reached 75,117 tok/s/GPU. Groups 2 and 24 reached 73,857 and 73,822 tok/s/GPU.
Group 8 was selected for BF16 and used for the matched FP8 arm.

| Microbatch | FSDP group | Allocator | FP8 variant | Tok/s/GPU | Allocated / reserved GiB |
|---:|---:|---|---|---:|---:|
| 8 | 2 | Expandable | No explicit checkpoint | 88,430 | 72.39 / 73.50 |
| 8 | 4 | Expandable | No explicit checkpoint | 89,061 | 72.43 / 73.66 |
| 8 | 8 | Expandable | No explicit checkpoint | 89,540 | 72.52 / 73.58 |
| 8 | 24 | Expandable | No explicit checkpoint | 88,395 | 72.87 / 74.54 |
| 4 | 8 | Expandable | No explicit checkpoint | 83,047 | 37.95 / 38.89 |
| 8 | 8 | Expandable | No explicit checkpoint, 128 MiB shard cache | 88,301 | 72.56 / 73.62 |
| 8 | 8 | Default | No explicit checkpoint | 88,407 | 72.54 / 74.87 |
| 8 | 8 | Expandable | GDN and specialized MLP replay, MLP-only tail 12, 128 MiB shard cache | 77,896 | 33.30 / 34.40 |

The FP8 replay candidate matched the logged loss and gradient norm of its cache-on no-recompute
control at all 12 updates. It cut peak reserved memory by 39.22 GiB and ran 11.8% slower than
that control. All candidates had zero allocator retries, including warmup. Independent FP8
tuning selected the matched recipe with the shard cache off.

## One-step profile

After timing, Nsight Systems captured update 7 in each precision on the same node and recipe.
The sample-index files matched the timing runs. The table reports disjoint GPU timeline averages
across all eight ranks. CUDA launch correlation IDs map GEMMs launched in the vocabulary
projection's NVTX ranges to that projection. The other categories use kernel names.

| GPU activity | BF16 ms | BF16 share | FP8 ms | FP8 share |
|---|---:|---:|---:|---:|
| Decoder GEMMs | 145.07 | 29.32% | 88.51 | 23.77% |
| Gated DeltaNet | 81.05 | 16.38% | 80.94 | 21.74% |
| BF16 vocabulary projection | 67.21 | 13.58% | 67.91 | 18.24% |
| Full attention | 39.76 | 8.04% | 24.00 | 6.45% |
| Standalone quantize/cast | 0.20 | 0.04% | 30.94 | 8.31% |
| Cross entropy | 9.55 | 1.93% | 10.03 | 2.69% |
| Exposed NCCL | 13.39 | 2.71% | 11.87 | 3.19% |
| Other kernels and memory movement | 77.85 | 15.74% | 47.43 | 12.74% |
| GPU idle | 60.69 | 12.27% | 10.65 | 2.86% |

Decoder GEMMs and full attention saved 56.56 ms and 15.75 ms per GPU in the capture,
with 30.74 ms added by standalone quantization and casts. Gated DeltaNet and the BF16
vocabulary projection together took 149 ms, about 40% of the FP8 trace, with nearly
the same absolute time in both formats. Those two operations limit the end-to-end gain.

FP8 launched 3,006 kernels per rank versus 1,748 in BF16. The trace reports cumulative
CUDA runtime launch API times of 124.6 ms and 169.4 ms per rank, respectively.
Launch calls can overlap GPU execution. The BF16 capture took 502 ms compared with
435–442 ms in its preceding four steady updates, so the observed idle times include
the effects of capture. The headline speedups use the uninstrumented confirmations.

Trace reports and attribution tables are under `.cache/qwen35-0p8b/profile-1324050/`
in the benchmark workspace. CPU export and analysis completed in job `1324590`.
