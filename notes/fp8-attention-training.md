# Qwen3.5-27B FP8 attention training

## Implementation

The `e2efp8-fp8-attention` branch starts at FP8 fusion head `e57d863`. It merges the promoted forward `755d2ac` and backward `da20ba8` revisions. The merge uses the forward branch's matched row correction and the backward branch's dO quantization and dQ/dK/dV paths. GitHub confirmed the three source heads over authenticated HTTPS before the runs.

Use `--fp8 --fp8-attention` in the dense FSDP2 benchmark. The shared Qwen3.5 dense and MoE attention implementation reads `TrainingCfg.fp8_attention`, which defaults to false. The setting applies to causal full attention with unpacked sequences and CP=1 on SM90. The sigmoid output gate runs after the attention operator and before the output projection. Checkpointing saves the Delta-Matching operator state and replays the input projections. The quantizers read aligned BF16 BSHD views directly and materialize a contiguous tensor when channel stride or alignment requires it.

## Experiment

Both FP8 arms use the released Qwen3.5-27B Hugging Face checkpoint, the same node of eight H200s, FSDP2, global batch 64, microbatch 4, sequence length 4096, seed 1234, and fused AdamW at a constant learning rate of `1e-5`. Gradients, gradient reduction, master weights and AdamW moments are FP32. The allocator is PyTorch's default, and training state stays on GPU. The pre-clip gradient norm is logged after normalization by the global valid-token count, with clipping at 1.0. The audit records allocator retry deltas on every rank and fails the arm if any retry occurs after step 2.

The benchmark runs each arm as a fresh process from the same checkpoint. The corpus has 19,200 distinct 4,096-token windows built from the first 78,643,201 Qwen-tokenized tokens of the pinned WikiText-103 train split (`b08601e04326c79dfdd32d625aee71d232d685c3`), including the final next-token label. The runner reloads the same corpus and shuffles with the same seed each time. The optional BF16 compute arm uses the same global batches with the previously measured BF16 recipe, microbatch 2 and a 12-layer MLP-only checkpoint tail. MFU uses the median step time after step 2, 156.9905 GFLOP per token and the common 989 TFLOP/s BF16 peak per H200.

## Exact first-batch replay

Slurm job `1290484` ran the original headroom recipe at `06e5227` and its original 12-layer MLP checkpoint tail on `pool1-00693`. Each arm loaded the same Hugging Face checkpoint. The job used the headroom run's token file and verified byte-identical sample indices and offsets. The KDA arm transplanted the merged kernels and operator into that historical recipe, with the matching selective checkpoint save rule.

| Full attention | Step 1 loss (nats) | Step 1 pre-clip norm |
| --- | ---: | ---: |
| FA4 BF16 | 1.8888 | 26.5919 |
| Published Delta-Matching v0.1.0 | 1.8911 | 155.2875 |
| Native KDA Delta-Matching | 1.8889 | 29.8697 |

The native KDA attention implementation reduced the first-step norm from 5.84× the FA4 control to 1.12× on the same data and training recipe. The first 29-layer-tail smoke run on the newer recipe used the same token data and sample order and gave a norm of 30.5076.

## Validation and allocator tuning

The combined kernel, operator and import-layer tests passed all 19 cases, including two comparisons with the published reference and an exact strided-projection regression. On a two-H200, two-update FSDP2 fixture with FP8 linears and BF16 attention, all 28 FP32 gradients, updated parameter shards, losses and gradient norms were bit-identical to `e57d863`. The fixture also checked FP32 master parameters and both FP32 AdamW moments.

Smoke job `1289946` stopped before step 1 because the merged dO producer omitted the thread-count argument required by the new shared quantizer. Commit `993ceed` passes the correct thread count to the producer and reducer. Both the 29- and 28-layer MLP-only tails incurred two allocator retries on every GPU at step 3. Their peak allocations were 126.20 GiB and 125.63 GiB respectively. The 24-layer tail moves one further four-layer group into full selective checkpointing.

With tail 24, job `1290572` on `pool1-00971` completed eight Delta-Matching updates with zero allocator retries on all ranks. Steps 3–8 took a median 8.735 seconds at 30,011 tokens/s, or 59.55% MFU. Peak allocated and reserved memory were 121.56 and 134.63 GiB per GPU. Commit `121c6cb` uses tail 24 by default when both FP8 switches are enabled.

The same job also ran the full model with FP8 linears and BF16 attention at tail 29 on the feature and starting revisions. Both microbatch loss scalars matched bit for bit on all eight ranks. All arms completed before a launcher edit caused a shell syntax error at exit. The metric and bit checks use the completed arm logs.

## 300-step matched-tail convergence

Job `1290664` ran both FP8 arms at tail 24 on `pool1-00350`, with code pinned to `121c6cb`. Both completed 300 optimizer steps and had zero allocator retries across all eight ranks, including warmup. Their batch-order hash was `cda16eda5eddd269c6849ff94189c50d058c6eb0880eaba0ebc0714dc311492c`.

| FP8 recipe attention | Loss step 1 → 300 | Mean loss (nats) | Pre-clip norm step 1 → 300 | Median seconds | Tokens/s | MFU | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FA4 BF16 | 1.9778 → 1.6890 | 1.678641 | 25.4463 → 0.4768 | 8.830 | 29,688 | 58.91% | 119.12 / 132.99 |
| Delta-Matching | 1.9772 → 1.6886 | 1.678783 | 30.0696 → 0.4960 | 8.757 | 29,935 | 59.40% | 121.56 / 134.63 |

Mean Delta-Matching minus FA4 loss was **+0.000142 nats**, with a mean absolute gap of **0.000396 nats**. Paired differences ranged from −0.0011 to +0.0022 nats, and the final gap was −0.0004 nats. Delta-Matching finished the measured run with similar training loss and 0.83% more tokens per second at the common tail.

| Steps | Mean loss gap (nats) | Median Delta-Matching / FA4 pre-clip norm |
| --- | ---: | ---: |
| 1–10 | −0.000370 | 0.986× |
| 11–50 | −0.000050 | 0.967× |
| 51–150 | +0.000085 | 0.996× |
| 151–300 | +0.000266 | 1.000× |

The first-step norm ratio was 1.182×. The persistent gap faded, with isolated higher norms later. Steps 119 and 293 were 3.2209 versus 0.5191 and 3.4872 versus 0.5768, respectively. Delta-Matching exceeded 2× the FA4 norm at five later steps and fell below 0.5× at two. Gradient clipping applied at 10 steps with Delta-Matching and 6 with FA4.

## BF16 compute control

The third arm in job `1290664` completed all 300 steps on the same node and verified the same batch-order hash. It used microbatch 2 and MLP-only tail 12 with FP32 gradients, reduction, master weights and optimizer moments. It had three allocator retries per GPU in warmup and zero afterward. The job released all eight GPUs after 2 h 39 min.

| Loss step 1 → 300 | Mean loss (nats) | Pre-clip norm step 1 → 300 | Median seconds | Tokens/s | MFU | Peak allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0085 → 1.6861 | 1.675485 | 31.0787 → 0.5274 | 11.389 | 23,017 | 45.67% | 125.42 / 135.40 |

BF16 compute averaged 0.003156 nats less loss than FP8 with FA4 and 0.003298 nats less than FP8 with Delta-Matching. It clipped on 11 steps. The BF16 compute comparison includes a different microbatch size and checkpoint tail.

## Throughput at each backend's best measured tail

Job `1290659` on `pool1-00909` held the node, checkpoint, shuffled WikiText batches, microbatch 4 and revision `121c6cb` fixed. All five settings ran 12 steps with zero post-warmup retries. Steps 3–12 are timed.

| FP8 recipe attention | MLP-only tail | Median seconds | Tokens/s | MFU | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FA4 BF16 | 29 | 8.912 | 29,416 | 58.37% | 124.20 / 136.42 |
| Delta-Matching | 24 | 8.873 | 29,542 | 58.62% | 121.56 / 134.63 |
| Delta-Matching | 25 | 8.861 | 29,582 | 58.70% | 122.13 / 135.52 |
| Delta-Matching | 26 | 8.849 | 29,622 | 58.78% | 123.30 / 136.37 |
| Delta-Matching | 27 | 8.852 | 29,614 | 58.76% | 124.47 / 136.39 |

Tail 26 led the short Delta-Matching sweep by 0.70% throughput and 0.41 MFU points versus stock FA4. Tail 26 had zero warmup retries. Tail 27 and stock FA4 each had one warmup retry per GPU. The tail-26 and tail-27 median step times were 3 ms apart.

The backward branch advanced to `116959f` during the 300-step run. Merge `d77de7d` combines its split K/V matched-correction streams, 256-thread dO producer and transposed matrix stores with the forward branch's descending head-major matched-correction dispatch and the training branch's strided input support. All 19 kernel, operator and import tests passed at the new merge. The convergence pair used the earlier pinned kernels at `121c6cb`.

The two integrated operator revisions returned bit-identical O, dQ, dK and dV on the same random operands at the dense 4K, 24:4 GQA, D=256 shape with batches 2 and 4. This check ran in job `1291198` on `pool1-00601`.

The follow-up same-node run `1291031` on `pool1-00397` measured another 12 steps per setting. All four arms were retry-free after warmup.

| FP8 recipe attention | MLP-only tail | Median seconds | Tokens/s | MFU | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FA4 BF16 | 29 | 8.771 | 29,889 | 59.31% | 124.20 / 136.42 |
| Delta-Matching, `116959f` backward | 27 | 8.707 | 30,106 | 59.74% | 124.47 / 136.39 |
| Delta-Matching, `116959f` backward | 26 | 8.719 | 30,068 | 59.66% | 123.30 / 136.37 |
| Delta-Matching, `da20ba8` backward | 26 | 8.715 | 30,081 | 59.69% | 123.30 / 136.37 |

The latest Delta-Matching arm at tail 27 delivered 0.73% more tokens/s and 0.43 more MFU points than FA4 at tail 29. Both had one warmup retry per GPU. Tail 26 had zero retries including warmup, and the two backward revisions differed by 4 ms in median step time at tail 26. For those two full-model runs, the first microbatch losses matched bit for bit on six of eight ranks, and the largest logged loss gap over 12 steps was 0.0020 nats. The operating default stays at tail 24, which completed the 300-step allocator check.

Per-step logs, Slurm and GPU gate output, input preparation, and the analysis script are in `.cache/fp8-attention-training/`. Slurm accepted submissions and warned that the account is approaching its stale-data submission threshold.
