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

The 300-step pair uses tail 24 for both FP8 arms to isolate the attention implementation. Slurm job `1290664` on `pool1-00350` runs the native Delta-Matching arm, the FA4 control, then the optional BF16 compute arm. The separate steady-state sweep in job `1290659` on `pool1-00909` runs 12 steps per setting with FA4 at its stock tail 29 and Delta-Matching at tails 24–27. Steps 3–12 are timed and any post-warmup retry disqualifies a setting.

## Throughput at each backend's best measured tail

The tail sweep held the node, checkpoint, shuffled WikiText batches, microbatch 4 and revision `121c6cb` fixed. All five settings ran 12 steps with zero post-warmup retries.

| FP8 recipe attention | MLP-only tail | Median seconds | Tokens/s | MFU | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FA4 BF16 | 29 | 8.912 | 29,416 | 58.37% | 124.20 / 136.42 |
| Delta-Matching | 24 | 8.873 | 29,542 | 58.62% | 121.56 / 134.63 |
| Delta-Matching | 25 | 8.861 | 29,582 | 58.70% | 122.13 / 135.52 |
| Delta-Matching | 26 | 8.849 | 29,622 | 58.78% | 123.30 / 136.37 |
| Delta-Matching | 27 | 8.852 | 29,614 | 58.76% | 124.47 / 136.39 |

Tail 26 led the short Delta-Matching sweep by 0.70% throughput and 0.41 MFU points versus stock FA4. Tail 26 had zero warmup retries. Tail 27 and stock FA4 each had one warmup retry per GPU. The tail-26 and tail-27 median step times were 3 ms apart.

The backward branch advanced to `116959f` during the 300-step run. Merge `d77de7d` combines its split K/V matched-correction streams, 256-thread dO producer and transposed matrix stores with the forward branch's descending head-major matched-correction dispatch and the training branch's strided input support. All 19 kernel, operator and import tests passed at the new merge. The running convergence pair uses the earlier pinned kernels at `121c6cb`.

Per-step logs, Slurm and GPU gate output, input preparation, and the analysis script are in `.cache/fp8-attention-training/`. Slurm accepted submissions and warned that the account is approaching its stale-data submission threshold.
