# Qwen3.5-9B FSDP2 on eight H200s

Use the official [Qwen/Qwen3.5-9B checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) and the same WikiText-103 training token shard as the [Qwen3.5 family benchmark](../../../qwen35-family/README.md). The recipe trains the text tower at sequence length 4,096 and global batch 64 with one eight-GPU FSDP2 mesh. Each arm starts from the released weights with seed 1234 and runs 12 AdamW updates. Throughput and memory use updates 7–12.

| Recipe | Microbatch | FSDP group size | Explicit recompute | FP8 shard cache |
| --- | ---: | ---: | --- | ---: |
| BF16 tuned | 4 | 8 | Disabled | — |
| FP8 matched | 4 | 8 | Disabled | 0 MiB |
| FP8 tuned | 4 | 4 | Disabled | 0 MiB |

Both precisions use expandable allocator segments, FP32 weight-gradient accumulation and FP32 reduce-scatter. FP8 uses blockwise linears, Delta-Matching attention and the stack's CuTe FP32 weight-gradient kernel. The vocabulary projection stays BF16.

Set `QWEN35_TORCHRUN` to the CUDA 13.2 environment's `torchrun`. Set `model_dir` to the checkpoint directory and `corpus_bin` to the tokenized corpus. Invoke the adapter through the project's shared GPU gate. Both `matched` and `tuned` run the two precisions in both orders and check the corpus-index hashes. Use an unused output directory for each call.

```bash
recipe="$repo/benchmarks/end2end/qwen3.5-9b-seq4096-bf16/h200-1n8g/fsdp2/run.sh"
"$gpu_gate" --priority normal --timing --gpus 8 --time 18 -- \
    bash "$recipe" matched "$model_dir" "$corpus_bin" "$matched_out"
"$gpu_gate" --priority normal --timing --gpus 8 --time 18 -- \
    bash "$recipe" tuned "$model_dir" "$corpus_bin" "$tuned_out"
```

Use `bf16` or `fp8` to run just the tuned configuration in that precision. The family runner loads the current committed source by default. The measurements use family commit `5c67803` over stack commit `dda6dbb`. The common summarizer reads the paired run directory and computes BF16 MFU with 48,725,753,856 model FLOPs per token and 989.4 TFLOP/s per H200.

## Paired results

Job `1323631`, ticket `7154`, ran both pairs on the same eight H200s. Each arm reports median throughput over updates 7–12. The precision rows take the median of the two arm medians. The paired speedup is the median of the two order-specific ratios. Memory uses the largest per-rank peaks over those timed updates and both repeats.

| Precision and recipe | Tok/s/GPU | BF16 MFU | Paired speedup | Allocated / reserved GiB | Loss | Warmup / timed retries |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| BF16 tuned, matched pair | 10,383.2 | 51.13% | 1.0000× | 123.71 / 132.46 | 2.1672 → 1.8897 | 0 / 0 |
| FP8 matched | 14,652.7 | — | 1.4112× | 113.96 / 118.76 | 2.1929 → 1.8945 | 0 / 0 |
| BF16 tuned, tuned pair | 10,356.0 | 51.00% | 1.0000× | 123.71 / 132.46 | 2.1672 → 1.8897 | 0 / 0 |
| FP8 tuned | 14,594.1 | — | 1.4092× | 112.00 / 116.42 | 2.1929 → 1.8946 | 0 / 0 |

The matched BF16→FP8 and FP8→BF16 ratios were 1.4086× and 1.4138×. The tuned ratios were 1.4085× and 1.4100×. The largest paired loss difference was 0.0257 in each comparison. Loss and pre-clip gradient norms were finite in every arm. Four-block FP8 grouping reduced reserved memory by 2.34 GiB with throughput within 0.4% of eight-block grouping.
