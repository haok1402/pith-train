# Qwen3.5-27B FP8 recompute recipes

## Stack recipe

Use microbatch 8, an MLP-only tail of four blocks, both mixer outputs replayed, and expandable CUDA allocator segments. On eight H200s with FSDP2, sequence length 4,096 and global batch 64, the recipe delivered **33,156–33,158 tok/s/node** at 7.906–7.907 s/step, about 1.44× the 23,017 tok/s BF16 FSDP2 baseline.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  benchmarks/end2end/qwen3.5-27b-seq4096-bf16/h200-1n8g/fsdp2/script.py \
  --model "$model" --weights "$weights" --dataset "$dataset" \
  --fp8 --fp8-attention --batch 64 --seq 4096 --micro-batch 8 \
  --fp8-mlp-recompute --checkpoint-replay-mixers both \
  --checkpoint-mlp-tail 4 --fp8-shard-cache-mib 1024
```

The paired comparison on stack `5b29d9f` used ten updates per arm, seed 1234 and the same batches. Throughput uses the median of steps 3–10. Both settings used expandable segments. Each arm had zero warmup and timed allocator retries, ended at loss 1.7409, and matched all 80 rank-step loss values.

| Setting | Tok/s/node (A/B, B/A) | Tok/s/GPU (A/B, B/A) | s/step (A/B, B/A) | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Tail 4 | 33,158 / 33,156 | 4,144.7 / 4,144.4 | 7.906 / 7.907 | 113.31 / 131.90 |
| Paired tail 2 | 32,966 / 33,036 | 4,120.7 / 4,129.6 | 7.952 / 7.935 | 108.13 / 126.37 |

In a separate comparison at matched microbatch 2 and tail 12, FP8 delivered 28,540 / 28,544 tok/s/node and BF16 delivered 22,685 / 22,928 tok/s/node. Tuned FP8 microbatch 8 with tail 4 delivered 33,128 / 33,160 tok/s/node on that same node at 7.913 / 7.906 s/step. Both FP8 settings matched their own loss bits across orders. The BF16 repeats had different loss bits from step 2.

## Earlier tuning with the default allocator

On eight H200s with FSDP2, sequence length 4,096 and global batch 64, microbatch 8 with specialized MLP backward replay and both mixer outputs replayed delivered **3,868.7 tok/s/GPU**. The MLP-only tail was two blocks. The same-node stock control delivered 3,751.1 tok/s/GPU. Both runs had zero timed allocator retries.

| Recipe | Tok/s/GPU | Peak allocated / reserved GiB | Optimizer updates |
| --- | ---: | ---: | ---: |
| Delta-Matching, microbatch 8, specialized MLP and mixer replay, tail 2 | 3,868.7 | 108.27 / 136.57 | 20 |
| Delta-Matching, microbatch 8, specialized MLP and mixer replay, tail 0 | 3,858.7 | 104.01 / 132.25 | 24 |
| Stock Delta-Matching, microbatch 4, tail 29 | 3,751.1 | 127.20 / 136.08 | 16 |

Use tail 0 to cut reserved memory by another 4.32 GiB in the measured run. Tails 4 and 6 incurred allocator retries after the two warmup steps.

With all explicit activation checkpoints disabled, microbatch 1 fit at 3,450.2 tok/s/GPU and 115.31/131.98 GiB. Microbatches 2 and 4 ran out of memory before the first update. To reproduce those settings, use `--checkpoint-mlp-tail 0 --checkpoint-keep-tail 64 --fp8-shard-cache-mib 0` alongside the FP8 flags. To reproduce BF16's matched FP8 study point, use microbatch 2 and `--checkpoint-mlp-tail 12 --fp8-shard-cache-mib 0`. That same-node matched FP8 run delivered 3,358.2 tok/s/GPU at 83.79/99.83 GiB.

Parameter gradients, reduction, master parameters, and AdamW moments remain FP32. Specialized MLP replay matched all outputs and gradients bit for bit at production MLP dimensions. Combined mixer and MLP replay matched all 53 FP32 parameter gradient and optimizer state shards over two updates on the reduced FSDP2 fixture.
