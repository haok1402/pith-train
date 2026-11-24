# Qwen3.5-27B FP8 recompute recipes

On eight H200s with FSDP2, sequence length 4,096 and global batch 64, microbatch 8 with specialized MLP backward replay and both mixer outputs replayed delivered **3,868.7 tok/s/GPU**. The MLP-only tail was two blocks. The same-node stock control delivered 3,751.1 tok/s/GPU. Both runs had zero timed allocator retries.

| Recipe | Tok/s/GPU | Peak allocated / reserved GiB | Optimizer updates |
| --- | ---: | ---: | ---: |
| Delta-Matching, microbatch 8, specialized MLP and mixer replay, tail 2 | 3,868.7 | 108.27 / 136.57 | 20 |
| Delta-Matching, microbatch 8, specialized MLP and mixer replay, tail 0 | 3,858.7 | 104.01 / 132.25 | 24 |
| Stock Delta-Matching, microbatch 4, tail 29 | 3,751.1 | 127.20 / 136.08 | 16 |

Pass these flags to the dense FSDP2 benchmark with the Qwen3.5-27B model, weight, and dataset paths.

```text
--fp8 --fp8-attention --batch 64 --seq 4096 --micro-batch 8
--fp8-mlp-recompute --checkpoint-replay-mixers both
--checkpoint-mlp-tail 2 --fp8-shard-cache-mib 1024
```

Use tail 0 to cut reserved memory by another 4.32 GiB in the measured run. Tails 4 and 6 incurred allocator retries after the two warmup steps.

With all explicit activation checkpoints disabled, microbatch 1 fit at 3,450.2 tok/s/GPU and 115.31/131.98 GiB. Microbatches 2 and 4 ran out of memory before the first update. To reproduce those settings, use `--checkpoint-mlp-tail 0 --checkpoint-keep-tail 64 --fp8-shard-cache-mib 0` alongside the FP8 flags. To reproduce BF16's matched FP8 study point, use microbatch 2 and `--checkpoint-mlp-tail 12 --fp8-shard-cache-mib 0`. That same-node matched FP8 run delivered 3,358.2 tok/s/GPU at 83.79/99.83 GiB.

Parameter gradients, reduction, master parameters, and AdamW moments remain FP32. Specialized MLP replay matched all outputs and gradients bit for bit at production MLP dimensions. Combined mixer and MLP replay matched all 53 FP32 parameter gradient and optimizer state shards over two updates on the reduced FSDP2 fixture.
