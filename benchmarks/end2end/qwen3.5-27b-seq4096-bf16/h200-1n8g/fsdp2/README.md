# Dense Qwen3.5 tensor parallelism

The dense FSDP2 benchmark enables tensor parallelism with `--tp 2` or `--tp 4`.
FSDP shards across the remaining ranks. `--async-tp` enables PyTorch
SymmetricMemory GEMM and collective overlap. The defaults use dense FSDP2
with TP disabled.

TP shards the residual stream across tokens, attention across complete
query and KV groups, DeltaNet across matching key and value heads, and
the embedding and LM head across vocabulary. TP groups must fit on one
node. The Qwen3.5-27B config has four KV heads, which limits this sharding
to TP4. The standalone BF16 path supports HF checkpoint import. DCP,
HSDP, and combinations with PP, CP or EP are currently unsupported.

Run the benchmark from the repository root with the PithTrain environment
active, using the scheduler to provide eight GPUs. For example:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/end2end/qwen3.5-27b-seq4096-bf16/h200-1n8g/fsdp2/script.py \
  --model "$MODEL_DIR" --weights "$MODEL_DIR" --dataset "$DATASET_DIR" \
  --tp 2 --async-tp --compile-blocks --seq 4096 --batch 64 --micro-batch 2 \
  --reduce-dtype bf16 --bf16-sharded-gradients --blocks-per-group 2 \
  --skip-embedding-backward-gather --steps 6
```

The benchmark uses fused AdamW by default. SymmetricMemory's CUDA backend
creates Unix sockets under `TMPDIR`. Use a sufficiently short path reachable
by every rank on the node.

On one eight-H200 node with 4K WikiText-103, async TP2 at microbatch 2
reached 47.54% MFU and 117.24 GiB peak allocated per GPU with zero
allocator retries over six steps. TP2 at microbatch 4 ran out of memory.
Async TP4 at microbatch 4 reached 42.59% MFU. The tuned FSDP2 control
reached 50.65% MFU, so the benchmark keeps TP optional.

On two GPUs, run `tp_parity.py --hf-roundtrip` with `torchrun` for the reduced
loss and gradient check. Use `--precision-reference` to add BF16 FSDP2 and
FP32 recurrence controls, or `--check-tp-partials` to also check the head-local
BF16 kernels and FP32 GEMM partials.
