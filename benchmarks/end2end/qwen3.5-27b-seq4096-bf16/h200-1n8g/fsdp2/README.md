# Dense Qwen3.5 tensor parallelism

The dense BF16 trainer accepts `--tp 2` or `--tp 4` with FSDP2 on the
remaining ranks. TP shards the residual stream across sequence tokens,
attention across complete query/KV head groups, DeltaNet across matching
key/value head groups, and the embedding and LM head across vocabulary.
TP groups must fit on one node. Qwen3.5-27B has four KV heads, which limits
this layout to TP4. The trainer supports HF checkpoint import.

The row projection writes FP32 GEMM partials and reduce-scatters them
in FP32. The column projection backward follows the same rule for input
gradients. These sequence collectives cast their results at the BF16
activation boundary. Linear weight GEMMs write FP32 gradients, and
FSDP2 reduces and accumulates FP32 shards. Master weights and both
AdamW moments remain FP32.

After the precision audit passes, launch the benchmark from the project
workspace with the GPU pool:

```bash
model="$PWD/inputs/Qwen3.5-27B"
bench/gpu-run.sh --gpus 8 --time 00:10:00 --timing -- \
  bash bench/run-tp.sh exact 8 benchmark --model "$model" --weights "$model" \
  --tp 2 --async-tp --compile-blocks --checkpoint selective \
  --seq 4096 --batch 64 --micro-batch 2 --blocks-per-group 2 \
  --skip-embedding-backward-gather --steps 6
```

`--async-tp` overlaps sequence collectives with GEMMs through PyTorch
SymmetricMemory, whose Unix sockets require a short `TMPDIR`. Use
`--tp-regather-inputs` to save only a rank's own sequence shard of
column projections and overlap the backward all-gather with input-gradient
compute. The benchmark uses fused AdamW at 1e-5, the default CUDA
allocator, and WikiText-103. It reports maximum per-rank allocation retries
each step and stops if a retry occurs after step 2.

Run the reduced parity audit with TP2 and an FSDP degree of two:

```bash
bench/gpu-run.sh --gpus 4 --time 00:12:00 --non-timing -- \
  bash bench/run-tp.sh exact 4 parity --async-tp --compile-blocks --checkpoint selective
```

The audit reports relative L2 error for every parameter and TP shard
against the unsharded FP32 model. It fails if any error exceeds the
corresponding stock BF16 error. Both attention types and masked tokens
are exercised across two microbatches. The audit also checks the FP32
shard mapping, master weights, gradients, and fused AdamW state.

`tp_parity.py --bisect --cases sync-compiled async-compiled` compares
sync and async TP on one fixture and mesh. The cancellation probe checks
FP32 row GEMM outputs and reduce-scatter in both modes.
