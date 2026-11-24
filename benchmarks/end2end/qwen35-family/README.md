# Dense Qwen3.5 BF16 and FP8 on eight H200s

Use the official BF16 checkpoint for each model and the shared 27B WikiText training token shard.
The launcher loads the same checkpoint anew for every arm, uses PithTrain's default seed, and
checks that the sample-index hashes match across arms. The workload is 4,096 tokens per sample and
64 samples per update on one eight-GPU node (`pp1-dp8-cp1-ep1`). FP8 enables the stack's blockwise
linears, Delta-Matching attention and FP32 weight-gradient kernel. The vocabulary projection uses
BF16 in both recipes.

0.8B, 2B and 4B use a tied embedding and vocabulary projection. The root FSDP unit owns one
`embed_tokens.weight` for both uses. The LM-head matrix gradient is computed in FP32 and FSDP adds
the embedding lookup gradient before reduce-scatter. The 9B config uses separate weights. Use the
same tie policy in BF16 and FP8. The loader accepts indexed shards and single-file safetensors.

## Launch

From the project root, set these variables for the pinned family checkout, model, corpus and run
directory. Use an unused directory per submission. Source the cluster environment before
submitting so the configured virtualenv and CUDA 13.2 compatibility libraries reach the task.

```bash
source .cache/fsdp2-fp8-verify/env.sh
export QWEN35_TORCHRUN="$fsdp2_fp8_venv/bin/torchrun"
export QWEN35_REVISION="$(git -C "$family_checkout" rev-parse HEAD)"
runner="$family_checkout/benchmarks/end2end/qwen35-family/run.sh"

# BF16 screen for the 24-layer models, microbatch 4, no recompute, group size 2.
bash bench/gpu-run.sh --priority normal --timing --gpus 8 --time 12 -- \
    bash "$runner" bf16 "$model_dir" "$corpus_bin" "$screen_dir" 8 '4:0:24:none:2'

# Once tuned, supply the winning BF16 recipe and use unique output directories.
bash bench/gpu-run.sh --priority normal --timing --gpus 8 --time 50 -- \
    bash "$runner" matched "$model_dir" "$corpus_bin" "$matched_dir" 12 "$bf16_arm"
bash bench/gpu-run.sh --priority normal --timing --gpus 8 --time 50 -- \
    bash "$runner" tuned "$model_dir" "$corpus_bin" "$tuned_dir" 12 "$bf16_arm" "$fp8_arm"
```

An arm is `MICROBATCH:MLP_ONLY_TAIL:KEPT_TAIL:MIXERS:FSDP_GROUP[:FP8_CACHE_MIB[:replay]]`.
`MIXERS` is `none`, `gated-delta`, `full` or `both`. It controls which mixer outputs the outer
checkpoint replays. `replay` enables the FP8 MLP's specialized replay. Use microbatch 1, 2, 4 or 8.
Set `MLP_ONLY_TAIL=0` and `KEPT_TAIL` to the model depth to disable explicit recompute. `matched`
copies the BF16 arm and uses the FP8 stack with cache and specialized replay off. `fp8` runs a
single FP8 candidate. `QWEN35_ARM_TIMEOUT` bounds each arm and defaults to `12m`. Use the shared
gate at normal priority and submit each pair as one task. The runner copies the pinned commit
into the run directory and executes the four arms sequentially in BF16/FP8 and FP8/BF16 order.

## Handoff

```bash
summary="$family_checkout/benchmarks/end2end/qwen35-family/summarize.py"
python "$summary" "$matched_dir" > "$matched_dir/summary.json"
python "$summary" "$tuned_dir" > "$tuned_dir/summary.json"
```

Final numbers use the median throughput over updates 7–12 per arm and the median of the two arm
medians per format. Report the median of the two paired speedups, retaining both order-specific
ratios in the JSON. The summary records the largest timed peak allocated and reserved GiB, allocator
retries, per-step cross-entropy and gradient norm, and the maximum paired loss delta. The BF16 MFU
uses Megatron's causal BSHD dense Gated DeltaNet estimator and H200's 989.4 TFLOP/s dense BF16 peak.
Share both JSON files, the winning arm specs, any failed candidates and the evidence behind any
attribution of time to GEMMs, Gated DeltaNet or attention. Keep raw logs in the run directories.
