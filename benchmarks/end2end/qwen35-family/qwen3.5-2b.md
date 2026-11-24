# Qwen3.5-2B on one eight-H200 node

Use the official Qwen/Qwen3.5-2B BF16 checkpoint at
`15852e8c16360a2fea060d615a32b45270f8a8fc` and the shared Qwen-tokenized WikiText-103 train shard.
The text tower uses a tied embedding and BF16 vocabulary projection. These recipes use FSDP2
`pp1-dp8-cp1-ep1`, sequence length 4,096, global batch 64, fused AdamW and seed 1234.
[Model checkpoint](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)
· [Shared harness](README.md)

| Recipe | Microbatch | FSDP group | Activation checkpointing | CUDA allocator | FP8 shard cache |
| --- | ---: | ---: | --- | --- | ---: |
| BF16 tuned | 8 | 8 | Off | Default | — |
| FP8 matched | 8 | 8 | Off | Default | 0 MiB |
| FP8 tuned | 8 | 12 | Off | Expandable segments | 256 MiB |

Each invocation runs BF16→FP8 and FP8→BF16 on the same node. Start with the verified CUDA 13.2
environment and the absolute path to the family's `torchrun` in `QWEN35_TORCHRUN`. Set `checkout`
to this checkout, `model_dir` to the local HF checkpoint directory, `corpus_bin` to the shared
token shard, and `matched_dir` and `tuned_dir` to fresh output directories. From the pool's
project root, run the following commands.

```bash
recipe_dir="$(realpath "$checkout")/benchmarks/end2end/qwen35-family"
export QWEN35_REVISION="$(git -C "$checkout" rev-parse '637d52e^{commit}')"
export QWEN35_2B_TORCHRUN="$QWEN35_TORCHRUN"
export QWEN35_TORCHRUN="$recipe_dir/qwen3.5-2b-torchrun.sh"
export QWEN35_2B_BF16_ALLOCATOR=default
bf16_arm='8:0:24:none:8'
fp8_arm='8:0:24:none:12:256'

QWEN35_2B_FP8_ALLOCATOR=default \
    bash bench/gpu-run.sh --priority normal --timing --gpus 8 --time 12 -- \
    bash "$recipe_dir/run.sh" matched "$model_dir" "$corpus_bin" "$matched_dir" 12 "$bf16_arm"

QWEN35_2B_FP8_ALLOCATOR=expandable \
    bash bench/gpu-run.sh --priority normal --timing --gpus 8 --time 12 -- \
    bash "$recipe_dir/run.sh" tuned "$model_dir" "$corpus_bin" "$tuned_dir" 12 "$bf16_arm" "$fp8_arm"

python3 "$recipe_dir/summarize.py" "$matched_dir"
python3 "$recipe_dir/summarize.py" "$tuned_dir"
```

Compare updates 7–12 in each arm. Check that the allocator retry counters stay at zero and that
cross-entropy and pre-clip gradient norms stay finite.
