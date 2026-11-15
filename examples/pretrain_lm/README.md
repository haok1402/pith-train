# Pretrain Language Model

Pretrain a Mixture-of-Experts (MoE) language model with pipeline, data, context and expert
parallelism (PP x DP x CP x EP) under MoE parallel folding, and FSDP.

## Step 1: Prepare Data

Tokenize the training corpus once per tokenizer:

```bash
bash examples/tokenize_corpus/launch.sh dclm-qwen3          # qwen3-30b-a3b
bash examples/tokenize_corpus/launch.sh dclm-qwen3.5        # qwen3.5-35b-a3b
bash examples/tokenize_corpus/launch.sh dclm-deepseek-v2    # deepseek-v2-lite
bash examples/tokenize_corpus/launch.sh dclm-gpt-oss        # gpt-oss-20b, gpt-oss-120b
```

See [tokenize_corpus](../tokenize_corpus/) for details.

## Step 2: Launch Training

```bash
bash examples/pretrain_lm/launch.sh qwen3-30b-a3b
srun -W 0 examples/pretrain_lm/launch.sh gpt-oss-120b     # multi-node, under SLURM
```

The launch script auto-detects available GPUs and works with both single-node and multi-node
(SLURM) setups.

## Available Models

| Model | Mesh | GPUs | Attention | Experts |
|---|---|---|---|---|
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `pp=1 ep=8` | 8 (H200/B200) | `dp=8` | `ep=8` |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `pp=1 ep=8` | 8 (H200/B200) | `dp=8` | `ep=8` |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | `pp=1 ep=8` | 8 | `dp=8` | `ep=8` |
| [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) | `pp=1 ep=8` | 8 | `dp=8` | `ep=8` |
| [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) | `pp=4 ep=8` | 32 | `dp=8` per stage | `ep=8` per stage |

Each model directory holds a `script.py` (the run: mesh, batch, optimizer, checkpoints) and a
`config.json` (the architecture). Edit `script.py` to customize.

Data parallelism is not set: with MoE parallel folding the `world / pp` GPUs of a pipeline stage
are factored twice, once for attention as `dp x cp` and once for the experts as `expert-dp x ep`.
So attention `dp = world / (pp * cp)` and `expert-dp = world / (pp * ep)`, and `cp` and `ep` are
chosen independently, each only dividing `world / pp`: EP may span CP. At `pp=1 ep=8` on one node
every GPU is an attention DP rank and the experts are sharded 8 ways over the same GPUs.

To change the mesh, see [Scaling a run](../../docs/user-guide.md#scaling-a-run). The end-to-end
benchmark suite, [benchmarks/end2end](../../benchmarks/end2end/), measures every model over the
layouts folding allows on 1, 2 and 4 H200 nodes, which meshes fit, and how fast each is; check it
before picking a mesh of your own. GPT-OSS has no context parallelism (`cp=1`).

## Output

Checkpoints are saved to `workspace/checkpoints/<model>/` at regular intervals. Training resumes
from the latest checkpoint automatically.
