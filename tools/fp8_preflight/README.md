# Dense H200 FP8 preflight

Run this from a Slurm login node with the PithTrain environment active.

```bash
export CUDA_HOME=/path/to/cuda-13.2
export DEEP_JIT_INCLUDE=/path/to/DeepGEMM/third-party/deep_jit/include
bash examples/pretrain_lm/launch.sh --fp8-preflight \
  /path/to/Qwen3.5-27B /path/to/Qwen3.5-27B /path/to/tokenized-corpus \
  /path/to/new-output-directory 5 --audit-losses
```

The supported recipe is eight H200s with FSDP2, FP8 linear layers and Delta-Matching
attention, sequence length 4096, and eight examples per rank. It loads the dense
Qwen3.5-27B HF checkpoint. The launcher seals kernels and weights in separate CPU
jobs, then requests the training GPUs on short QoS. The kernel job has 10 minutes
for cold imports, CuTe export and native DeepGEMM compilation. The weight job
has 8 minutes for cold imports and four concurrent workers reading and verifying
the 50 GiB rank packs. A successful stage can be reused if the next stage fails.

`PITHTRAIN_PYTHON` selects the interpreter. `ELFUTILS_INCLUDE` adds an elfutils
header directory to the DeepGEMM build. `PITHTRAIN_PREFLIGHT_CACHE` lets
subsequent launches reuse a verified bundle. `PITHTRAIN_WEIGHT_SEED` can reuse
compatible rank packs on the same filesystem after checking their contents
against the source checkpoint. `PITHTRAIN_DEEPGEMM_SEED` accepts a prior
DeepGEMM receipt and cache. CPU preflight recomputes the required JIT keys and
checks the hashes of selected seed artifacts. `PITHTRAIN_PREFLIGHT_ONLY=1` stops after CPU
verification. Set `PITHTRAIN_GPU_TIME` for runs longer than the smoke.

The bundle records the source and compiler identity, CuTe launch ABI, DeepGEMM
cache entries, and rank-local weight digests. The training worker resolves CuTe
and DeepGEMM specializations from the sealed bundle. A missing artifact ends the
job with the launch signature or cache compiler invocation in the log. Torch
Inductor's runtime compile and first use of cross-entropy have separate timings.

`preflight.log` and `weights.log` contain the CPU jobs and seal results. `run.log` records
per-rank timings and loss, and `gpu.csv` contains a 200 ms utilization timeline
for each GPU. Generate a summary after a completed training job:

```bash
python -m tools.fp8_preflight.receipt /path/to/output \
  --baseline /path/to/prior-run.log
```
