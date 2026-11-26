# Dense H200 FP8 preflight

Run this from a Slurm login node with the PithTrain environment active.

```bash
export CUDA_HOME=/path/to/cuda-13.2
export DEEP_JIT_INCLUDE=/path/to/DeepGEMM/third-party/deep_jit/include
export PITHTRAIN_INDUCTOR_SEED=/path/to/prior/inductor-cache
export PITHTRAIN_INDUCTOR_MANIFEST=/path/to/inductor-manifest.json
bash examples/pretrain_lm/launch.sh --fp8-preflight \
  /path/to/Qwen3.5-27B /path/to/Qwen3.5-27B /path/to/tokenized-corpus \
  /path/to/new-output-directory 5 --audit-losses
```

The supported recipe is eight H200s with FSDP2, FP8 linear layers and Delta-Matching
attention, sequence length 4096, and eight examples per rank. It loads the dense
Qwen3.5-27B HF checkpoint. The launcher seals kernels and weights in separate CPU
jobs, then requests the training GPUs. The kernel job has 10 minutes
for cold imports, CuTe export and native DeepGEMM compilation. The weight job
has 8 minutes for cold imports and four concurrent workers reading and verifying
the 50 GiB rank packs. A successful stage can be reused if the next stage fails.

Set `PITHTRAIN_SLURM_ACCOUNT`, `PITHTRAIN_CPU_PARTITION`, and
`PITHTRAIN_GPU_PARTITION` for your site. The launcher also accepts
`PITHTRAIN_CPU_QOS`, `PITHTRAIN_GPU_QOS`, `PITHTRAIN_GPU_CONSTRAINT`, and
`PITHTRAIN_SLURM_JOB_PREFIX`. Unset scheduler options use Slurm defaults.

`PITHTRAIN_PYTHON` selects the interpreter. `ELFUTILS_INCLUDE` adds an elfutils
header directory to the DeepGEMM build. `PITHTRAIN_PREFLIGHT_CACHE` lets
subsequent launches reuse a verified bundle. `PITHTRAIN_WEIGHT_SEED` can reuse
compatible rank packs on the same filesystem after checking their contents
against the source checkpoint. `PITHTRAIN_DEEPGEMM_SEED` accepts a prior
DeepGEMM receipt and cache. CPU preflight recomputes the required JIT keys and
checks the hashes of selected seed artifacts. `PITHTRAIN_PREFLIGHT_ONLY=1` stops after CPU
verification. Set `PITHTRAIN_GPU_TIME` for runs longer than the smoke.

The Inductor manifest records the SHA-256 of the original `config.json` in
`model_config`, the compiler source keys in `torch_key` (hex) and `triton_key`,
and a `files` mapping from relative `fxgraph` paths to SHA-256 digests. Preflight
checks the target and covers both attention mixers in forward and backward and
the residual stage in forward, on every CUDA device index. The worker logs the
actual FX cache hits and fails on an uncovered graph or a guard mismatch.

`PITHTRAIN_RUNTIME_SEED` can supply the runtime's `venv-N.tar.zst` archives and
their `sha256` manifest. The seed also records the base interpreter path in
`python` and the venv-relative site-packages directory in `site`. CPU preflight
verifies the archives. The worker starts utilization capture and extracts them
once per node, then uses the staged interpreter on all ranks. The default
staging directory is `/dev/shm`. Set `PITHTRAIN_NODE_STORAGE` to use another
node-local filesystem.

The bundle records the source and compiler identity, CuTe launch ABI, DeepGEMM
cache entries, rank-local weight digests, and the resolved HF text configuration.
The dense worker uses that configuration directly and imports only its model family.
The training worker resolves CuTe
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

The wrapper check uses the same launch fixtures as preflight. The CPU mode checks
the exported signatures and loads each shared object. The CUDA mode requires one
H200 and `--gpu-utilization` to capture the timeline from its first import onward.
If a job-local sampler already started before staging, pass its CSV with
`--recorded-gpu-utilization` and use `--staged-runtime` with the staged interpreter.
Use the sealed DeepGEMM cache and the version-only NVCC shim for that check.
