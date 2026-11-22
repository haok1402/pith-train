# Qwen3.5-27B setup status

- Fork: `pith-train/` at `f66bcc9`. Dense support landed in `a4225d9`. The PP8/4K recipe on `origin/examples-4k-h200` at `088b32d` targets two H200 nodes.
- Environment: `pith-train/.venv`, ready and import-verified on the CPU partition. Python 3.12, PyTorch 2.15.0+cu132, Transformers 5.24.0, DeepGEMM 2.8.2. The first build hit missing ELF headers. Job `1263153` succeeded with workspace-local Ubuntu headers through `bench/install-env.sh`. Source `bench/env.sh` inside your Slurm job. Use this venv with another checkout by putting that checkout first on `PYTHONPATH`.
- Weights: `inputs/Qwen3.5-27B/`, ready. All 11 safetensors files match the index and header sizes. HF revision `81a287ca484a5e71148f5beb7f367bb5286d70f9`, 55.56 GB total.
- Survey: [survey.md](survey.md), complete. At 4K, 50% MFU requires approximately 25,199 aggregate tokens/s.
- Integrated target: 50% BF16 MFU at 4K with real-text loss sanity. [Shared ablation table](RESULTS.md).
- PP8 control: historical Muon job `1263388` OOMed on step 2. The detached Muon retry `1263873` was stopped after the switch to AdamW. Detached replacement `1264441` uses fused AdamW, batch 64, WikiText 4K and expandable segments. [Submission log](../logs/pp8-adamw-4k-real-retry-submit.log).
- The dense FSDP2 baseline met the target in job `1263478` on `pool1-00668`. At 4K and global batch 64, the median of steps 3–5 was 12.192 s, 21,502 aggregate tokens/s, 118.66 GiB peak allocated per GPU, and 42.66% MFU. [Method and configuration](fsdp2.md), [training log](../logs/fsdp2-full-compile-20261122T080428.log).
- FSDP2 real-text control `1264511` is running on `pool1-00782`. The detached B2 gate-fusion trial `1264522` is submitted. Both use fused AdamW at 4K and batch 64. See [RESULTS.md](RESULTS.md).
- Slurm printed a stale-data threshold warning. Submissions are currently accepted.

## GPU jobs

Lease fix live since 00:36:21 PST on 2026-11-22. Liveness = per-ticket flock lease.

Submit from the login node with [gpu-run.sh](../bench/gpu-run.sh). It sets CUDA 13.2, requests 28 CPUs per H200, and runs a command-bound interactive job named `deltamatch-e2efp8`. The local FIFO gate is in `locks/gpu-run/`. The head checks pending and running interactive jobs at most once every 15 seconds across all waiters. It submits when one of the eight slots is free and releases the line when Slurm accepts the job. A confirmed cap rejection keeps the same ticket at the head. Waiting prints the queue position. Slurm enforces the four-node cap and determines when accepted jobs start. Two concurrent one-GPU `nvidia-smi` checks passed, jobs `1263696` and `1263788` on `pool1-00698`.

```bash
# One H200 smoke check or kernel micro-benchmark.
bench/gpu-run.sh --gpus 1 --time 00:02:00 -- nvidia-smi

# Two H200 proxy.
bench/gpu-run.sh --gpus 2 --time 00:10:00 -- bash path/to/your-proxy.sh

# Detached eight-H200 BF16 allocator retry after the restart.
setsid nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash bench/allocate-bf16.sh --seq 4096 --batch 64 --steps 6 \
  --dataset "$PWD/inputs/integration-wikitext-train" \
  > logs/pp8-adamw-4k-real-retry-submit.log 2>&1 < /dev/null &
```

After maintenance, launch each GPU command detached with `setsid nohup`, redirect to a run log, and record the job ID when the controller returns it.
