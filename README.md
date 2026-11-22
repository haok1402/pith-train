# Delta-Matching Qwen3.5-27B BF16 setup

Check out this branch as a benchmark workspace. The launcher expects the PithTrain fork in the nested `pith-train/` checkout and the NRT1 CUDA 13.2 installation at `../local/cuda-13.2`.

```bash
git clone https://github.com/haok1402/pith-train pith-train
git -C pith-train checkout --detach f66bcc9ce7803954d14608162a31d0fb10dcd3e6
mkdir -p logs
srun -A nvr_elm_llm -p cpu -q cpu-short -N1 -n1 -c16 --mem=64G \
  --time=00:25:00 -o logs/uv-%j.log bash bench/install-env.sh
srun -A nvr_elm_llm -p cpu -q cpu-short -N1 -n1 -c4 --mem=8G \
  --time=01:15:00 -o logs/weights-%j.log bash inputs/fetch-qwen35.sh
```

[STATUS.md](notes/STATUS.md) has the GPU launcher, detached benchmark command, and measured state. [survey.md](notes/survey.md) describes the dense execution path and ablations. The BF16 driver adapts the fork's PP8 recipe at `088b32d` to a short 4K synthetic-data benchmark using the released weights.

`/usr/bin/python3 bench/summarize.py logs/bf16-JOB_ID.log` summarizes the steady-state step time, throughput, allocated memory, and MFU.

The integration tests use the pinned WikiText-103 train fixture and fused AdamW. Prepare the fixture in a CPU Slurm job with `bash bench/prepare-integration-text.sh`. The command accepts `E2EFP8_TEXT_PYTHON` for an environment with NumPy, PyArrow and tokenizers. [RESULTS.md](notes/RESULTS.md) records the full-node ablations and the held-out WikiText loss reference.
