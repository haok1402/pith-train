# FP8 wgrad publication gate

Run this gate on the exact PithTrain commit before publishing every Hopper FP8 wgrad pin.
Keep the output directory as the receipt. Publish only when `result.json` says `"pass": true`.
A missing or failed worker leaves the gate incomplete.

From the repository root, use the training Python environment and CUDA 13.2.

```bash
python -m tools.fp8_wgrad_gate.gate run \
  --matrix "$DENSE_MATRIX" --patched "$PATCHED_DEEPGEMM_PACKAGE" \
  --evaluator "$FROZEN_EVALUATOR" --protocol "$DENSE_REPLAY_PROTOCOL" \
  --changed "$AFFECTED_CASE_IDS" --output "$NEW_RECEIPT_DIR" \
  --cpu-command "$CPU_LAUNCH" --gpu-command "$ONE_GPU_LAUNCH" \
  --smoke-command "$SHORT_ONE_GPU_LAUNCH"
```

Use the 55-case `representative-dense-wgrad-v3/matrix.json`, the accompanying frozen
`evaluate.py` and the sustained replay helper used by dense-shared-v1. The package argument points
to the installed `deep_gemm` directory. Supply `CUDA_HOME` and `DEEP_JIT_INCLUDE`, and set
`ELFUTILS_INCLUDE` if the compiler needs it. Launch commands are executable prefixes that append
the worker's arguments. They receive `WGRAD_GATE_STEP` and `WGRAD_GATE_EXCLUDE_NODE`. Honor the
exclusion for reverse timing. Use a CPU job for preparation and independent single-H200 jobs for
the GPU workers, with the integration worker on short QoS. Use the cluster's account and job
prefix in the external launch commands.

- **Exactness.** All 55 block-scale geometries must match the pinned patched DeepGEMM FP32 bits
  on recipe and signed-zero/subnormal inputs. The guard suite also covers odd M, K and N tails,
  offsets, zero work and all 132 visible SM counts. For an illegal reference layout the oracle
  makes aligned, zero-padded copies and trims unused output rows and columns.
- **Timing.** Supply a comma-separated matrix ID for every touched production route and every
  distinct affected geometry. The script adds the nearest production geometry to each ID by
  summed log2 distance in M, N and K. Use `dispatch` for a change to the shared entry guard.
  Both orders use the same operands and output address, a 700 W H200, 32-launch CUDA graphs,
  at least 2 s of reference preconditioning, 250 ms of warmup per arm and at least 750 ms per
  timed arm. Each order keeps two A/B and two A/A pairs. A case fails if both order medians
  exceed +0.70%. The receipt retains one-order excursions and telemetry.
- **Memcheck.** Compute Sanitizer must exit successfully and report zero errors. The caller-stream
  dependency and offset cases must still match, including output guard regions.
- **Integration.** The preflight launcher must load the sealed CPU artifacts for the eager,
  one-rank FSDP2 fixture. It compares both accumulated microbatch losses and the FP32 gradient
  shard against an independent backward using patched DeepGEMM. The eager fixture's Inductor
  manifest is empty. CuTe and DeepGEMM cache misses fail the worker.
- **Stock comparison.** Add `--stock "$UNMODIFIED_DEEPGEMM_PACKAGE"` for the one-time 55-case
  whole-linear-backward measurement. Both arms include gradient quantization and dgrad. The
  stock arm also repeats activation scales per channel and accumulates 1D×1D into a zeroed
  FP32 main gradient. The ratio has no publication threshold.
