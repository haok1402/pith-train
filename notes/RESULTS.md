# Qwen3.5-27B BF16 integration results

The target is at least 50% MFU on one eight-H200 node at sequence length 4,096, with real-text loss checked against the released HF model. MFU is aggregate tokens/s × 156.9905 GFLOP/token ÷ (8 × 989 TFLOP/s). The count includes 48 Gated DeltaNet layers and 16 causal gated GQA layers. The exact 50% cutoff is approximately 25,199 aggregate tokens/s. [Calculator](../bench/summarize.py), [survey](survey.md).

The matched integration runs use fused AdamW, learning rate 1e-5, global batch 64 and the same WikiText-103 train fixture. Peak is the maximum allocated GiB on any GPU in the timed window. The real-text comparison records pre-update mean loss against HF on the same tokens and checks training for finite loss and gradients. The initial eight-sample 4K WikiText-103 test reference is **1.793438 nats**, job `1263942`. [HF reference log](../logs/fsdp2-hf-reference-gated.log). The dense FSDP2 control scored **1.7957** on those eight examples, a difference of **+0.002262 nats**, with finite gradient norm 30.7754. [FSDP2 check, 1264087](../logs/fsdp2-real-wikitext-1264087.log).

The released HF model scored **1.813772 nats** on the first 16 seeded 4K WikiText test samples for the B2 comparison. The identical integrated FSDP2 test batch is waiting for resources as job `1266084`. [HF reference, 1264958](../logs/hf-wikitext-reference16-submit.log), [FSDP2 B2 submission](../logs/fsdp2-integration-b2-gates-loss16.log).

| Technique | Batch / microbatch | Workload and timing | Step, s | Aggregate tok/s | Peak GiB/GPU | MFU | Δ MFU vs matched control | Real-text sanity | Job / evidence |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Stock PP8 DualPipeV with AdamW and expandable allocator segments | 64 / 1 | WikiText train. **Does not fit on 1 node.** Rank 0 OOM in step 2. | — | — | 135.18 at OOM | — | — | First loss 1.9220; finite gradient norm 35.7535 | [1264441](../logs/bf16-1264441.log) |
| Dense 1F1B PP4×DP2, AdamW, 15/17/17/15 layers, selective MLP recompute | 256 / 2 | WikiText. HBM OOM after warmup. | — | — | 135.12 at OOM | — | — | First logged loss 1.9815, finite gradient | [1264537](../logs/ppdense-full-adamw-wikitext-pp4-mbs2-20261122T083725.log) |
| Dense 1F1B PP2×DP4, AdamW, 30/34 layers, selective MLP recompute | 256 / 1 | WikiText. HBM OOM after warmup. | — | — | 135.10 at OOM | — | — | First logged loss 1.9851, finite gradient | [1264597](../logs/ppdense-full-adamw-wikitext-pp2-mbs1-20261122T083725.log) |
| Dense 1F1B PP2×DP4 with per-microbatch gradient sync, AdamW | 64 / 2 | WikiText. HBM OOM before steady timing. | — | — | 135.17 at OOM | — | — | — | [1265478](../logs/ppdense-full-adamw-wikitext-pp2-mbs2-grad-sync-20261122T085649-1265478.log) |
| Dense 1F1B with AdamW | — | Full-node experiment in progress in pipeline worktree | — | — | — | — | — | Pending | [Pipeline ledger](ppdense.md) |
| Dense FSDP2, compiled, FP32 reduction, reshards after forward, fused AdamW | 64 / 1 | Synthetic, median steps 3–5 | 12.192 | 21,502 | 118.66 | 42.66% | — | Pending matched WikiText run | [1263478](../logs/fsdp2-full-compile-20261122T080428.log) |
| FSDP2, B2 plus selective checkpointing, BF16 reduction, skip embedding backward gather | 64 / 2 | Synthetic, median steps 3–5 | 11.897 | 22,034 | 127.90 | 43.72% | +1.06 pp vs FP32 B1 synthetic FSDP2 | Pending matched B2 WikiText check | [Training log](../logs/fsdp2-full-mbs2-sac-20261122T082348.log) |
| FSDP2 matched real-text control, compiled and FP32 reduction | 64 / 1 | WikiText train, median steps 3–6 | 12.314 | 21,288 | 118.49 | **42.24%** | Reference | 1.9215 → 1.6987, finite; held-out control +0.002262 nats vs HF | [1264511](../logs/fsdp2-integration-control-real-retry.log) |
| Integrated FSDP2, gated projection fusion, BF16 reduction, selective checkpointing, skip embedding backward gather | 64 / 2 | WikiText train, median steps 3–6 | **10.391** | **25,228** | **127.72** | **50.06%** | +7.82 pp vs the FP32 B1 real-text control | 1.9208 → 1.6993, finite; −0.0007 nats vs the B1 control at step 1 | [1264522](../logs/fsdp2-integration-b2-gates-real-retry.log) |
| Exact 32-step integrated FSDP2 repeat | 64 / 2 | WikiText train, median steps 3–32 on pool1-00793 | **10.508** | **24,946** | **127.72** | **49.50%** | −0.56 pp vs the short run on a different node | 1.9208 → 1.6942, finite loss and gradients throughout | [1265084](../logs/fsdp2-integration-b2-gates-repeat32.log) |
| Integrated B2 with the default CUDA allocator | 64 / 2 | WikiText train, median steps 3–6 | 11.712 | 22,382 | 127.90 | 44.41% | −5.65 pp vs the expandable-segments short run, on a different node | 1.9208 → 1.6987, finite | [1265114](../logs/fsdp2-integration-b2-default-alloc-1265114.log), [submission](../logs/fsdp2-integration-b2-default-alloc-submit.log) |
| Integrated B2 with `NCCL_MAX_CTAS=16` | 64 / 2 | WikiText train, median steps 3–6 on pool1-00624 | 10.505 | 24,954 | 127.72 | 49.51% | Within 0.02 pp of the repeat, on a different node | 1.9208 → 1.6992, finite | [1265777](../logs/fsdp2-integration-b2-nccl16-1265777.log), [submission](../logs/fsdp2-integration-b2-nccl16-submit.log) |
| Integrated B2 with the LM-head all-gather retained across microbatches | 64 / 2 | WikiText train. Same-node, same-commit A/B at `0a2cf9f`, queued. | — | — | — | — | Pending | Pending | [Submission](../logs/fsdp2-integration-b2-head-ablate-submit.log) |
| FSDP2 compact or larger communication buckets, BF16 gradient shards | 64 / 2 | WikiText train on pool1-00090. 1, 2 and 4 consecutive blocks per bucket, plus DeltaNet triples with attention singletons. | — | — | — | — | Pending | Pending | [1265621](../logs/fsdp2-full-tune-lease.log) |
| FSDP2 forward all-gather copy-out on a side stream | 64 / 2 planned | WikiText full-node comparison follows proxy correctness and timing. | — | — | — | — | Pending | Pending | [Proxy submission](../logs/fsdp2-copyout-proxy-submit.log) |
| FSDP2 B2, matched selective-checkpoint control with separate gate projections | 64 / 2 | WikiText train, median steps 3–6 on pool1-00651 | 10.506 | 24,952 | 127.72 | 49.51% | Within 0.02 pp of the fused repeat, on a different node | 1.9208 → 1.6990, finite | [1265794](../logs/fsdp2-integration-b2-control-real.log) |
| Integration candidate, B2 with BF16 sharded gradients and reduction overlap, checkpointing disabled | 64 / 2 | WikiText train. Waiting for resources. | — | — | — | — | Pending | Pending | [1266023](../logs/fsdp2-integration-b2-no-ac-real.log) |
| Tensor parallelism | — | Parity and proxy work in progress | — | — | — | — | — | Pending | [TP ledger](tp.md) |
| KDA Gated DeltaNet forward candidate | — / 2 | 4K operator timing 1.095×; full-node ablation pending the branch | — | — | — | — | — | Pending full-model check | [Operator timing](../../kda-tasks/gated_delta_rule_pinned_fwd/runs/c1-timing-repeat.log) |
| KDA cross-entropy first candidate | — / 2 | 4K operator timing 0.971×; original kernel remains the control | — | — | — | — | — | — | [Operator timing](../../kda-tasks/cross_entropy/runs/c01-dense.log) |
| Integrated FSDP2 and selected KDA kernels | — | Full-node trial pending kernel branches | — | — | — | — | — | Pending | Branch `e2efp8-integration` |

The WikiText-103 train fixture is pinned to revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The real-text sanity fixture uses the raw test split. [Dataset](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1). The detailed proxy and operator measurements remain in [fsdp2.md](fsdp2.md), [ppdense.md](ppdense.md), and [kernels.md](kernels.md). Proxy MFU and operator speedups guide which changes to run at 27B.

Stock PP8 does not fit on 1 node with either optimizer tested. The historical Muon job `1263388` OOMed on step 2. The matched AdamW run had 430.81 MiB device memory free and 279.91 MiB reserved but unallocated when the 2.37 GiB request failed. The smaller-batch retry was dropped before Slurm submission.

The first AdamW launchers exited before Slurm submission because the local gate lost tickets across PID namespaces. All three replacements are detached through the repaired lease-based gate. [Gate handoff](gate-handoff.md).

## Full-model profile

Nsight Systems captured one training step of 27B B1 FSDP2 with BF16 reduction on eight H200s. Across ranks, the median GPU window was 11.761 s. Compute kernels occupied 11.392 s and NCCL occupied 6.195 s, with 0.273 s of NCCL outside compute. All-gather occupied 4.153 s and reduce-scatter 2.042 s. These are unions of kernel intervals. GEMMs occupy 73.5–73.8% of each GPU window. FSDP packing and copy-out sum to 6.246 s across eight GPUs, which may overlap with compute. The matched B2 CTA-cap trial tests whether lowering NCCL occupancy reduces contention during overlap. A B2 profile is queued on the instrumented `d155195` checkout with the same training flags. [Profile run 1264719](../logs/fsdp2-nsys-1264719.log), [interval analysis 1265088](../logs/integration-trace-1265088.log), [GEMM breakdown 1265124](../logs/fsdp2-step-1264719-corrected-1265124.log), [B2 profile submission](../logs/fsdp2-integration-b2-nsys-submit.log).

The two-GPU B2 `max-autotune-no-cudagraphs` arm timed out during compilation. Inductor's ATen GEMM candidate led 51 of the 58 logged shape trials, including the main SwiGLU forward shapes. End-to-end compile mode remains unmeasured. [Autotune log, 1264787](../logs/fsdp2-max-autotune-1264787.log).

## Reproduce the 50.06% run

Job `1264522` ran on `pool1-00776` at commit `0d17785ea033be0e40dfabd4393f554404e1c534`, pushed as [e2efp8-integration](https://github.com/haok1402/pith-train/tree/e2efp8-integration). Its four timed steps ranged from 25,127 to 25,320 tokens/s. The exact repeat completed as job `1265084` on `pool1-00793`. Its 30 timed steps ranged from 24,156 to 25,115 tokens/s. The median was 253 tokens/s below the 50% target. Timed losses ranged from 1.6176 to 1.7895 nats. Existing detached ticket 222 was promoted in place. [Short run](../logs/fsdp2-integration-b2-gates-real-retry.log), [32-step repeat](../logs/fsdp2-integration-b2-gates-repeat32.log).

- Eight H200s, 224 CPUs, interactive QoS, one node. CUDA 13.2 with the compatibility libraries first on `LD_LIBRARY_PATH`, through `bench/env.sh`. PyTorch 2.15.0+cu132, Transformers 5.24.0, FlashAttention-4 4.0.0b28 and Triton 3.9 with default FP fusion enabled. The runner sets `OMP_NUM_THREADS=4`, `TORCHINDUCTOR_COMPILE_THREADS=2`, and preloads the system `libstdc++.so.6`.
- Fused AdamW, learning rate 1e-5, seed 1234, BF16, FSDP2 across eight GPUs. Sequence length 4,096, global batch 64, microbatch 2. Default compiled full blocks, selective checkpointing, BF16 reduction, reshard after forward, skip the embedding backward gather, expandable allocator segments.
- Released model revision `81a287ca484a5e71148f5beb7f367bb5286d70f9`. The tokenized WikiText train fixture has 2,048 4K sequences and SHA-256 `005a1380fc258400929a3ee5fdf8015a025a4d87f1edc86ffef7c11c54b22a95`. The separate direct B2 HF comparison uses the first 16 seeded sequences from the WikiText test fixture.

```bash
setsid nohup env E2EFP8_INTEGRATION_TREE=wt-integration PYTORCH_ALLOC_CONF=expandable_segments:True \
  bench/gpu-run.sh --gpus 8 --time 00:20:00 -- \
  bash bench/run-integration.sh train --steps 32 --micro-batch 2 --checkpoint selective \
  --reduce-dtype bf16 --skip-embedding-backward-gather \
  > logs/fsdp2-integration-b2-gates-repeat32.log 2>&1 < /dev/null &
```
