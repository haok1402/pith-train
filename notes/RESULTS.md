# Qwen3.5-27B BF16 integration results

The target is at least 50% MFU on one eight-H200 node at sequence length 4,096, with real-text loss checked against the released HF model. MFU is aggregate tokens/s × 156.9905 GFLOP/token ÷ (8 × 989 TFLOP/s). The count includes 48 Gated DeltaNet layers and 16 causal gated GQA layers. The exact 50% cutoff is approximately 25,199 aggregate tokens/s. [Calculator](../bench/summarize.py), [survey](survey.md).

New full-node trials use fused AdamW, learning rate 1e-5 and the same WikiText-103 train fixture. End-to-end MFU gains use the same global batch, data and timing window. Peak is the maximum allocated GiB on any GPU in the timed window. The real-text comparison records pre-update mean loss against HF on the same tokens and checks training for finite loss and gradients. The initial eight-sample 4K WikiText-103 test reference is **1.793438 nats**, job `1263942`. [HF reference log](../logs/fsdp2-hf-reference-gated.log). The dense FSDP2 control scored **1.7957** on those eight examples, a difference of **+0.002262 nats**, with finite gradient norm 30.7754. [FSDP2 check, 1264087](../logs/fsdp2-real-wikitext-1264087.log).

| Technique | Batch / microbatch | Workload and timing | Step, s | Aggregate tok/s | Peak GiB/GPU | MFU | Δ MFU vs matched control | Real-text sanity | Job / evidence |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PP8 DualPipeV with AdamW and expandable allocator segments | 64 / 1 | WikiText train. Submitted. | — | — | — | — | — | Pending | [1264441](../logs/pp8-adamw-4k-real-retry-submit.log) |
| Dense 1F1B with AdamW | — | Full-node experiment in progress in pipeline worktree | — | — | — | — | — | Pending | [Pipeline ledger](ppdense.md) |
| Dense FSDP2, compiled, FP32 reduction, reshards after forward, fused AdamW | 64 / 1 | Synthetic, median steps 3–5 | 12.192 | 21,502 | 118.66 | 42.66% | — | Pending matched WikiText run | [1263478](../logs/fsdp2-full-compile-20261122T080428.log) |
| FSDP2, B2 plus selective checkpointing, BF16 reduction, skip embedding backward gather | 64 / 2 | Synthetic, median steps 3–5 | 11.897 | 22,034 | 127.90 | 43.72% | +1.06 pp vs FP32 B1 synthetic FSDP2 | Pending matched B2 WikiText check | [Training log](../logs/fsdp2-full-mbs2-sac-20261122T082348.log) |
| FSDP2 matched real-text control, compiled and FP32 reduction | 64 / 1 | WikiText train. Running. | — | — | — | — | — | Pending | [1264511](../logs/fsdp2-integration-control-real-retry.log) |
| Integration candidate, gated projection fusion, BF16 reduction, selective checkpointing, skip embedding backward gather | 64 / 2 | WikiText train. Submitted. | — | — | — | — | Pending matched control | Pending | [1264522](../logs/fsdp2-integration-b2-gates-real-retry.log) |
| Tensor parallelism | — | Parity and proxy work in progress | — | — | — | — | — | Pending | [TP ledger](tp.md) |
| SM90 dense gate fusion and KDA variants | — | Full-node ablations pending | — | — | — | — | — | Pending | [Kernel inventory](qwen35-kernels.md) |
| Integrated FSDP2 and selected kernels | — | Full-node trial pending | — | — | — | — | — | Pending | Branch `e2efp8-integration` |

The WikiText-103 train fixture is pinned to revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The real-text sanity fixture uses the raw test split. [Dataset](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1). The detailed proxy and operator measurements remain in [fsdp2.md](fsdp2.md), [ppdense.md](ppdense.md), and [kernels.md](kernels.md). Proxy MFU and operator speedups guide which changes to run at 27B.

Historical Muon diagnostics: stock PP8 job `1263388` OOMed on step 2, and job `1263873` was stopped after the AdamW decision. The PP4 1F1B Muon run `1263528` also OOMed on step 2.

The first AdamW launchers exited before Slurm submission because the local gate lost tickets across PID namespaces. All three replacements are detached through the repaired lease-based gate. [Gate handoff](gate-handoff.md).
