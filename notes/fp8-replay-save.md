# Qwen3.5-27B FP8 replay saves

**MLP projection saving is unsafe pending diagnosis.** Step-5 loss bits diverged on two nodes despite exact first-step gradients. The branch exposes this configuration under an explicitly unsafe flag for diagnosis.

At the BF16-matched settings, the original FP8 stack ran about **1.25×** the BF16 throughput. Saving the projection outputs of 52 mixers and 24 MLPs brought it to **1.35×** the 23,017 tok/s/node BF16 reference. The 24-MLP run used 132.25 GiB of reserved memory per H200.

## Matched step profile

Both recipes used eight H200s, sequence length 4,096, global batch 64, microbatch 2, 12 layers with MLP-only checkpointing, and the same checkpoint and batch order. The step-5 Nsight pair ran on `pool1-00886` in Slurm `1324107`. BF16 used PR #152 at `0aeb575`. FP8 used stack `3208623` with wgrad pin `85422f8`, Delta-Matching attention, and the FP8 shard cache disabled.

BF16 measured 22,899 tok/s/node at 11.448 s/step. FP8 measured 28,509 tok/s/node at 9.195 s/step. The table assigns disjoint GPU wall-clock time and averages it across the eight ranks. Exposed FSDP includes packing and collectives outside compute overlap.

| Category | BF16 s | FP8 s | BF16 ÷ FP8 |
| --- | ---: | ---: | ---: |
| Forward GEMMs | 2.650 | 1.513 | 1.752× |
| Dgrad GEMMs | 2.762 | 1.540 | 1.793× |
| Wgrad GEMMs | 2.770 | 1.604 | 1.726× |
| Replayed GEMMs | 0.678 | 2.040 | 0.332× |
| Other checkpoint replay | 0.220 | 0.255 | 0.862× |
| Gated DeltaNet | 0.593 | 0.588 | 1.009× |
| Attention | 0.355 | 0.212 | 1.671× |
| Quantize and cast | 0.001 | 0.514 | 0.002× |
| Norms and elementwise | 0.598 | 0.399 | 1.499× |
| Loss and optimizer | 0.037 | 0.040 | 0.933× |
| Other memory | 0.021 | 0.005 | 4.378× |
| Exposed FSDP | 0.687 | 0.394 | 1.744× |
| Idle | 0.074 | 0.088 | 0.845× |
| Total | 11.446 | 9.192 | 1.245× |

FP8 sped up the original forward, dgrad, and wgrad GEMMs by 1.76× and saved 3.52 s. The extra replayed GEMMs took back 1.36 s, and quantizing and casting added 0.51 s. Gated DeltaNet took about 0.59 s in both recipes. BF16's selective checkpoint policy saved the MLP `aten.mm` and `aten.addmm` outputs. FP8 used custom ops and replayed 768 extra outer-checkpoint MLP GEMMs per GPU, comprising 512 gate and up projections plus 256 down projections. The FP8 backward also ran 512 inner-checkpoint mixer projection GEMMs per GPU.

## Guarded save sequence

The outer selective checkpoint saves `pithtrain::fp8_linear_fwd` and `pithtrain::fp8_linear_fwd_prequantized` only while running the selected mixer or MLP projections. A scheduler pass puts the compiled mixer projection outputs in Inductor's `never_reuse_but_free_buffers` set. The outer checkpoint retains detached aliases, so an in-place storage reuse can overwrite a saved value without advancing its version counter. The MLP projections in this arm run eagerly.

The mixer ladder used protected builds throughout on `pool1-00203` in Slurm `1326838`. Saving 16, 32, and all 52 checkpointed mixers gave 28,734, 29,067, and 29,471 tok/s/node. The control gave 28,448 tok/s/node. Saving all 52 cut the median step by 0.32 s, from 9.215 to 8.895 s, with no retries. Peak memory rose from 83.79/99.81 to 101.06/118.36 GiB allocated/reserved per GPU.

The next paired test added MLP saves to the 52-mixer control. Both orders ran on `pool1-00828` in Slurm `1328906`, pool ticket `7328`, using protected source based on `6741832` and wgrad pin `c623c6b`. Each successful arm ran eight steps and the timings use steps 3–8. The allocator used its default settings.

| Arm | Tok/s/node | Tok/s/GPU | s/step | Peak allocated/reserved GiB | Retries | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 mixers, opening control | 29,733 | 3,717 | 8.817 | 101.06 / 118.36 | 0 | 1.6991 |
| 52 mixers + 64 MLPs | — | — | — | 130.33 / 136.47 | 40 across ranks | OOM at step 2 |
| 52 mixers + 24 MLPs, first order | 31,001 | 3,875 | 8.456 | 115.68 / 132.25 | 0 | 1.6993 |
| 52 mixers + 24 MLPs, reverse order | 30,952 | 3,869 | 8.470 | 115.68 / 132.25 | 0 | 1.6991 |
| 52 mixers, closing control | 29,644 | 3,706 | 8.843 | 101.06 / 118.36 | 0 | 1.6991 |

The 24 MLPs saved about 0.37 s/step, or 4.3% in node throughput, with another 14.62 GiB allocated and 13.90 GiB reserved per GPU. This is a separate matched-settings experiment from the tuned MB8 stack.

## Correctness and open checks

- In a five-layer, full-width fixture loaded from the 27B checkpoint, the guarded save matched the guarded no-recompute reference at all 70 parameters. Relative gradient L2 was zero and loss bits matched. The full 64-layer MB2 no-recompute reference ran out of memory, including with expandable segments.
- The guard can change Inductor's fused RMSNorm reduction order. In a four-layer fixture it changed gradient relative L2 by 0.207% versus the unguarded build. Direct kernel replay localized the rounding difference to a changed FP32 reduction tile, with the same BF16 casts. The no-recompute comparison therefore uses the same guarded build.
- The 16-, 32-, and 52-mixer ladder arms matched the opening control's 6,808 first-step FP32 gradient shards and 64 rank-step loss records. The closing control differed first on rank 3's first forward microbatch despite the same batch hash. Two UUID-tagged repeats on `pool1-00665` in ticket `7321` matched the opening control bit for bit. The original `pool1-00203` job did not record UUIDs, so the cause remains unattributed.
- The two 24-MLP timing arms matched all 6,808 first-step gradient shards. The first arm's forward loss bits diverged at step 5 across all ranks, while the reverse order matched both controls through step 8. In a later run on `pool1-00298` (ticket `7351`), two 24-MLP repeats matched the control at every batch, loss, and pre-clip gradient fingerprint through step 5 and at all post-step-4 weight fingerprints. The first timing arm on `pool1-00828` did not log GPU UUIDs. The repeat on that host was canceled in favor of the six-arm bisect below.
- The opt-in branch at `3c9533c` with wgrad pin `bcab26e` completed five steps with zero retries on `pool1-00665` (Slurm `1329272`, ticket `7358`). It matched all 6,808 first-step gradient shards and all loss bits through step 4. At step 5 it diverged on all eight ranks, with loss 1.7488 versus 1.7485 and a fingerprint different from the divergence on `pool1-00828`. The drift has occurred on two nodes and remains unattributed.

### Eight-step same-node bisect

Slurm `1330580` (ticket `7368`) ran the protected control, mixer-only save, and mixer-plus-24-MLP save twice each on `pool1-00614`. All six arms used the same checkpoint, batches, compiler guard, and `c623c6b` wgrad bytes. Every step fingerprinted the batches, forward loss, 6,808 pre-clip FP32 gradient shards, and 6,808 updated weight shards. All six ran eight steps with zero allocator retries. The two controls and two mixer-only arms matched each other across all eight steps.

The MLP-save repeats first differed in pre-clip gradients at steps 2 and 3 respectively, with identical forward loss bits at that step. The first differing parameter in model order was rank 0's `embed_tokens.weight` gradient shard, shape `[31040, 5120]` and stride `(5120, 1)`. The first and second repeats differed in 6,719 and 6,727 gradient shards at onset. The updated weights differed in the same step.

On rank 0, all three layer-63 MLP weight gradients matched at onset in both repeats. Layer 63's attention Q and K projection gradients differed in the first repeat, and its Q and K norm gradients differed in the second. The highest-indexed changed parameter in model order was `layers.63.input_layernorm.weight` in both repeats. The first producing operation and whether any saved activation storage changed between save and replay remain open.

### Saved MLP operand check

The final diagnostic paired the mixer-only and MLP-save recipes on `pool1-00521` in Slurm `1332257` (ticket `7418`). It fingerprinted all 12 saved SwiGLU operands in layers 40–63 at save and backward unpack, including BF16 gate and up pre-activations, FP8 weights and activations, and their scales. All 18,432 operand pairs per arm matched across steps 2 and 3. All 4,608 selective-checkpoint output cache entries in the MLP-save arm also matched at store and retrieval. The two arms matched at MLP dY and dX, and at every model gradient and weight through step 3. The drift did not reproduce during the operand check, so the cause remains open and MLP saving remains unsafe.

## Run the experiment

Use the benchmark script inside an eight-H200 allocation, with the tokenized corpus and checkpoint supplied as `DATA` and `MODEL`.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    benchmarks/end2end/qwen3.5-27b-seq4096-bf16/h200-1n8g/fsdp2/script.py \
    --model "$MODEL" --weights "$MODEL" --dataset "$DATA" \
    --seq 4096 --batch 64 --steps 8 --micro-batch 2 --checkpoint-mlp-tail 12 \
    --fp8 --fp8-attention --fp8-shard-cache-mib 0 --unsafe-fp8-replay-save
```
