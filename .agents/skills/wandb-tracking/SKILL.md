---
name: wandb-tracking
description: Read, analyze, and manage Weights & Biases (wandb) experiment data for PithTrain runs. Use when the user pastes a wandb.ai URL, or asks to list runs, compare runs or loss curves (per-step and final delta), check whether a variant matches a baseline, read training throughput (tokens-per-second), pull a run's console log after a crash (output.log, stdout+stderr with tracebacks), or build and manage a saved view of which runs to show.
---

# wandb-tracking

Work with wandb experiment data for PithTrain via the scripts below: runs, metrics, console output, and saved views.

The scripts below are the primitives; compose them for the question asked. **Do not produce an unsolicited full report.** Answer the specific question.

## Prerequisites

- wandb must be authenticated: `wandb.Api()` reads credentials from `~/.netrc`.
- Scripts live in `scripts/` beside this file.

## Primitives

| Question | Command |
|---|---|
| What runs / metric keys exist? | `runs.py list <entity/project>` |
| Per-step + final delta of a metric across runs | `runs.py compare <entity/project> --metric KEY --ref RUN --runs RUN[,RUN...]` |
| Download console output (`output.log`, stdout+stderr) for runs | `runs.py logs <entity/project> --runs RUN[,RUN...]` |
| List views / what each view shows | `views.py list <entity/project>` |
| What does one `?nw=` view show? | `views.py show <entity/project> <nw_slug>` |
| Create / update a view | `views.py set <entity/project> --show RUN[,RUN...] --name NAME [--update NW]` |
| Delete a view | `views.py delete <entity/project> <nw_slug>` |

`RUN` is a run **id** or **display name** everywhere. `entity/project` example: `pithtrain/pr74`.

## Runs and metrics

Run `runs.py list` first. It prints `id | state | steps | name` and the **union of logged metric keys** to feed `runs.py compare --metric`. PithTrain logs `train/cross-entropy-loss`, `train/gradient-norm`, `train/load-balance-loss`, `train/learning-rate`, `train/step`, `infra/step-time`, `infra/tokens-per-second`, `infra/peak-gpu-memory`.

### Delta reporting (comparing loss curves)

`runs.py compare` is the canonical report: it matters for loss curves because a raw final number hides the trajectory. Each `--runs` entry is compared against `--ref` (`delta = run - ref`); for a pairwise A-vs-B use `--ref B --runs A`. It reports, per run:

- **per-step delta range** across all steps (the extreme is usually an early transient).
- **final delta** at the last step, plus **settling** (mean of the last 5): where it's headed.

These are facts, not a verdict. Report them to the user as prose plus a small table, and always state the step count / horizon: a 64-step run only rules out *large* effects, so say so, and judge the deltas against that horizon yourself.

## Throughput

Throughput is the logged metric `infra/tokens-per-second`. Pass **`--drop-first`**: step 0 is a warmup outlier (first-step compile/alloc). Steady-state fp8-vs-bf16 and pow2-vs-full-mantissa (the two fp8 scale formats) deltas then fall out of `runs.py compare`. (At small scale, expect fp8 *slower* than bf16, since quant overhead outweighs the GEMM matrix-multiply win.)

## Console output (stdout + stderr)

`runs.py logs` downloads `output.log` per run: the run's console output (**both stdout and stderr**), captured and synced upstream. A crashed run's traceback lands here too, so this is the first place to look when a run died. `--file` selects a different run file: `config.yaml`, `wandb-metadata.json` (host/git/command/GPU), `wandb-summary.json`, `requirements.txt`. **Offset gotcha:** `output.log` is 1-indexed and starts at step 2; the wandb scalar `_step` is 0-indexed (0..N-1). Cross-reference by content, not by line number.

## Views (`?nw=<slug>`)

A view (the `?nw=<slug>` saved workspace) narrows which of a project's runs are shown. Driving `views.py`:

- From a pasted `?nw=<slug>` URL, pass the `<slug>` to `show` (or `list` to see every view's `shown` / `hidden` runs).
- `set --show RUN[,...] --name NAME` creates a view of those runs and prints its `?nw=` URL. `--from-view NW` starts from an existing view's layout/panels; `--update NW` edits a view in place (**same URL**), otherwise a new slug is minted.
- Creating/updating a view is an outward-facing write to the user's project, so confirm the target runs + name first. Reversible with `delete`.

See [references/views.md](references/views.md) if you need to modify `views.py` itself.
