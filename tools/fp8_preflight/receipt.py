"""Summarize startup utilization, phase timings and dense recipe loss fingerprints."""

import argparse
import csv
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tools.fp8_preflight.artifacts import write_json

EVENT = re.compile(r"STARTUP rank=(\d+) time=(\S+) phase=(\S+) state=(\S+)(?: (\w+)=([^\s]+))?")
FINGERPRINT = re.compile(r"audit step=(\d+) rank=(\d+) .*?outcome=ok loss_bits=([0-9a-f,]+)")


def fingerprints(log):
    return {(int(step), int(rank)): bits for step, rank, bits in FINGERPRINT.findall(log)}


def summarize(run, baseline):
    log = (run / "run.log").read_text()
    durations = defaultdict(lambda: defaultdict(list))
    first_step = {}
    ranks = {}
    intervals = defaultdict(list)
    for rank, when, phase, state, key, value in EVENT.findall(log):
        timestamp = datetime.fromisoformat(when)
        if phase == "gpu" and key == "uuid":
            ranks[rank] = "GPU-" + value.removeprefix("GPU-")
        if state == "end" and key == "seconds":
            durations[phase][rank].append(float(value))
        if phase == "train_step" and state == "end":
            first_step.setdefault(rank, timestamp)
    with (run / "gpu.csv").open(newline="") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) != 6:
                continue
            when, _, uuid, util, _, _ = (field.strip() for field in row)
            try:
                timestamp = datetime.fromisoformat(when.replace("/", "-")).replace(tzinfo=UTC)
                intervals[uuid].append((timestamp, int(util)))
            except ValueError:
                continue

    low = {}
    for rank, cutoff in first_step.items():
        samples = intervals[ranks[rank]]
        if not samples or samples[-1][0] < cutoff:
            raise ValueError(f"GPU timeline is incomplete for rank {rank}")
        elapsed = 0
        for (begin, util), (end, _) in zip(samples, samples[1:]):
            if begin >= cutoff:
                break
            if util < 20:
                elapsed += (min(cutoff, end) - begin).total_seconds()
        low[rank] = round(elapsed, 3)

    losses = re.findall(r"cross-entropy-loss ([\d.]+)", log)
    actual = fingerprints(log)
    result = {
        "job": (run / "slurm-job").read_text().strip(),
        "losses": losses,
        "first_step_end_by_rank": {rank: value.isoformat() for rank, value in first_step.items()},
        "gpu_uuid_by_rank": ranks,
        "seconds_below_20_percent_through_step_one_by_rank": low,
        "low_gpu_minutes": round(sum(low.values()) / 60, 3),
        "phase_seconds": durations,
        "fingerprints": len(actual),
    }
    if baseline is not None:
        expected = fingerprints(baseline.read_text())
        result["fingerprints_equal_to_baseline"] = bool(expected) and expected == actual
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    result = summarize(args.run, args.baseline)
    write_json(args.run / "startup-receipt.json", result)
    if len(result["seconds_below_20_percent_through_step_one_by_rank"]) != 8:
        raise SystemExit("Incomplete eight-rank startup trace")
    if args.baseline and not result["fingerprints_equal_to_baseline"]:
        raise SystemExit("Recipe loss fingerprints differ from the baseline")


if __name__ == "__main__":
    main()
