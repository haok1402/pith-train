"""Measure exposed NCCL time in one profiled dense FSDP2 training step."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sqlite3
from statistics import median
import sys

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "pith-train/.agents/skills/analyze-nsys-profile/scripts"
    ),
)
from common import sweep_overlap, union_duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Nsight Systems SQLite export")
    args = parser.parse_args()

    with sqlite3.connect(f"{args.trace.resolve().as_uri()}?mode=ro", uri=True) as connection:
        windows = connection.execute("""
            SELECT e.globalTid / 0x1000000 % 0x1000000, e.start, e.end,
                   COALESCE(e.text, strings.value)
            FROM NVTX_EVENTS e
            LEFT JOIN StringIds strings ON strings.id = e.textId
            WHERE e.start >= 0 AND e.end > e.start
              AND COALESCE(e.text, strings.value) LIKE 'rank=%'
        """).fetchall()
        if not windows:
            parser.error("no per-rank training-step NVTX ranges found")

        timings = []
        print("rank  window_ms  compute_ms  nccl_ms  exposed_nccl_ms  all_gather_ms  reduce_scatter_ms")
        for pid, start, end, label in sorted(windows, key=lambda row: int(re.match(r"rank=(\d+)", row[3])[1])):
            rows = connection.execute("""
                SELECT MAX(k.start, ?), MIN(k.end, ?), s.value
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.shortName = s.id
                WHERE k.start >= 0 AND k.start < ? AND k.end > ?
                  AND k.globalPid / 0x1000000 % 0x1000000 = ?
            """, (start, end, end, start, pid)).fetchall()
            compute, communication = [], []
            collective = defaultdict(list)
            for first, last, name in rows:
                if name.lower().startswith("nccl"):
                    communication.append((first, last))
                    for operation in ("allgather", "reducescatter"):
                        if operation in name.lower().replace("_", ""):
                            collective[operation].append((first, last))
                else:
                    compute.append((first, last))
            nccl = union_duration(communication)
            exposed = nccl - sweep_overlap(communication, compute)
            row = [
                end - start,
                union_duration(compute),
                nccl,
                exposed,
                union_duration(collective["allgather"]),
                union_duration(collective["reducescatter"]),
            ]
            timings.append(row)
            rank = re.match(r"rank=(\d+)", label)[1]
            print(f"{rank:>4}  " + "  ".join(f"{value / 1e6:9.2f}" for value in row))
    print(" median  " + "  ".join(f"{median(column) / 1e6:9.2f}" for column in zip(*timings)))


if __name__ == "__main__":
    main()
