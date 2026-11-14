"""
Whole-step all-to-all overlap of one bench.py capture, per rank.

The window is the "workload ... schedule ..." NVTX range bench.py puts around model.step, so the
optimizer and the gradient clip are excluded and the end-of-step FSDP reduce-scatter is included
(as "other comm"). EP all-to-all kernels are the NCCL kernels launched inside a stage2_* (dispatch)
or stage4_* (combine) NVTX range, labeled with the analyze-nsys-profile skill's innermost_nvtx;
compute is every non-NCCL kernel. A NCCL all-to-all kernel's duration includes the time it waits
for its slowest peer, so exposed all-to-all time includes the cost of imbalance across ranks.

usage: python analyze.py <report.sqlite> [--json]
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".agents/skills/analyze-nsys-profile/scripts"))  # fmt: skip

from common import innermost_nvtx, kernels_in_window, stage_of, sweep_overlap, union_duration  # noqa: E402


def windows(con: sqlite3.Connection) -> list[tuple[int, int, int, str]]:
    cur = con.cursor()
    cur.execute(
        """
        SELECT e.globalTid / 0x1000000 % 0x1000000 AS pid, e.start, e.end, COALESCE(e.text, s.value)
        FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
        WHERE e.start >= 0 AND COALESCE(e.text, s.value) LIKE 'workload % schedule %'
        ORDER BY pid
        """
    )
    return cur.fetchall()


def chunk_ranges(con, pid, start, end):
    cur = con.cursor()
    cur.execute(
        """
        SELECT e.start, e.end, COALESCE(e.text, s.value)
        FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
        WHERE e.globalTid / 0x1000000 % 0x1000000 = ? AND e.start >= ? AND e.end <= ?
          AND (COALESCE(e.text, s.value) LIKE 'forward chunk%' OR COALESCE(e.text, s.value) LIKE 'backward chunk%')
        ORDER BY e.start
        """,
        (pid, start, end),
    )
    return cur.fetchall()


def analyze(path: str) -> dict:
    con = sqlite3.connect(path)
    rows = []
    label = None
    for pid, start, end, name in windows(con):
        label = name
        ks = kernels_in_window(con, pid, start, end)
        comm = [k for k in ks if k["name"].startswith("nccl")]
        comp = [(k["start"], k["end"]) for k in ks if not k["name"].startswith("nccl")]
        labels = innermost_nvtx(con, pid, [k["launch_start"] or k["start"] for k in comm])
        a2a, other = [], []
        for k, nv in zip(comm, labels):
            stage = stage_of(nv)
            if stage is not None and stage[:6] in ("stage2", "stage4"):
                a2a.append((k["start"], k["end"]))
            else:
                other.append((k["start"], k["end"]))
        a2a_t = union_duration(a2a)
        hidden = sweep_overlap(a2a, comp)
        busy = union_duration(comp)
        everything = union_duration(comp + [(k["start"], k["end"]) for k in comm])
        # Paired F/B ranges (overlapped schedule) vs single F or B ranges.
        pairs = singles = 0
        for _, _, nm in chunk_ranges(con, pid, start, end):
            if "backward chunk" in nm and nm.startswith("forward chunk"):
                pairs += 1
            else:
                singles += 1
        rows.append(
            dict(
                pid=pid,
                span_ms=(end - start) / 1e6,
                compute_busy_ms=busy / 1e6,
                a2a_ms=a2a_t / 1e6,
                a2a_hidden_ms=hidden / 1e6,
                a2a_exposed_ms=(a2a_t - hidden) / 1e6,
                overlap_pct=100.0 * hidden / a2a_t if a2a_t else None,
                other_comm_exposed_ms=(union_duration(other) - sweep_overlap(other, comp)) / 1e6,
                gpu_idle_ms=((end - start) - everything) / 1e6,
                paired_ranges=pairs,
                single_ranges=singles,
            )
        )
    con.close()
    n = len(rows)

    def agg(key, fn):
        vals = [r[key] for r in rows if r[key] is not None]
        return fn(vals) if vals else None

    summary = dict(
        report=path,
        label=label,
        ranks=n,
        span_ms_max=agg("span_ms", max),
        compute_busy_ms_mean=agg("compute_busy_ms", lambda v: sum(v) / len(v)),
        compute_busy_ms_min=agg("compute_busy_ms", min),
        compute_busy_ms_max=agg("compute_busy_ms", max),
        a2a_ms_mean=agg("a2a_ms", lambda v: sum(v) / len(v)),
        a2a_exposed_ms_mean=agg("a2a_exposed_ms", lambda v: sum(v) / len(v)),
        a2a_exposed_ms_max=agg("a2a_exposed_ms", max),
        overlap_pct_mean=agg("overlap_pct", lambda v: sum(v) / len(v)),
        other_comm_exposed_ms_mean=agg("other_comm_exposed_ms", lambda v: sum(v) / len(v)),
        gpu_idle_ms_mean=agg("gpu_idle_ms", lambda v: sum(v) / len(v)),
        paired_ranges=rows[0]["paired_ranges"] if rows else None,
        single_ranges=rows[0]["single_ranges"] if rows else None,
        per_rank=rows,
    )
    return summary


if __name__ == "__main__":
    s = analyze(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(s))
    else:
        for k, v in s.items():
            if k != "per_rank":
                print(f"{k}: {v}")
        for r in s["per_rank"]:
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
