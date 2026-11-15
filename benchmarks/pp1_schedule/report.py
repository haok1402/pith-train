"""
Build the measured tables of RESULTS.md from bench.py's records (workspace/pp1/*.jsonl, OOM files
included) and analyze_all.sh's nsys analysis (workspace/pp1/analysis.jsonl). Every number printed
comes from those files; a cell with no measurement says why ("OOM", "n/a", "-").

usage: python benchmarks/pp1_schedule/report.py > workspace/pp1/tables.md
"""

import glob
import json
import os
import re
from collections import defaultdict

W = "workspace/pp1"
MODELS = {"dsv2": "deepseek-v2-lite", "qwen3": "qwen3-30b-a3b"}
SCHEDS = ("overlapped", "sequential")
# Probe suites that ran every length (SUITE DONE): a schedule without an OOM fits all of them.
DONE_PROBES = {
    os.path.basename(f)[: -len(".log")]
    for f in glob.glob("logging/pp1/probe-*.log")
    if "SUITE DONE" in open(f, errors="replace").read()
}


def load(pattern):
    recs = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def suite_records(name):
    return load(f"{W}/{name}.jsonl") + load(f"{W}/{name}.oom.rank*.jsonl")


def paired(recs, workload):
    """
    The two schedules' records of a workload measured on one host (the latest such host), or the
    OOM marker per schedule.
    """
    out = {}
    oks = defaultdict(dict)
    for r in recs:
        if r.get("workload") != workload:
            continue
        if r.get("status") == "oom":
            out.setdefault(r["schedule"], "OOM")
        elif r.get("status") == "ok" and r.get("host"):
            oks[r["host"]][r["schedule"]] = r
    hosts = [h for h, d in oks.items() if all(s in d or out.get(s) == "OOM" for s in SCHEDS)]
    if hosts:
        h = hosts[-1]
        for s in SCHEDS:
            if s in oks[h]:
                out[s] = oks[h][s]
    return out


def suite_of_report(path, prefix_to_suite):
    base = os.path.basename(path)
    m = re.match(r"(j\d+\.\d+)-nsys-report", base)
    if m:
        return prefix_to_suite.get(m.group(1))
    m = re.match(r"(main|probe)-(.+?)(-j\d+\.\d+)?-a\d+\.\d+\.nsys-rep", base)
    return m.group(2) if m else None


def nsys_index():
    """
    (suite, workload, schedule) -> the latest analysis row.
    """
    prefix_to_suite = {}
    for f in glob.glob(f"{W}/main-*.jsonl") + glob.glob(f"{W}/main-*.oom.rank*.jsonl"):
        suite = os.path.basename(f).split(".")[0][len("main-") :]
        for r in load(f):
            m = re.search(r"(j\d+\.\d+)", r.get("nsys_prefix") or "")
            if m:
                prefix_to_suite[m.group(1)] = suite
    for log in glob.glob(f"{W}/main-*-j*-a*.log"):
        m = re.match(r"main-(.+)-(j\d+\.\d+)-a\d+\.log", os.path.basename(log))
        if m:
            prefix_to_suite.setdefault(m.group(2), m.group(1))
    idx = {}
    for r in load(f"{W}/analysis.jsonl"):
        label = r.get("label") or ""
        m = re.match(r"workload (\S+) schedule (\S+)", label)
        suite = suite_of_report(r["report"], prefix_to_suite)
        if not m or not suite:
            continue
        idx[(suite, m.group(1), m.group(2))] = r
    return idx


def fmt(x, spec="%.2f"):
    return "-" if x is None else spec % x


def main_table(model_key, ep, nsys):
    # The recapture (-r2) is loaded last, so where it has a complete pair, paired() takes it.
    recs = suite_records(f"main-{model_key}-ep{ep}") + suite_records(f"main-{model_key}-ep{ep}-r2")
    suite = f"{model_key}-ep{ep}"
    workloads = []
    for r in recs:
        if r.get("workload") not in workloads and r.get("workload") != "setup":
            workloads.append(r["workload"])
    rows = []
    header = "| workload | M | max pack | step s, median (min-max of 3) (ovl / seq) | seq vs ovl | tok/s/GPU (ovl / seq) | peak alloc GiB (ovl / seq) | peak reserved GiB (ovl / seq) | exposed a2a ms (ovl / seq) | a2a overlap % (ovl) | compute ms per rank min-max (ovl) | host |"  # fmt: skip
    rows.append(header)
    rows.append("|" + "|".join(["---"] * (header.count("|") - 1)) + "|")
    for w in workloads:
        p = paired(recs, w)
        o, s = p.get("overlapped"), p.get("sequential")
        if o is None and s is None:
            continue

        def cell(r, key, spec="%.2f"):
            if r == "OOM":
                return "OOM"
            if r is None:
                return "-"
            return fmt(r.get(key), spec)

        def steps(r):
            if not isinstance(r, dict):
                return cell(r, "step_time")
            st = r.get("step_times") or [r["step_time"]]
            return f"{r['step_time']:.3f} ({min(st):.2f}-{max(st):.2f})"

        M = next((r.get("M") for r in (o, s) if isinstance(r, dict)), next((x.get("M") for x in recs if x.get("workload") == w), "-"))  # fmt: skip
        mp = next((x.get("max_pack") for x in recs if x.get("workload") == w and x.get("max_pack")), 2048)  # fmt: skip
        delta = "-"
        if isinstance(o, dict) and isinstance(s, dict):
            delta = "%+.1f%%" % (100 * (s["step_time"] / o["step_time"] - 1))
        no = nsys.get((suite + "-r2", w, "overlapped")) or nsys.get((suite, w, "overlapped"))
        ns = nsys.get((suite + "-r2", w, "sequential")) or nsys.get((suite, w, "sequential"))
        exp = f"{fmt(no and no.get('a2a_exposed_ms_mean'), '%.0f')} / {fmt(ns and ns.get('a2a_exposed_ms_mean'), '%.0f')}"  # fmt: skip
        ovp = fmt(no and no.get("overlap_pct_mean"), "%.0f")
        busy = "-" if not no else f"{no['compute_busy_ms_min']:.0f}-{no['compute_busy_ms_max']:.0f}"
        host = next((r.get("host", "")[-3:] for r in (o, s) if isinstance(r, dict)), "-")
        rows.append(
            f"| {w} | {M} | {mp} | {steps(o)} / {steps(s)} | {delta} | "
            f"{cell(o, 'tokens_per_s_per_gpu', '%.0f')} / {cell(s, 'tokens_per_s_per_gpu', '%.0f')} | "
            f"{cell(o, 'peak_alloc_gib', '%.1f')} / {cell(s, 'peak_alloc_gib', '%.1f')} | "
            f"{cell(o, 'peak_reserved_gib', '%.1f')} / {cell(s, 'peak_reserved_gib', '%.1f')} | {exp} | {ovp} | {busy} | {host} |"
        )
    return "\n".join(rows)


def imbalance_table(model_key, ep):
    recs = suite_records(f"main-{model_key}-ep{ep}") + suite_records(f"main-{model_key}-ep{ep}-r2")
    rows = ["| M | schedule | even s | intra s (cost) | full s (cost) |", "|---|---|---|---|---|"]
    for M in (1, 2, 4, 8):
        for sched in SCHEDS:
            vals = {}
            for kind in ("even", "intra", "full"):
                p = paired(recs, f"pack16k-M{M}-{kind}")
                vals[kind] = p.get(sched)
            e = vals["even"]

            def c(v):
                if v == "OOM":
                    return "OOM"
                if not isinstance(v, dict):
                    return "-"
                if isinstance(e, dict):
                    return f"{v['step_time']:.3f} ({v['step_time'] - e['step_time']:+.3f})"
                return f"{v['step_time']:.3f}"

            ev = "OOM" if e == "OOM" else (f"{e['step_time']:.3f}" if isinstance(e, dict) else "-")
            rows.append(f"| {M} | {sched} | {ev} | {c(vals['intra'])} | {c(vals['full'])} |")
    return "\n".join(rows)


def probe_table():
    rows = ["| model | mesh | M | schedule | largest fitting pack (tokens/rank/micro-batch) | its step s | its peak alloc / reserved GiB | first OOM |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for f in sorted(glob.glob(f"{W}/probe-*.jsonl")):
        if ".oom." in f:
            continue
        name = os.path.basename(f)[: -len(".jsonl")]
        recs = suite_records(name)
        m = re.match(r"probe-(\w+)-ep(\d+)-M(\d+)", name)
        if not m:
            continue
        for sched in SCHEDS:
            ok = [r for r in recs if r.get("schedule") == sched and r.get("status") == "ok"]
            oom = [r for r in recs if r.get("schedule") == sched and r.get("status") == "oom"]
            best = max(ok, key=lambda r: r["probe_len"]) if ok else None
            first_oom = min((r["probe_len"] for r in oom), default=None)
            done = first_oom is not None or name in DONE_PROBES
            fit = best["probe_len"] if best else "none fits"
            if not done:
                fit = f">= {best['probe_len']} (probe unfinished)" if best else "pending"
            rows.append(
                f"| {MODELS[m.group(1)]} | {'EP8' if m.group(2) == '8' else 'DP8 (EP1)'} | {m.group(3)} | {sched} | "
                f"{fit} | {fmt(best and best['step_time'], '%.3f')} | "
                f"{fmt(best and best['peak_alloc_gib'], '%.1f')} / {fmt(best and best['peak_reserved_gib'], '%.1f')} | {first_oom or ('-' if done else 'pending')} |"
            )
    return "\n".join(rows)


def nsys_table(nsys):
    rows = ["| suite | workload | schedule | report | a2a ms | exposed a2a ms | overlap % | GPU idle ms |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for (suite, w, s), r in sorted(nsys.items()):
        rows.append(
            f"| {suite} | {w} | {s} | `{r['report']}` | {fmt(r.get('a2a_ms_mean'), '%.0f')} | "
            f"{fmt(r.get('a2a_exposed_ms_mean'), '%.0f')} | {fmt(r.get('overlap_pct_mean'), '%.0f')} | {fmt(r.get('gpu_idle_ms_mean'), '%.0f')} |"
        )
    return "\n".join(rows)


def matched_probe_table(name):
    """
    Both schedules at the same pack lengths of one probe suite, so speed compares like for like.
    """
    recs = suite_records(name)
    lens = sorted({r["probe_len"] for r in recs if r.get("probe_len")})
    rows = ["| pack tokens | overlapped s (3 steps) | sequential s (3 steps) | peak alloc / reserved GiB (ovl) | (seq) |", "|---|---|---|---|---|"]  # fmt: skip
    for L in lens:
        cells = []
        mem = []
        for sched in SCHEDS:
            rr = [r for r in recs if r.get("probe_len") == L and r.get("schedule") == sched]
            ok = [r for r in rr if r["status"] == "ok"]
            if ok:
                r = ok[-1]
                cells.append(
                    f"{r['step_time']:.3f} ({', '.join('%.2f' % x for x in r['step_times'])})"
                )
                mem.append(f"{r['peak_alloc_gib']:.1f} / {r['peak_reserved_gib']:.1f}")
            elif rr:
                cells.append("OOM")
                mem.append("-")
            else:
                cells.append("not run (shorter length OOMed)" if any(x.get("schedule") == sched and x["status"] == "oom" for x in recs) else "pending")  # fmt: skip
                mem.append("-")
        rows.append(f"| {L} | {cells[0]} | {cells[1]} | {mem[0]} | {mem[1]} |")
    return "\n".join(rows)


def dp_vs_ep_table(model_key):
    ep = suite_records(f"main-{model_key}-ep8") + suite_records(f"main-{model_key}-ep8-r2")
    dp = suite_records(f"main-{model_key}-ep1")
    names = []
    for r in ep + dp:
        if r.get("workload") not in names and r.get("workload") != "setup":
            names.append(r["workload"])
    rows = ["| workload | EP8 overlapped s | EP8 sequential s | DP8 overlapped s | DP8 sequential s | best DP8 vs best EP8 | peak alloc GiB EP8 / DP8 (overlapped) |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for w in names:
        pe, pd = paired(ep, w), paired(dp, w)

        def v(p, sched):
            x = p.get(sched)
            return "OOM" if x == "OOM" else ("-" if x is None else f"{x['step_time']:.3f}")

        def best(p):
            xs = [p[k]["step_time"] for k in SCHEDS if isinstance(p.get(k), dict)]
            return min(xs) if xs else None

        be, bd = best(pe), best(pd)
        cmp = "-" if be is None or bd is None else "%+.1f%%" % (100 * (bd / be - 1))
        me = pe.get("overlapped")
        md = pd.get("overlapped")
        mem = f"{me['peak_alloc_gib']:.1f}" if isinstance(me, dict) else str(me or "-")
        mem += " / " + (f"{md['peak_alloc_gib']:.1f}" if isinstance(md, dict) else str(md or "-"))
        rows.append(f"| {w} | {v(pe, 'overlapped')} | {v(pe, 'sequential')} | {v(pd, 'overlapped')} | {v(pd, 'sequential')} | {cmp} | {mem} |")  # fmt: skip
    return "\n".join(rows)


if __name__ == "__main__":
    nsys = nsys_index()
    for mk in ("dsv2", "qwen3"):
        for ep in (8, 1):
            print(f"\n### {MODELS[mk]} {'EP8' if ep == 8 else 'DP8 (EP1)'}\n")
            print(main_table(mk, ep, nsys))
            print(f"\nImbalance cost, {MODELS[mk]} {'EP8' if ep == 8 else 'DP8'}: step time of uneven packs minus even packs, same tokens and M.\n")  # fmt: skip
            print(imbalance_table(mk, ep))
    for mk in ("dsv2", "qwen3"):
        print(f"\n### DP8 (EP1) vs EP8, {MODELS[mk]}\n")
        print(dp_vs_ep_table(mk))
    print("\n### deepseek-v2-lite EP8, 2 packs per step, both schedules at the same lengths\n")
    print(matched_probe_table("probe-dsv2-ep8-M2"))
    print("\n### Largest pack before OOM\n")
    print(probe_table())
    print("\n### nsys reports\n")
    print(nsys_table(nsys))
