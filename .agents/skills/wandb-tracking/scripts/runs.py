"""
List, compare, and pull files for runs in a wandb project.

    python scripts/runs.py list    <entity/project>
    python scripts/runs.py compare <entity/project> --metric KEY --ref RUN --runs RUN[,RUN...] [--drop-first]
    python scripts/runs.py logs    <entity/project> --runs RUN[,RUN...] [--file NAME]

RUN is a run id or display name.

compare reports, per run, the per-step delta range, the final delta and what it
settles into, and whether that final delta sits inside the run's own noise.
--drop-first drops step 0 (a warmup outlier for throughput; do not use it for loss).

logs downloads output.log (a run's stdout+stderr, crash tracebacks included) to
workspace/wandb-logs/<run-name>/. --file picks another run file (config.yaml,
wandb-metadata.json, wandb-summary.json, requirements.txt).
"""

import argparse
import os
import statistics

import wandb


def run_maps(api, path):
    runs = list(api.runs(path))
    return {r.id: r.name for r in runs}, {r.name: r.id for r in runs}


def to_id(id2name, name2id, sel, path):
    if sel in id2name:
        return sel
    if sel in name2id:
        return name2id[sel]
    raise SystemExit(f"run not found in {path}: {sel!r}")


def series(run, key):
    """
    Per-step {step: value} for a metric, via scan_history (returns every step).
    """
    d = {}
    for row in run.scan_history(keys=["_step", key]):
        v = row.get(key)
        if v is not None:
            d[int(row["_step"])] = v
    return d


def noise_floor(vals, tail=20):
    """
    The run's own noise floor: mean step-to-step change over the last tail
    steps, restricted to the tail so the steep early-training descent does not
    inflate it.
    """
    v = vals[-tail:] if len(vals) > tail else vals
    if len(v) < 2:
        return 0.0
    return statistics.mean(abs(v[i] - v[i - 1]) for i in range(1, len(v)))


def fmt(x):
    return f"{x:+,.5f}" if abs(x) < 1000 else f"{x:+,.0f}"


def cmd_list(api, path):
    runs = list(api.runs(path))
    print(f"{path}: {len(runs)} runs\n")
    print(f"{'id':<12} {'state':<10} {'steps':>6}  name")
    keys = set()
    for r in sorted(runs, key=lambda x: x.name):
        print(f"{r.id:<12} {r.state:<10} {str(r.summary.get('_step')):>6}  {r.name}")
        keys |= {k for k in r.summary.keys() if not k.startswith("_")}
    print("\nlogged metric keys:")
    for k in sorted(keys):
        print(f"  {k}")


def cmd_compare(api, path, args):
    id2name, name2id = run_maps(api, path)
    ref_id = to_id(id2name, name2id, args.ref, path)
    ref = series(api.run(f"{path}/{ref_id}"), args.metric)
    vars_ = {
        sel: series(api.run(f"{path}/{to_id(id2name, name2id, sel, path)}"), args.metric)
        for sel in args.runs.split(",")
    }

    steps = sorted(set(ref).intersection(*[set(v) for v in vars_.values()]))
    if args.drop_first and steps:
        steps = steps[1:]
    if not steps:
        raise SystemExit("no common steps with this metric")
    fs = steps[-1]

    print(
        f"metric: {args.metric}   ref: {id2name[ref_id]}   steps: {len(steps)} "
        f"[{steps[0]}-{fs}]{'  (dropped step 0)' if args.drop_first else ''}\n"
    )

    ref_noise = noise_floor([ref[s] for s in steps])
    for sel, var in vars_.items():
        deltas = [var[s] - ref[s] for s in steps]
        final = deltas[-1]
        settling = statistics.mean(deltas[-min(5, len(deltas)) :])
        verdict = "INSIDE" if abs(final) < ref_noise else "ABOVE"
        print(f"=== {sel} vs {id2name[ref_id]} ===")
        print(
            f"  per-step delta : range [{fmt(min(deltas))}, {fmt(max(deltas))}]   "
            f"final(step {fs}) {fmt(final)}   settling {fmt(settling)}"
        )
        print(
            f"  vs noise       : steady-state jitter {ref_noise:,.5f}  ->  "
            f"final delta is {verdict} the noise\n"
        )


def cmd_logs(api, path, args):
    id2name, name2id = run_maps(api, path)
    for sel in args.runs.split(","):
        r = api.run(f"{path}/{to_id(id2name, name2id, sel, path)}")
        dest = f"workspace/wandb-logs/{r.name}"
        os.makedirs(dest, exist_ok=True)
        f = r.file(args.file).download(root=dest, replace=True)
        n = sum(1 for _ in open(f.name))
        print(f"{r.name}: {f.name}  ({os.path.getsize(f.name):,} bytes, {n} lines)")


def main():
    ap = argparse.ArgumentParser(usage=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").add_argument("path")

    sp = sub.add_parser("compare")
    sp.add_argument("path")
    sp.add_argument("--metric", required=True)
    sp.add_argument("--ref", required=True)
    sp.add_argument("--runs", required=True, help="comma-separated ids or names")
    sp.add_argument("--drop-first", action="store_true")

    sp = sub.add_parser("logs")
    sp.add_argument("path")
    sp.add_argument("--runs", required=True, help="comma-separated ids or names")
    sp.add_argument("--file", default="output.log", help="which wandb file to download")

    args = ap.parse_args()
    api = wandb.Api()
    if args.cmd == "list":
        cmd_list(api, args.path)
    elif args.cmd == "compare":
        cmd_compare(api, args.path, args)
    elif args.cmd == "logs":
        cmd_logs(api, args.path, args)


if __name__ == "__main__":
    main()
