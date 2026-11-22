"""FIFO submission gate for the shared NRT1 interactive QoS."""

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

CAP = 8
POLL_SECONDS = 15
HOST = socket.gethostname()
ACCEPTED = re.compile(
    r"^(?:srun(?:\.real)?: (?:verbose: )?(?:job \d+ (?:queued|has been allocated)|"
    r"jobid=\d+ returned by the controller|granted job allocation \d+)|"
    r"gpu-run: Slurm job \d+ started)",
    re.IGNORECASE,
)
CAP_REASON = "QOSMaxSubmitJobPerUserLimit"
GENERIC_DENIAL = "Unable to allocate resources: Job violates accounting/QOS policy"


def process_start(pid):
    try:
        # Field 22 follows the command name, which may contain spaces and parentheses.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (FileNotFoundError, ProcessLookupError):
        return None


@contextlib.contextmanager
def locked_state(gate):
    with (gate / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = gate / "queue.json"
        before = path.read_text() if path.exists() else "{}"
        state = {"next_ticket": 1, "line": [], "next_poll": 0, "free": 0} | json.loads(before)
        yield state
        after = json.dumps(state)
        if after != before:
            temporary = gate / f".queue-{os.getpid()}.tmp"
            temporary.write_text(after)
            temporary.replace(path)


def join(gate):
    with locked_state(gate) as state:
        ticket = state["next_ticket"]
        state["next_ticket"] += 1
        state["line"].append([ticket, HOST, os.getpid(), process_start(os.getpid())])
    return ticket


def leave(gate, ticket):
    with locked_state(gate) as state:
        state["line"] = [entry for entry in state["line"] if entry[0] != ticket]


def check_capacity():
    result = subprocess.run(
        ["squeue", "--noheader", "--array", "--user", str(os.getuid()), "--qos", "interactive",
         "--states", "PENDING,RUNNING", "--format", "%i"],
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        return None
    return max(0, CAP - len(result.stdout.split()))


def wait_for_slot(gate, ticket, unconfirmed_status):
    last_position = None
    while True:
        poll = False
        with locked_state(gate) as state:
            state["line"] = [
                entry for entry in state["line"]
                if entry[1] != HOST or process_start(entry[2]) == entry[3]
            ]
            position = 1 + next(i for i, entry in enumerate(state["line"]) if entry[0] == ticket)
            if position == 1:
                now = time.time()
                if now >= state["next_poll"]:
                    # Persist the probe time before squeue, including if this waiter is interrupted.
                    state["next_poll"] = now + POLL_SECONDS
                    state["free"] = 0
                    poll = True
                elif state["free"] > 0:
                    state["free"] -= 1
                    return 0
        if poll:
            free = check_capacity()
            with locked_state(gate) as state:
                state["free"] = free or 0
                if free and unconfirmed_status is None:
                    state["free"] -= 1
                    return 0
            if free is not None and unconfirmed_status is not None:
                if free > 0:
                    return unconfirmed_status
                unconfirmed_status = None
            continue
        if position != last_position:
            print(f"gpu-run: waiting for interactive, queue position {position}", file=sys.stderr)
            last_position = position
        time.sleep(1)


def submit(gate, ticket, gpus, wall, command):
    argv = [
        "srun", "-v", "-A", "nvr_elm_llm", "-p", "batch", "-q", "interactive",
        "-J", "deltamatch-e2efp8", "-N1", "-n1", f"--gpus={gpus}",
        f"--cpus-per-task={28 * gpus}", "--constraint=nvidia_h200", "--cpu-bind=none",
        f"--time={wall}", "-W", "0",
        "bash", "-c",
        r'printf "gpu-run: Slurm job %s started on %s\n" "$SLURM_JOB_ID" "$SLURMD_NODENAME" >&2; exec "$@"',
        "gpu-run", *command,
    ]
    proc = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1)
    accepted = cap_rejected = generic_denial = False
    try:
        with proc.stderr:
            for line in proc.stderr:
                print(line, end="", file=sys.stderr, flush=True)
                if not accepted and ACCEPTED.match(line):
                    accepted = True
                    leave(gate, ticket)
                cap_rejected |= CAP_REASON in line
                generic_denial |= GENERIC_DENIAL in line
        status = proc.wait()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        proc.wait()
        raise
    return status if status >= 0 else 128 - status, accepted, cap_rejected, generic_denial


def main():
    parser = argparse.ArgumentParser(usage="gpu-run.sh --gpus N --time T -- command [args...]")
    parser.add_argument("--gpus", required=True, type=int, choices=range(1, 9))
    parser.add_argument("--time", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if len(args.command) < 2 or args.command[0] != "--":
        parser.error("provide a command after --")

    gate = Path(os.environ["E2EFP8_ROOT"]) / "locks" / "gpu-run"
    gate.mkdir(parents=True, exist_ok=True)
    ticket = join(gate)
    unconfirmed_status = None
    try:
        while True:
            status = wait_for_slot(gate, ticket, unconfirmed_status)
            if status:
                return status
            status, accepted, cap_rejected, generic_denial = submit(
                gate, ticket, args.gpus, args.time, args.command[1:]
            )
            if not status or accepted or not (cap_rejected or generic_denial):
                return status
            with locked_state(gate) as state:
                state["free"] = 0
                state["next_poll"] = time.time() + POLL_SECONDS
            unconfirmed_status = None if cap_rejected else status
    finally:
        leave(gate, ticket)


def stop(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
