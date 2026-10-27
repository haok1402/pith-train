"""
Launch-config tuning that keeps every rank on the same kernels.

A kernel family that can run with several launch configs is a Tunable. Each launch asks
tunable.pick(key, rows) for its config, key being what the model fixes (kind, dimensions, layout)
and rows its row count per group: once the key is tuned, one table lookup. Until then pick() returns
the family's default, the same on every rank, and remembers the key.

tune_kernels() is called on every rank before the first training step and after it. The first call
installs the choices cached for this setup (a file every node of the cluster sees, named by the GPU
and a fingerprint of the library versions, the candidates and the kernel source), so a repeated
setup runs tuned from its first step. The second pools the keys every rank used that the cache
lacked, splits the timing of their candidates across the ranks on synthetic inputs (dropping any
whose results differ from the default's), pools the timings, installs the same choice on every rank
and adds it to the cache. Nothing is timed after that. Every exchange happens inside tune_kernels(),
which all ranks reach; none happens while a launch picks its config.
"""

import hashlib
import inspect
import json
import os
from importlib import metadata
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

__all__ = ["Tunable", "tune_kernels"]

MIN_BUCKET, MAX_BUCKET = 3, 17
"""
Row buckets: rows per group in [2^b, 2^(b+1)), from 2^3 = 8 to 2^17 = 131072 rows.
"""

BUCKET_SPREAD = 2
"""
Buckets tuned on each side of every bucket the first step used, so later steps whose row counts move
(routing, sequence length) still land on a tuned entry.
"""

MARGIN = 0.03
"""
A candidate replaces the default only if it is this much faster, so near-ties keep the default.
"""

WARMUP, ITERS = 3, 10

DEFAULT_CACHE_DIR = Path.home() / ".cache/pithtrain/tuning"
"""
Where tuned choices are kept unless PITHTRAIN_TUNING_DIR says otherwise: a directory every node of
the cluster sees (the home directory), so tuning is paid once per setup, not once per launch.
"""

TUNABLES: dict[str, "Tunable"] = {}
CACHED: list = []  # this setup's cache entries, the same on every rank once tune_kernels() has read them


class Tunable:
    """
    A kernel family whose launch config tune_kernels() may choose. candidates lists its configs in a
    fixed order; default(key) is the config it runs until tuned; prepare(key, rows) builds synthetic
    inputs with that many rows per group and returns launch(config), which runs the kernel once on
    them (compiling on first use) and returns its outputs, or None when the inputs would not fit.
    """

    def __init__(self, name: str, candidates: tuple, default: Callable, prepare: Callable) -> None:
        self.name, self.candidates, self.default, self.prepare = name, candidates, default, prepare
        self.table: dict[tuple, object] = {}  # (key, bucket) -> installed config
        self.tuned: dict[tuple, list] = {}  # key -> its installed buckets
        self.seen: dict[tuple, set] = {}  # key -> buckets pick() was asked for on this rank
        self.fingerprint = hashlib.sha256((repr(candidates) + inspect.getsource(inspect.getmodule(prepare))).encode()).hexdigest()[:16]
        TUNABLES[name] = self
        self.install(CACHED)

    def install(self, entries: list) -> dict:
        """
        Install the cache entries of this family (its name and fingerprint: a changed kernel or
        candidate list matches none), skipping buckets already installed. Returns {(name, key,
        bucket): (config, default)} of those installed.
        """
        installed = {}
        for name, fingerprint, key, bucket, idx in entries:
            if name == self.name and fingerprint == self.fingerprint and bucket not in self.tuned.get(key, ()):
                self.table[(key, bucket)] = self.candidates[idx]
                self.tuned.setdefault(key, []).append(bucket)
                installed[(name, key, bucket)] = (self.candidates[idx], self.default(key))
        return installed

    def pick(self, key: tuple, rows: int):
        """
        The config of one launch with rows rows per group: once tuned, one table lookup.
        """
        bucket = min(max(rows.bit_length() - 1, MIN_BUCKET), MAX_BUCKET)
        config = self.table.get((key, bucket))
        return config if config is not None else self.pick_untuned(key, bucket)

    def pick_untuned(self, key: tuple, bucket: int):
        """
        A launch whose key and bucket have no entry: before tuning (recording the key for
        tune_kernels()) the default, after it the nearest tuned bucket of the key, or the default
        for a key no rank used in the first step.
        """
        self.seen.setdefault(key, set()).add(bucket)
        if key in self.tuned:
            nearest = self.table[(key, min(self.tuned[key], key=lambda b: (abs(b - bucket), b)))]
            self.table[(key, bucket)] = nearest  # every rank computes the same, so the lookup is cached
            return nearest
        return self.default(key)


def gather(value: object) -> list:
    """
    Every rank's value in rank order (this process's alone when not distributed).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def cache_file() -> Path:
    """
    This setup's cache file, named by the GPU model and a fingerprint of the library versions and of
    this module, so an upgrade or a change to the tuning rules starts a new file. (A changed kernel
    or candidate list is told apart by the family fingerprint each entry carries.)
    """
    versions = []
    for package in ("torch", "nvidia-cutlass-dsl", "quack-kernels"):
        try:
            versions.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{package}==none")
    fingerprint = hashlib.sha256(repr((versions, inspect.getsource(inspect.getmodule(cache_file)))).encode()).hexdigest()[:16]
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return Path(os.environ.get("PITHTRAIN_TUNING_DIR", DEFAULT_CACHE_DIR)) / f"{gpu}-{fingerprint}.json"


def as_tuple(value):
    """
    A JSON list back as the tuple it was written from, nested lists included.
    """
    return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value


def read_cache() -> list:
    """
    The cached entries [(name, family fingerprint, key, bucket, candidate index)], or [] when
    there are none.
    """
    try:
        return [(name, fingerprint, as_tuple(key), bucket, idx) for name, fingerprint, key, bucket, idx in json.loads(cache_file().read_text())]
    except (OSError, ValueError):
        return []


def write_cache(entries: list) -> None:
    """
    Write the entries, replacing the file in one rename so a concurrent reader sees all or nothing.
    """
    path = cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(sorted(entries, key=repr)))
    os.replace(tmp, path)


def time_launch(launch: Callable, config) -> float:
    """
    Median microseconds of one launch, after WARMUP launches.
    """
    for _ in range(WARMUP):
        launch(config)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
    for start, end in events:
        start.record()
        launch(config)
        end.record()
    torch.cuda.synchronize()
    return sorted(start.elapsed_time(end) * 1e3 for start, end in events)[ITERS // 2]


def same_results(outputs: tuple, reference: tuple) -> bool:
    """
    Whether outputs match reference up to accumulation order: within 2^-6 of reference's
    largest magnitude, elementwise.
    """
    return all(torch.allclose(out.float(), ref.float(), rtol=0, atol=2**-6 * max(ref.float().abs().max().item(), 1e-30)) for out, ref in zip(outputs, reference))


def time_candidate(tunable: Tunable, key: tuple, bucket: int, config) -> float | None:
    """
    Median microseconds of config at a bucket's size (1.5x its lower bound), or None when the inputs
    would not fit, the config fails, or its results differ from the default's.
    """
    launch = tunable.prepare(key, 3 << (bucket - 1))
    if launch is None:
        return None
    try:
        reference = tuple(out.clone() for out in launch(tunable.default(key)))
        if not same_results(launch(config), reference):
            return None
        return time_launch(launch, config)
    except Exception:  # a config this kernel kind cannot take
        return None


def tune_kernels() -> dict:
    """
    Install this setup's cached choices, then tune every key any rank has used that they lack. Call
    on every rank at the same points: before the first training step (installs the cache, so that
    step runs the cached choices) and after it (tunes what it used and the cache lacked). Rank 0
    reads the cache for every rank; each rank times its share of the (key, candidate) tasks; every
    rank installs the same pooled choice (per key and bucket the fastest candidate, unless the
    default is within MARGIN), rank 0 adds it to the cache, every rank compiles the configs it will
    launch, and all ranks check they installed the same choices, which also lines them up so the
    next step starts together. Returns {(name, key, bucket): (config, default)} of this call.
    """
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)
    if not CACHED:  # the first call reads the cache: rank 0's read, the same on every rank
        CACHED.extend(gather(read_cache() if rank == 0 else None)[0])
    chosen = {}
    for tunable in TUNABLES.values():
        chosen.update(tunable.install(CACHED))
    pending = sorted(((name, key, sorted(buckets)) for name, tunable in TUNABLES.items() for key, buckets in tunable.seen.items() if key not in tunable.tuned), key=repr)
    buckets_of: dict[tuple, set] = {}
    for name, key, buckets in (entry for entries in gather(pending) for entry in entries):
        for bucket in buckets:
            buckets_of.setdefault((name, key), set()).update(range(max(bucket - BUCKET_SPREAD, MIN_BUCKET), min(bucket + BUCKET_SPREAD, MAX_BUCKET) + 1))
    tasks = [(name, key, idx) for (name, key) in sorted(buckets_of, key=repr) for idx in range(len(TUNABLES[name].candidates))]

    timings = []
    for name, key, idx in tasks[rank::world_size]:
        tunable = TUNABLES[name]
        timings += [(name, key, bucket, idx, time_candidate(tunable, key, bucket, tunable.candidates[idx])) for bucket in sorted(buckets_of[(name, key)])]
    torch.cuda.empty_cache()

    times_of: dict[tuple, dict] = {}
    for name, key, bucket, idx, us in (timing for rank_timings in gather(timings) for timing in rank_timings):
        if us is not None:
            times_of.setdefault((name, key, bucket), {})[idx] = us
    for (name, key, bucket), times in sorted(times_of.items(), key=repr):
        tunable = TUNABLES[name]
        default, fastest = tunable.default(key), min(times, key=lambda idx: (times[idx], idx))
        default_idx = tunable.candidates.index(default) if default in tunable.candidates else None
        config = default if default_idx in times and times[default_idx] <= times[fastest] * (1 + MARGIN) else tunable.candidates[fastest]
        tunable.table[(key, bucket)] = config
        tunable.tuned.setdefault(key, []).append(bucket)
        chosen[(name, key, bucket)] = (config, default)

    if rank == 0 and times_of:
        entries = {(name, tunable.fingerprint, key, bucket): tunable.candidates.index(config) for name, tunable in TUNABLES.items() for (key, bucket), config in tunable.table.items()}
        for name, fingerprint, key, bucket, idx in read_cache():  # keep what other runs added meanwhile
            entries.setdefault((name, fingerprint, key, bucket), idx)
        write_cache([(*entry, idx) for entry, idx in entries.items()])
    for name, key, config in sorted({(name, key, config) for (name, key, _), (config, _) in chosen.items() if key in TUNABLES[name].seen}, key=repr):
        launch = TUNABLES[name].prepare(key, 1 << MIN_BUCKET)
        if launch is not None:
            launch(config)  # compile now, not in the next step
    torch.cuda.synchronize()
    digest = hashlib.sha256(repr(sorted(chosen.items(), key=repr)).encode()).hexdigest()
    if len(set(gather(digest))) != 1:
        raise RuntimeError("tune_kernels: ranks installed different configs")
    return chosen
