"""
Host-side memoization of FlashAttention-4's Python entry (memoized_launch): a repeated call
signature replays the recorded compiled-kernel launches instead of running FA4's Python. Pinned to
the validated FA4 version; process-global and not thread-safe. FA4 runs on SM90 and SM100.
"""

import dataclasses
import importlib.metadata
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["FAST_PATH", "MISSING", "memoized_launch", "plans", "signature"]

# FA4's entry spends tens of microseconds per call deriving tile sizes and the compile key, a pure
# function of the call's signature, while its kernels take a few microseconds to launch. A recorded
# call becomes a plan only if FA4 did nothing but allocate, take views and call cached kernels;
# nothing captured is replayed except compiled kernels and immutable arguments.

VALIDATED_FA4 = "4.0.0b28"


def fast_path_enabled() -> bool:
    try:
        return importlib.metadata.version("flash-attn-4") == VALIDATED_FA4
    except importlib.metadata.PackageNotFoundError:
        return False


FAST_PATH = fast_path_enabled()

# aten ops FA4's entry may run without disqualifying a plan: allocations and views.
ALLOC_OPS = {"empty.memory_format", "empty_like.default", "empty_strided.default", "zeros.default", "zeros_like.default", "new_empty.default", "new_zeros.default"}
ZERO_OPS = {"zero_.default"}
VIEW_OPS = {"detach.default", "alias.default", "view.default", "_unsafe_view.default", "t.default", "transpose.int", "permute.default", "as_strided.default", "slice.Tensor", "select.int", "squeeze.dim", "unsqueeze.default", "expand.default"}


class Recording(TorchDispatchMode):
    """Logs the aten ops run under it: allocations (and whether zero-filled) and anything else."""

    def __init__(self) -> None:
        super().__init__()
        self.allocs: List[torch.Tensor] = []
        self.zeroed: set = set()
        self.other: List[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func).removeprefix("aten.")
        if name in ALLOC_OPS:
            self.allocs.append(out)
            if name.startswith(("zeros", "new_zeros")):
                self.zeroed.add(out.data_ptr())
        elif name in ZERO_OPS:
            self.zeroed.add(args[0].data_ptr())
        elif name not in VIEW_OPS:
            self.other.append(name)
        return out


class Recorder:
    def __init__(self, fn: Callable, log: list) -> None:
        self.fn, self.log = fn, log

    def __call__(self, *args, **kwargs):
        self.log.append((self.fn, args, kwargs))
        return self.fn(*args, **kwargs)


class RecordingCache:
    """Stands in for one FA4 compile cache during a recorded call."""

    def __init__(self, inner: Any, log: list) -> None:
        self.inner, self.log, self.inserted = inner, log, False

    def __contains__(self, key) -> bool:
        return key in self.inner

    def __getitem__(self, key):
        return Recorder(self.inner[key], self.log)

    def __setitem__(self, key, fn) -> None:
        self.inserted = True
        self.inner[key] = fn

    def __getattr__(self, name):
        return getattr(self.inner, name)


@dataclasses.dataclass(frozen=True)
class Plan:
    allocs: Tuple[Tuple[torch.Size, Tuple[int, ...], torch.dtype, bool], ...]
    calls: Tuple[Tuple[Callable, Any, Any], ...]
    outputs: Tuple[Any, ...]
    # calls flattened for replay: (kernel, arguments with the constants filled in, (position,
    # template) of every argument rebuilt per call)
    fast_calls: Tuple[Tuple[Callable, Tuple[Any, ...], Tuple[Tuple[int, Any], ...]], ...] = dataclasses.field(compare=False, default=())


class Unsupported(Exception):
    pass


def template(x: Any, inputs: Dict[str, torch.Tensor], allocs: List[torch.Tensor]) -> Any:
    """How to rebuild x from the live inputs and fresh allocations of a later call."""
    if isinstance(x, torch.Tensor):
        same = lambda t: t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        meta = (x.shape, x.stride(), x.storage_offset())
        for ref, t in inputs.items():
            if t is not None and same(t) and meta == (t.shape, t.stride(), t.storage_offset()):
                return ("in", ref, None)
        for ref, t in enumerate(allocs):
            if same(t):
                view = None if meta == (t.shape, t.stride(), t.storage_offset()) else (tuple(x.shape), tuple(x.stride()), x.storage_offset())
                return ("new", ref, view)
        # Inputs only ever pass through whole: a view of one (or a copy) is not planned.
        raise Unsupported("argument tensor that is no input and no allocation of FA4")
    if isinstance(x, (list, tuple)):
        parts = tuple(template(e, inputs, allocs) for e in x)
        if hasattr(x, "_fields"):
            return ("namedtuple", type(x), parts)
        return ("seq", type(x), parts)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        fields = {f.name: template(getattr(x, f.name), inputs, allocs) for f in dataclasses.fields(x)}
        if all(v[0] == "const" for v in fields.values()):
            return ("const", x)
        return ("dataclass", type(x), fields)
    if x is None or isinstance(x, (bool, int, float, str)):
        return ("const", x)
    if hasattr(x, "__dict__") and any(isinstance(v, torch.Tensor) for v in vars(x).values()):
        raise Unsupported(f"argument object {type(x).__name__} holds tensors")
    return ("const", x)


def build(t: Any, inputs: Dict[str, torch.Tensor], allocs: List[torch.Tensor]) -> Any:
    kind = t[0]
    if kind == "const":
        return t[1]
    if kind in ("in", "new"):
        base = inputs[t[1]] if kind == "in" else allocs[t[1]]
        if t[2] is None:
            return base.detach()
        shape, stride, offset = t[2]
        return base.detach().as_strided(shape, stride, offset)
    if kind == "namedtuple":
        return t[1](*(build(e, inputs, allocs) for e in t[2]))
    if kind == "seq":
        return t[1](build(e, inputs, allocs) for e in t[2])
    return t[1](**{k: build(v, inputs, allocs) for k, v in t[2].items()})


def record(entry: Callable[[], Tuple[torch.Tensor, ...]], inputs: Dict[str, torch.Tensor]):
    """Run FA4's entry once, returning (its outputs, a plan or None)."""
    # Imported here, not at the top: FA4 loads cutlass, and importing pithtrain.kernels must not.
    from flash_attn.cute import interface as _fa4
    owners = [f for f in vars(_fa4).values() if callable(f) and hasattr(f, "compile_cache")]
    log: list = []
    caches = [RecordingCache(f.compile_cache, log) for f in owners]
    for f, cache in zip(owners, caches):
        f.compile_cache = cache
    try:
        with Recording() as rec:
            outs = entry()
    finally:
        for f, cache in zip(owners, caches):
            f.compile_cache = cache.inner
    if any(c.inserted for c in caches):
        return outs, MISSING  # compiled a kernel on this call: record the next one
    if rec.other or not log:
        return outs, None  # did more than allocate, take views and launch cached kernels
    try:
        allocs = rec.allocs
        alloc_meta = tuple((t.shape, t.stride(), t.dtype, t.data_ptr() in rec.zeroed) for t in allocs)
        calls = tuple((fn, template(args, inputs, allocs), template(kwargs, inputs, allocs)) for fn, args, kwargs in log)
        outputs = tuple(template(o, inputs, allocs) for o in outs)
    except Unsupported:
        return outs, None
    if any(o[0] != "new" for o in outputs) or any(kwargs for _, _, kwargs in log):
        return outs, None
    fast_calls = []
    for fn, args, _ in calls:
        consts = tuple(a[1] if a[0] == "const" else None for a in args[2])
        slots = tuple((i, a) for i, a in enumerate(args[2]) if a[0] != "const")
        fast_calls.append((fn, consts, slots))
    return outs, Plan(alloc_meta, calls, outputs, tuple(fast_calls))


def replay(plan: Plan, inputs: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
    allocs = [torch.zeros(shape, dtype=dtype, device=device) if zeroed else torch.empty_strided(shape, stride, dtype=dtype, device=device) for shape, stride, dtype, zeroed in plan.allocs]
    detached = {name: t.detach() for name, t in inputs.items() if t is not None}
    for fn, consts, slots in plan.fast_calls:
        args = list(consts)
        for i, t in slots:
            kind, ref, view = t
            if kind == "in" and view is None:
                args[i] = detached[ref]
            elif kind == "new" and view is None:
                args[i] = allocs[ref]
            else:
                args[i] = build(t, inputs, allocs)
        fn(*args)
    return tuple(allocs[o[1]] if o[2] is None else build(o, inputs, allocs) for o in plan.outputs)


MISSING, _SEEN = object(), object()
plans: Dict[Tuple, Optional[Plan]] = {}


def signature(kind: str, inputs: Dict[str, Optional[torch.Tensor]], *scalars) -> Tuple:
    # Plans pass inputs through whole, so their layout and alignment (FA4 copies misaligned ones)
    # are all that matters; how inputs alias each other does not.
    return (kind, scalars, inputs["q"].device, tuple(None if t is None else (t.shape, t.stride(), t.dtype, t.data_ptr() % 16) for t in inputs.values()))


def memoized_launch(kind: str, inputs: Dict[str, Optional[torch.Tensor]], scalars: Tuple, entry: Callable[[], Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    entry() (a call into FA4's Python entry), replayed through a recorded host-side plan once one
    exists for this call's signature. Returns entry()'s output tuple.

    kind: the FA4 entry point's name. scalars: a hashable tuple of every non-tensor argument.
    inputs: name -> tensor (or None) for every tensor argument, "q" included. entry must be a
    function of exactly these and return freshly allocated tensors.

    The signature is kind, scalars, the device and each input's shape, strides, dtype and address
    modulo 16. Its first call runs entry(), the second runs it while recording, and once a clean
    recording exists (FA4 only allocated, took views and called compiled kernels) later calls
    allocate the same buffers afresh and call the same kernels on the live inputs. The outputs are
    entry()'s (bit for bit where FA4 is deterministic). With any FA4 version other than the
    validated one, every call is entry().
    """
    if not FAST_PATH:
        return entry()
    key = signature(kind, inputs, *scalars)
    plan = plans.get(key, MISSING)
    if plan is MISSING:
        # First call of the signature: plain entry, which compiles whatever it needs.
        plans[key] = _SEEN
        return entry()
    if plan is _SEEN:
        outs, plan = record(entry, inputs)
        if plan is not MISSING:
            plans[key] = plan
        return outs
    if plan is None:
        return entry()
    return replay(plan, inputs, inputs["q"].device)
