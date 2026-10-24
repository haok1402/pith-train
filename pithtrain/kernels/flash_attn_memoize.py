"""
Record / replay of FlashAttention-4's host path (``memoized_launch``): host-side memoization of
FA4's compile caches, so a repeated call signature skips FA4's Python entry and launches the
compiled kernels directly. FA4 runs on SM90 and SM100, so this has no architecture suffix; the
custom ops that call it are ``pithtrain.operators.flash_attn_v4``.

It wraps ``flash_attn.cute.interface``'s compile caches during one recorded call. That is global
and not thread-safe, and the path is pinned to the FA4 version it was validated on
(``_VALIDATED_FA4``); any other version keeps calling FA4's entry.
"""

import dataclasses
import importlib.metadata
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from flash_attn.cute import interface as _fa4
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["FAST_PATH", "MISSING", "memoized_launch", "plans", "signature"]

# fmt: off

# ------------------------------------------------------------
# MEMOIZED CALL PATH
# ------------------------------------------------------------
#
# FA4's Python entry spends tens of microseconds per call deriving tile sizes, building the
# compile key and looking the compiled kernels up, all of which is a pure function of the call's
# signature; the kernels themselves cost a few microseconds to launch. The first call of each
# signature runs FA4's own entry while recording what it did: every compiled-kernel call (the
# compile caches are wrapped for that one call) and every aten op (a dispatch mode). If it did
# nothing but allocate (empty or zeros), take views and call cached kernels, the recording becomes
# a plan; later calls of the signature allocate the same buffers afresh, exactly as FA4 does
# (zero-filled where FA4 zero-fills them), and call the same compiled kernels with the same
# argument layout and the live tensors. Nothing captured is replayed except compiled kernels
# and immutable arguments. A signature whose first clean recording shows anything else, or a
# version other than the one this was validated on, keeps calling FA4's entry.

_VALIDATED_FA4 = "4.0.0b28"


def _fast_path_enabled() -> bool:
    try:
        return importlib.metadata.version("flash-attn-4") == _VALIDATED_FA4
    except importlib.metadata.PackageNotFoundError:
        return False


FAST_PATH = _fast_path_enabled()

# aten ops FA4's entry may run without disqualifying a plan: allocations and views.
_ALLOC_OPS = {"empty.memory_format", "empty_like.default", "empty_strided.default", "zeros.default", "zeros_like.default", "new_empty.default", "new_zeros.default"}
_ZERO_OPS = {"zero_.default"}
_VIEW_OPS = {"detach.default", "alias.default", "view.default", "_unsafe_view.default", "t.default", "transpose.int", "permute.default", "as_strided.default", "slice.Tensor", "select.int", "squeeze.dim", "unsqueeze.default", "expand.default"}


class _Recording(TorchDispatchMode):
    """Logs the aten ops run under it: allocations (and whether zero-filled) and anything else."""

    def __init__(self) -> None:
        super().__init__()
        self.allocs: List[torch.Tensor] = []
        self.zeroed: set = set()
        self.other: List[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func).removeprefix("aten.")
        if name in _ALLOC_OPS:
            self.allocs.append(out)
            if name.startswith(("zeros", "new_zeros")):
                self.zeroed.add(out.data_ptr())
        elif name in _ZERO_OPS:
            self.zeroed.add(args[0].data_ptr())
        elif name not in _VIEW_OPS:
            self.other.append(name)
        return out


class _Recorder:
    def __init__(self, fn: Callable, log: list) -> None:
        self.fn, self.log = fn, log

    def __call__(self, *args, **kwargs):
        self.log.append((self.fn, args, kwargs))
        return self.fn(*args, **kwargs)


class _RecordingCache:
    """Stands in for one FA4 compile cache during a recorded call."""

    def __init__(self, inner: Any, log: list) -> None:
        self.inner, self.log, self.inserted = inner, log, False

    def __contains__(self, key) -> bool:
        return key in self.inner

    def __getitem__(self, key):
        return _Recorder(self.inner[key], self.log)

    def __setitem__(self, key, fn) -> None:
        self.inserted = True
        self.inner[key] = fn

    def __getattr__(self, name):
        return getattr(self.inner, name)


@dataclasses.dataclass(frozen=True)
class _Plan:
    allocs: Tuple[Tuple[torch.Size, Tuple[int, ...], torch.dtype, bool], ...]
    calls: Tuple[Tuple[Callable, Any, Any], ...]
    outputs: Tuple[Any, ...]
    # calls flattened for replay: (kernel, argument list with the constants filled in, the
    # (position, template) of every argument rebuilt per call)
    fast_calls: Tuple[Tuple[Callable, Tuple[Any, ...], Tuple[Tuple[int, Any], ...]], ...] = dataclasses.field(compare=False, default=())


class _Unsupported(Exception):
    pass


def _template(x: Any, inputs: Dict[str, torch.Tensor], allocs: List[torch.Tensor]) -> Any:
    """How to rebuild x from the live inputs and fresh allocations of a later call."""
    if isinstance(x, torch.Tensor):
        same = lambda t: t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()  # noqa: E731
        meta = (x.shape, x.stride(), x.storage_offset())
        for ref, t in inputs.items():
            if t is not None and same(t) and meta == (t.shape, t.stride(), t.storage_offset()):
                return ("in", ref, None)
        for ref, t in enumerate(allocs):
            if same(t):
                view = None if meta == (t.shape, t.stride(), t.storage_offset()) else (tuple(x.shape), tuple(x.stride()), x.storage_offset())
                return ("new", ref, view)
        # Inputs only ever pass through whole: a view of one (or a copy) is not planned.
        raise _Unsupported("argument tensor that is no input and no allocation of FA4")
    if isinstance(x, (list, tuple)):
        parts = tuple(_template(e, inputs, allocs) for e in x)
        if hasattr(x, "_fields"):
            return ("namedtuple", type(x), parts)
        return ("seq", type(x), parts)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        fields = {f.name: _template(getattr(x, f.name), inputs, allocs) for f in dataclasses.fields(x)}
        if all(v[0] == "const" for v in fields.values()):
            return ("const", x)
        return ("dataclass", type(x), fields)
    if x is None or isinstance(x, (bool, int, float, str)):
        return ("const", x)
    if hasattr(x, "__dict__") and any(isinstance(v, torch.Tensor) for v in vars(x).values()):
        raise _Unsupported(f"argument object {type(x).__name__} holds tensors")
    return ("const", x)


def _build(t: Any, inputs: Dict[str, torch.Tensor], allocs: List[torch.Tensor]) -> Any:
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
        return t[1](*(_build(e, inputs, allocs) for e in t[2]))
    if kind == "seq":
        return t[1](_build(e, inputs, allocs) for e in t[2])
    return t[1](**{k: _build(v, inputs, allocs) for k, v in t[2].items()})


def _record(entry: Callable[[], Tuple[torch.Tensor, ...]], inputs: Dict[str, torch.Tensor]):
    """Run FA4's entry once, returning (its outputs, a plan or None)."""
    owners = [f for f in vars(_fa4).values() if callable(f) and hasattr(f, "compile_cache")]
    log: list = []
    caches = [_RecordingCache(f.compile_cache, log) for f in owners]
    for f, cache in zip(owners, caches):
        f.compile_cache = cache
    try:
        with _Recording() as rec:
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
        calls = tuple((fn, _template(args, inputs, allocs), _template(kwargs, inputs, allocs)) for fn, args, kwargs in log)
        outputs = tuple(_template(o, inputs, allocs) for o in outs)
    except _Unsupported:
        return outs, None
    if any(o[0] != "new" for o in outputs) or any(kwargs for _, _, kwargs in log):
        return outs, None
    fast_calls = []
    for fn, args, _ in calls:
        consts = tuple(a[1] if a[0] == "const" else None for a in args[2])
        slots = tuple((i, a) for i, a in enumerate(args[2]) if a[0] != "const")
        fast_calls.append((fn, consts, slots))
    return outs, _Plan(alloc_meta, calls, outputs, tuple(fast_calls))


def _replay(plan: _Plan, inputs: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, ...]:
    allocs = [
        torch.zeros(shape, dtype=dtype, device=device) if zeroed else torch.empty_strided(shape, stride, dtype=dtype, device=device)
        for shape, stride, dtype, zeroed in plan.allocs
    ]
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
                args[i] = _build(t, inputs, allocs)
        fn(*args)
    return tuple(allocs[o[1]] if o[2] is None else _build(o, inputs, allocs) for o in plan.outputs)


MISSING, _SEEN = object(), object()
plans: Dict[Tuple, Optional[_Plan]] = {}


def signature(kind: str, inputs: Dict[str, Optional[torch.Tensor]], *scalars) -> Tuple:
    # Plans pass inputs through whole, so their layout and alignment (FA4 copies misaligned ones)
    # are all that matters; how inputs alias each other does not.
    return (kind, scalars, inputs["q"].device, tuple(
        None if t is None else (t.shape, t.stride(), t.dtype, t.data_ptr() % 16)
        for t in inputs.values()
    ))


def memoized_launch(kind: str, inputs: Dict[str, Optional[torch.Tensor]], scalars: Tuple, entry: Callable[[], Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """entry() through the plan recorded for this signature, once one exists."""
    if not FAST_PATH:
        return entry()
    key = signature(kind, inputs, *scalars)
    plan = plans.get(key, MISSING)
    if plan is MISSING:
        # First call of the signature: plain entry, which compiles whatever it needs.
        plans[key] = _SEEN
        return entry()
    if plan is _SEEN:
        outs, plan = _record(entry, inputs)
        if plan is not MISSING:
            plans[key] = plan
        return outs
    if plan is None:
        return entry()
    return _replay(plan, inputs, inputs["q"].device)
