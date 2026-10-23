"""
FSDP2 forward methods with a lean path for the repeated calls of a step.

``torch.distributed.fsdp.register_fsdp_forward_method`` runs FSDP's full ``_pre_forward`` /
``_post_forward`` around every call. DualPipeV calls each layer's stage methods once per
micro-batch, and under its ZeRO-1 setup (``reshard_after_forward=False``) the parameters stay
unsharded from a group's first call of the step to the end-of-step reduce, so from the second call
on most of that machinery is a no-op that still costs ~240 us of host time per call (debug
logging and profiler ranges with formatted names, the root pre-forward's stream waits and input
device moves, unshard / reshard checks).

``register_fsdp_forward_method`` here keeps FSDP's full path whenever any of its preconditions
does not hold (a group not yet unsharded or with an all-gather in flight, a backward-state
module, a pending forward root or all-gather state, forward prefetching, an output cast, no grad
mode, tracing). Otherwise it performs exactly the per-call side effects that matter, through
FSDP's own functions and in its order:

- the state's and each param group's training-state transitions (IDLE -> FORWARD -> IDLE);
- the mixed-precision input cast, whenever any floating input tensor is not already in the param
  dtype (fp32 router probabilities are cast as before); skipped only when no input needs it;
- ``RegisterPostBackwardFunction`` on the grad-requiring inputs (per-micro-batch post-backward
  gradient accumulation is unchanged);
- the post-forward order record each group's backward prefetch reads;
- the pre-backward hook on every grad-requiring output.

A call to a module still in PRE_BACKWARD (the pipeline defers FSDP's root post-backward callback
to the end of the step, so every module stays there after its first backward) gets FSDP's
activation-checkpointing branch, which with every group unsharded is the input cast alone.

What it skips is, under those preconditions, state-free: the root pre-forward's all-gather stream
waits only order all-gathers, and every all-gather comes from a call that takes the full path (its
group is sharded then), which does the waits itself. Inside another state's forward (the root model's, in the sequential
passes) FSDP itself skips the root work for the call.

``install_lean_hooks`` covers the calls that still reach FSDP's hooks (first calls, pre- and
post-backward): their ``_dynamo_disable`` decorator builds a new ``torch._dynamo.disable`` wrapper
around the hook on every call; it is built once instead, the same wrapper around the same function,
and the per-micro-batch post-backward (no gradient reduce) skips its debug log and profiler ranges.
"""

import functools
from typing import Any

import torch
import torch.nn as nn
from torch._functorch.utils import unwrap_dead_wrappers
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import register_fsdp_forward_method as _upstream_register
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    TrainingState,
    _dynamo_disable,
    is_bw,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    RegisterPostBackwardFunction,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState


def _seq(x: Any) -> bool:
    return type(x) in (tuple, list) or (isinstance(x, tuple) and hasattr(type(x), "_fields"))


def _rebuild(x: Any, elems) -> Any:
    return type(x)(*elems) if hasattr(type(x), "_fields") else type(x)(elems)


_SCALARS = frozenset((bool, int, float, str, type(None)))
_NOT_PLAIN, _CAST = 1, 2


def _scan(x: Any, dtype: torch.dtype | None, grads: list) -> int:
    """
    One walk of x. Returns _NOT_PLAIN unless x is made only of tensors, tuples / named tuples /
    lists, None and scalars, and _CAST if FSDP's input cast to dtype (None: no cast) would change a
    tensor of it. Appends its grad-requiring tensors to grads in collect_grad_tensors' (depth-first)
    order. Scalar elements, such as the split sizes in routing metadata, are skipped in place.
    """
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            grads.append(x)
        if dtype is not None and x.dtype != dtype and torch.is_floating_point(x):
            return _CAST
        return 0
    if type(x) in _SCALARS or isinstance(x, (bool, int, float, str)):
        return 0
    if not _seq(x):
        return _NOT_PLAIN
    flags = 0
    for e in x:
        if type(e) not in _SCALARS:
            flags |= _scan(e, dtype, grads)
    return flags


def _scan_call(args: tuple, kwargs: dict, dtype: torch.dtype | None, grads: list) -> int:
    """_scan over a call's arguments: args, then the kwargs values."""
    flags = _scan(args, dtype, grads)
    for v in kwargs.values():
        flags |= _scan(v, dtype, grads)
    return flags


def _cast(x: Any, dtype: torch.dtype) -> Any:
    """FSDP's _cast_fp_tensor over a plain structure: the same x.to(dtype) per floating tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if torch.is_floating_point(x) and x.dtype != dtype else x
    if _seq(x):
        return _rebuild(x, [_cast(e, dtype) for e in x])
    return x


def _replace(x: Any, it) -> Any:
    """x with its grad-requiring tensors replaced, in order, by those of it."""
    if isinstance(x, torch.Tensor):
        return next(it) if x.requires_grad else x
    if _seq(x):
        return _rebuild(x, [_replace(e, it) for e in x])
    return x


def _backward_state_call_ok(state: Any) -> bool:
    """
    A call to a module already in PRE_BACKWARD. The pipeline defers FSDP's root post-backward
    callback to the end of the step, so a module stays in that state from its first backward on,
    and FSDP's forward hooks then take their activation-checkpointing branch: unshard any sharded
    group, cast the inputs, cast the output. With every group unsharded and no output dtype, only
    the input cast is left.
    """
    if torch.compiler.is_compiling() or state._mp_policy.output_dtype is not None:
        return False
    return all(group.is_unsharded for group in state._fsdp_param_groups)


def _cast_dtype(state: Any) -> torch.dtype | None:
    """The dtype FSDP casts forward inputs to, or None when it does not cast them."""
    policy = state._mp_policy
    return policy.param_dtype if policy.cast_forward_inputs else None


def _cast_inputs(state: Any, args: tuple, kwargs: dict, flags: int) -> tuple[tuple, dict]:
    """FSDP's _cast_forward_inputs, given the _scan_call flags of the arguments."""
    if flags & _NOT_PLAIN:
        return state._cast_forward_inputs(args, kwargs)
    if flags & _CAST:
        dtype = state._mp_policy.param_dtype
        return _cast(args, dtype), {k: _cast(v, dtype) for k, v in kwargs.items()}
    return args, kwargs


# RegisterPostBackwardFunction.apply minus Function.apply's bind_default_args, whose only effect
# here is an inspect.signature per call: forward(param_group, *inputs) has no defaults, so binding
# positional arguments returns them unchanged.
_register_post_backward_apply = super(torch.autograd.Function, RegisterPostBackwardFunction).apply


def _repeat_call_ok(state: Any) -> bool:
    if not torch.is_grad_enabled() or torch.compiler.is_compiling() or is_bw():
        return False
    if torch._C._are_functorch_transforms_active():
        return False
    if state._training_state != TrainingState.IDLE or state._states_to_forward_prefetch:
        return False
    if state._mp_policy.output_dtype is not None:
        return False
    # With no forward root yet, FSDP makes this call the root: its post-forward then frees the
    # last all-gather result, so take the full path while one is held. Inside another state's
    # forward (the pipeline's sequential passes run the layers inside the model's forward), FSDP
    # does no root work for this call at all.
    root = state._state_ctx.iter_forward_root
    if root is state:
        return False
    for group in state._fsdp_param_groups:
        if root is None and group.comm_ctx.all_gather_state is not None:
            return False
        if group._training_state != TrainingState.IDLE or group._all_gather_result is not None:
            return False
        if not group.is_unsharded or group._reshard_after_forward:
            return False
    return True


def register_fsdp_forward_method(module: nn.Module, method_name: str) -> None:
    """
    Drop-in for torch.distributed.fsdp.register_fsdp_forward_method with the lean repeat path.
    """
    if not isinstance(module, FSDPModule):
        _upstream_register(module, method_name)
        return
    orig_method = getattr(module, method_name)

    @functools.wraps(orig_method)
    def wrapped_method(self, *args, **kwargs):
        state = self._get_fsdp_state()
        if state._training_state == TrainingState.PRE_BACKWARD and _backward_state_call_ok(state):
            dtype = _cast_dtype(state)
            if dtype is not None:
                flags = _scan_call(args, kwargs, dtype, [])
                args, kwargs = _cast_inputs(state, args, kwargs, flags)
            return orig_method(*args, **kwargs)
        if not _repeat_call_ok(state):
            args, kwargs = state._pre_forward(self, args, kwargs)
            out = orig_method(*args, **kwargs)
            return state._post_forward(self, args, out)
        groups = state._fsdp_param_groups
        tensors = []
        flags = _scan_call(args, kwargs, _cast_dtype(state), tensors)
        plain = not flags & _NOT_PLAIN
        state._training_state = TrainingState.FORWARD
        args, kwargs = _cast_inputs(state, args, kwargs, flags)
        recollect = bool(flags & _CAST)  # the cast replaced the floating tensors it changed
        for group in groups:
            group._training_state = TrainingState.FORWARD
            if not plain:
                args, kwargs = group._register_post_backward_hook(args, kwargs)
                continue
            # FSDP's _register_post_backward_hook: one RegisterPostBackwardFunction over all
            # grad-requiring inputs (args, then kwargs), its outputs swapped in place of them.
            if recollect:
                tensors = []
                _scan_call(args, kwargs, None, tensors)
            recollect = True  # this group replaces them for the next one
            if tensors:
                applied = _register_post_backward_apply(*unwrap_dead_wrappers((group, *tensors)))
                it = iter(applied)
                args = _replace(args, it)
                kwargs = {k: _replace(v, it) for k, v in kwargs.items()}
        out = orig_method(*args, **kwargs)
        for group in groups:
            group._record_post_forward()
            group._training_state = TrainingState.IDLE
        outs = []
        if not _scan(out, None, outs) & _NOT_PLAIN:
            # FSDP's _register_pre_backward_hook: the hook on every grad-requiring output.
            for t in outs:
                t.register_hook(state._pre_backward)
        else:
            out = state._register_pre_backward_hook(out)
        state._training_state = TrainingState.IDLE
        return out

    setattr(module, method_name, wrapped_method.__get__(module, type(module)))


# The FSDP hooks decorated with _dynamo_disable that run during a training step.
_HOOKS = (
    (FSDPState, "_pre_forward"),
    (FSDPState, "_post_forward"),
    (FSDPState, "_pre_backward"),
    (FSDPState, "_root_post_backward_final_callback"),
    (FSDPParamGroup, "pre_backward"),
    (FSDPParamGroup, "post_backward"),
)
_PER_CALL_DISABLE = _dynamo_disable(lambda: None).__code__


def _lean_post_backward(post_backward):
    """
    FSDPParamGroup.post_backward with FSDP's own body for a backward that reduces nothing (the
    pipeline's per-micro-batch backwards run with gradient sync off until the end of the step):
    accumulate, reshard if asked, switch to the accumulated gradient. Only its debug log and two
    profiler ranges are left out (13.2 -> 2.0 us per call). Any other call is FSDP's own.
    """

    @functools.wraps(post_backward)
    def lean(self, *unused):
        if self.reduce_grads:
            return post_backward(self, *unused)
        self._training_state = TrainingState.POST_BACKWARD
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()
        if self.reshard_after_backward:
            self.reshard()
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_accumulated_grad_if_needed()

    return lean


def install_lean_hooks() -> None:
    """
    Replace each hook's per-call ``torch._dynamo.disable(func, recursive=True)`` with that wrapper
    built once, around FSDP's function, or around _lean_post_backward's for post_backward. Hooks
    whose decorator is not FSDP's per-call one are left alone.
    """
    for cls, name in _HOOKS:
        hook = cls.__dict__[name]
        if getattr(hook, "__code__", None) is not _PER_CALL_DISABLE:
            continue
        func = hook.__wrapped__
        if (cls, name) == (FSDPParamGroup, "post_backward"):
            func = _lean_post_backward(func)
        setattr(
            cls, name, torch._dynamo.disable(func, recursive=True, reason="skipping FSDP hooks")
        )
