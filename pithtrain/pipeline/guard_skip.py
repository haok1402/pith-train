"""
Skip Dynamo guard evaluation on steps that repeat an already guarded step.

Every compiled region (the per-layer stage 1 / stage 5 / fused stage 5+1 functions, the gpt-oss
router, the ring-attention regions under CP) pays Dynamo's frame entry on each call, about 150 us
of host time, on a step that is host-bound. Under
``torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)`` Dynamo evaluates only the
"differentiating" guards of each code object's cache: the TENSOR_MATCH guards on non-parameter
inputs (shape, stride, dtype, device, requires_grad, dispatch keys) plus every guard that has
already failed once, which is how the entries of different layer variants of one code object
(dense vs MoE layer, sliding vs full attention, linear vs full attention) tell themselves apart:
compiling the second variant failed the first entry's guard on the attribute that differs, so
that guard stays checked. The call then runs the same compiled artifact the guarded call would.

What skipping hides is any recompile condition that is neither a tensor property nor a guard that
failed during warm-up. Each one is either constant for the process or part of the signature
below, so a step runs unguarded only when an identical signature already completed a fully
guarded step in this process:

- process constants after setup_*: ``distributed.ep_size`` / ``cp_size`` / ``pp_size`` (EP=1 and
  CP code paths), ``training.fp8`` and the FP8 recipe flags read at trace time. The signature
  carries them anyway, so a change trips it.
- per step, all recomputed on every step: grad and inference mode (``forward_only`` inference
  compiles separate entries, and Qwen3.5's linear attention branches on grad mode), the training
  flag of every module, the ambient Torch state Dynamo guards (autocast, default dtype and device,
  deterministic mode, TF32 / reduced-precision flags, thread count, the identity of every active
  torch-function and dispatch mode and saved-tensor hook), the identity and scalar settings of
  every gate's router replay and load-balance objects, the layer variants and module structures,
  ``cu_seqlens`` per micro-batch (packed micro-batches take the varlen attention path), and each
  micro-batch's input metadata.
- within a step: the layer variants (fingerprinted in the signature) and the ring-attention
  ``combine_partial`` None / offset variants all occur during the guarded first step, so their
  differentiating guards have failed once and stay checked.
- first vs later steps: the first step of a signature runs with full guards, compiling every
  entry it needs; only afterwards is the signature trusted. Should anything compile during a
  skipped step anyway, every signature loses its trust and runs guarded again.

Skipping is off for a model whose decoder layers differ in module structure (qwen3.5 linear vs
full attention, deepseek-v2 dense vs MoE): the differentiating guards copied from one variant's
entry look up submodules the other variant lacks, and Dynamo raises (KeyError 'self_attn' on a
qwen3.5 linear-attention layer) instead of moving on to the next entry. Variants that differ only
in attribute values (gpt-oss sliding vs full attention) keep it.

Skipping is only enabled on the torch release it was validated on.
"""

from typing import Any, Hashable, List, Tuple

import torch
from torch._dynamo.utils import counters

from pithtrain.contexts import distributed, training

_VALIDATED_TORCH = "2.13."


def _enabled() -> bool:
    return torch.__version__.startswith(_VALIDATED_TORCH)


def _compiled_graphs() -> int:
    # Bumped once per graph Dynamo produces: every compile and every recompile.
    return counters["stats"]["unique_graphs"]


def _mode_ids(modes) -> Tuple:
    return tuple((type(m).__name__, id(m)) for m in modes)


def _ambient() -> Tuple:
    """
    Process-global Torch state Dynamo guards on every compiled region (grad mode aside), with
    the identity of every active torch-function / dispatch mode and saved-tensor hook.
    """
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    try:
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(True)
    except (AttributeError, TypeError):
        hooks = None
    matmul = torch.backends.cuda.matmul
    return (
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.is_autocast_cache_enabled(),
        torch.get_default_dtype(),
        str(torch.get_default_device()),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul.allow_tf32,
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
        _mode_ids(_get_current_function_mode_stack()),
        _mode_ids(_get_current_dispatch_mode_stack()),
        None if hooks is None else tuple(id(h) for h in hooks),
    )


def _config(obj: Any) -> Tuple | None:
    """Identity, type and scalar settings of a router-replay / load-balance object."""
    if obj is None:
        return None
    scalars = tuple(
        sorted(
            (k, v)
            for k, v in getattr(obj, "__dict__", {}).items()
            if v is None or isinstance(v, (bool, int, float, str))
        )
    )
    return (id(obj), type(obj).__name__, scalars)


def _model_state(model: torch.nn.Module) -> Tuple[Tuple, Tuple, Tuple, bool]:
    """
    One pass over the module tree: (training flag of every module, router replay / load-balance
    configuration of every gate, the decoder layers' variants and module structures, whether those
    structures are all the same).

    A module's structure is its type and, recursively, its named children's structures, so any
    submodule added, removed, renamed or swapped changes it. A layer's variant is that structure
    plus the attributes that select a different cache entry of one compiled method (sliding/full
    or linear/full attention, CP layout).
    """
    modes, gates, layers = [], [], []

    def visit(m: torch.nn.Module) -> Tuple:
        modes.append(m.training)
        if hasattr(m, "router_replay"):
            gates.append(
                (_config(m.router_replay), _config(getattr(m, "load_balance_loss_fn", None)))
            )
        children = tuple(
            (name, visit(child)) for name, child in m._modules.items() if child is not None
        )
        structure = (type(m).__name__, children)
        if hasattr(m, "idx") and hasattr(m, "mlp"):
            attn = getattr(m, "self_attn", None)
            layers.append(
                (
                    structure,
                    getattr(attn, "is_sliding", None),
                    getattr(m, "is_linear", None),
                    getattr(m, "to_contiguous", None),
                    getattr(m, "to_zigzag", None),
                )
            )
        return structure

    visit(model)
    uniform = len({layer[0] for layer in layers}) <= 1
    return tuple(modes), tuple(gates), tuple(layers), uniform


def _tensor(t: torch.Tensor | None) -> Tuple | None:
    if t is None:
        return None
    return (t.shape, t.stride(), t.dtype, t.device, t.requires_grad)


class GuardSkip:
    """
    Per-process record of step signatures that completed a fully guarded step.
    """

    def __init__(self) -> None:
        self.trusted: set[Hashable] = set()
        self.enabled = _enabled()

    def signature(self, model: torch.nn.Module, microbatches: List[Any]) -> Tuple | None:
        """
        The step's signature, recomputed from scratch on every call, or None when the step may
        not run unguarded at all (decoder layers of different module structure).
        """
        modes, gates, layers, uniform = _model_state(model)
        if not uniform:
            return None
        microbatch_meta = tuple(
            (tuple(_tensor(t) for t in mb.model_inputs), _tensor(mb.cu_seqlens))
            for mb in microbatches
        )
        return (
            torch.is_grad_enabled(),
            _ambient(),
            modes,
            training.fp8,
            getattr(training, "fp8_param_gather", None),
            distributed.ep_size,
            distributed.cp_size,
            distributed.pp_size,
            gates,
            layers,
            microbatch_meta,
        )

    def run(self, model: torch.nn.Module, microbatches: List[Any], fn, *args, **kwargs):
        """
        fn(*args, **kwargs) unguarded when this step's signature is trusted, guarded otherwise.
        """
        if not self.enabled:
            return fn(*args, **kwargs)
        key = self.signature(model, microbatches)
        if key is None:
            return fn(*args, **kwargs)
        trusted = key in self.trusted
        before = _compiled_graphs()
        if trusted:
            with torch.compiler.set_stance("default", skip_guard_eval_unsafe=True):
                out = fn(*args, **kwargs)
            if _compiled_graphs() != before:
                # Something compiled although this signature had run guarded: trust nothing
                # until each signature has run guarded again.
                self.trusted.clear()
                return out
        else:
            out = fn(*args, **kwargs)
        self.trusted.add(key)
        return out
