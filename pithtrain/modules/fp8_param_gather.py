"""
FP8 parameter all-gather for FSDP2: gather FP8 weight bytes instead of bf16.

Port of Megatron-LM's ``--fp8-param-gather`` to PithTrain's 128x128 block-scaled FP8 weights. With
it on, no bf16 copy of an FP8 weight exists and the parameter all-gather moves one byte per element:

1. ``precompute_fp8_param_scales`` runs after every change to the fp32 master (weight init,
   checkpoint load, ``optimizer.step()``). Each rank computes the per-tile amax of its own shard,
   and one packed all-reduce MAX per FSDP shard group leaves every rank with the full grid, from
   which the scales follow. Tiles may straddle shards, so the grid is reduced, never gathered. The
   shard is then cast to FP8 into a persistent buffer, as Megatron's distributed optimizer does
   right after its step, so the forward pass does no quantization work at all.
2. FSDP2 calls ``fsdp_pre_all_gather`` on the sharded parameter, which hands it that FP8 shard,
   and all-gathers the FP8 bytes.
3. ``fsdp_post_all_gather`` lays the gathered bytes and their transpose out back to back in one
   buffer of exactly the bf16 weight's size, and the unsharded parameter is a bf16 view of it. The
   parameter keeps its usual shape and dtype, so autograd and FSDP's gradient path are unchanged,
   but its bytes are the FP8 operands. The module reads them through two FP8 buffers viewing the
   same storage, built once: FSDP frees and reallocates the storage in place on every reshard and
   unshard, so the views stay valid and no per-call work is needed to recover the operands.

Every step rounds fp32 to bf16 before taking an amax or casting, so the FP8 bytes and scales are
bit-identical to ``fp8cast_blockwise_transpose[_batched]`` of the bf16 weight that FSDP gathers
with the flag off. The flag changes memory and communication, not numerics.

The hooks are plain functions set as attributes of the sharded local tensor, which FSDP2 picks up
at lazy init; no tensor subclass is involved, so optimizers, checkpointing and ``torch.compile``
only ever see plain tensors.

The kernels (amax, cast, scales, pack + transpose) are the ``fp8_weight_quant`` picker's
(``pithtrain/kernels/fp8_weight_quant.py``).
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, free_storage

from pithtrain.kernels.fp8_weight_quant import (
    fp8_pack_transpose,
    fp8_scales,
    fp8_scales_multi,
    fp8_shard_amax,
    fp8_shard_amax_multi,
    fp8_shard_cast,
)

__all__ = [
    "fp8_param_operands",
    "install_fp8_param_gather",
    "install_raw_fp8_param_gather",
    "precompute_fp8_param_scales",
]

BLOCK = 128


# ---------------------------------------------------------------------------
# FSDP2 wiring
# ---------------------------------------------------------------------------


@dataclass
class _FP8ParamState:
    """
    Everything one wrapped weight needs, shared by its two FSDP hooks and the precompute.
    """

    module: nn.Module
    fsdp_param: FSDPParam
    shape: torch.Size
    """Global shape, (N, K) or (G, N, K)."""
    row_offset: int
    """First global row of N this rank holds; 0 for a 3-D weight, sharded over G."""
    group_offset: int
    """First global group of G this rank holds; 0 for a 2-D weight."""
    amax: torch.Tensor
    """(G, ceil(N/128), ceil(K/128)) view into the packed buffer of the shard group."""
    version: int = -1
    """Version counter of the local shard when amax was last computed."""
    packed: Optional[torch.Tensor] = None
    """Flat byte buffer [fp8 | fp8_t] behind the unsharded parameter."""
    fp8_shard: Optional[torch.Tensor] = None
    """This rank's shard cast to FP8, of FSDP's padded shard size; the all-gather input."""

    def local_shard(self) -> torch.Tensor:
        """The local fp32 shard as (g_local, local_rows, K)."""
        local = self.fsdp_param._sharded_local_tensor
        return local.view(1, *local.shape) if len(self.shape) == 2 else local

    def pre_all_gather(
        self,
        mesh: Any,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: Any,
    ) -> tuple[tuple[torch.Tensor, ...], None]:
        if self._shard_freed():
            self.cast_shard()
        assert self.fsdp_param._sharded_local_tensor._version == self.version, (
            "FP8 shard is stale: call precompute_fp8_param_scales after changing the weights"
        )
        return (self.fp8_shard,), None

    def _shard_freed(self) -> bool:
        return self.fp8_shard is not None and self.fp8_shard.untyped_storage().size() == 0

    def cast_shard(self) -> None:
        """Cast the local fp32 shard into fp8_shard with the current amax grid."""
        padded = self.fsdp_param.padded_sharded_param_size
        if self._shard_freed():
            self.fp8_shard.untyped_storage().resize_(self.fp8_shard.numel())
        if self.fp8_shard is None:
            # Uneven shards must still hand FSDP the padded size; the padding rows are dropped
            # after the gather, so they are left uninitialised.
            device = self.fsdp_param.sharded_param.device
            self.fp8_shard = torch.empty(padded, dtype=torch.uint8, device=device)
        out = self.fp8_shard.view(torch.float8_e4m3fn)
        shard = self.local_shard()
        if len(self.shape) == 2:
            dst = out.view(1, *padded)[:, : shard.shape[1]]
        else:
            dst = out[: shard.shape[0]]
        fp8_shard_cast(shard, self.amax, dst, self.row_offset, self.group_offset)

    def post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: None,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> Optional[tuple[torch.Tensor, tuple[torch.Tensor, ...]]]:
        assert param_dtype == torch.bfloat16, param_dtype
        (gathered,) = all_gather_outputs
        gathered = gathered[: self.shape[0]].view(torch.float8_e4m3fn).view(self.shape)
        gathered = gathered.view(-1, *self.shape[-2:])
        numel = gathered.numel()
        if out is None:
            self.packed = torch.empty(2 * numel, dtype=torch.uint8, device=gathered.device)
            data = self.packed.view(torch.float8_e4m3fn)
            *lead, N, K = self.shape
            fp8, fp8_t = data[:numel].view(*lead, N, K), data[numel:].view(*lead, K, N)
            self.module.register_buffer("weight_fp8", fp8, persistent=False)
            self.module.register_buffer("weight_t_fp8", fp8_t, persistent=False)
        fp8_pack_transpose(gathered, self.packed)
        # The bytes now live in the packed buffer; drop the gather output until the next unshard,
        # which reallocates it. Both run on the current stream, so the free is ordered.
        for t in self.fsdp_param.all_gather_outputs:
            free_storage(t)
        # FSDP copied the FP8 shard into its all-gather buffer, so it is dead until the next
        # precompute recasts it: release it rather than hold 1 B/param through the step.
        free_storage(self.fp8_shard)
        if out is not None:
            return None
        return self.packed.view(torch.bfloat16).view(self.shape), (self.packed,)


_states: list[_FP8ParamState] = []
_groups: list[tuple[dist.ProcessGroup, torch.Tensor, list[_FP8ParamState]]] = []


def install_fp8_param_gather(model: nn.Module, fp8_classes: tuple[type, ...]) -> None:
    """
    Make FSDP2 all-gather the weight of every module of fp8_classes in FP8.

    Call after fully_shard and before the first forward. The scales are computed here once; call
    precompute_fp8_param_scales again after every later change to the weights.
    """
    _states.clear()
    _groups.clear()
    by_group: dict[int, tuple[dist.ProcessGroup, list[tuple[_FP8ParamState, int]]]] = {}
    for fsdp_module in model.modules():
        if not isinstance(fsdp_module, FSDPModule):
            continue
        param_group = fsdp_module._get_fsdp_state()._fsdp_param_group
        if param_group is None:
            continue
        for fsdp_param in param_group.fsdp_params:
            module = fsdp_param._module_info.module
            if not isinstance(module, fp8_classes) or fsdp_param._module_info.param_name != "weight":  # fmt: skip
                continue
            assert fsdp_param.fsdp_placement.dim == 0
            shape = fsdp_param._orig_size
            assert len(shape) in (2, 3), shape
            mesh_info = fsdp_param.mesh_info
            rank, size = mesh_info.shard_mesh_rank, mesh_info.shard_mesh_size
            chunk = (shape[0] + size - 1) // size
            offset = min(rank * chunk, shape[0])
            G, N, K = (1, *shape) if len(shape) == 2 else shape
            grid = (G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)
            device = fsdp_param.sharded_param.device
            scale = torch.empty(grid, dtype=torch.float32, device=device)
            scale_t = torch.empty(grid[0], grid[2], grid[1], dtype=torch.float32, device=device)
            if len(shape) == 2:
                scale, scale_t = scale[0], scale_t[0]
            module.register_buffer("weight_scale", scale, persistent=False)
            module.register_buffer("weight_scale_t", scale_t, persistent=False)
            module.fp8_param_gather = True
            row_offset, group_offset = (offset, 0) if len(shape) == 2 else (0, offset)
            state = _FP8ParamState(module, fsdp_param, shape, row_offset, group_offset, torch.empty(0))  # fmt: skip
            group = mesh_info.shard_process_group
            entry = by_group.setdefault(id(group), (group, []))
            entry[1].append((state, G * grid[1] * grid[2]))
            local = fsdp_param._sharded_local_tensor
            local.fsdp_pre_all_gather = state.pre_all_gather
            local.fsdp_post_all_gather = state.post_all_gather
            fsdp_param._init_extensions()
            _states.append(state)

    # One packed amax buffer per shard group, so each group reduces in a single collective.
    for group, entries in by_group.values():
        device = entries[0][0].fsdp_param.sharded_param.device
        buffer = torch.zeros(sum(n for _, n in entries), dtype=torch.float32, device=device)
        start = 0
        for state, n in entries:
            G, N, K = (1, *state.shape) if len(state.shape) == 2 else state.shape
            state.amax = buffer[start : start + n].view(G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)  # fmt: skip
            start += n
        _groups.append((group, buffer, [s for s, _ in entries]))

    precompute_fp8_param_scales()


@torch.no_grad()
def precompute_fp8_param_scales() -> None:
    """
    Recompute the FP8 scales and the FP8 shard of every installed weight from its fp32 shard.

    Collective over each FSDP shard group, so every rank must call it at the same point.
    """
    for group, buffer, states in _groups:
        buffer.zero_()
        launches = _multi_launches(states)
        if launches is not None:
            launches[0]()
        else:
            for state in states:
                shard, amax = state.local_shard(), state.amax
                fp8_shard_amax(shard, amax, state.row_offset, state.group_offset)
        if group.size() > 1:
            dist.all_reduce(buffer, op=dist.ReduceOp.MAX, group=group)
        if launches is not None:
            launches[1]()
        for state in states:
            if launches is None:
                fp8_scales(state.amax, *_scale_grids(state.module))
            state.cast_shard()
            state.version = state.fsdp_param._sharded_local_tensor._version


def _scale_grids(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """The module's scale buffers as (G, R, C) and (G, C, R)."""
    scale, scale_t = module.weight_scale, module.weight_scale_t
    if scale.dim() == 2:
        scale, scale_t = scale.unsqueeze(0), scale_t.unsqueeze(0)
    return scale, scale_t


_multi_cache: dict[int, tuple[tuple[int, ...], tuple[Any, Any]]] = {}


def _multi_launches(states: list[_FP8ParamState]) -> Optional[tuple[Any, Any]]:
    """
    One amax launch and one scales launch covering every weight of a shard group (CuTe-DSL, with a
    device table of pointers built once), or None where the kernels have no such launch (off SM90),
    so the caller loops over the weights. Rebuilt if any shard moved, e.g. after an RL offload.
    """
    shards = [s.local_shard() for s in states]
    key = tuple(t.data_ptr() for t in shards)
    cached = _multi_cache.get(id(states))
    if cached is None or cached[0] != key:
        amaxes = [s.amax for s in states]
        grids = [_scale_grids(s.module) for s in states]
        amax_launch = fp8_shard_amax_multi(
            shards, amaxes, [s.row_offset for s in states], [s.group_offset for s in states]
        )
        if amax_launch is None:
            return None
        scales_launch = fp8_scales_multi(amaxes, [g[0] for g in grids], [g[1] for g in grids])
        cached = (key, (amax_launch, scales_launch))
        _multi_cache[id(states)] = cached
    return cached[1]


# ---------------------------------------------------------------------------
# Raw expert parameters (gpt-oss's gate_up_proj / down_proj)
# ---------------------------------------------------------------------------
#
# The FP8 parameter all-gather above for expert weights kept as raw
# parameters, such as gpt-oss's ``gate_up_proj`` / ``down_proj``.
#
# ``install_fp8_param_gather`` wraps the ``weight`` of FP8Linear / FP8GroupedLinear modules and hangs
# the gathered FP8 operands off that module as ``weight_fp8``, ``weight_scale`` and so on. A module
# holding several raw 3-D expert weights cannot take that naming, and turning the weights into
# submodules would rename them in every checkpoint (``experts.gate_up_proj`` ->
# ``experts.gate_up_proj.weight``). Instead, a module lists its raw FP8 weights in a class attribute
# ``fp8_raw_params``, and each gets a small holder module (kept outside the module tree, so neither
# ``state_dict`` nor ``named_modules`` sees it) that plays the part of the FP8 module for
# the gather above: the same ``_FP8ParamState`` hooks, the same packed per-shard-group amax reduction,
# recomputed by the same ``precompute_fp8_param_scales``. FQNs, checkpoints and the optimizer are
# unchanged.
#
# The module reads the operands with :func:`fp8_param_operands`, which returns ``None`` when the


_HOLDERS = "_fp8_param_holders"


class _Holder(nn.Module):
    """Stands in for an FP8 module: the gather registers the operand buffers on it."""

    fp8_param_gather: bool = True


def fp8_param_operands(module: nn.Module, name: str) -> Optional[tuple[torch.Tensor, ...]]:
    """(fp8, scale, fp8_t, scale_t) of the raw weight ``name`` when it is gathered in FP8."""
    holder = module.__dict__.get(_HOLDERS, {}).get(name)
    if holder is None:
        return None
    return holder.weight_fp8, holder.weight_scale, holder.weight_t_fp8, holder.weight_scale_t


def install_raw_fp8_param_gather(model: nn.Module) -> None:
    """
    Gather in FP8 every raw parameter named in a module's ``fp8_raw_params``.

    Call after ``install_fp8_param_gather`` (which resets the registry and must therefore
    run first), after fully_shard and before the first forward. Collective over each shard group.
    """
    by_group: dict[int, tuple[dist.ProcessGroup, list[tuple[_FP8ParamState, int]]]] = {}
    for fsdp_module in model.modules():
        if not isinstance(fsdp_module, FSDPModule):
            continue
        param_group = fsdp_module._get_fsdp_state()._fsdp_param_group
        if param_group is None:
            continue
        for fsdp_param in param_group.fsdp_params:
            module = fsdp_param._module_info.module
            name = fsdp_param._module_info.param_name
            if name not in getattr(module, "fp8_raw_params", ()):
                continue
            assert fsdp_param.fsdp_placement.dim == 0
            shape = fsdp_param._orig_size
            assert len(shape) == 3, shape
            mesh_info = fsdp_param.mesh_info
            rank, size = mesh_info.shard_mesh_rank, mesh_info.shard_mesh_size
            chunk = (shape[0] + size - 1) // size
            offset = min(rank * chunk, shape[0])
            G, N, K = shape
            grid = (G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)
            device = fsdp_param.sharded_param.device
            holder = _Holder()
            holder.register_buffer("weight_scale", torch.empty(grid, dtype=torch.float32, device=device), persistent=False)  # fmt: skip
            holder.register_buffer("weight_scale_t", torch.empty(G, grid[2], grid[1], dtype=torch.float32, device=device), persistent=False)  # fmt: skip
            # Plain dict in __dict__: not a submodule, so invisible to state_dict and FSDP.
            module.__dict__.setdefault(_HOLDERS, {})[name] = holder
            # A raw parameter's FP8 shard is released after each gather, as for every FP8 weight;
            # that matters most for expert weights that the expert-dp axis does not shard (EP8 on
            # one node: the shard is the whole local weight, 2.4 GB of gpt-oss-20b).
            state = _FP8ParamState(holder, fsdp_param, shape, 0, offset, torch.empty(0))
            group = mesh_info.shard_process_group
            entry = by_group.setdefault(id(group), (group, []))
            entry[1].append((state, G * grid[1] * grid[2]))
            local = fsdp_param._sharded_local_tensor
            local.fsdp_pre_all_gather = state.pre_all_gather
            local.fsdp_post_all_gather = state.post_all_gather
            fsdp_param._init_extensions()
            _states.append(state)

    for group, entries in by_group.values():
        device = entries[0][0].fsdp_param.sharded_param.device
        buffer = torch.zeros(sum(n for _, n in entries), dtype=torch.float32, device=device)
        start = 0
        for state, n in entries:
            G, N, K = state.shape
            state.amax = buffer[start : start + n].view(G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)  # fmt: skip
            start += n
        _groups.append((group, buffer, [s for s, _ in entries]))

    if by_group:
        precompute_fp8_param_scales()
