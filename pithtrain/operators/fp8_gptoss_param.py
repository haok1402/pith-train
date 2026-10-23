"""
FP8 parameter all-gather (``pithtrain/operators/fp8_param.py``) for expert weights kept as raw
parameters, such as gpt-oss's ``gate_up_proj`` / ``down_proj``.

``install_fp8_param_gather`` wraps the ``weight`` of FP8Linear / FP8GroupedLinear modules and hangs
the gathered FP8 operands off that module as ``weight_fp8``, ``weight_scale`` and so on. A module
holding several raw 3-D expert weights cannot take that naming, and turning the weights into
submodules would rename them in every checkpoint (``experts.gate_up_proj`` ->
``experts.gate_up_proj.weight``). Instead, a module lists its raw FP8 weights in a class attribute
``fp8_raw_params``, and each gets a small holder module (kept outside the module tree, so neither
``state_dict`` nor ``named_modules`` sees it) that plays the part of the FP8 module for
``fp8_param``: the same ``_FP8ParamState`` hooks, the same packed per-shard-group amax reduction,
recomputed by the same ``precompute_fp8_param_scales``. FQNs, checkpoints and the optimizer are
unchanged.

The module reads the operands with :func:`fp8_param_operands`, which returns ``None`` when the
weight is not installed (flag off), so the caller falls back to its own quantization cache.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from pithtrain.operators import fp8_param
from pithtrain.operators.fp8_param import BLOCK, _FP8ParamState

__all__ = ["install_raw_fp8_param_gather", "fp8_param_operands"]

_HOLDERS = "_fp8_param_holders"


class _Holder(nn.Module):
    """Stands in for an FP8 module: fp8_param registers the operand buffers on it."""

    fp8_param_gather: bool = True


class _RawFP8ParamState(_FP8ParamState):
    """
    _FP8ParamState for a raw parameter. Its FP8 shard is released after each gather by the base
    class, as for every FP8 weight.

    Releasing the shard matters most for expert weights that the expert-dp axis does not shard
    (EP8 on one node: the shard is the whole local weight, 2.4 GB of gpt-oss-20b).
    """

    pass


def fp8_param_operands(module: nn.Module, name: str) -> Optional[tuple[torch.Tensor, ...]]:
    """(fp8, scale, fp8_t, scale_t) of the raw weight ``name`` when it is gathered in FP8."""
    holder = module.__dict__.get(_HOLDERS, {}).get(name)
    if holder is None:
        return None
    return holder.weight_fp8, holder.weight_scale, holder.weight_t_fp8, holder.weight_scale_t


def install_raw_fp8_param_gather(model: nn.Module) -> None:
    """
    Gather in FP8 every raw parameter named in a module's ``fp8_raw_params``.

    Call after ``install_fp8_param_gather`` (which resets fp8_param's registry and must therefore
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
            state = _RawFP8ParamState(holder, fsdp_param, shape, 0, offset, torch.empty(0))
            group = mesh_info.shard_process_group
            entry = by_group.setdefault(id(group), (group, []))
            entry[1].append((state, G * grid[1] * grid[2]))
            local = fsdp_param._sharded_local_tensor
            local.fsdp_pre_all_gather = state.pre_all_gather
            local.fsdp_post_all_gather = state.post_all_gather
            fsdp_param._init_extensions()
            fp8_param._states.append(state)

    for group, entries in by_group.values():
        device = entries[0][0].fsdp_param.sharded_param.device
        buffer = torch.zeros(sum(n for _, n in entries), dtype=torch.float32, device=device)
        start = 0
        for state, n in entries:
            G, N, K = state.shape
            state.amax = buffer[start : start + n].view(G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)  # fmt: skip
            start += n
        fp8_param._groups.append((group, buffer, [s for s, _ in entries]))

    if by_group:
        fp8_param.precompute_fp8_param_scales()
