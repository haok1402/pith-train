"""
fp32 main_grad accumulation for expert weights kept as raw parameters (gpt-oss's gate_up_proj /
down_proj), the counterpart of ``wgrad_accumulation.install_wgrad_accumulation`` for modules that
list them in ``fp8_raw_params``.

Without it the FP8 expert wgrad writes a bf16 ``.grad`` per micro-batch and FSDP2 adds it into an
fp32 accumulator of its own, so a step holds the bf16 gradient (2 B/param) next to the fp32 sum
(4 B/param). With it the k-grouped FP8 wgrad's fp32 result is added straight into ``main_grad``,
which FSDP2 reduces in place of a ``.grad``: 2 B/param less (4.5 GiB for gpt-oss-20b at EP8 on one
node), and the per-micro-batch gradients are no longer rounded to bf16 before they are summed
(the same change wgrad_accumulation made for GroupedLinear / FP8GroupedLinear).

The buffers hang off the module in ``__dict__`` (not parameters, buffers or submodules), and the
FSDP wiring is wgrad_accumulation's: allocated at the first backward of the step, handed to FSDP2
as the accumulated gradient when the group reduces, released after.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard

from pithtrain.contexts import training
from pithtrain.operators.linear import ARCH_MAJOR
from pithtrain.operators.wgrad_accumulation import _attach_main_grads

__all__ = ["install_raw_main_grad", "raw_main_grad"]

_MAIN_GRADS = "_raw_main_grads"


def raw_main_grad(module: nn.Module, name: str) -> Optional[torch.Tensor]:
    """The fp32 main_grad of raw parameter ``name`` if installed (None otherwise). Its storage is
    only allocated from the first backward of a step on, so users check it when they write."""
    return module.__dict__.get(_MAIN_GRADS, {}).get(name)


def install_raw_main_grad(model: nn.Module) -> None:
    """Give every raw parameter named in a module's ``fp8_raw_params`` an fp32 main_grad, after
    fully_shard (alongside install_wgrad_accumulation). Only the Hopper FP8 expert path
    (fp8_gptoss) and the BF16 CuTe experts (bf16_cute) write it, so nothing is installed
    otherwise."""
    from pithtrain.operators import bf16_cute

    if not ((training.fp8 and ARCH_MAJOR == 9) or bf16_cute.enabled()):
        return
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        for group in fully_shard.state(module)._fsdp_param_groups:
            pairs = []
            for fsdp_param in group.fsdp_params:
                owner = fsdp_param._module_info.module
                name = fsdp_param._module_info.param_name
                if name not in getattr(owner, "fp8_raw_params", ()):
                    continue
                mg = torch.empty(fsdp_param._orig_size, dtype=torch.float32, device=fsdp_param.device)  # fmt: skip
                mg.untyped_storage().resize_(0)
                owner.__dict__.setdefault(_MAIN_GRADS, {})[name] = mg
                pairs.append((fsdp_param, mg))
            if pairs:
                _attach_main_grads(group, pairs)
