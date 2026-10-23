"""
Weight-gradient accumulation into a layer-owned fp32 buffer.

BF16Linear, GroupedLinear and FP8GroupedLinear add their weight gradient straight into the fp32
main_grad buffer instead of writing a bf16 .grad that FSDP2 then adds into its own fp32 buffer. It
also keeps the sum across micro-batches in fp32 under DualPipeV, where FSDP2 would otherwise add
most micro-batches together in bf16. setup_model enables it for every run, BF16 or FP8.

install_wgrad_accumulation connects these layers to FSDP2 after fully_shard by extending two methods
of each parameter group:
- pre_backward allocates and zeroes main_grad at the first backward of the step.
- post_backward, when the group reduces, hands main_grad to FSDP2 as the accumulated gradient of the
  parameter and releases its memory once FSDP2 has copied it into the reduce-scatter input.

main_grad keeps the same tensor object across steps, so compiled graphs that take it as an input
stay valid. Without install, main_grad is None and the layer behaves like its torch.nn version.
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from pithtrain.operators.grouped_linear import FP8GroupedLinear, GroupedLinear
from pithtrain.operators.linear import BF16Linear

__all__ = ["install_wgrad_accumulation"]


def install_wgrad_accumulation(model: nn.Module) -> None:
    """
    Give every BF16Linear, GroupedLinear and FP8GroupedLinear weight in the model a main_grad,
    after fully_shard.
    """
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        for group in fully_shard.state(module)._fsdp_param_groups:
            assert isinstance(group, FSDPParamGroup)
            pairs = []
            for fsdp_param in group.fsdp_params:
                assert isinstance(fsdp_param, FSDPParam)
                owner = fsdp_param._module_info.module
                if fsdp_param._module_info.param_name != "weight":
                    continue
                if not isinstance(owner, (BF16Linear, GroupedLinear, FP8GroupedLinear)):
                    continue
                size, device = fsdp_param._orig_size, fsdp_param.device
                owner.main_grad = torch.empty(size, dtype=torch.float32, device=device)
                if isinstance(owner, FP8GroupedLinear) and _wgrad_runs_transposed(size):
                    # The SM90 k-grouped wgrad computes this weight's gradient transposed (see
                    # grouped_linear._k_grouped_fp8_gemm_nt); keep main_grad in that layout so it
                    # accumulates in place instead of transposing twice per micro-batch.
                    g, n, k = size
                    owner.main_grad = torch.empty((g, k, n), dtype=torch.float32, device=device).transpose(1, 2)  # fmt: skip
                owner.main_grad.untyped_storage().resize_(0)
                pairs.append((fsdp_param, owner.main_grad))
            if pairs:
                _attach_main_grads(group, pairs)


def _wgrad_runs_transposed(size: torch.Size) -> bool:
    """
    Whether the Hopper k-grouped FP8 wgrad of a (G, N, K) weight takes the transposed branch of
    grouped_linear's deadlock guard: odd 128-block count on N, even on K.
    """
    if torch.cuda.get_device_capability()[0] != 9 or len(size) != 3:
        return False
    _, n, k = size
    return -(-n // 128) % 2 == 1 and -(-k // 128) % 2 == 0


def _attach_main_grads(group: FSDPParamGroup, pairs: List[Tuple[FSDPParam, torch.Tensor]]) -> None:
    """
    Make the group allocate main_grad at the start of backward and reduce it in place of a .grad.
    """
    orig_pre_backward, orig_post_backward = group.pre_backward, group.post_backward

    def pre_backward(*args: Any) -> None:
        for _, main_grad in pairs:
            if main_grad.untyped_storage().size() == 0:
                main_grad.untyped_storage().resize_(main_grad.numel() * main_grad.element_size())
                main_grad.zero_()
        orig_pre_backward(*args)

    def post_backward(*args: Any) -> None:
        if not group.reduce_grads:
            orig_post_backward(*args)
            return
        # FSDP2 reduces a group in one dtype, so move the bf16 gradients into fp32 first.
        for fsdp_param in group.fsdp_params:
            if hasattr(fsdp_param, "_unsharded_param"):
                fsdp_param.accumulate_unsharded_grad_if_needed()
                fsdp_param.to_accumulated_grad_if_needed()
        # Keep any gradient the weight got outside its layer, e.g. kv_b_proj in MLA ring attention.
        for fsdp_param, main_grad in pairs:
            if main_grad.untyped_storage().size() > 0:
                if fsdp_param.unsharded_accumulated_grad is not None:
                    main_grad.add_(fsdp_param.unsharded_accumulated_grad)
                # A transposed main_grad is made contiguous once per step for the reduce-scatter.
                fsdp_param.unsharded_accumulated_grad = main_grad.contiguous()
        orig_post_backward(*args)
        # FSDP2 has copied main_grad into the reduce-scatter input on this stream.
        for _, main_grad in pairs:
            main_grad.untyped_storage().resize_(0)

    group.pre_backward = pre_backward
    group.post_backward = post_backward
