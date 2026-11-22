"""Round an FP32 parameter for BF16 compute without rounding its gradient."""

import torch


@torch.library.custom_op("pithtrain::stock_bf16_value", mutates_args=())
def stock_bf16_value(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.to(torch.bfloat16).float()


@stock_bf16_value.register_fake
def _fake_stock_bf16_value(parameter):
    return torch.empty_like(parameter, dtype=torch.float32)


def _backward(ctx, grad):
    return grad


stock_bf16_value.register_autograd(_backward)
