import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["fp8_wgrad", "fp8_wgrad_supported"]


def fp8_wgrad_supported() -> bool:
    """
    Whether the device supports the fresh FP32 FP8 weight-gradient kernel (SM90).
    """
    return ARCH_MAJOR == 9


def fp8_wgrad(grad: torch.Tensor, grad_scale: torch.Tensor, activation: torch.Tensor, activation_scale: torch.Tensor, out: torch.Tensor) -> None:
    """
    Overwrite row-major FP32 out (M, N) with the FP8 weight gradient grad @ activation.T.

    grad (M, K) and activation (N, K) are K-major e4m3 on the same CUDA device. K is a multiple
    of 16 and N a multiple of 8. grad_scale is FP32 (M, ceil(K / 128)), with a scale per row
    per 128 tokens. activation_scale is FP32 (ceil(N / 128), ceil(K / 128)) for 128x128 scales,
    or (N, ceil(K / 128)) for per-row scales. Scale strides may be arbitrary. Inputs are read-only.
    SM90 only. Zero K writes positive zeros.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.fp8_wgrad_sm90 import fp8_wgrad_sm90
            return fp8_wgrad_sm90(grad, grad_scale, activation, activation_scale, out)
        case 10:
            raise NotImplementedError("fp8_wgrad requires SM90")
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
