import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["fp8_wgrad", "fp8_wgrad_supported"]

BLOCK_K = 128


def fp8_wgrad_supported() -> bool:
    """
    Whether the device supports the fresh FP32 FP8 weight-gradient kernel (SM90).
    """
    return ARCH_MAJOR == 9


def tma_rows(tensor: torch.Tensor) -> bool:
    """
    Whether a nonempty row-major tensor has a TMA-compatible base address and row pitch.
    """
    pitch = tensor.stride(0) * tensor.element_size()
    return tensor.stride(1) == 1 and tensor.stride(0) >= tensor.shape[1] and pitch < 2**40 and pitch % 16 == 0 and tensor.data_ptr() % 16 == 0


def fp8_wgrad_fallback(grad: torch.Tensor, grad_scale: torch.Tensor, activation: torch.Tensor, activation_scale: torch.Tensor, out: torch.Tensor, rowwise: bool) -> None:
    """
    FP32 wgrad from the original FP8 codes and scales, applying one scale product per K block.
    """
    if not rowwise:
        activation_scale = activation_scale.repeat_interleave(BLOCK_K, dim=0)[:activation.shape[0]]
    out.zero_()
    with torch.autocast(device_type="cuda", enabled=False):
        for block, start in enumerate(range(0, grad.shape[1], BLOCK_K)):
            span = slice(start, start + BLOCK_K)
            product = grad[:, span].float() @ activation[:, span].float().T
            scale = grad_scale[:, block, None] * activation_scale[None, :, block]
            out.addcmul_(product, scale)
    # Match the kernel's canonical positive zero.
    out.add_(0.0)


def fp8_wgrad(grad: torch.Tensor, grad_scale: torch.Tensor, activation: torch.Tensor, activation_scale: torch.Tensor, out: torch.Tensor) -> None:
    """
    Overwrite row-major FP32 out (M, N) with the FP8 weight gradient grad @ activation.T.

    grad (M, K) and activation (N, K) are e4m3 on the output's CUDA device. grad_scale is FP32
    (M, ceil(K / 128)); activation_scale is FP32 (ceil(N / 128), ceil(K / 128)) for 128x128 scales
    or (N, ceil(K / 128)) for per-row scales. Inputs are read-only and must not alias out.
    On SM90, use CuTe with aligned K-major rows (including padded K tails), N divisible by eight
    and an even visible SM count. Other layouts and SM budgets use an FP32 fallback.
    Zero K writes positive zeros.
    """
    if not fp8_wgrad_supported():
        raise NotImplementedError("fp8_wgrad requires SM90")
    if grad.ndim != 2 or activation.ndim != 2:
        raise ValueError("FP8 wgrad requires 2-D FP8 operands")
    M, K = grad.shape
    N = activation.shape[0]
    blocks = (K + BLOCK_K - 1) // BLOCK_K
    rowwise = activation_scale.shape == (N, blocks)
    if activation.shape[1] != K or out.shape != (M, N):
        raise ValueError("FP8 wgrad requires matching (M, K), (N, K) and (M, N) shapes")
    if grad_scale.shape != (M, blocks) or not (rowwise or activation_scale.shape == ((N + BLOCK_K - 1) // BLOCK_K, blocks)):
        raise ValueError("FP8 wgrad requires one scale per row or activation tile per 128 tokens")
    if grad.dtype != torch.float8_e4m3fn or activation.dtype != grad.dtype:
        raise TypeError("FP8 wgrad requires E4M3 operands")
    if any(tensor.dtype != torch.float32 for tensor in (grad_scale, activation_scale, out)):
        raise TypeError("FP8 wgrad requires FP32 scales and output")
    if not out.is_cuda or any(tensor.device != out.device for tensor in (grad, grad_scale, activation, activation_scale)):
        raise ValueError("FP8 wgrad requires operands, scales and output on the same CUDA device")
    if not M or not N:
        return
    if (N > 1 and out.stride(1) != 1) or (M > 1 and out.stride(0) < N):
        raise ValueError("FP8 wgrad requires a nonoverlapping row-major output")
    if not K:
        out.zero_()
        return
    # Storage sharing can race the CuTe stores and can also corrupt later fallback K blocks.
    if any(torch._C._overlaps(out, tensor) for tensor in (grad, grad_scale, activation, activation_scale)):
        raise ValueError("FP8 wgrad output must not share storage with its operands or scales")
    if N % 8 == 0 and max(M, N, K) < 2**31 and all(tma_rows(tensor) for tensor in (grad, activation, out)):
        num_sms = torch.cuda.get_device_properties(out.device).multi_processor_count
        if num_sms >= 2 and num_sms % 2 == 0:
            from pithtrain.kernels.fp8_wgrad_sm90 import fp8_wgrad_sm90, fp8_wgrad_sm90_can_tile
            if fp8_wgrad_sm90_can_tile(M, blocks, rowwise):
                return fp8_wgrad_sm90(grad, grad_scale, activation, activation_scale, out, num_sms=num_sms)
    fp8_wgrad_fallback(grad, grad_scale, activation, activation_scale, out, rowwise)
