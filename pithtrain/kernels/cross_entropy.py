import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["cross_entropy_fwd_bwd"]


def cross_entropy_fwd_bwd(inp: torch.Tensor, target: torch.Tensor, ignore_index: int, sum_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cross-entropy of the logits inp against target, overwriting the logits with the gradient of
    that mean. Returns (loss, grad).

    inp: (N, V) bf16, fp16 or fp32 logits with V >= 1 (SM90 raises ValueError for V == 0; SM100 does
    not check), overwritten in place when row-contiguous (otherwise a contiguous copy is overwritten
    and inp is left alone). target: integer (N,), any stride, each in [0, V) or ignore_index (any
    other value reads out of bounds); not modified. N * V may pass 2^31: both architectures offset
    rows in 64 bits, so a long pack runs as one launch over all N rows.

    loss: fp32 scalar, the sum over non-ignored rows of logsumexp(x) - x[target] divided by n =
    max(non-ignored rows, 1). grad: inp itself (or the copy), (softmax(x) - onehot(target)) / n per
    non-ignored row computed in fp32 and rounded once, zero rows where target == ignore_index.
    Matches F.cross_entropy(inp.float(), target, ignore_index=ignore_index) and its gradient to
    fp32 rounding; deterministic. With sum_grad, multiply the rounded grad by n rounded to inp's
    dtype, then round once more. SM90 BF16 performs the multiply in the kernel. The loss still
    returns the mean. No (N, V) temporary is allocated for row-contiguous inp.
    """
    scale_in_kernel = sum_grad and inp.dtype == torch.bfloat16 and ARCH_MAJOR == 9
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.cross_entropy_sm90 import cross_entropy_fwd_bwd_sm90
            loss, grad = cross_entropy_fwd_bwd_sm90(inp, target, ignore_index, scale_in_kernel)
        case 10:
            from pithtrain.kernels.cross_entropy_sm100 import cross_entropy_fwd_bwd_sm100
            loss, grad = cross_entropy_fwd_bwd_sm100(inp, target, ignore_index)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x")
    if sum_grad and not scale_in_kernel:
        grad.mul_((target != ignore_index).sum().clamp_min(1).float().to(grad.dtype))
    return loss, grad
