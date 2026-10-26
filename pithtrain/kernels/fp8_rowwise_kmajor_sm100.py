"""Triton implementation of fp8cast_rowwise_kmajor (contract in fp8_rowwise_kmajor), the reference the SM90 kernel is bit-identical to."""

import torch

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_rowwise_colwise import fp8cast_rowwise_colwise_kernel

__all__ = ["fp8cast_rowwise_kmajor_sm100"]


def fp8cast_rowwise_kmajor_sm100(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_kmajor in Triton; both scales come back row-major."""
    assert x.ndim == 2 and x.shape[0] % 128 == 0, (f"Expected 2D tensor with M%128==0, got shape {x.shape}")
    M, N = x.shape
    BLOCK = 128

    num_row_groups = M // BLOCK
    scale_tok_cols = (N + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    kmajor_fp8 = torch.empty(M * N, dtype=torch.float8_e4m3fn, device=x.device)
    kmajor_scale = torch.empty((N, num_row_groups), dtype=torch.float32, device=x.device)

    block_starts = torch.arange(0, M, BLOCK, device=x.device, dtype=torch.int64)
    block_to_group = torch.searchsorted(grouped_mm_offs.to(torch.int64), block_starts, right=True).to(torch.int32)
    block_to_group.clamp_(max=grouped_mm_offs.shape[0] - 1)

    _BLOCK_N = 128
    grid = (num_row_groups, (N + _BLOCK_N - 1) // _BLOCK_N)

    fp8cast_rowwise_colwise_kernel[grid](x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_ch_ptr=kmajor_fp8, scale_ch_ptr=kmajor_scale, M=M, N=N, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_tok_row=N, scale_tok_cols=scale_tok_cols, stride_out_ch_row=0, scale_ch_cols=num_row_groups, block_to_group_ptr=block_to_group, cumsum_ptr=grouped_mm_offs, BLOCK_N=_BLOCK_N, NATIVE_E8M0=NATIVE_E8M0, WRITE_KMAJOR=True, num_warps=4, num_stages=2)

    return out_tok, scale_tok, kmajor_fp8, kmajor_scale
