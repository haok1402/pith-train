from functools import partial
from typing import List, Optional

import deep_gemm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.kernels.fp8_gemm import (
    k_grouped_fp8_gemm_nt,
    m_grouped_fp8_gemm_nt,
    quantize_grouped_input,
)
from pithtrain.kernels.fp8_rowwise_colwise import fp8cast_rowwise_colwise
from pithtrain.kernels.fp8_rowwise_kmajor import fp8cast_rowwise_kmajor
from pithtrain.kernels.fp8_weight_quant import fp8cast_blockwise_transpose_batched


class GroupedLinearFunc(torch.autograd.Function):
    """
    Custom autograd Function for BF16 grouped linear layer (MoE experts).

    Forward: output      = grouped_mm(input, weight.T)      [2D-3D, jagged on M]
    Dgrad:   grad_input  = grouped_mm(grad_output, weight)  [2D-3D, jagged on M]
    Wgrad:   weight_grad = grouped_mm(grad_output.T, input) [2D-2D, jagged on K]

    The wgrad is split from dgrad so DualPipeV's zero-bubble W-phase can defer
    it via WeightGradStore, freeing the critical path during stage3_b.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        main_grad: Optional[torch.Tensor] = None,
        ks: Optional[List[int]] = None,
        ks_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)
        ctx.save_for_backward(input, weight, grouped_mm_offs)
        ctx.main_grad, ctx.ks, ctx.ks_tensor = main_grad, ks, ks_tensor
        return output

    @staticmethod
    def backward(ctx, dy):
        input, weight, offs = ctx.saved_tensors
        dgrad = F.grouped_mm(dy, weight, offs=offs)
        main_grad, ks, ks_tensor = ctx.main_grad, ctx.ks, ctx.ks_tensor

        def wgrad_fn(dy: torch.Tensor, x: torch.Tensor, offs: torch.Tensor) -> None:
            if main_grad is not None:
                dy, x = dy.contiguous(), x.contiguous()
                deep_gemm.k_grouped_bf16_gemm_tn_contiguous(dy, x, main_grad, ks, ks_tensor, c=main_grad)  # fmt: skip
                return
            wgrad = F.grouped_mm(dy.transpose(0, 1), x, offs=offs)
            weight.grad = wgrad if weight.grad is None else weight.grad.add_(wgrad)

        if WeightGradStore.enabled:
            WeightGradStore.put(partial(wgrad_fn, dy.detach(), input.detach(), offs.detach()))
        else:
            wgrad_fn(dy, input, offs)

        return dgrad, None, None, None, None, None


class GroupedLinear(nn.Module):
    """
    Grouped linear layer that partitions input data and applies a distinct
    linear transformation per group. This is useful for the MLP layers in
    the mixture-of-experts models.
    """

    main_grad: Optional[torch.Tensor] = None

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: Optional[List[int]] = None,
        ks_tensor: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
        quantized_input: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # Use a matmul instead of new_empty to preserve the autograd graph.
            # With 0 tokens the result is (0, out_features) and gradients are zero,
            # but the grad_fn must exist so that run_backward does not crash.
            weight = self.weight[0] if self.main_grad is None else self.weight[0].detach()
            return input @ weight.T
        if self.main_grad is not None and torch.is_grad_enabled():
            assert ks is not None and ks_tensor is not None
            return GroupedLinearFunc.apply(input, self.weight, grouped_mm_offs, self.main_grad, ks, ks_tensor)  # fmt: skip
        return GroupedLinearFunc.apply(input, self.weight, grouped_mm_offs)


def fp8_grouped_wgrad(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    ks: list,
    ks_tensor: torch.Tensor,
    weight: torch.Tensor,
    shape: tuple[int, int, int],
) -> None:
    """
    Weight gradient of an FP8 grouped linear from per-channel FP8 grad_output (a) and input (b),
    deferred to DualPipeV's W phase when WeightGradStore is on. With an fp32 main_grad on the
    weight (FP8GroupedLinear sets weight.main_grad when wgrad_accumulation installed one), the
    GEMM accumulates straight into it, so no bf16 .grad and no fp32 temporary exist; otherwise the
    result is added into weight.grad in bf16. Blackwell (TN, MN-Major) takes the per-channel data
    directly; on Hopper (NT, K-Major) the quantization kernel already wrote it K-major.
    """
    if ARCH_MAJOR >= 10:
        k_grouped_gemm = deep_gemm.k_grouped_fp8_gemm_tn_contiguous
    else:
        k_grouped_gemm = k_grouped_fp8_gemm_nt
    main_grad = getattr(weight, "main_grad", None)

    def grad_weight_fn(a, b, ks, ks_tensor):
        if main_grad is not None:
            k_grouped_gemm(a, b, main_grad, ks, ks_tensor, c=main_grad)
            return
        c = torch.zeros(shape, device=a[0].device, dtype=torch.float32)
        k_grouped_gemm(a, b, c, ks, ks_tensor, c=c)
        weight_grad_bf16 = c.to(torch.bfloat16)
        if weight.grad is None:
            weight.grad = weight_grad_bf16
        else:
            weight.grad += weight_grad_bf16

    if WeightGradStore.enabled:
        a = (a[0].detach(), a[1].detach())
        b = (b[0].detach(), b[1].detach())
        WeightGradStore.put(partial(grad_weight_fn, a, b, ks, ks_tensor.detach()))
    else:
        grad_weight_fn(a, b, ks, ks_tensor)


class FP8GroupedLinearFunc(torch.autograd.Function):
    """
    Custom autograd Function for FP8 grouped linear layer (MoE experts).

    Forward:  output = grouped_mm(input, weight.T)  via m_grouped FP8 GEMM NT
    Dgrad:    grad_input = grouped_mm(grad_output, weight_t.T)  via m_grouped FP8 GEMM NT
    Wgrad:    weight_grad = grouped_mm(grad_output.T, input)  via k_grouped FP8 GEMM
              (Blackwell: TN/MN-Major, Hopper: NT/K-Major)
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        quantized_weight: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        group_indices: torch.Tensor | None = None,
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ):
        weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = quantized_weight
        M, K = input.shape
        num_groups, N, _ = weight.shape

        # Quantize input to FP8 (fused: single read for both forward and wgrad), unless the caller
        # already did so for several projections of the same input (see quantize_grouped_input).
        if quantized_input is None:
            quantized_input = quantize_grouped_input(input, grouped_mm_offs)
        input_fp8, scale_input, input_ch_fp8, scale_input_ch = quantized_input

        assert ARCH_MAJOR >= 10 or group_indices is not None, (
            "group_indices is required on Hopper (SM90); call precompute_group_indices() once "
            "at the caller level and pass it to all grouped projections"
        )

        # Forward: m_grouped GEMM NT contiguous
        output = torch.empty((M, N), device=input.device, dtype=input.dtype)
        m_grouped_fp8_gemm_nt(
            (input_fp8, scale_input),
            (weight_fp8, scale_weight),
            output,
            grouped_mm_offs,
            M,
            group_indices=group_indices,
        )

        ctx.save_for_backward(
            weight_t_fp8, scale_weight_t, grouped_mm_offs, input_ch_fp8, scale_input_ch, ks_tensor
        )
        ctx.weight_ref = weight
        ctx.ks = ks
        ctx.group_indices = group_indices
        ctx.M = M
        ctx.K = K
        ctx.N = N
        ctx.num_groups = num_groups

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight_t_fp8, scale_weight_t, grouped_mm_offs, input_ch_fp8, scale_input_ch, ks_tensor = (
            ctx.saved_tensors
        )
        weight = ctx.weight_ref
        ks = ctx.ks
        group_indices = ctx.group_indices
        M = ctx.M
        K, N, num_groups = ctx.K, ctx.N, ctx.num_groups

        # Quantize grad_output (fused: single read).
        # Produces both per-token (for dgrad) and per-channel (for wgrad)
        # FP8 tensors in one pass, eliminating a redundant BF16 read.
        # On Hopper the colwise output is K-major (fused transpose).
        if ARCH_MAJOR >= 10:
            grad_fp8, scale_grad, grad_ch_fp8, scale_grad_ch = fp8cast_rowwise_colwise(grad_output)
        else:
            grad_fp8, scale_grad, grad_ch_fp8, scale_grad_ch = fp8cast_rowwise_kmajor(
                grad_output, grouped_mm_offs
            )

        # Dgrad: m_grouped GEMM NT contiguous with pre-transposed weight
        grad_input = torch.empty((M, K), device=grad_output.device, dtype=grad_output.dtype)
        m_grouped_fp8_gemm_nt(
            (grad_fp8, scale_grad),
            (weight_t_fp8, scale_weight_t),
            grad_input,
            grouped_mm_offs,
            M,
            group_indices=group_indices,
        )

        # Wgrad: k_grouped GEMM on the per-channel FP8 of grad_output and input.
        a_wgrad = (grad_ch_fp8, scale_grad_ch)
        b_wgrad = (input_ch_fp8, scale_input_ch)

        fp8_grouped_wgrad(a_wgrad, b_wgrad, ks, ks_tensor, weight, (num_groups, N, K))

        return grad_input, None, None, None, None, None, None, None


class FP8GroupedLinear(nn.Module):
    """
    FP8 grouped linear layer for MoE experts.

    Drop-in replacement for ``GroupedLinear`` using FP8 GEMM via DeepGEMM.
    Weight shape: ``(num_groups, out_features, in_features)``.
    Quantized weights are cached and reused across micro-batches within a single pipeline step.
    With fp8_param_gather set (see ``fp8_param``), the weight arrives already quantized.
    """

    fp8_param_gather: bool = False
    main_grad: Optional[torch.Tensor] = None
    """fp32 gradient buffer set by wgrad_accumulation; the wgrad GEMM accumulates into it."""

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))
        self._wq_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._wq_version: int = -1

    def _get_quantized_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Every FP8 expert path reads the weight through here, so this is where the weight that
        # the backward sees learns its main_grad (the unsharded parameter outlives the step).
        if self.main_grad is not None:
            self.weight.main_grad = self.main_grad
        if self.fp8_param_gather:
            return self.weight_fp8, self.weight_scale, self.weight_t_fp8, self.weight_scale_t
        if torch.compiler.is_compiling():
            return fp8cast_blockwise_transpose_batched(self.weight)
        ver = FP8WeightCacheControl.version
        if self._wq_version == ver:
            return self._wq_cache
        result = fp8cast_blockwise_transpose_batched(self.weight)
        self._wq_cache = result
        self._wq_version = ver
        return result

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor | None = None,
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # Preserve autograd graph with 0 tokens (same pattern as GroupedLinear).
            weight = self.weight[0] if self.main_grad is None else self.weight[0].detach()
            return input @ weight.T
        quantized_weight = self._get_quantized_weight()
        return FP8GroupedLinearFunc.apply(
            input, self.weight, grouped_mm_offs, ks, ks_tensor, quantized_weight, group_indices,
            quantized_input,
        )  # fmt: skip
