"""
FP8 routed experts of gpt-oss on Hopper, as one autograd node.

gpt-oss keeps its experts as raw parameters (an interleaved gate_up projection with a bias, a down
projection with a bias) and activates with a clamped SwiGLU, so it cannot use the SwiGLU experts
of ``grouped_linear.FP8SwiGLUExpertsFunc``. This is the same design for its layout:

    forward:  x -FP8-> gate_up GEMM -> [bias + clamped SwiGLU + FP8 quantization] -> down GEMM
    backward: dy -FP8-> dh GEMM -> [bias + clamped SwiGLU backward + FP8 quantization
              + bias-gradient partials] -> dx GEMM, both wgrads k-grouped from the saved FP8

The bracketed steps are single kernels (``fp8_cute/clamped_swiglu_quant.py``), so the biased
gate_up, the activation h and the gate_up gradient never reach HBM in bf16. The down bias is added
in place into the down GEMM's output (one fp32 add rounded to bf16, as ``indexed_bias_add`` does,
without materializing ``bias[group_ids]``); its gradient is ``indexed_bias_add``'s segment sum.
Every FP8 operand is bit-identical to the unfused path (FP8GroupedLinearFunc -> indexed_bias_add
-> clamped_swiglu -> FP8GroupedLinearFunc -> indexed_bias_add), so the forward output, the
weight/input gradients and the down bias gradient are too; the gate_up bias gradient sums the same
bf16 values per expert in a different fp32 order.
"""

import torch
import triton
import triton.language as tl

from pithtrain.operators.deepgemm_quantize import fp8cast_rowwise_kmajor
from pithtrain.operators.grouped_linear import (
    _fp8_grouped_gemm,
    _k_grouped_fp8_gemm_nt,
    quantize_grouped_input,
)
from pithtrain.operators.indexed_bias_add import _segment_sum_along_rows
from pithtrain.pipeline.execution import WeightGradStore


def _kgrouped_safe(n: int) -> int:
    """
    Channel count to lay out a k-grouped wgrad operand with so DeepGEMM can take it as the first
    (M-side) operand: a multiple of 256.

    DeepGEMM's SM90 k-grouped FP8 GEMM deadlocks (the kernel never finishes, so the next launch
    or sync hangs) whenever ceil(M / 128) * ceil(N / block_n) is odd: it runs 2-CTA clusters
    (cluster_m = 2) and an odd tile count leaves one CTA without its partner. block_n depends on
    the problem (136 or 152 for gpt-oss), but block_m is 128, so an even M-block count is always
    safe. gpt-oss hits it with its 2880-wide hidden and intermediate (23 M-blocks; 19 or 7
    N-blocks at block_n = 152), measured on H200 with workspace/kda-tasks/round3/gptoss/scripts/
    kg_case.py: e.g. (M, N) = (2880, 2880), (2880, 1024), (5760, 2880) hang, (2880, 2944),
    (2944, 2944), (1024, 2880), (2880, 5760) complete.
    """
    return -(-n // 256) * 256


def _fp8_wgrad(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    ks: list,
    ks_tensor: torch.Tensor,
    weight: torch.Tensor,
    main_grad: torch.Tensor | None = None,
) -> None:
    """
    weight.grad += a^T b per expert (into main_grad in fp32 instead when it holds storage, see
    fp8_gptoss_wgrad) (k-grouped FP8 GEMM, deferred to DualPipeV's W phase like
    grouped_linear._fp8_grouped_wgrad), for weight (G, N, K), a's K-major channels NA >= N and
    b's KB >= K (either may be laid out padded, see _kgrouped_safe). The operand with a multiple
    of 256 channels goes first, so DeepGEMM's tile count is even; if that is b, the product is
    computed transposed (b^T a).
    """
    G, N, K = weight.shape
    NA, KB = a[1].shape[0], b[1].shape[0]
    swap = NA % 256 != 0 and KB % 256 == 0

    def fn(a, b, ks, ks_tensor):
        if swap:
            c = torch.zeros((G, KB, NA), device=a[0].device, dtype=torch.float32)
            _k_grouped_fp8_gemm_nt(b, a, c, ks, ks_tensor, c=c)
            grad = c[:, :K, :N].transpose(1, 2)
        else:
            c = torch.zeros((G, NA, KB), device=a[0].device, dtype=torch.float32)
            _k_grouped_fp8_gemm_nt(a, b, c, ks, ks_tensor, c=c)
            grad = c[:, :N, :K]
        if main_grad is not None and main_grad.untyped_storage().size() > 0:
            main_grad.add_(grad)
            return
        grad = grad.to(torch.bfloat16)
        if weight.grad is None:
            weight.grad = grad.contiguous()
        else:
            weight.grad += grad

    if WeightGradStore.enabled:
        a = (a[0].detach(), a[1].detach())
        b = (b[0].detach(), b[1].detach())
        WeightGradStore.put(lambda: fn(a, b, ks, ks_tensor.detach()))
    else:
        fn(a, b, ks, ks_tensor)


@triton.jit
def _group_bias_add_kernel(y_ptr, bias_ptr, gi_ptr, H, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):  # fmt: skip
    """y[r] += bias[gi[r]] in place, each sum in fp32 rounded once to y's dtype."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols[None, :] < H
    g = tl.load(gi_ptr + rows).to(tl.int64)
    offs = rows.to(tl.int64)[:, None] * H + cols[None, :]
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(bias_ptr + g[:, None] * H + cols[None, :], mask=mask).to(tl.float32)
    tl.store(y_ptr + offs, (y + b).to(y_ptr.dtype.element_ty), mask=mask)


def _group_bias_add_(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """In-place ``y += bias[group_indices]``, bit-identical to ``y + bias[group_indices]`` for
    bf16 y and bias (M a multiple of 128, as grouped rows are)."""
    M, H = y.shape
    if y.dtype != bias.dtype or M % 128 != 0 or not y.is_contiguous() or not bias.is_contiguous():  # fmt: skip
        y.copy_(y + bias[group_indices])
        return
    _group_bias_add_kernel[(M // 32, triton.cdiv(H, 256))](y, bias, group_indices, H, BLOCK_M=32, BLOCK_H=256)  # fmt: skip


__all__ = ["FP8ClampedSwiGLUExpertsFunc", "fp8_clamped_swiglu_experts"]

RECOMPUTE_GATE_UP: bool = False  # True: recompute gate_up in backward, -3.1 GB, +5% step time
"""
Keep x's rowwise FP8 (1 B/element of hidden) instead of the bf16 gate_up GEMM output
(2 B/element of 2 * intermediate, 4x more for gpt-oss) and redo the gate_up GEMM in backward.
Bit-identical (the GEMM is deterministic); trades one extra grouped GEMM per layer and micro-batch
for 3 GiB at the gpt-oss-20b PP1/EP8 peak. Off by default.
"""


class FP8ClampedSwiGLUExpertsFunc(torch.autograd.Function):
    """
    gate_up (+bias) -> clamped SwiGLU -> down (no bias) for gpt-oss FP8 experts on Hopper. See the
    module docstring. With quantized_input (from fp8_scatter_quant), x is only a gradient carrier.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        w_gate_up: torch.Tensor,
        b_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor,
        q_gate_up: tuple[torch.Tensor, ...],
        q_down: tuple[torch.Tensor, ...],
        alpha: float,
        limit: float,
        quantized_input: tuple[torch.Tensor, ...] | None = None,
        main_grads: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
    ) -> torch.Tensor:
        from pithtrain.operators.fp8_cute.clamped_swiglu_quant import (
            clamped_swiglu_quant_rowwise,
        )

        if quantized_input is None:
            quantized_input = quantize_grouped_input(x, grouped_mm_offs)
        x_fp8, x_scale, x_ch, x_ch_scale = quantized_input
        two_inter, hidden = w_gate_up.shape[1], w_down.shape[1]
        offs, gi = grouped_mm_offs, group_indices
        gu = _fp8_grouped_gemm((x_fp8, x_scale), q_gate_up[:2], two_inter, offs, gi)
        # Only the rowwise h for the down GEMM: its K-major copy (down_proj's wgrad operand) is
        # recomputed from gu in backward rather than kept, the same bytes.
        h_fp8, h_scale = clamped_swiglu_quant_rowwise(gu, b_gate_up, offs, alpha, limit)
        y = _fp8_grouped_gemm((h_fp8, h_scale), q_down[:2], hidden, offs, gi)
        _group_bias_add_(y, b_down, gi)
        recompute = RECOMPUTE_GATE_UP
        saved = (x_fp8, x_scale, q_gate_up[0], q_gate_up[1]) if recompute else (gu,)
        ctx.save_for_backward(
            b_gate_up, x_ch, x_ch_scale, offs, ks_tensor,
            q_gate_up[2], q_gate_up[3], q_down[2], q_down[3], *saved,
        )  # fmt: skip
        ctx.recompute = recompute
        ctx.weights = (w_gate_up, w_down)
        ctx.main_grads = main_grads
        ctx.ks, ctx.group_indices = ks, gi
        ctx.alpha, ctx.limit = alpha, limit
        ctx.b_down_dtype = b_down.dtype
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        from pithtrain.operators.fp8_cute.clamped_swiglu_quant import (
            clamped_swiglu_bwd_quant,
            clamped_swiglu_quant_kmajor,
        )

        (
            b_gate_up, x_ch, x_ch_scale, offs, ks_tensor,
            wgu_t, sgu_t, wd_t, sd_t, *saved,
        ) = ctx.saved_tensors  # fmt: skip
        w_gate_up, w_down = ctx.weights
        ks, gi = ctx.ks, ctx.group_indices
        inter, hidden = w_down.shape[2], w_down.shape[1]
        if ctx.recompute:
            x_fp8, x_scale, wgu, sgu = saved
            gu = _fp8_grouped_gemm((x_fp8, x_scale), (wgu, sgu), w_gate_up.shape[1], offs, gi)
            del saved, x_fp8, x_scale
        else:
            (gu,) = saved

        dy = dy.contiguous()
        d_b_down = _segment_sum_along_rows(dy, offs, ctx.b_down_dtype)
        dy_fp8, dy_scale, dy_ch, dy_ch_scale = fp8cast_rowwise_kmajor(dy, offs)
        dh = _fp8_grouped_gemm((dy_fp8, dy_scale), (wd_t, sd_t), inter, offs, gi)
        # h's K-major copy, laid out with DeepGEMM-safe channels (see _kgrouped_safe).
        kp = _kgrouped_safe(inter)
        h_ch, h_ch_scale = clamped_swiglu_quant_kmajor(gu, b_gate_up, offs, ctx.alpha, ctx.limit, kp)  # fmt: skip
        mg_gate_up, mg_down = ctx.main_grads
        _fp8_wgrad((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), ks, ks_tensor, w_down, mg_down)
        del h_ch, h_ch_scale
        del dy_fp8, dy_scale, dy_ch, dy_ch_scale
        kp = _kgrouped_safe(w_gate_up.shape[1])
        dguq, d_bias = clamped_swiglu_bwd_quant(dh, gu, b_gate_up, offs, ctx.alpha, ctx.limit, kp)
        del dh
        dx = _fp8_grouped_gemm(dguq[:2], (wgu_t, sgu_t), hidden, offs, gi)
        _fp8_wgrad(dguq[2:], (x_ch, x_ch_scale), ks, ks_tensor, w_gate_up, mg_gate_up)
        return dx, None, d_bias, None, d_b_down, *([None] * 10)


def fp8_clamped_swiglu_experts(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    b_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    q_gate_up: tuple[torch.Tensor, ...],
    q_down: tuple[torch.Tensor, ...],
    alpha: float,
    limit: float,
    quantized_input: tuple[torch.Tensor, ...] | None = None,
    main_grads: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
) -> torch.Tensor:
    """
    down(clamped_swiglu(gate_up(x) + b_gate_up)) + b_down through FP8ClampedSwiGLUExpertsFunc.
    q_* are the (fp8, scale, fp8_t, scale_t) quantized weights.
    """
    return FP8ClampedSwiGLUExpertsFunc.apply(
        x, w_gate_up, b_gate_up, w_down, b_down, grouped_mm_offs, ks, ks_tensor, group_indices,
        q_gate_up, q_down, alpha, limit, quantized_input, main_grads,
    )  # fmt: skip
