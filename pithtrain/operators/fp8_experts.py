"""
FP8 routed experts on Hopper as single autograd nodes, with the activation fused into the FP8
quantization of its neighbours, and the fused stage-3 input path that feeds them.

- :class:`FP8SwiGLUExpertsFunc` (``fp8_swiglu_experts``): separate gate / up / down
  FP8GroupedLinear weights (DeepSeek-V2-Lite, Qwen3), SwiGLU fused through the kernel library's
  ``silu_mul_quant_*``.
- :class:`FP8SwiGLUGateUpExpertsFunc` (``fp8_swiglu_gate_up_experts``): one fused (G, 2I, H) gate_up
  weight (Qwen3.5).
- :class:`FP8ClampedSwiGLUExpertsFunc` (``fp8_clamped_swiglu_experts``): gpt-oss's raw biased
  interleaved gate_up and down parameters and its clamped SwiGLU, fused through
  ``clamped_swiglu_quant_*``.
- :class:`FP8ScatterQuantFunc` (``fp8_scatter_quant``): received tokens -> quantized grouped input
  in one pass (``scatter_quant``), with ``scatter_quant_backward``.

Only where ``fp8_fused_experts_supported()`` (the models check it); elsewhere FP8 training takes
the unfused FP8GroupedLinear path.

Every FP8 operand is bit-identical to the unfused path (FP8GroupedLinear -> activation ->
FP8GroupedLinear), so the outputs and gradients are too, except where noted (gpt-oss's gate_up
bias gradient sums the same bf16 values in a different fp32 order).

The gpt-oss experts: the down bias is added in place into the down GEMM's output (one fp32 add
rounded to bf16, as ``indexed_bias_add`` does, without materializing ``bias[group_ids]``); its
gradient is ``indexed_bias_add``'s segment sum. Their wgrads go through :func:`fp8_padded_wgrad`,
whose operands are laid out padded to the channel counts the k-grouped GEMM takes without a copy.
"""

from typing import Optional

import torch

from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels import (
    clamped_swiglu_bwd_quant,
    clamped_swiglu_quant_kmajor,
    clamped_swiglu_quant_rowwise,
    fp8_m_grouped_gemm,
    group_bias_add_,
    k_grouped_fp8_gemm,
    quantize_grouped_input,
    scatter_quant,
    segment_sum_along_rows,
    silu_mul_bwd_quant,
    silu_mul_bwd_quant_gu,
    silu_mul_quant_fwd,
    silu_mul_quant_gu_fwd,
)
from pithtrain.operators.grouped_linear import FP8GroupedLinear, fp8_grouped_wgrad

__all__ = [
    "FP8ClampedSwiGLUExpertsFunc",
    "FP8ScatterQuantFunc",
    "FP8SwiGLUExpertsFunc",
    "FP8SwiGLUGateUpExpertsFunc",
    "RECOMPUTE_GATE_UP",
    "fp8_clamped_swiglu_experts",
    "fp8_padded_wgrad",
    "fp8_scatter_quant",
    "fp8_swiglu_experts",
    "fp8_swiglu_gate_up_experts",
    "scatter_quant_backward",
]


# ---------------------------------------------------------------------------
# SwiGLU experts, separate gate / up weights
# ---------------------------------------------------------------------------


class FP8SwiGLUExpertsFunc(torch.autograd.Function):
    """
    gate_proj, up_proj -> SwiGLU -> down_proj for FP8 routed experts on Hopper, as one autograd node.

    SwiGLU is fused into the quantization of its neighbours (``silu_mul_quant_fwd`` and co.): forward
    quantizes h for down_proj straight from g and u, and backward quantizes dg and du for the
    gate/up GEMMs straight from dh, so h, dg and du never reach HBM in bf16. The x quantization is
    shared by gate_proj and up_proj. Every kernel is bit-identical to the unfused path, so the
    result equals FP8GroupedLinear(gate/up) -> silu_mul -> FP8GroupedLinear(down).

    The K-major FP8 h that down_proj's wgrad reads is not kept from the forward but quantized
    again from the saved g and u in backward (one more silu_mul_quant pass, deterministic, so the
    bytes are the same), which drops it from the saved activations: 24 MiB per layer and
    micro-batch at 32k routed tokens, for no measurable step time. recompute_h = False keeps it.
    """

    recompute_h: bool = True

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor,
        q_gate: tuple[torch.Tensor, ...],
        q_up: tuple[torch.Tensor, ...],
        q_down: tuple[torch.Tensor, ...],
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if quantized_input is None:
            quantized_input = quantize_grouped_input(x, grouped_mm_offs)
        x_fp8, x_scale, x_ch, x_ch_scale = quantized_input
        inter, hidden = w_gate.shape[1], w_down.shape[1]
        g = fp8_m_grouped_gemm((x_fp8, x_scale), q_gate[:2], inter, grouped_mm_offs, group_indices)
        u = fp8_m_grouped_gemm((x_fp8, x_scale), q_up[:2], inter, grouped_mm_offs, group_indices)
        h_fp8, h_scale, h_ch, h_ch_scale = silu_mul_quant_fwd(g, u, grouped_mm_offs)
        y = fp8_m_grouped_gemm((h_fp8, h_scale), q_down[:2], hidden, grouped_mm_offs, group_indices)
        del h_fp8, h_scale
        if FP8SwiGLUExpertsFunc.recompute_h:
            h_ch = h_ch_scale = None
        ctx.save_for_backward(
            g, u, x_ch, x_ch_scale, h_ch, h_ch_scale, grouped_mm_offs, ks_tensor,
            q_gate[2], q_gate[3], q_up[2], q_up[3], q_down[2], q_down[3],
        )  # fmt: skip
        ctx.weights = (w_gate, w_up, w_down)
        ctx.ks, ctx.group_indices = ks, group_indices
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (
            g, u, x_ch, x_ch_scale, h_ch, h_ch_scale, offs, ks_tensor,
            wg_t, sg_t, wu_t, su_t, wd_t, sd_t,
        ) = ctx.saved_tensors  # fmt: skip
        w_gate, w_up, w_down = ctx.weights
        ks, gi = ctx.ks, ctx.group_indices
        inter, hidden = w_gate.shape[1], w_down.shape[1]

        # Locals are dropped as soon as their last reader is queued, to keep the transient peak of
        # the backward low; the deferred wgrads keep only the K-major operands they need.
        if h_ch is None:
            _, _, h_ch, h_ch_scale = silu_mul_quant_fwd(g, u, offs)
        dy_fp8, dy_scale, dy_ch, dy_ch_scale = quantize_grouped_input(dy, offs)
        del dy
        dh = fp8_m_grouped_gemm((dy_fp8, dy_scale), (wd_t, sd_t), inter, offs, gi)
        del dy_fp8, dy_scale
        fp8_grouped_wgrad((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), ks, ks_tensor, w_down, tuple(w_down.shape))  # fmt: skip
        del dy_ch, dy_ch_scale, h_ch, h_ch_scale
        dgq, duq = silu_mul_bwd_quant(dh, g, u, offs)
        del dh, g, u
        dx = fp8_m_grouped_gemm(dgq[:2], (wg_t, sg_t), hidden, offs, gi)
        dgq = dgq[2:]
        dx += fp8_m_grouped_gemm(duq[:2], (wu_t, su_t), hidden, offs, gi)
        duq = duq[2:]
        fp8_grouped_wgrad(dgq, (x_ch, x_ch_scale), ks, ks_tensor, w_gate, tuple(w_gate.shape))
        fp8_grouped_wgrad(duq, (x_ch, x_ch_scale), ks, ks_tensor, w_up, tuple(w_up.shape))
        return dx, None, None, None, None, None, None, None, None, None, None, None


def fp8_swiglu_experts(
    gate_proj: "FP8GroupedLinear",
    up_proj: "FP8GroupedLinear",
    down_proj: "FP8GroupedLinear",
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    quantized_input: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """
    down_proj(silu(gate_proj(x)) * up_proj(x)) through FP8SwiGLUExpertsFunc. With quantized_input
    (from fp8_scatter_quant) x is only a gradient carrier and is never read.
    """
    return FP8SwiGLUExpertsFunc.apply(
        x, gate_proj.weight, up_proj.weight, down_proj.weight, grouped_mm_offs, ks, ks_tensor,
        group_indices, gate_proj._get_quantized_weight(), up_proj._get_quantized_weight(),
        down_proj._get_quantized_weight(), quantized_input,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Fused stage-3 input: expansion + scatter + quantization
# ---------------------------------------------------------------------------


def scatter_quant_backward(
    grad_x: torch.Tensor,
    reverse_shuffle_idxs: torch.Tensor,
    expand_idx: Optional[torch.Tensor],
    num_src_rows: int,
) -> torch.Tensor:
    """Gradient w.r.t. ``gathered`` given the gradient of the (never materialized) bf16 grouped
    input ``x`` ``(M, N)``: the backward of ``scatter_for_grouped_gemm`` (gather by
    ``reverse_shuffle_idxs``) composed with that of ``padded_index_gather`` (scatter-add by
    ``expand_idx``). Padding rows of ``grad_x`` are ignored. Needs no forward activations."""
    g = grad_x[reverse_shuffle_idxs]
    if expand_idx is None:
        return g
    out = grad_x.new_zeros(num_src_rows, grad_x.shape[1])
    # scatter_add_ (the kernel _PaddedIndexGather.backward uses): ~4.5x faster than index_add_
    # for bf16 rows here (94 vs 550 us at m=32768, N=2048)
    out.scatter_add_(0, expand_idx.unsqueeze(-1).expand_as(g), g)
    return out


class FP8ScatterQuantFunc(torch.autograd.Function):
    """
    Received tokens -> quantized grouped expert input in one pass (``scatter_quant``): the
    dedup expansion (padded_index_gather), the scatter by expert (scatter_for_grouped_gemm) and the
    FP8 quantization (quantize_grouped_input) fused, so the bf16 expert input is never written.

    Returns a zero-storage (M, N) stand-in for that bf16 input, which only carries its gradient
    back to the received tokens, followed by scatter_for_grouped_gemm's outputs with the bf16
    tokens replaced by the quantized tuple. Rows are placed in token order within each expert.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        gathered: torch.Tensor,
        expert_idxs: torch.Tensor,
        num_groups: int,
        expand_idx: torch.Tensor | None,
        group_counts: list[int] | None,
    ):
        xq, rsi, offs, ks, ks_tensor = scatter_quant(
            gathered, expert_idxs, num_groups, expand_idx, group_counts=group_counts
        )
        x = gathered.new_zeros(()).expand(xq[0].shape)
        empty = rsi.new_empty(0)
        ctx.save_for_backward(rsi, expand_idx if expand_idx is not None else empty)
        ctx.has_expand, ctx.num_src_rows = expand_idx is not None, gathered.shape[0]
        ctx.mark_non_differentiable(rsi, offs, ks_tensor, *xq)
        return x, rsi, offs, ks_tensor, ks, xq

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor, *_):
        rsi, expand_idx = ctx.saved_tensors
        expand_idx = expand_idx if ctx.has_expand else None
        return (
            scatter_quant_backward(grad_x, rsi, expand_idx, ctx.num_src_rows),
            None,
            None,
            None,
            None,
        )


def fp8_scatter_quant(
    gathered: torch.Tensor,
    expert_idxs: torch.Tensor,
    num_groups: int,
    expand_idx: torch.Tensor | None = None,
    group_counts: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor, tuple[torch.Tensor, ...]]:
    """
    (x, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor, quantized_x) for FP8 experts on
    Hopper; see FP8ScatterQuantFunc. x is the gradient carrier to hand the experts with quantized_x.
    """
    x, rsi, offs, ks_tensor, ks, xq = FP8ScatterQuantFunc.apply(
        gathered, expert_idxs, num_groups, expand_idx, group_counts
    )
    return x, rsi, offs, ks, ks_tensor, xq


# ---------------------------------------------------------------------------
# SwiGLU experts, fused gate_up weight (Qwen3.5)
# ---------------------------------------------------------------------------


class FP8SwiGLUGateUpExpertsFunc(torch.autograd.Function):
    """
    gate_up_proj -> SwiGLU -> down_proj for FP8 routed experts with a fused gate_up weight
    (G, 2I, H), on Hopper, as one autograd node. The GEMMs are those of the unfused path (one
    gate_up GEMM, one down GEMM; in backward one dgrad and one wgrad each); SwiGLU lives inside the
    quantization kernels above.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor,
        q_gate_up: tuple[torch.Tensor, ...],
        q_down: tuple[torch.Tensor, ...],
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if quantized_input is None:
            quantized_input = quantize_grouped_input(x, grouped_mm_offs)
        x_fp8, x_scale, x_ch, x_ch_scale = quantized_input
        inter2, hidden = w_gate_up.shape[1], w_down.shape[1]
        gu = fp8_m_grouped_gemm((x_fp8, x_scale), q_gate_up[:2], inter2, grouped_mm_offs, group_indices)  # fmt: skip
        h_fp8, h_scale, _, _ = silu_mul_quant_gu_fwd(gu, grouped_mm_offs)
        y = fp8_m_grouped_gemm((h_fp8, h_scale), q_down[:2], hidden, grouped_mm_offs, group_indices)
        # The K-major h for down_proj's wgrad is a pure function of gu, which is saved anyway:
        # backward quantizes it again (bit-identical) instead of keeping it alive meanwhile.
        ctx.save_for_backward(
            gu, x_ch, x_ch_scale, grouped_mm_offs, ks_tensor,
            q_gate_up[2], q_gate_up[3], q_down[2], q_down[3],
        )  # fmt: skip
        ctx.weights = (w_gate_up, w_down)
        ctx.ks, ctx.group_indices = ks, group_indices
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        gu, x_ch, x_ch_scale, offs, ks_tensor, wgu_t, sgu_t, wd_t, sd_t = ctx.saved_tensors
        w_gate_up, w_down = ctx.weights
        ks, gi = ctx.ks, ctx.group_indices
        inter, hidden = w_down.shape[2], w_down.shape[1]

        dy_fp8, dy_scale, dy_ch, dy_ch_scale = quantize_grouped_input(dy, offs)
        dh = fp8_m_grouped_gemm((dy_fp8, dy_scale), (wd_t, sd_t), inter, offs, gi)
        _, _, h_ch, h_ch_scale = silu_mul_quant_gu_fwd(gu, offs)
        fp8_grouped_wgrad((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), ks, ks_tensor, w_down, tuple(w_down.shape))  # fmt: skip
        dguq = silu_mul_bwd_quant_gu(dh, gu, offs)
        del dh
        dx = fp8_m_grouped_gemm(dguq[:2], (wgu_t, sgu_t), hidden, offs, gi)
        fp8_grouped_wgrad(dguq[2:], (x_ch, x_ch_scale), ks, ks_tensor, w_gate_up, tuple(w_gate_up.shape))  # fmt: skip
        return dx, None, None, None, None, None, None, None, None, None


def fp8_swiglu_gate_up_experts(
    gate_up_proj: torch.nn.Module,
    down_proj: torch.nn.Module,
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    quantized_input: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """
    down_proj(silu(g) * u) with [g | u] = gate_up_proj(x), both FP8GroupedLinear, through
    FP8SwiGLUGateUpExpertsFunc. With quantized_input (from fp8_scatter_quant) x is only a gradient
    carrier and is never read.
    """
    return FP8SwiGLUGateUpExpertsFunc.apply(
        x, gate_up_proj.weight, down_proj.weight, grouped_mm_offs, ks, ks_tensor, group_indices,
        gate_up_proj._get_quantized_weight(), down_proj._get_quantized_weight(), quantized_input,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Clamped SwiGLU experts with biases (gpt-oss)
# ---------------------------------------------------------------------------


def fp8_padded_wgrad(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    ks: list,
    ks_tensor: torch.Tensor,
    weight: torch.Tensor,
    main_grad: torch.Tensor | None = None,
) -> None:
    """
    weight.grad += a^T b per expert (into main_grad in fp32 instead when it holds storage, see
    modules/wgrad_accumulation.install_raw_main_grad) (k-grouped FP8 GEMM, deferred to DualPipeV's
    W phase like grouped_linear.fp8_grouped_wgrad), for weight (G, N, K), a's per-channel FP8 with
    NA >= N channels and b's with KB >= K (either may be laid out padded: the fused ops'
    ``padded=True``); ``k_grouped_fp8_gemm`` drops the padding.
    """
    G, N, K = weight.shape

    def fn(a, b, ks, ks_tensor):
        if main_grad is not None and main_grad.untyped_storage().size() > 0:
            k_grouped_fp8_gemm(a, b, main_grad, ks, ks_tensor)
            return
        grad = torch.zeros((G, N, K), device=a[0].device, dtype=torch.float32)
        k_grouped_fp8_gemm(a, b, grad, ks, ks_tensor)
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
        if quantized_input is None:
            quantized_input = quantize_grouped_input(x, grouped_mm_offs)
        x_fp8, x_scale, x_ch, x_ch_scale = quantized_input
        two_inter, hidden = w_gate_up.shape[1], w_down.shape[1]
        offs, gi = grouped_mm_offs, group_indices
        gu = fp8_m_grouped_gemm((x_fp8, x_scale), q_gate_up[:2], two_inter, offs, gi)
        # Only the rowwise h for the down GEMM: its K-major copy (down_proj's wgrad operand) is
        # recomputed from gu in backward rather than kept, the same bytes.
        h_fp8, h_scale = clamped_swiglu_quant_rowwise(gu, b_gate_up, offs, alpha, limit)
        y = fp8_m_grouped_gemm((h_fp8, h_scale), q_down[:2], hidden, offs, gi)
        group_bias_add_(y, b_down, gi)
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
        (
            b_gate_up, x_ch, x_ch_scale, offs, ks_tensor,
            wgu_t, sgu_t, wd_t, sd_t, *saved,
        ) = ctx.saved_tensors  # fmt: skip
        w_gate_up, w_down = ctx.weights
        ks, gi = ctx.ks, ctx.group_indices
        inter, hidden = w_down.shape[2], w_down.shape[1]
        if ctx.recompute:
            x_fp8, x_scale, wgu, sgu = saved
            gu = fp8_m_grouped_gemm((x_fp8, x_scale), (wgu, sgu), w_gate_up.shape[1], offs, gi)
            del saved, x_fp8, x_scale
        else:
            (gu,) = saved

        dy = dy.contiguous()
        d_b_down = segment_sum_along_rows(dy, offs, ctx.b_down_dtype)
        dy_fp8, dy_scale, dy_ch, dy_ch_scale = quantize_grouped_input(dy, offs)
        dh = fp8_m_grouped_gemm((dy_fp8, dy_scale), (wd_t, sd_t), inter, offs, gi)
        # h's K-major copy, laid out padded for the k-grouped wgrad.
        h_ch, h_ch_scale = clamped_swiglu_quant_kmajor(gu, b_gate_up, offs, ctx.alpha, ctx.limit, padded=True)  # fmt: skip
        mg_gate_up, mg_down = ctx.main_grads
        fp8_padded_wgrad((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), ks, ks_tensor, w_down, mg_down)
        del h_ch, h_ch_scale
        del dy_fp8, dy_scale, dy_ch, dy_ch_scale
        dguq, d_bias = clamped_swiglu_bwd_quant(
            dh, gu, b_gate_up, offs, ctx.alpha, ctx.limit, padded=True
        )
        del dh
        dx = fp8_m_grouped_gemm(dguq[:2], (wgu_t, sgu_t), hidden, offs, gi)
        fp8_padded_wgrad(dguq[2:], (x_ch, x_ch_scale), ks, ks_tensor, w_gate_up, mg_gate_up)
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
