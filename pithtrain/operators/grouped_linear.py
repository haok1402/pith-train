from functools import partial
from typing import List, Optional

import deep_gemm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pithtrain.operators.deepgemm_quantize import (
    fp8cast_blockwise_transpose_batched,
    fp8cast_rowwise_colwise,
    fp8cast_rowwise_kmajor,
)
from pithtrain.operators.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.operators.linear import ARCH_MAJOR
from pithtrain.pipeline.execution import WeightGradStore


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


def _m_grouped_fp8_gemm_nt(a, b, d, grouped_mm_offs, M, group_indices=None):
    """Dispatch m_grouped FP8 GEMM NT to the right API for the current GPU arch."""
    if ARCH_MAJOR >= 10:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a,
            b,
            d,
            grouped_mm_offs,
            use_psum_layout=True,
            expected_m_for_psum_layout=M,
        )
    else:
        assert group_indices is not None
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, group_indices)


_K_GROUPED_MAX_ROWS = 65536
"""
Rows per call of DeepGEMM's SM90 k-grouped FP8 GEMM. The kernel fails to launch
(CUDA_ERROR_INVALID_VALUE) once the rows of all groups reach 116224 whatever N and K are, which
imbalanced routing reaches (an EP rank receiving ~3.5x its share), so larger calls are split.
"""


def _pad_k_major_channels(
    x: tuple[torch.Tensor, torch.Tensor], ks: list, channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A K-major grouped operand (flat data of (n, k_g) row-major blocks, scales (n, rows/128) with
    any strides) re-laid out with ``channels`` >= n channels; the extra channels are zero data
    with unit scales, so they contribute exact zeros to the product.
    """
    data, scale = x
    n, blocks = scale.shape
    rows = sum(ks)
    out = data.new_zeros(channels * rows)
    row = 0
    for k in ks:
        if k:
            dst = out[row * channels : (row + k) * channels].view(channels, k)
            dst[:n].copy_(data[row * n : (row + k) * n].view(n, k))
        row += k
    # MN-major (column-major) like the quantization kernels write them.
    out_scale = torch.ones((blocks, channels), device=scale.device, dtype=scale.dtype).T
    out_scale[:n].copy_(scale)
    return out, out_scale


def _k_grouped_call(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
) -> None:
    """
    One deep_gemm.k_grouped_fp8_gemm_nt_contiguous call accumulating into d (c is d), made safe
    from a DeepGEMM SM90 deadlock: the kernel runs 2-CTA clusters along M (block_m = 128,
    cluster_m = 2) and never finishes when ceil(M / 128) * ceil(N / block_n) is odd, M being the
    first operand's channels (found by the gptoss agent: e.g. (M, N) = (2880, 2880), (2880, 1024),
    (5760, 2880) hang; workspace/kda-tasks/round3/gptoss/docs/orchestrator_diffs.md). An even
    M-block count is always safe, so when a's is odd the call computes the transposed product
    b a^T if b's is even, and otherwise pads a's channels to a multiple of 256 with zeros. The
    result is the same sum of the same FP8 products; padded rows of the output are discarded.
    """
    m, n = a[1].shape[0], b[1].shape[0]
    if -(-m // 128) % 2 == 0:
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, c=d)
    elif -(-n // 128) % 2 == 0:
        # d is usually a main_grad that install_wgrad_accumulation stores transposed for exactly
        # this case, so its transpose is contiguous and accumulates in place, copy-free.
        dt = d.transpose(1, 2)
        if dt.is_contiguous():
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(b, a, dt, ks, ks_tensor, c=dt)
            return
        dt = dt.contiguous()
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(b, a, dt, ks, ks_tensor, c=dt)
        d.copy_(dt.transpose(1, 2))
    else:
        mp = -(-m // 256) * 256
        dp = d.new_zeros((d.shape[0], mp, d.shape[2]))
        dp[:, :m].copy_(d)
        ap = _pad_k_major_channels(a, ks, mp)
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(ap, b, dp, ks, ks_tensor, c=dp)
        d.copy_(dp[:, :m])


def _k_grouped_fp8_gemm_nt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    c: torch.Tensor,
) -> None:
    """
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous with c accumulated into d (d is c), split into calls
    of at most _K_GROUPED_MAX_ROWS rows, each made through _k_grouped_call (deadlock guard). Each group's operand is a (channels, k) row-major block of
    the flat K-major buffer and its scales a column range of (channels, rows/128), so runs of whole
    groups are contiguous slices; a group longer than the limit is cut along k and accumulates
    chunk by chunk into its slice of d. Scale slices keep the parent's strides, so MN-major scales
    (fp8_cute/scale_layout.py) stay in the layout DeepGEMM reads without a transpose.
    """
    assert d is c
    if sum(ks) <= _K_GROUPED_MAX_ROWS:
        _k_grouped_call(a, b, d, ks, ks_tensor)
        return
    channels = (a[1].shape[0], b[1].shape[0])

    def call(g0: int, g1: int, row0: int, row1: int, chunk: tuple[int, int] | None) -> None:
        ops = []
        for (data, scale), n in zip((a, b), channels):
            if chunk is None:
                data = data[row0 * n : row1 * n]
                scale = scale[:, row0 // 128 : row1 // 128]
            else:
                k, (r0, r1) = row1 - row0, chunk
                data = data[row0 * n : row1 * n].view(n, k)[:, r0:r1].contiguous().view(-1)
                scale = scale[:, (row0 + r0) // 128 : (row0 + r1) // 128]
            ops.append((data, scale))
        sub_ks = ks[g0:g1] if chunk is None else [chunk[1] - chunk[0]]
        sub_t = ks_tensor[g0:g1] if chunk is None else torch.tensor(sub_ks, dtype=ks_tensor.dtype, device=ks_tensor.device)  # fmt: skip
        out = d[g0:g1]
        _k_grouped_call(ops[0], ops[1], out, sub_ks, sub_t)

    g0, row0, row = 0, 0, 0
    for g, k in enumerate(ks):
        if k > _K_GROUPED_MAX_ROWS:
            if g > g0:
                call(g0, g, row0, row, None)
            for r0 in range(0, k, _K_GROUPED_MAX_ROWS):
                call(g, g + 1, row, row + k, (r0, min(r0 + _K_GROUPED_MAX_ROWS, k)))
            g0, row0 = g + 1, row + k
        elif row + k - row0 > _K_GROUPED_MAX_ROWS:
            call(g0, g, row0, row, None)
            g0, row0 = g, row
        row += k
    if len(ks) > g0:
        call(g0, len(ks), row0, row, None)


def _fp8_grouped_wgrad(
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
        k_grouped_gemm = _k_grouped_fp8_gemm_nt
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
        _m_grouped_fp8_gemm_nt(
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
        _m_grouped_fp8_gemm_nt(
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

        _fp8_grouped_wgrad(a_wgrad, b_wgrad, ks, ks_tensor, weight, (num_groups, N, K))

        return grad_input, None, None, None, None, None, None, None


def quantize_grouped_input(
    input: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a grouped-GEMM input for FP8GroupedLinear: per-token FP8 for the m-grouped forward and
    per-channel FP8 for the k-grouped wgrad, from one read. On Hopper the per-channel output is
    written directly in K-major layout.

    Projections that share an input (gate_proj and up_proj) can quantize it once and pass the
    result to each as quantized_input; the bytes are the same either way.
    """
    if ARCH_MAJOR >= 10:
        return fp8cast_rowwise_colwise(input)
    return fp8cast_rowwise_kmajor(input, grouped_mm_offs)


def _fp8_grouped_gemm(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    n: int,
    grouped_mm_offs: torch.Tensor,
    group_indices: torch.Tensor | None,
) -> torch.Tensor:
    """m-grouped FP8 GEMM ``a @ b.T`` per group into a fresh bf16 (M, n) tensor."""
    M = a[0].shape[0]
    out = torch.empty((M, n), device=a[0].device, dtype=torch.bfloat16)
    _m_grouped_fp8_gemm_nt(a, b, out, grouped_mm_offs, M, group_indices=group_indices)
    return out


class FP8SwiGLUExpertsFunc(torch.autograd.Function):
    """
    gate_proj, up_proj -> SwiGLU -> down_proj for FP8 routed experts on Hopper, as one autograd node.

    SwiGLU is fused into the quantization of its neighbours (fp8_cute/silu_mul_quant.py): forward
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
        from pithtrain.operators.fp8_cute.silu_mul_quant import silu_mul_quant_fwd

        if quantized_input is None:
            quantized_input = quantize_grouped_input(x, grouped_mm_offs)
        x_fp8, x_scale, x_ch, x_ch_scale = quantized_input
        inter, hidden = w_gate.shape[1], w_down.shape[1]
        g = _fp8_grouped_gemm((x_fp8, x_scale), q_gate[:2], inter, grouped_mm_offs, group_indices)
        u = _fp8_grouped_gemm((x_fp8, x_scale), q_up[:2], inter, grouped_mm_offs, group_indices)
        h_fp8, h_scale, h_ch, h_ch_scale = silu_mul_quant_fwd(g, u, grouped_mm_offs)
        y = _fp8_grouped_gemm((h_fp8, h_scale), q_down[:2], hidden, grouped_mm_offs, group_indices)
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
        from pithtrain.operators.fp8_cute.silu_mul_quant import silu_mul_bwd_quant

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
            from pithtrain.operators.fp8_cute.silu_mul_quant import silu_mul_quant_fwd

            _, _, h_ch, h_ch_scale = silu_mul_quant_fwd(g, u, offs)
        dy_fp8, dy_scale, dy_ch, dy_ch_scale = fp8cast_rowwise_kmajor(dy, offs)
        del dy
        dh = _fp8_grouped_gemm((dy_fp8, dy_scale), (wd_t, sd_t), inter, offs, gi)
        del dy_fp8, dy_scale
        _fp8_grouped_wgrad((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), ks, ks_tensor, w_down, tuple(w_down.shape))  # fmt: skip
        del dy_ch, dy_ch_scale, h_ch, h_ch_scale
        dgq, duq = silu_mul_bwd_quant(dh, g, u, offs)
        del dh, g, u
        dx = _fp8_grouped_gemm(dgq[:2], (wg_t, sg_t), hidden, offs, gi)
        dgq = dgq[2:]
        dx += _fp8_grouped_gemm(duq[:2], (wu_t, su_t), hidden, offs, gi)
        duq = duq[2:]
        _fp8_grouped_wgrad(dgq, (x_ch, x_ch_scale), ks, ks_tensor, w_gate, tuple(w_gate.shape))
        _fp8_grouped_wgrad(duq, (x_ch, x_ch_scale), ks, ks_tensor, w_up, tuple(w_up.shape))
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


class FP8ScatterQuantFunc(torch.autograd.Function):
    """
    Received tokens -> quantized grouped expert input in one pass (fp8_cute/scatter_quant.py): the
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
        from pithtrain.operators.fp8_cute.scatter_quant import scatter_quant

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
        from pithtrain.operators.fp8_cute.scatter_quant import scatter_quant_backward

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
