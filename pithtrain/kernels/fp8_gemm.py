"""
DeepGEMM FP8 GEMM launchers and the recipes built on them, shared by the FP8 linear layers, the
FP8 expert ops and the MLA pass-latent ring attention so each GEMM convention has one source.

- Dense: :func:`fp8_gemm_nt` (``out = a @ b.T``) and the linear recipes
  :func:`fp8_act_weight_gemm` (forward) / :func:`fp8_dgrad_wgrad` (backward).
- m-grouped (expert forward / dgrad): :func:`fp8_m_grouped_gemm`, with the reusable per-row group
  ids of :func:`m_grouped_group_indices` as an optional hint.
- k-grouped (expert wgrad): :func:`k_grouped_fp8_gemm`, which splits calls above
  ``K_GROUPED_MAX_ROWS`` and avoids DeepGEMM's SM90 2-CTA-cluster deadlock, and
  :func:`fp8_grouped_wgrad_buffer`, the main_grad layout it accumulates into in place.
- Grouped-GEMM input prep: :func:`quantize_grouped_input`.
- :func:`fp8_fused_experts_supported`: whether the fused FP8 expert ops run on this device.

DeepGEMM picks its SM90 or SM100 kernels itself; the m- / k-grouped entry points and the grouped
input layout differ by architecture and are chosen here, so no caller branches on it.

Scale layout: every 1x128 activation scale has logical shape ``(rows, ceil(K / 128))`` fp32 and
unspecified strides (MN-major from the SM90 kernels, ``kernels/fp8_scale_layout.py``; row-major
from the SM100 kernels). The GEMMs here take either as returned; callers never read the strides.
"""

from typing import Tuple

import deep_gemm
import torch

from pithtrain.kernels.arch import ARCH_MAJOR, is_sm90
from pithtrain.kernels.fp8_rowwise_blockwise_transpose import fp8cast_rowwise_blockwise_transpose
from pithtrain.kernels.fp8_rowwise_colwise import fp8cast_rowwise_colwise
from pithtrain.kernels.fp8_rowwise_kmajor import fp8cast_rowwise_kmajor
from pithtrain.kernels.fp8_rowwise_transpose import fp8cast_rowwise_transpose

__all__ = [
    "K_GROUPED_MAX_ROWS",
    "fp8_act_weight_gemm",
    "fp8_dgrad_wgrad",
    "fp8_fused_experts_supported",
    "fp8_gemm_nt",
    "fp8_grouped_wgrad_buffer",
    "fp8_m_grouped_gemm",
    "k_grouped_fp8_gemm",
    "k_grouped_fp8_gemm_nt",
    "k_grouped_runs_transposed",
    "k_grouped_safe_size",
    "m_grouped_fp8_gemm_nt",
    "m_grouped_group_indices",
    "pad_k_major_channels",
    "quantize_grouped_input",
]


def fp8_gemm_nt(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    ``out = a @ b.T`` into a fresh ``(m, n)`` tensor of ``dtype`` on ``device``, through DeepGEMM's
    ``fp8_fp4_gemm_nt`` (fp32 accumulation, one rounding to ``dtype``). ``dtype`` must be bf16:
    DeepGEMM's SM90 kernel rejects any other output dtype.

    - ``a_fp8``: e4m3 ``(m, K)`` row-major; ``a_scale``: its 1x128 scale, fp32 of logical shape
      ``(m, ceil(K / 128))`` as this library's rowwise quantizations return it (any strides:
      MN-major is read as is, another layout is transposed by DeepGEMM first).
    - ``b_fp8``: e4m3 ``(n, K)`` row-major; ``b_scale``: its 128x128 scale, fp32
      ``(ceil(n / 128), ceil(K / 128))`` row-major (:func:`fp8cast_blockwise_transpose`, or the
      blockwise transpose of :func:`fp8cast_rowwise_blockwise_transpose`).

    Equals the fp32 product of the dequantized operands up to FP8 GEMM accumulation error (not bit
    for bit); deterministic. No input is modified.
    """
    out = torch.empty((m, n), device=device, dtype=dtype)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, a_scale), (b_fp8, b_scale), out)
    return out


def fp8_act_weight_gemm(
    input_2d: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward FP8 linear ``y = x @ W.T``: ``(output, input_t_fp8, scale_input_t)``.

    - ``input_2d``: ``x``, 2-D ``(m, K)`` bf16; quantized with
      :func:`fp8cast_rowwise_blockwise_transpose` (rowwise 1x128), not modified.
    - ``weight_fp8``, ``scale_weight``: ``W`` ``(n, K)`` e4m3 with its 128x128 scale fp32
      ``(ceil(n / 128), ceil(K / 128))`` row-major, as :func:`fp8cast_blockwise_transpose` returns.
    - ``output``: fresh ``(m, n)`` in ``input_2d``'s dtype, :func:`fp8_gemm_nt` of the quantized
      operands.
    - ``input_t_fp8``, ``scale_input_t``: ``x.T`` ``(K, m)`` e4m3 quantized per 128x128 tile,
      scale fp32 ``(ceil(K / 128), ceil(m / 128))`` row-major: the activation operand
      :func:`fp8_dgrad_wgrad` takes for the weight gradient.

    Deterministic; values as in :func:`fp8cast_rowwise_blockwise_transpose` and
    :func:`fp8_gemm_nt`.
    """
    m, n = input_2d.shape[0], weight_fp8.shape[0]
    input_fp8, scale_input, input_t_fp8, scale_input_t = fp8cast_rowwise_blockwise_transpose(
        input_2d
    )
    output = fp8_gemm_nt(
        input_fp8, scale_input, weight_fp8, scale_weight, m, n, input_2d.device, input_2d.dtype
    )
    return output, input_t_fp8, scale_input_t


def fp8_dgrad_wgrad(
    grad_2d: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward FP8 linear of ``y = x @ W.T``: ``(grad_input, weight_grad)``.

    - ``grad_2d``: the gradient of ``y``, 2-D ``(m, n)`` bf16; quantized rowwise (and its
      transpose rowwise) with :func:`fp8cast_rowwise_transpose`, not modified.
    - ``weight_t_fp8``, ``scale_weight_t``: ``W.T`` ``(k, n)`` e4m3 with its 128x128 scale fp32
      ``(ceil(k / 128), ceil(n / 128))`` row-major (the transpose pair of
      :func:`fp8cast_blockwise_transpose`).
    - ``input_t_fp8``, ``scale_input_t``: ``x.T`` ``(k, m)`` blockwise, as
      :func:`fp8_act_weight_gemm` returns them (saved from the forward, or recomputed).
    - ``k``: the input features, ``W``'s second dimension.
    - ``grad_input = grad @ W``: fresh ``(m, k)``; ``weight_grad = grad.T @ x``: fresh ``(n, k)``
      (not accumulated anywhere); both in ``grad_2d``'s dtype, through :func:`fp8_gemm_nt`.

    Deterministic.
    """
    m, n = grad_2d.shape
    grad_fp8, scale_grad, grad_t_fp8, scale_grad_t = fp8cast_rowwise_transpose(grad_2d)
    grad_input = fp8_gemm_nt(
        grad_fp8, scale_grad, weight_t_fp8, scale_weight_t, m, k, grad_2d.device, grad_2d.dtype
    )
    weight_grad = fp8_gemm_nt(
        grad_t_fp8, scale_grad_t, input_t_fp8, scale_input_t, n, k, grad_2d.device, grad_2d.dtype
    )
    return grad_input, weight_grad


def m_grouped_fp8_gemm_nt(a, b, d, grouped_mm_offs, M, group_indices=None):
    """
    m-grouped FP8 GEMM NT into ``d``, through the DeepGEMM entry point of the current GPU arch.
    SM90's takes per-row group ids: ``group_indices`` when given (a hint reused across the
    projections of a layer), else computed here with :func:`m_grouped_group_indices`.
    """
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
        if group_indices is None:
            group_indices = m_grouped_group_indices(grouped_mm_offs, M)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, group_indices)


K_GROUPED_MAX_ROWS = 65536
"""
Rows per call of DeepGEMM's SM90 k-grouped FP8 GEMM. The kernel fails to launch
(CUDA_ERROR_INVALID_VALUE) once the rows of all groups reach 116224 whatever N and K are, which
imbalanced routing reaches (an EP rank receiving ~3.5x its share), so larger calls are split.
"""


def pad_k_major_channels(
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


def k_grouped_runs_transposed(m: int, n: int) -> bool:
    """
    Whether the SM90 k-grouped FP8 GEMM of an (m, n) output (m the first operand's channels) runs as
    the transposed product in :func:`_k_grouped_call`: an odd 128-block count on m (which would
    deadlock) and an even one on n. ``install_wgrad_accumulation`` lays out main_grad for it.
    """
    return -(-m // 128) % 2 == 1 and -(-n // 128) % 2 == 0


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
    elif k_grouped_runs_transposed(m, n):
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
        ap = pad_k_major_channels(a, ks, mp)
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(ap, b, dp, ks, ks_tensor, c=dp)
        d.copy_(dp[:, :m])


def k_grouped_fp8_gemm_nt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    c: torch.Tensor,
) -> None:
    """
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous with c accumulated into d (d is c), split into calls
    of at most K_GROUPED_MAX_ROWS rows, each made through _k_grouped_call (deadlock guard). Each group's operand is a (channels, k) row-major block of
    the flat K-major buffer and its scales a column range of (channels, rows/128), so runs of whole
    groups are contiguous slices; a group longer than the limit is cut along k and accumulates
    chunk by chunk into its slice of d. Scale slices keep the parent's strides, so MN-major scales
    (kernels/fp8_scale_layout.py) stay in the layout DeepGEMM reads without a transpose.
    """
    assert d is c
    if sum(ks) <= K_GROUPED_MAX_ROWS:
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
        if k > K_GROUPED_MAX_ROWS:
            if g > g0:
                call(g0, g, row0, row, None)
            for r0 in range(0, k, K_GROUPED_MAX_ROWS):
                call(g, g + 1, row, row + k, (r0, min(r0 + K_GROUPED_MAX_ROWS, k)))
            g0, row0 = g + 1, row + k
        elif row + k - row0 > K_GROUPED_MAX_ROWS:
            call(g0, g, row0, row, None)
            g0, row0 = g, row
        row += k
    if len(ks) > g0:
        call(g0, len(ks), row0, row, None)


def k_grouped_fp8_gemm(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
) -> None:
    """
    The FP8 grouped weight gradient, accumulated in place: ``d[g] += a_g^T b_g`` in fp32 for each
    group ``g`` of ``ks`` rows.

    - ``d``: fp32 ``(G, N, K)``, any layout :func:`fp8_grouped_wgrad_buffer` returns or contiguous.
    - ``a``, ``b``: the per-channel FP8 data and 1x128 scales (``x_ch_fp8, x_ch_scale``) of
      :func:`quantize_grouped_input` (or of the fused quantization ops) of the grad_output (N
      channels) and of the input (K channels), over the same grouped rows. ``ks`` (host) and
      ``ks_tensor`` (int32, device) are the per-group row counts, multiples of 128.
    - Either operand may be laid out with more channels than ``d`` has (zero-padded, as the fused
      ops' ``padded=True`` writes it); the extra channels are dropped. Only the fused SM90 ops
      produce such operands (where ``fp8_fused_experts_supported()``), so SM100 takes none.

    SM100 runs DeepGEMM's TN k-grouped GEMM. SM90 runs the NT one with the row-limit split and the
    2-CTA deadlock guard of :func:`k_grouped_fp8_gemm_nt`; with padded operands it computes the
    product into an fp32 temporary (transposed when only ``b``'s channels are a multiple of 256,
    so DeepGEMM's M-side tile count is even) and adds its unpadded part into ``d``.
    """
    if ARCH_MAJOR >= 10:
        deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c=d)
        return
    G, N, K = d.shape
    NA, KB = a[1].shape[0], b[1].shape[0]
    if (NA, KB) == (N, K):
        k_grouped_fp8_gemm_nt(a, b, d, ks, ks_tensor, c=d)
        return
    if NA % 256 != 0 and KB % 256 == 0:
        c = torch.zeros((G, KB, NA), device=d.device, dtype=torch.float32)
        k_grouped_fp8_gemm_nt(b, a, c, ks, ks_tensor, c=c)
        d.add_(c[:, :K, :N].transpose(1, 2))
    else:
        c = torch.zeros((G, NA, KB), device=d.device, dtype=torch.float32)
        k_grouped_fp8_gemm_nt(a, b, c, ks, ks_tensor, c=c)
        d.add_(c[:, :N, :K])


def fp8_grouped_wgrad_buffer(size: torch.Size, device: torch.device) -> torch.Tensor:
    """
    Uninitialized fp32 main_grad of shape ``size`` for an FP8 grouped weight ``(G, N, K)``, laid out
    so :func:`k_grouped_fp8_gemm` accumulates into it in place: where the SM90 deadlock guard
    computes the product transposed, its storage is ``(G, K, N)`` and the result its transpose
    view (``.contiguous()`` gives the ``(G, N, K)`` gradient); otherwise contiguous.
    """
    if len(size) == 3 and is_sm90():
        g, n, k = size
        if k_grouped_runs_transposed(n, k):
            return torch.empty((g, k, n), dtype=torch.float32, device=device).transpose(1, 2)
    return torch.empty(size, dtype=torch.float32, device=device)


def fp8_fused_experts_supported() -> bool:
    """
    Whether the fused FP8 expert ops run on this device: SwiGLU and gpt-oss's clamped SwiGLU
    fused into the quantization of their neighbouring grouped GEMMs (``operators/fp8_experts.py``),
    the fused scatter + quantization of the stage-3 input (:func:`scatter_quant`), the fused dense
    SwiGLU MLP (``operators/fp8_swiglu_mlp.py``) and the fp32 main_grad of raw expert parameters
    they write. They are built on DeepGEMM's SM90 K-major k-grouped GEMM: True on SM90, False
    elsewhere, where FP8 training takes the unfused ``FP8GroupedLinear`` path.
    """
    return is_sm90()


def quantize_grouped_input(
    input: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The FP8 operands of a grouped-GEMM input from one read of ``input``: ``(x_fp8, x_scale,
    x_ch_fp8, x_ch_scale)``.

    - ``input``: bf16 ``(M, K)`` contiguous, rows grouped by expert with each group padded to a
      multiple of 128 rows; ``grouped_mm_offs`` the int32 cumulative group ends (last = M).
    - ``x_fp8`` ``(M, K)`` e4m3 and ``x_scale`` ``(M, ceil(K / 128))`` fp32: the rowwise 1x128 E8M0
      quantization, the m-grouped GEMM's ``a`` operand (:func:`fp8_m_grouped_gemm`).
    - ``x_ch_fp8`` / ``x_ch_scale``: the per-channel (128x1 along rows, within each group)
      quantization, the k-grouped wgrad operand (:func:`k_grouped_fp8_gemm`), in the layout this
      device's k-grouped GEMM reads: on SM90 K-major per group (group ``g``'s ``(K, ks[g])``
      row-major block flat at ``offs[g-1] * K``, scale ``(K, M / 128)``), on SM100 ``(M, K)``
      with scale ``(M / 128, K)``. Scale strides are unspecified (see the module docstring).

    Values per 128-element block: ``scale`` = the power of two at or above ``max(amax, 1e-4) *
    fp32(1 / 448)`` (at least 2^-126), ``fp8 = e4m3_satfinite(x * (1 / scale))``, round to nearest
    even (exact: the scale is a power of two). Projections that share an input (gate_proj and
    up_proj) can quantize it once and pass the result to each as quantized_input; the bytes are the
    same.
    """
    if ARCH_MAJOR >= 10:
        return fp8cast_rowwise_colwise(input)
    return fp8cast_rowwise_kmajor(input, grouped_mm_offs)


def fp8_m_grouped_gemm(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    n: int,
    grouped_mm_offs: torch.Tensor,
    group_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    m-grouped FP8 GEMM into a fresh bf16 ``(M, n)`` tensor: rows ``[offs[g-1], offs[g])`` of the
    output are ``a_g @ b[g].T``, fp32 accumulation.

    - ``a``: ``(fp8 (M, K), 1x128 scale (M, ceil(K / 128)))`` as :func:`quantize_grouped_input` (or
      a fused quantization op) returns them; M a multiple of 128 per group.
    - ``b``: ``(fp8 (G, n, K), 128x128 scale (G, ceil(n / 128), ceil(K / 128)))``, the weight as
      ``fp8cast_blockwise_transpose_batched`` returns it.
    - ``grouped_mm_offs``: int32 ``(G,)`` cumulative padded group ends. ``group_indices``: the
      optional hint of :func:`m_grouped_group_indices`, computed here when None.
    """
    M = a[0].shape[0]
    out = torch.empty((M, n), device=a[0].device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt(a, b, out, grouped_mm_offs, M, group_indices=group_indices)
    return out


def k_grouped_safe_size(n: int) -> int:
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


def m_grouped_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor | None:
    """
    The group-id hint of :func:`fp8_m_grouped_gemm`, computed once and reused by every m-grouped
    GEMM over the same rows; None where this device's m-grouped GEMM takes none (SM100).

    On SM90: int32 ``(M,)`` per-row group ids of the grouped rows, e.g. ``grouped_mm_offs = [128,
    256, 384, 512]`` -> ``[0,0,...,1,1,...,2,2,...,3,3,...]``; rows at or past the last offset get
    the last group. The one-launch CuTe kernel (``group_indices_sm90``) computes it for the offsets
    it supports, otherwise ``arange`` + ``searchsorted`` + ``clamp`` give the same values.
    """
    if not is_sm90():
        return None
    from pithtrain.kernels import group_indices_sm90 as cgi

    if cgi.supports_group_indices(grouped_mm_offs, M):
        return cgi.group_indices(grouped_mm_offs, M)  # same values, one launch
    row_indices = torch.arange(M, device=grouped_mm_offs.device)
    gi = torch.searchsorted(grouped_mm_offs, row_indices, right=True).to(torch.int32)
    gi.clamp_(max=grouped_mm_offs.shape[0] - 1)
    return gi
