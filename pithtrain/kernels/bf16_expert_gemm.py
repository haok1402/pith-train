"""
The SM90-only BF16 CuTe experts' six grouped GEMMs (fused SwiGLU epilogues, scattered stores,
fp32 main_grad wgrads): the contract functions of ``bf16_expert_gemm_sm90``.

Each function imports ``bf16_expert_gemm_sm90`` inside its body, so importing this module (and
``pithtrain.kernels``) never loads cutlass or QuACK; callers take these kernels only where
``bf16_experts_supported`` says yes. ``operators/bf16_experts.py`` composes them into the BF16
expert autograd node. SM100 has no implementation (``bf16_experts_supported`` is False there).
"""

__all__ = [
    "bf16_expert_dgated_backward",
    "bf16_expert_dgrad",
    "bf16_expert_down_forward_scattered",
    "bf16_expert_gated_forward",
    "bf16_expert_wgrad_down",
    "bf16_expert_wgrad_gate_up",
]


def bf16_expert_gated_forward(x, w_gate, w_up, cu, a_idx, bias=None, act=()):
    """
    Gate/up grouped GEMM of the gathered rows ``x[a_idx]`` with the SwiGLU fused in its epilogue:
    fresh bf16 ``(preact (M, 2I), h (M, I))``, M = ``a_idx.shape[0]``.

    - ``x``: bf16 ``(T, H)`` row-major, the received tokens; ``a_idx``: int32 ``(M,)`` and ``cu``:
      int32 ``(G + 1,)``, the ``a_idx`` / ``cu_seqlens`` of :func:`bf16_expert_group_slots` (grouped
      row ``r`` of expert ``g``, ``cu[g] <= r < cu[g + 1]``, reads ``x[a_idx[r]]``).
    - ``w_gate``, ``w_up``: bf16 ``(G, I, H)``, same strides, each 16-byte aligned and a nonzero
      multiple of 16 bytes apart (asserted); either address order. H and I multiples of 8.
    - ``preact[r, 2i]`` / ``preact[r, 2i + 1]``: ``x[a_idx[r]] . w_gate[g, i]`` /
      ``x[a_idx[r]] . w_up[g, i]`` accumulated in fp32 and rounded once to bf16, always in (gate,
      up) order whatever the weights' address order.
    - ``h``: QuACK's ``swiglu`` (fp32) of the bf16-rounded gate and up, rounded to bf16; the
      backward (:func:`bf16_expert_dgated_backward`) recomputes it bit for bit from ``preact``.
    - ``act = (alpha, limit)`` selects gpt-oss's form: ``bias`` bf16 ``(G, 2I)``, interleaved like
      ``preact``, is added to the fp32 accumulator before the one rounding, and ``h`` is the
      clamped SwiGLU ``swiglu_oai``; ``w_gate`` must then be the lower address (asserted).

    Inputs are not modified; M == 0 launches nothing. Deterministic.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_gemm_sm90`` (cutlass, QuACK) on first use.
    """
    from pithtrain.kernels.bf16_expert_gemm_sm90 import bf16_expert_gated_forward_sm90

    return bf16_expert_gated_forward_sm90(x, w_gate, w_up, cu, a_idx, bias, act)


def bf16_expert_down_forward_scattered(h, w_down, cu, slot_of, bias=None):
    """
    Down grouped GEMM with its output rows scattered back to received-slot order: a fresh bf16
    ``(M, H)`` ``out`` with ``out[slot_of[r]] = h[r] @ w_down[g].T (+ bias[g])`` for each grouped
    row ``r`` of expert ``g``.

    - ``h``: bf16 ``(M, I)`` row-major, the grouped rows; ``w_down``: bf16 ``(G, H, I)``;
      ``cu``: int32 ``(G + 1,)`` and ``slot_of``: int32 ``(M,)`` of
      :func:`bf16_expert_group_slots` (a permutation, so every row of ``out`` is written once).
    - ``bias``: optional bf16 ``(G, H)`` (gpt-oss), added to the fp32 accumulator before the one
      rounding to bf16.

    Inputs are not modified; M == 0 launches nothing. Deterministic.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_gemm_sm90`` (cutlass, QuACK) on first use.
    """
    from pithtrain.kernels.bf16_expert_gemm_sm90 import bf16_expert_down_forward_scattered_sm90

    return bf16_expert_down_forward_scattered_sm90(h, w_down, cu, slot_of, bias)


def bf16_expert_dgated_backward(dy, w_down, preact, cu, slot_of, w_gate, w_up, act=()):
    """
    Down-projection dgrad with the SwiGLU backward fused in its epilogue: ``(dpre, h, swapped)``.

    - ``dy``: bf16 ``(M, H)`` row-major, the output gradient in received-slot order;
      ``w_down``: bf16 ``(G, H, I)``; ``preact``: the bf16 ``(M, 2I)`` of
      :func:`bf16_expert_gated_forward`; ``cu``, ``slot_of``: as in
      :func:`bf16_expert_down_forward_scattered`. ``w_gate`` / ``w_up`` are read only for their
      address order.
    - For grouped row ``r`` of expert ``g``: ``dh = dy[slot_of[r]] @ w_down[g]`` in fp32, then
      QuACK's ``dswiglu`` (``dswiglu_oai`` with ``act = (alpha, limit)``, which requires
      ``w_gate`` to be the lower address) of the saved gate / up gives ``dg``, ``du`` and ``h``.
    - ``dpre``: fresh bf16 ``(M, 2I)``, pairs in the weights' address order: ``(dg, du)``, or
      ``(du, dg)`` when ``swapped`` (``w_up`` below ``w_gate``), the order
      :func:`bf16_expert_dgrad` and :func:`bf16_expert_wgrad_gate_up` read. ``h``: fresh bf16
      ``(M, I)``, bit-identical to the forward's ``h`` (the down wgrad input).

    Inputs are not modified; M == 0 launches nothing. Deterministic.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_gemm_sm90`` (cutlass, QuACK) on first use.
    """
    from pithtrain.kernels.bf16_expert_gemm_sm90 import bf16_expert_dgated_backward_sm90

    return bf16_expert_dgated_backward_sm90(dy, w_down, preact, cu, slot_of, w_gate, w_up, act)


def bf16_expert_dgrad(dpre, w_gate, w_up, cu, dpre_swapped):
    """
    Gate/up dgrad: a fresh bf16 ``(M, H)`` ``dx`` in grouped-row order, ``dx[r] = dg[r] @
    w_gate[g] + du[r] @ w_up[g]`` for each row ``r`` of expert ``g``, one fp32 accumulation over
    ``K = 2I`` rounded once.

    - ``dpre``, ``dpre_swapped``: as :func:`bf16_expert_dgated_backward` returned them; the weights
      must still lie in the address order it saw (asserted). ``w_gate``, ``w_up``: bf16
      ``(G, I, H)`` paired as in :func:`bf16_expert_gated_forward`; ``cu``: int32 ``(G + 1,)``.
    - The caller sums ``dx`` back to the received tokens (:func:`bf16_expert_gather_sum`).

    Inputs are not modified; M == 0 launches nothing. Deterministic.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_gemm_sm90`` (cutlass, QuACK) on first use.
    """
    from pithtrain.kernels.bf16_expert_gemm_sm90 import bf16_expert_dgrad_sm90

    return bf16_expert_dgrad_sm90(dpre, w_gate, w_up, cu, dpre_swapped)


def bf16_expert_wgrad_gate_up(x, dpre, dpre_swapped, mg_gate, mg_up, cu, a_idx):
    """
    In place: the gate / up weight gradients accumulated into their fp32 main_grads,
    ``mg_gate[g] += dg_g^T @ x[a_idx]_g`` and ``mg_up[g] += du_g^T @ x[a_idx]_g`` over the
    grouped rows of each expert (fp32 accumulation added to the existing values). Returns None.

    - ``x``, ``a_idx``, ``cu``: as in :func:`bf16_expert_gated_forward`; ``dpre``,
      ``dpre_swapped``: as :func:`bf16_expert_dgated_backward` returned them.
    - ``mg_gate``, ``mg_up``: fp32 ``(G, I, H)``, same strides, 16-byte aligned and a nonzero
      multiple of 16 bytes apart, either address order (the lanes are routed to the right one).

    Other inputs are not modified; M == 0 is a no-op. Deterministic.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_gemm_sm90`` (cutlass, QuACK) on first use.
    """
    from pithtrain.kernels.bf16_expert_gemm_sm90 import bf16_expert_wgrad_gate_up_sm90

    return bf16_expert_wgrad_gate_up_sm90(x, dpre, dpre_swapped, mg_gate, mg_up, cu, a_idx)


def bf16_expert_wgrad_down(dy, h, mg_down, cu, slot_of):
    """
    In place: the down weight gradient accumulated into its fp32 main_grad, ``mg_down[g] +=
    dy[slot_of]_g^T @ h_g`` over the grouped rows of each expert. Returns None.

    - ``dy``: bf16 ``(M, H)`` in received-slot order; ``h``: bf16 ``(M, I)`` grouped rows (the
      ``h`` of :func:`bf16_expert_dgated_backward`); ``cu``, ``slot_of``: as in
      :func:`bf16_expert_down_forward_scattered`. ``mg_down``: fp32 ``(G, H, I)``.

    Other inputs are not modified; M == 0 is a no-op. Deterministic.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_gemm_sm90`` (cutlass, QuACK) on first use.
    """
    from pithtrain.kernels.bf16_expert_gemm_sm90 import bf16_expert_wgrad_down_sm90

    return bf16_expert_wgrad_down_sm90(dy, h, mg_down, cu, slot_of)
