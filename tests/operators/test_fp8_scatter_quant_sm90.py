"""Fused expand + scatter + rowwise/K-major FP8 quantization against the three-step path.

The three-step reference is ``padded_index_gather`` -> ``scatter_for_grouped_gemm`` ->
``fp8cast_rowwise_kmajor`` (Triton). The Triton scatter places rows within an expert group in
atomic order, so the fused kernel (stable placement) is checked three ways:

* placement-independent outputs equal the three-step path outright: grouped_mm_offs, ks_tensor,
  ks, and every token's rowwise FP8 row + scale (looked up through each side's own placement);
* the placement is the stable one (token order within each group);
* all four quantized tensors, padding rows included, are bit-identical to the three-step path with
  the same placement (reference scatter by the fused reverse_shuffle_idxs, then Triton quant).
"""

import pytest
import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90 as cute_quant
from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100 as triton_quant
from pithtrain.kernels.fp8_scale_layout import is_mn_major
from pithtrain.kernels.fp8_scatter_quant_sm90 import scatter_quant_sm90 as scatter_quant
from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm

pytestmark = pytest.mark.skipif(not is_sm90(), reason="Hopper (SM90) K-major path")


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _routing(
    sizes: list[int], dedup: int, segmented: bool, seed: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """expert_idxs with the given per-expert sizes, plus an expand_idx with ~``dedup`` slots per
    source row. ``segmented`` mimics EP dispatch (runs per sender); otherwise a random order."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    eidx = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    m = eidx.numel()
    if segmented:
        # 8 senders, each sending a contiguous run per expert
        sender = torch.randint(0, 8, (m,), generator=g)
        eidx = eidx[torch.argsort(sender * len(sizes) + eidx, stable=True)]
    else:
        eidx = eidx[torch.randperm(m, generator=g)]
    m_src = max(1, (m + dedup - 1) // dedup)
    expand = torch.randint(0, m_src, (m,), generator=g)
    expand[: min(m, m_src)] = torch.arange(min(m, m_src))  # every source row used at least once
    expand = expand[torch.randperm(m, generator=g)]
    return eidx.cuda(), expand.cuda(), m_src


def _tokens(rows: int, N: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, N, device="cuda", generator=g)
    rs = torch.exp2(torch.randint(-12, 12, (rows, 1), device="cuda", generator=g).float())
    x = x * rs
    mask = torch.rand(rows, N, device="cuda", generator=g) < 1e-3
    x = torch.where(mask, x * 1e4, x)
    x[3::97] = 0.0
    x[:, : min(N, 3)] = 0.0
    return x.to(torch.bfloat16)


def _check(sizes: list[int], N: int, dedup: int, segmented: bool, host_counts: bool) -> None:
    G = len(sizes)
    seed = sum(sizes) * 7 + N + dedup
    eidx, expand, m_src = _routing(sizes, dedup, segmented, seed)
    gathered = _tokens(m_src, N, seed)
    counts = sizes if host_counts else None
    use_expand = dedup > 1 or segmented

    # ---- three-step reference ----
    x = padded_index_gather(gathered, expand) if use_expand else gathered[expand]
    ref_out, ref_rsi, ref_offs, ref_ks, ref_kst = scatter_for_grouped_gemm(
        x, eidx, G, group_counts=counts
    )
    ref_q = triton_quant(ref_out, ref_offs)

    src = gathered if use_expand else x
    xq, rsi, offs, ks, kst = scatter_quant(
        src, eidx, G, expand if use_expand else None, group_counts=counts
    )

    assert torch.equal(offs, ref_offs) and torch.equal(kst, ref_kst) and ks == ref_ks
    assert rsi.dtype == torch.int64 and rsi.shape == ref_rsi.shape

    # stable placement
    starts = torch.cat([offs.new_zeros(1), offs[:-1]]).long()
    order = torch.argsort(eidx, stable=True)
    rank = torch.empty_like(order)
    pos = torch.arange(eidx.numel(), device="cuda")
    first = torch.searchsorted(eidx[order], torch.arange(G, device="cuda"))
    rank[order] = pos - first[eidx[order]]
    assert torch.equal(rsi, starts[eidx] + rank)

    # bit-identical to the three-step path with the same placement (padding rows included)
    M = ref_out.shape[0]
    same = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    same[rsi] = x
    exp_q = triton_quant(same, offs)  # values (row-major scales)
    lay_q = cute_quant(same, offs)  # layout: what quantize_grouped_input returns (MN-major scales)
    names = ("x_fp8", "x_scale", "x_ch", "x_ch_scale")
    for name, a, b, c in zip(names, xq, exp_q, lay_q):
        assert a.shape == c.shape and a.dtype == c.dtype and a.stride() == c.stride(), name
        assert torch.equal(_u8(a), _u8(b)), f"{name}: {(_u8(a) != _u8(b)).sum().item()} diffs"
    assert is_mn_major(xq[1]) and is_mn_major(xq[3])

    # placement-independent part equals the actual three-step output
    assert torch.equal(_u8(xq[0])[rsi], _u8(ref_q[0])[ref_rsi])
    assert torch.equal(xq[1][rsi], ref_q[1][ref_rsi])
    if torch.equal(rsi, ref_rsi):  # the atomic scatter happened to be stable
        for a, b in zip(xq, ref_q):
            assert torch.equal(_u8(a), _u8(b))


BALANCED = [2048] * 16
IMBALANCED = [0, 128, 384, 0, 0, 1024, 256, 128, 0, 2944, 128, 0, 640, 384, 128, 0]
RAGGED = [1, 0, 300, 129, 0, 77, 1000, 5, 0, 0, 255, 256, 17, 0, 640, 3]


@pytest.mark.parametrize("dedup", [1, 2, 4, 8])
@pytest.mark.parametrize("host_counts", [True, False])
@pytest.mark.parametrize("sizes", [BALANCED, IMBALANCED, RAGGED], ids=["bal", "imb", "ragged"])
def test_scatter_quant(sizes: list[int], host_counts: bool, dedup: int) -> None:
    _check(sizes, 2048, dedup, segmented=True, host_counts=host_counts)


@pytest.mark.parametrize(
    "sizes,N",
    [
        ([4096] * 16, 2048),  # M = 65536
        ([8192, 0, 0, 0, 8192, 0, 0, 16384], 2048),
        (RAGGED, 768),
        (RAGGED, 200),  # N not a multiple of 128
        ([5, 0, 3], 136),
        ([0, 0, 1], 8),
        ([300] * 128, 256),  # many groups
        ([7] * 300, 128),
        ([70000], 64),  # one group, > 2^16 tokens
    ],
)
def test_scatter_quant_shapes(sizes: list[int], N: int) -> None:
    _check(sizes, N, 3, segmented=False, host_counts=False)


def test_scatter_quant_identity_and_determinism() -> None:
    eidx, _, _ = _routing(IMBALANCED, 1, False, 0)
    x = _tokens(eidx.numel(), 2048, 1)
    a = scatter_quant(x, eidx, 16)
    b = scatter_quant(x, eidx, 16)
    for u, v in zip(a[0], b[0]):
        assert torch.equal(_u8(u), _u8(v))
    assert torch.equal(a[1], b[1])
    _check(IMBALANCED, 2048, 1, segmented=False, host_counts=True)


def test_scatter_quant_back_to_back() -> None:
    """Many launches with a changing chunk count reuse the device-side ticket/epoch scratch."""
    for it, sizes in enumerate([[2048] * 16, [5, 0, 3] + [0] * 13, [4096] * 16, [1] * 16] * 3):
        _check(sizes, 128, 2, segmented=it % 2 == 0, host_counts=True)


def test_scatter_quant_cuda_graph() -> None:
    """With host counts the op is capturable; replays see fresh tokens and routing."""
    sizes = IMBALANCED
    eidx, expand, m_src = _routing(sizes, 2, True, 5)
    gathered = _tokens(m_src, 512, 5)
    scatter_quant(gathered, eidx, 16, expand, group_counts=sizes)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = scatter_quant(gathered, eidx, 16, expand, group_counts=sizes)
    for seed in (6, 7):
        e2, x2, _ = _routing(sizes, 2, True, seed)
        eidx.copy_(e2)
        expand.copy_(x2)
        gathered.copy_(_tokens(m_src, 512, seed))
        graph.replay()
        ref = scatter_quant(gathered, eidx, 16, expand, group_counts=sizes)
        for a, b in zip(out[0], ref[0]):
            assert torch.equal(_u8(a), _u8(b))
        assert torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])


@pytest.mark.parametrize("dedup", [1, 4])
def test_scatter_quant_backward(dedup: int) -> None:
    """scatter_quant_backward equals autograd through padded_index_gather + scatter (same dy per
    token; integer-valued fp32 so the scatter-add order cannot change the sums)."""
    from pithtrain.operators.fp8_experts import scatter_quant_backward

    sizes, N = RAGGED, 256
    eidx, expand, m_src = _routing(sizes, dedup, True, 3)
    g = torch.randn(m_src, N, device="cuda").requires_grad_(True)
    out, ref_rsi, offs, ks, _ = scatter_for_grouped_gemm(padded_index_gather(g, expand), eidx, 16)
    dy_tok = torch.randint(-8, 8, (eidx.numel(), N), device="cuda").float()
    d_out = torch.randint(-8, 8, out.shape, device="cuda").float()  # padding rows: garbage
    d_out[ref_rsi] = dy_tok
    out.backward(d_out)

    _, rsi, offs2, _, _ = scatter_quant(g.detach().bfloat16(), eidx, 16, expand)
    d_x = torch.randint(-8, 8, out.shape, device="cuda").float()
    d_x[rsi] = dy_tok
    assert torch.equal(scatter_quant_backward(d_x, rsi, expand, m_src), g.grad)
