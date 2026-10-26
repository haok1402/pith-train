"""
Verifier of ``scatter_quant`` (feature ``fp8_scatter_quant``) against plain PyTorch.

Contract: dedup expansion (``gathered[expand_idx]``) + scatter of the tokens into per-expert
groups padded with zero rows to 128 + ``quantize_grouped_input`` of the result (SM90 layout:
rowwise 1x128 plus the per-channel FP8 K-major per group), returning ``(quantized 4-tuple,
grouped_mm_offs, reverse_shuffle_idxs, ks, ks_tensor)``.

Reference: a PyTorch bincount of ``expert_idxs`` gives ``ks`` (padded group sizes), ``ks_tensor``
and ``grouped_mm_offs``. Row placement within a group is the kernel's choice, so
``reverse_shuffle_idxs`` is only required to be a valid placement (each token inside its expert's
rows, all distinct); the reference scatters the expanded tokens through it into a zero bf16
matrix and quantizes that with a PyTorch implementation of the recipe, and every quantized output
(padding rows included, which must be zero bytes) must equal it exactly. On SM90 the placement is
stable (token order within each expert); that is checked too.

Recipe: per 128-element block ``amax = max |x|``, ``scale = 2^ceil(log2(max(amax, 1e-4) / 448))``
(computed as the kernels do: fp32 product with 1/448, then a power-of-two ceil by bit arithmetic;
the same value), ``q = e4m3_satfinite(x * (1 / scale))`` (round to nearest even).

    pytest tests/kernels/test_fp8_scatter_quant.py -v
"""

import pytest
import torch

from pithtrain.kernels import fp8_fused_experts_supported, scatter_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8_fused_experts_supported(),
    reason="the fused FP8 expert ops run only where fp8_fused_experts_supported()",
)

BLOCK = 128

# ---------------------------------------------------------------- PyTorch quantization reference


def _e8m0(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The recipe's power-of-two scale of an fp32 amax, and its exact reciprocal."""
    a = torch.maximum(amax, torch.full_like(amax, 1e-4))
    s = a * torch.full_like(a, 1.0 / 448.0)
    bits = s.view(torch.int32)
    bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    scale = torch.maximum(bits.view(torch.float32), torch.full_like(a, 2.0**-126))
    rcp = ((254 << 23) - scale.view(torch.int32)).view(torch.float32)
    return scale, rcp


def _q(x: torch.Tensor, rcp: torch.Tensor) -> torch.Tensor:
    return (x * rcp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_grouped_ref(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``quantize_grouped_input`` on SM90 in PyTorch: ``(x_fp8 (M, K) as uint8, x_scale (M,
    ceil(K/128)), x_ch flat (M*K,) uint8 K-major per group, x_ch_scale (K, M/128))``."""
    M, K = x.shape
    nb = -(-K // BLOCK)
    xf = torch.zeros(M, nb * BLOCK, dtype=torch.float32, device=x.device)
    xf[:, :K] = x.float()
    s_row, r_row = _e8m0(xf.abs().view(M, nb, BLOCK).amax(-1))
    q_row = _q(xf.view(M, nb, BLOCK), r_row[..., None]).view(M, nb * BLOCK)[:, :K]
    xc = xf[:, :K].reshape(M // BLOCK, BLOCK, K)
    s_ch, r_ch = _e8m0(xc.abs().amax(1))
    q_ch = _q(xc, r_ch[:, None, :]).view(M, K)
    flat = torch.zeros(M * K, dtype=torch.uint8, device=x.device)
    start = 0
    for end in offs.tolist():
        flat[start * K : end * K].view(K, end - start).copy_(q_ch[start:end].T)
        start = end
    return q_row.contiguous(), s_row, flat, s_ch.T.contiguous()


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


# --------------------------------------------------------------------------------- inputs


def _routing(
    sizes: list[int], dedup: int, segmented: bool, seed: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """expert_idxs with the given per-expert sizes, plus an expand_idx with ~``dedup`` slots per
    source row. ``segmented`` mimics EP dispatch (runs per sender); otherwise a random order."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    eidx = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    m = eidx.numel()
    if segmented:
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
    x[5::101] = -x[5::101].abs()
    return x.to(torch.bfloat16)


# ---------------------------------------------------------------------------------- checker


def _ref_layout(eidx: torch.Tensor, G: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    counts = torch.bincount(eidx, minlength=G)
    ks_t = ((counts + BLOCK - 1) // BLOCK * BLOCK).to(torch.int32)
    return ks_t.cumsum(0).to(torch.int32), ks_t, ks_t.tolist()


def check(
    gathered: torch.Tensor,
    eidx: torch.Tensor,
    G: int,
    expand: torch.Tensor | None,
    counts: list[int] | None,
    out: tuple,
) -> None:
    xq, rsi, offs, ks, kst = out
    m, N = eidx.shape[0], gathered.shape[1]
    x = gathered if expand is None else gathered[expand]

    ref_offs, ref_kst, ref_ks = _ref_layout(eidx, G)
    assert offs.dtype == torch.int32 and torch.equal(offs, ref_offs)
    assert kst.dtype == torch.int32 and torch.equal(kst, ref_kst)
    assert isinstance(ks, list) and ks == ref_ks
    if counts is not None:
        assert ks == [(c + BLOCK - 1) // BLOCK * BLOCK for c in counts]
    M = sum(ref_ks)

    # reverse_shuffle_idxs: a valid placement (inside the token's group, all rows distinct)
    assert rsi.dtype == torch.int64 and rsi.shape == (m,)
    starts = (ref_offs - ref_kst).long()
    counts_t = torch.bincount(eidx, minlength=G)
    rel = rsi - starts[eidx]
    assert bool((rel >= 0).all()) and bool((rel < counts_t[eidx]).all())
    assert torch.unique(rsi).numel() == m

    # every quantized output, padding rows included, against PyTorch through that placement
    dense = torch.zeros(M, N, dtype=torch.bfloat16, device=gathered.device)
    dense[rsi] = x
    ref = quantize_grouped_ref(dense, ref_offs)
    shapes = [(M, N), (M, -(-N // BLOCK)), (M * N,), (N, M // BLOCK)]
    names = ("x_fp8", "x_scale", "x_ch_fp8", "x_ch_scale")
    for name, a, b, shape in zip(names, xq, ref, shapes):
        assert tuple(a.shape) == shape, (name, a.shape, shape)
        assert a.dtype == (torch.float8_e4m3fn if name.endswith("fp8") else torch.float32), name
        if not torch.equal(_u8(a), b):
            raise AssertionError(f"{name}: {(_u8(a) != b).sum().item()} mismatches")
    pad = torch.ones(M, dtype=torch.bool, device=gathered.device)
    pad[rsi] = False
    assert not bool(_u8(xq[0])[pad].any()), "padding rows must be zero bytes"


def check_stable(eidx: torch.Tensor, G: int, rsi: torch.Tensor, offs: torch.Tensor) -> None:
    """Token order within each expert (the SM90 placement)."""
    kst = torch.diff(offs, prepend=offs.new_zeros(1))
    starts = (offs - kst).long()
    order = torch.argsort(eidx, stable=True)
    rank = torch.empty_like(order)
    pos = torch.arange(eidx.numel(), device=eidx.device)
    first = torch.searchsorted(eidx[order], torch.arange(G, device=eidx.device))
    rank[order] = pos - first[eidx[order]]
    assert torch.equal(rsi, starts[eidx] + rank)


def _run(sizes: list[int], N: int, dedup: int, segmented: bool, host_counts: bool) -> None:
    G = len(sizes)
    seed = sum(sizes) * 7 + N + dedup
    eidx, expand, m_src = _routing(sizes, dedup, segmented, seed)
    use_expand = dedup > 1 or segmented
    counts = sizes if host_counts else None
    if use_expand:
        gathered = _tokens(m_src, N, seed)
        out = scatter_quant(gathered, eidx, G, expand, group_counts=counts)
        check(gathered, eidx, G, expand, counts, out)
    else:
        gathered = _tokens(eidx.numel(), N, seed)
        out = scatter_quant(gathered, eidx, G, padding_alignment=128, group_counts=counts)
        check(gathered, eidx, G, None, counts, out)
    check_stable(eidx, G, out[1], out[2])


BALANCED = [2048] * 16
IMBALANCED = [0, 128, 384, 0, 0, 1024, 256, 128, 0, 2944, 128, 0, 640, 384, 128, 0]
RAGGED = [1, 0, 300, 129, 0, 77, 1000, 5, 0, 0, 255, 256, 17, 0, 640, 3]


# ------------------------------------------------------------------------------------ tests


@pytest.mark.parametrize("dedup", [1, 2, 4, 8])
@pytest.mark.parametrize("host_counts", [True, False])
@pytest.mark.parametrize("sizes", [BALANCED, IMBALANCED, RAGGED], ids=["bal", "imb", "ragged"])
def test_scatter_quant(sizes: list[int], host_counts: bool, dedup: int) -> None:
    _run(sizes, 2048, dedup, segmented=True, host_counts=host_counts)


@pytest.mark.parametrize(
    "sizes,N",
    [
        ([4096] * 16, 2048),  # M = 65536
        ([8192, 0, 0, 0, 8192, 0, 0, 16384], 2048),
        (RAGGED, 768),
        (RAGGED, 2880),  # gpt-oss hidden
        (RAGGED, 200),  # N not a multiple of 128
        ([5, 0, 3], 136),
        ([0, 0, 1], 8),
        ([300] * 128, 256),  # many groups
        ([7] * 300, 128),
        ([70000], 64),  # one group, > 2^16 tokens
    ],
)
def test_scatter_quant_shapes(sizes: list[int], N: int) -> None:
    _run(sizes, N, 3, segmented=False, host_counts=False)


@pytest.mark.parametrize("host_counts", [True, False])
def test_scatter_quant_identity(host_counts: bool) -> None:
    """No expand_idx: ``gathered`` holds one row per token."""
    _run(IMBALANCED, 2048, 1, segmented=False, host_counts=host_counts)
    _run(RAGGED, 2880, 1, segmented=False, host_counts=host_counts)


def test_scatter_quant_n_not_multiple_of_8() -> None:
    """N % 8 != 0: placement may differ, outputs are still checked through reverse_shuffle_idxs."""
    sizes, G, N = RAGGED, len(RAGGED), 100
    eidx, expand, m_src = _routing(sizes, 2, True, 9)
    gathered = _tokens(m_src, N, 9)
    check(gathered, eidx, G, expand, None, scatter_quant(gathered, eidx, G, expand))


def test_scatter_quant_determinism() -> None:
    eidx, _, _ = _routing(IMBALANCED, 1, False, 0)
    x = _tokens(eidx.numel(), 2048, 1)
    a = scatter_quant(x, eidx, 16)
    b = scatter_quant(x, eidx, 16)
    for u, v in zip(a[0], b[0]):
        assert torch.equal(_u8(u), _u8(v))
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and a[3] == b[3]


def test_scatter_quant_back_to_back() -> None:
    """Many launches with a changing token count."""
    for it, sizes in enumerate([[2048] * 16, [5, 0, 3] + [0] * 13, [4096] * 16, [1] * 16] * 3):
        _run(sizes, 128, 2, segmented=it % 2 == 0, host_counts=True)


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
        check(gathered, eidx, 16, expand, sizes, out)


def test_scatter_quant_zeros() -> None:
    """All-zero tokens: zero bytes everywhere and the scale floor."""
    eidx, expand, m_src = _routing(RAGGED, 2, True, 11)
    gathered = torch.zeros(m_src, 768, dtype=torch.bfloat16, device="cuda")
    out = scatter_quant(gathered, eidx, len(RAGGED), expand)
    check(gathered, eidx, len(RAGGED), expand, None, out)
    floor = _e8m0(torch.zeros(1, device="cuda"))[0].item()
    assert bool((out[0][1] == floor).all()) and bool((out[0][3] == floor).all())
