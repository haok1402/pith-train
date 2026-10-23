"""
Parity of the BF16 CuTe expert GEMMs (bf16_cute.gemm) with an fp32 reference, on ragged groups
(empty, 1-row, not a multiple of the tile, imbalanced, more than 65536 rows), gathered rows, and
both address orders of the gate / up weights and of their main_grads.

Tolerance: a bf16 output's relative error vs the fp32 reference must be at most 1.5x that of
F.grouped_mm (the unfused GEMM, on the same inputs with the groups padded to 128 rows as the unfused path
calls it); an fp32 accumulation into a main_grad at most 1.5x that of DeepGEMM's k-grouped GEMM
(the unfused wgrad) or 1e-5, whichever is larger. The h the dgated backward re-emits must equal the
forward's h bit for bit, and every kernel must be deterministic.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.operators.bf16_cute import QUACK_VERSION, _quack_version
from pithtrain.operators.silu_mul import silu_mul

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9
    or _quack_version() != QUACK_VERSION,
    reason="needs SM90 and the validated quack-kernels",
)

COUNTS = {
    "balanced": [1024] * 32,
    "ragged": [0, 1, 300, 129, 0, 1000, 7, 512, 1, 0, 2048, 33],
    "imbalanced": [6000, 3, 0, 120, 1, 250, 0, 9],
    "large": [9000] * 8,  # 72000 rows per call
    "q35": [512] * 64,
}


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def pairs(G, n, k, scale):
    """Two (G, n, k) bf16 weights, returned in both address orders."""
    a = (torch.randn(G, n, k, device="cuda") * scale).to(torch.bfloat16)
    b = (torch.randn(G, n, k, device="cuda") * scale).to(torch.bfloat16)
    return [(a, b), (b, a)]


def grouped_mm_padded(x, w, counts, transpose_w=True):
    """The unfused F.grouped_mm call: groups padded to 128 rows, the padding rows dropped."""
    pad = [(c + 127) // 128 * 128 for c in counts]
    xp = x.new_zeros(sum(pad), x.shape[1])
    src = dst = 0
    for c, p in zip(counts, pad):
        xp[dst : dst + c] = x[src : src + c]
        src, dst = src + c, dst + p
    offs = torch.tensor(pad, device="cuda").cumsum(0).to(torch.int32)
    yp = F.grouped_mm(xp, w.transpose(1, 2) if transpose_w else w, offs=offs)
    out, src, dst = [], 0, 0
    for c, p in zip(counts, pad):
        out.append(yp[dst : dst + c])
        dst += p
    return torch.cat(out)


def per_group(x, counts, fn):
    out, s = [], 0
    for e, c in enumerate(counts):
        out.append(fn(e, x[s : s + c].float()))
        s += c
    return torch.cat(out)


def cu_of(counts):
    return torch.tensor([0] + counts, device="cuda").cumsum(0).to(torch.int32)


def setup(counts, H, inter, seed=0):
    torch.manual_seed(seed)
    M, T = sum(counts), max(sum(counts) // 2, 1)
    xr = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    a_idx = torch.randint(0, T, (M,), device="cuda", dtype=torch.int32)
    return xr, a_idx, cu_of(counts)


def _down(h, wd, cu, bias=None):
    """The down GEMM in grouped order (identity slots)."""
    from pithtrain.operators.bf16_cute import gemm

    ident = torch.arange(h.shape[0], device=h.device, dtype=torch.int32)
    return gemm.down_forward_scattered(h, wd, cu, ident, bias)


@pytest.mark.parametrize("case", list(COUNTS))
@pytest.mark.parametrize("order", [0, 1])
def test_gated_down_dgated_dgrad(case, order):
    from pithtrain.operators.bf16_cute import gemm

    counts = COUNTS[case]
    G, H = len(counts), 2048
    inter = 512 if case == "q35" else 768
    xr, a_idx, cu = setup(counts, H, inter)
    M = sum(counts)
    wg, wu = pairs(G, inter, H, H**-0.5)[order]
    wd = (torch.randn(G, H, inter, device="cuda") * inter**-0.5).to(torch.bfloat16)
    xg = xr[a_idx.long()]

    # forward gated: preact pairs (gate, up), h = silu(g) * u from the rounded preact
    pre, h = gemm.gated_forward(xr, wg, wu, cu, a_idx)
    g_ref = per_group(xg, counts, lambda e, a: a @ wg[e].float().T)
    u_ref = per_group(xg, counts, lambda e, a: a @ wu[e].float().T)
    g_torch = grouped_mm_padded(xg, wg, counts)
    assert rel(pre[:, 0::2], g_ref) <= 1.5 * rel(g_torch, g_ref) + 1e-7
    assert rel(pre[:, 1::2], u_ref) <= 1.5 * rel(grouped_mm_padded(xg, wu, counts), u_ref) + 1e-7
    g16, u16 = pre[:, 0::2].float(), pre[:, 1::2].float()
    h_ref = F.silu(g16) * u16
    h_r5 = silu_mul(pre[:, 0::2].contiguous(), pre[:, 1::2].contiguous())  # r5-base's kernel
    assert rel(h, h_ref) <= 1.5 * rel(h_r5, h_ref)

    # forward down
    y = _down(h, wd, cu)
    y_ref = per_group(h, counts, lambda e, a: a @ wd[e].float().T)
    assert rel(y, y_ref) <= 1.5 * rel(grouped_mm_padded(h, wd, counts), y_ref) + 1e-7

    # backward dgated: dy gathered by slot_of, dpre in the weights' address order, h re-emitted
    dy = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    slot_of = torch.randperm(M, device="cuda").to(torch.int32)
    dpre, h2, swapped = gemm.dgated_backward(dy, wd, pre, cu, slot_of, wg, wu)
    assert swapped == (wu.data_ptr() < wg.data_ptr())
    assert torch.equal(h, h2), "the backward must recompute the forward's h bit for bit"
    dyg = dy[slot_of.long()]
    dh_ref = per_group(dyg, counts, lambda e, a: a @ wd[e].float())
    sg = torch.sigmoid(g16)
    dg_ref = dh_ref * u16 * sg * (1 + g16 * (1 - sg))
    du_ref = dh_ref * g16 * sg
    dg, du = (dpre[:, 1::2], dpre[:, 0::2]) if swapped else (dpre[:, 0::2], dpre[:, 1::2])
    dh_torch = grouped_mm_padded(dyg, wd, counts, transpose_w=False).float()
    dg_torch = (dh_torch * u16 * sg * (1 + g16 * (1 - sg))).to(torch.bfloat16)
    du_torch = (dh_torch * g16 * sg).to(torch.bfloat16)
    assert rel(dg, dg_ref) <= 1.5 * rel(dg_torch, dg_ref) + 1e-7
    assert rel(du, du_ref) <= 1.5 * rel(du_torch, du_ref) + 1e-7

    # backward dgrad over K = 2I: one GEMM summing the gate and up halves in fp32
    dx = gemm.dgrad(dpre, wg, wu, cu, swapped)
    dgf, duf = dg.float(), du.float()
    dx_ref = per_group(torch.cat([dgf, duf], 1), counts, lambda e, a: a[:, :inter] @ wg[e].float() + a[:, inter:] @ wu[e].float())  # fmt: skip
    dx_torch = grouped_mm_padded(dg.contiguous(), wg, counts, transpose_w=False) + grouped_mm_padded(du.contiguous(), wu, counts, transpose_w=False)  # fmt: skip
    assert rel(dx, dx_ref) <= 1.5 * rel(dx_torch, dx_ref) + 1e-7

    # determinism
    for a, b in zip((pre, h, y, dpre, dx), (*gemm.gated_forward(xr, wg, wu, cu, a_idx), _down(h, wd, cu), gemm.dgated_backward(dy, wd, pre, cu, slot_of, wg, wu)[0], gemm.dgrad(dpre, wg, wu, cu, swapped))):  # fmt: skip
        assert torch.equal(a, b)


@pytest.mark.parametrize("case", ["balanced", "ragged", "imbalanced", "large"])
@pytest.mark.parametrize("w_order", [0, 1])
@pytest.mark.parametrize("mg_order", [0, 1])
def test_wgrads(case, w_order, mg_order):
    import deep_gemm

    from pithtrain.operators.bf16_cute import gemm

    counts = COUNTS[case]
    G, H, inter = len(counts), 2048, 768
    xr, a_idx, cu = setup(counts, H, inter, seed=1)
    M = sum(counts)
    wg, wu = pairs(G, inter, H, H**-0.5)[w_order]
    swapped = wu.data_ptr() < wg.data_ptr()
    dpre = torch.randn(M, 2 * inter, device="cuda", dtype=torch.bfloat16)
    h = torch.randn(M, inter, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    slot_of = torch.randperm(M, device="cuda").to(torch.int32)
    m_a, m_b = (torch.full((G, inter, H), 0.5, device="cuda") for _ in range(2))
    mg_g, mg_u = (m_a, m_b) if mg_order == 0 else (m_b, m_a)
    mg_d = torch.full((G, H, inter), 0.5, device="cuda")
    gemm.wgrad_gate_up(xr, dpre, swapped, mg_g, mg_u, cu, a_idx)
    gemm.wgrad_down(dy, h, mg_d, cu, slot_of)

    xg, dyg = xr[a_idx.long()].float(), dy[slot_of.long()].float()
    lane_g, lane_u = (1, 0) if swapped else (0, 1)
    dg, du = dpre[:, lane_g::2], dpre[:, lane_u::2]
    cs = cu.tolist()
    g_ref = 0.5 + torch.stack([dg[cs[e] : cs[e + 1]].float().T @ xg[cs[e] : cs[e + 1]] for e in range(G)])  # fmt: skip
    u_ref = 0.5 + torch.stack([du[cs[e] : cs[e + 1]].float().T @ xg[cs[e] : cs[e + 1]] for e in range(G)])  # fmt: skip
    d_ref = 0.5 + torch.stack([dyg[cs[e] : cs[e + 1]].T @ h[cs[e] : cs[e + 1]].float() for e in range(G)])  # fmt: skip

    def deepgemm_err(a, b, ref):
        """r5-base's wgrad (DeepGEMM k-grouped += fp32) on the same rows, groups zero-padded to
        128 rows as r5-base's scatter pads them."""
        pad = [(n + 127) // 128 * 128 for n in counts]
        ap, bp = a.new_zeros(sum(pad), a.shape[1]), b.new_zeros(sum(pad), b.shape[1])
        src = dst = 0
        for n, p in zip(counts, pad):
            ap[dst : dst + n], bp[dst : dst + n] = a[src : src + n], b[src : src + n]
            src, dst = src + n, dst + p
        c = torch.full_like(ref, 0.5)
        ks_t = torch.tensor(pad, device="cuda", dtype=torch.int32)
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(ap, bp, c, pad, ks_t, c=c)
        return rel(c, ref)

    tol = lambda e: max(1.5 * e, 1e-6)  # noqa: E731
    assert rel(mg_g, g_ref) <= tol(deepgemm_err(dg, xg.to(torch.bfloat16), g_ref))
    assert rel(mg_u, u_ref) <= tol(deepgemm_err(du, xg.to(torch.bfloat16), u_ref))
    assert rel(mg_d, d_ref) <= tol(deepgemm_err(dyg.to(torch.bfloat16), h, d_ref))

    # determinism: a second accumulation adds exactly the same values
    before = [t.clone() for t in (mg_g, mg_u, mg_d)]
    for t in (mg_g, mg_u, mg_d):
        t.fill_(0.5)
    gemm.wgrad_gate_up(xr, dpre, swapped, mg_g, mg_u, cu, a_idx)
    gemm.wgrad_down(dy, h, mg_d, cu, slot_of)
    for a, b in zip(before, (mg_g, mg_u, mg_d)):
        assert torch.equal(a, b)


def test_wrong_group_offsets_fail():
    """A GEMM run with shifted group offsets misses the reference: the parity check catches it."""
    from pithtrain.operators.bf16_cute import gemm

    counts = COUNTS["ragged"]
    G, H, inter = len(counts), 2048, 768
    xr, a_idx, cu = setup(counts, H, inter)
    wg, wu = pairs(G, inter, H, H**-0.5)[0]
    xg = xr[a_idx.long()]
    g_ref = per_group(xg, counts, lambda e, a: a @ wg[e].float().T)
    bad = cu.clone()
    bad[1:-1] += 1
    pre, _ = gemm.gated_forward(xr, wg, wu, bad, a_idx)
    assert rel(pre[:, 0::2], g_ref) > 1.5 * rel(grouped_mm_padded(xg, wg, counts), g_ref)


@pytest.mark.parametrize("limit", [7.0, 2.0])
def test_gptoss_gated_bias_clamp(limit):
    """gpt-oss kernels: interleaved gate/up rows of one weight, a per-expert bias added in fp32
    before the single rounding, the clamped SwiGLU of the given limit, and the bias on the down
    projection; the dgated backward re-emits the forward's h bit for bit."""
    from pithtrain.operators.bf16_cute import gemm

    counts = [700, 0, 1, 2048, 300, 129, 900, 64]
    G, H, inter = len(counts), 2880, 2880
    xr, a_idx, cu = setup(counts, H, inter, seed=3)
    M = sum(counts)
    w = (torch.randn(G, 2 * inter, H, device="cuda") * H**-0.5).to(torch.bfloat16)
    bias = (torch.randn(G, 2 * inter, device="cuda") * 2).to(torch.bfloat16)
    wd = (torch.randn(G, H, inter, device="cuda") * inter**-0.5).to(torch.bfloat16)
    bd = torch.randn(G, H, device="cuda").to(torch.bfloat16)
    act = (1.702, limit)
    pre, h = gemm.gated_forward(xr, w[:, 0::2], w[:, 1::2], cu, a_idx, bias, act)
    xg = xr[a_idx.long()]
    gu_ref = per_group(xg, counts, lambda e, a: a @ w[e].float().T + bias[e].float())
    gu_torch = grouped_mm_padded(xg, w, counts).float() + per_group(xg, counts, lambda e, a: bias[e].float().expand(a.shape[0], -1))  # fmt: skip
    assert rel(pre, gu_ref) <= 1.5 * rel(
        gu_torch.to(torch.bfloat16), gu_ref
    )  # r5-base adds in bf16
    g, u = pre[:, 0::2].float(), pre[:, 1::2].float()
    gc, uc = g.clamp(max=limit), u.clamp(-limit, limit)
    h_ref = (uc + 1) * gc * torch.sigmoid(1.702 * gc)
    assert rel(h, h_ref) <= 1.5 * rel(h_ref.to(torch.bfloat16), h_ref) + 1e-3
    y = _down(h, wd, cu, bd)
    y_ref = per_group(h, counts, lambda e, a: a @ wd[e].float().T + bd[e].float())
    bd_rows = per_group(h, counts, lambda e, a: bd[e].float().expand(a.shape[0], -1))
    y_torch = (grouped_mm_padded(h, wd, counts).float() + bd_rows).to(torch.bfloat16)  # r5-base
    assert rel(y, y_ref) <= 1.5 * rel(y_torch, y_ref)
    dy = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    slot_of = torch.randperm(M, device="cuda").to(torch.int32)
    dpre, h2, swapped = gemm.dgated_backward(dy, wd, pre, cu, slot_of, w[:, 0::2], w[:, 1::2], act)
    assert not swapped
    assert torch.equal(h, h2)
    # clamp gradients: zero where the preact saturates
    assert torch.all(dpre[:, 0::2][g > limit] == 0)
    assert torch.all(dpre[:, 1::2][u.abs() > limit] == 0)


@pytest.mark.parametrize("case", ["balanced", "ragged", "imbalanced", "q35"])
def test_down_scattered(case):
    """The down GEMM writing row r to row slot_of[r] equals the grouped GEMM then the reorder."""
    from pithtrain.operators.bf16_cute import gemm

    counts = COUNTS[case]
    G, H = len(counts), 2048
    inter = 512 if case == "q35" else 768
    M = sum(counts)
    torch.manual_seed(4)
    h = torch.randn(M, inter, device="cuda", dtype=torch.bfloat16)
    wd = (torch.randn(G, H, inter, device="cuda") * inter**-0.5).to(torch.bfloat16)
    cu = cu_of(counts)
    slot_of = torch.randperm(M, device="cuda").to(torch.int32)
    for bias in (None, torch.randn(G, H, device="cuda").to(torch.bfloat16)):
        y = _down(h, wd, cu, bias)
        out = gemm.down_forward_scattered(h, wd, cu, slot_of, bias)
        ref = torch.empty_like(y)
        ref[slot_of.long()] = y
        assert torch.equal(out, ref)
