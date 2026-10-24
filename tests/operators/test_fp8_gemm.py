"""
kernels.fp8_gemm.k_grouped_fp8_gemm_nt splits DeepGEMM's SM90 k-grouped FP8 GEMM into calls of at most
K_GROUPED_MAX_ROWS rows, since the kernel fails to launch at 116224 rows. Splitting at group
boundaries must not change the result; splitting inside a group must match a dequantized fp32
reference.
"""

import pytest
import torch

from pithtrain.kernels import fp8_gemm
from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.kernels.fp8_rowwise_kmajor import fp8cast_rowwise_kmajor

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="SM90 k-grouped path")


def operands(counts: list[int], n: int, k: int):
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    m = sum(counts)
    torch.manual_seed(0)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    _, _, a, sa = fp8cast_rowwise_kmajor(dy, offs)
    _, _, b, sb = fp8cast_rowwise_kmajor(x, offs)
    ks_t = torch.tensor(counts, device="cuda", dtype=torch.int32)
    return (a, sa), (b, sb), ks_t


def wgrad(counts, n, k, limit, monkeypatch) -> torch.Tensor:
    monkeypatch.setattr(fp8_gemm, "K_GROUPED_MAX_ROWS", limit)
    a, b, ks_t = operands(counts, n, k)
    out = torch.ones(len(counts), n, k, device="cuda")
    fp8_gemm.k_grouped_fp8_gemm_nt(a, b, out, counts, ks_t, c=out)
    return out


def dequant(data: torch.Tensor, scale: torch.Tensor, counts: list[int], ch: int) -> list:
    """Per group (ch, k_g) fp32 from the K-major flat layout and (ch, rows/128) scales."""
    out, row = [], 0
    for k in counts:
        block = data[row * ch : (row + k) * ch].view(ch, k).float()
        s = scale[:, row // 128 : (row + k) // 128].repeat_interleave(128, dim=1)
        out.append(block * s)
        row += k
    return out


@pytest.mark.parametrize("counts", [[1024, 0, 512, 2048, 128, 0, 768, 256], [128] * 40])
def test_group_boundary_split_is_exact(counts, monkeypatch) -> None:
    whole = wgrad(counts, 256, 384, 1 << 30, monkeypatch)
    split = wgrad(counts, 256, 384, 1024, monkeypatch)
    assert torch.equal(whole, split)


@pytest.mark.parametrize("counts", [[131072, 128, 0, 256], [8192] * 16, [300032, 0, 1024]])
def test_beyond_kernel_limit_matches_reference(counts, monkeypatch) -> None:
    n, k = 128, 256
    got = wgrad(counts, n, k, 65536, monkeypatch)
    a, b, _ = operands(counts, n, k)
    ra, rb = dequant(*a, counts, n), dequant(*b, counts, k)
    for g in range(len(counts)):
        ref = 1.0 + ra[g] @ rb[g].T
        err = (got[g] - ref).norm() / ref.norm()
        assert err < 1e-3, (g, err.item())  # DeepGEMM FP8 accumulation alone is ~1.2e-4
