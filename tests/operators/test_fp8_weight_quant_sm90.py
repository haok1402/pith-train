"""Bit-exactness of the CuTe-DSL weight quantization kernels (``fp8_weight_quant_sm90``) against their
Triton references."""

import pytest
import torch

from pithtrain.kernels import fp8_weight_quant_sm90 as cq
from pithtrain.kernels import fp8_weight_quant_sm100 as ref
from pithtrain.kernels.arch import is_sm90

pytestmark = pytest.mark.skipif(not is_sm90(), reason="SM90 CuTe-DSL kernels")

BLOCK = 128


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return a.shape == b.shape and a.stride() == b.stride() and torch.equal(a, b)


def _weight(shape: tuple, dtype: torch.dtype = torch.bfloat16, kind: str = "randn") -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(hash((shape, kind)) % 2**31)
    x = torch.randn(shape, device="cuda", generator=g, dtype=torch.float32)
    if kind == "outliers":
        mask = torch.rand(shape, device="cuda", generator=g) < 1e-3
        x = torch.where(mask, x * 3e4, x)
        x.view(-1)[0] = 6e4
    elif kind == "zeros":
        x[..., : shape[-1] // 2] = 0.0
        if x.dim() == 3:
            x[0] = 0.0
    elif kind == "tiny":
        x = x * 1e-6
    elif kind == "wide":
        x = x * torch.exp2(torch.randint(-30, 30, shape, device="cuda", generator=g).float())
    return x.to(dtype)


SHAPES_2D = [(4096, 2048), (512, 2048), (2048, 4096), (576, 10944), (128, 128), (100, 300), (130, 7), (1, 1)]  # fmt: skip
SHAPES_3D = [(16, 768, 2048), (16, 2048, 768), (5, 384, 192), (3, 100, 260), (1, 129, 136)]
KINDS = ["randn", "outliers", "zeros", "tiny", "wide"]


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("kind", KINDS)
def test_blockwise_transpose_2d(shape: tuple, kind: str) -> None:
    x = _weight(shape, kind=kind)
    for a, b in zip(
        ref.fp8cast_blockwise_transpose_sm100(x), cq.fp8cast_blockwise_transpose_sm90(x)
    ):
        assert _same(a, b)


@pytest.mark.parametrize("shape", [(512, 384), (200, 136)])
def test_blockwise_transpose_2d_fp32(shape: tuple) -> None:
    x = _weight(shape, dtype=torch.float32, kind="wide")
    for a, b in zip(
        ref.fp8cast_blockwise_transpose_sm100(x), cq.fp8cast_blockwise_transpose_sm90(x)
    ):
        assert _same(a, b)


def test_blockwise_transpose_2d_all_zeros() -> None:
    x = torch.zeros(256, 384, device="cuda", dtype=torch.bfloat16)
    for a, b in zip(
        ref.fp8cast_blockwise_transpose_sm100(x), cq.fp8cast_blockwise_transpose_sm90(x)
    ):
        assert _same(a, b)


@pytest.mark.parametrize("shape", SHAPES_3D)
@pytest.mark.parametrize("kind", KINDS)
def test_blockwise_transpose_3d(shape: tuple, kind: str) -> None:
    x = _weight(shape, kind=kind)
    ra = ref.fp8cast_blockwise_transpose_batched_sm100(x)
    for a, b in zip(ra, cq.fp8cast_blockwise_transpose_batched_sm90(x)):
        assert _same(a, b)


def _shards(shape: tuple, n: int) -> list[tuple[torch.Tensor, int, int]]:
    """FSDP-style dim-0 shards of an fp32 master: (local (g, rows, K), row_offset, group_offset)."""
    w = _weight(shape, dtype=torch.float32, kind="wide")
    chunk = (shape[0] + n - 1) // n
    out = []
    for r in range(n):
        lo, hi = min(r * chunk, shape[0]), min(r * chunk + chunk, shape[0])
        local = w[lo:hi].contiguous()
        if len(shape) == 2:
            out.append((local.view(1, *local.shape), lo, 0))
        else:
            out.append((local, 0, lo))
    return out


SHARDED = [((4096, 2048), 16), ((512, 2048), 16), ((2048, 4096), 16), ((16, 768, 2048), 2),
           ((16, 2048, 768), 2), ((576, 10944), 5), ((5, 384, 192), 2), ((300, 260), 7),
           ((300, 258), 7), ((3, 130, 66), 2)]  # fmt: skip


def _grid(shape: tuple) -> tuple:
    G, N, K = (1, *shape) if len(shape) == 2 else shape
    return G, (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK


@pytest.mark.parametrize("shape,n", SHARDED)
def test_shard_amax_cast(shape: tuple, n: int) -> None:
    shards = _shards(shape, n)
    amax_r = torch.zeros(_grid(shape), device="cuda")
    amax_c = torch.zeros(_grid(shape), device="cuda")
    for local, ro, go in shards:
        # Each rank writes only its own tiles into a zeroed grid; compare per shard, then reduce.
        a_r, a_c = torch.zeros_like(amax_r), torch.zeros_like(amax_c)
        ref.fp8_shard_amax_sm100(local, a_r, ro, go)
        cq.fp8_shard_amax_sm90(local, a_c, ro, go)
        assert _same(a_r, a_c)
        amax_r = torch.maximum(amax_r, a_r)
        amax_c = torch.maximum(amax_c, a_c)
    for local, ro, go in shards:
        o_r = torch.empty(local.shape, dtype=torch.float8_e4m3fn, device="cuda")
        o_c = torch.empty_like(o_r)
        ref.fp8_shard_cast_sm100(local, amax_r, o_r, ro, go)
        cq.fp8_shard_cast_sm90(local, amax_c, o_c, ro, go)
        assert _same(o_r, o_c)


@pytest.mark.parametrize("shape,n", SHARDED)
def test_shard_amax_multi(shape: tuple, n: int) -> None:
    shards = _shards(shape, n)
    # Pretend every shard is a different weight of one group: separate grids, one launch.
    amax_r = [torch.zeros(_grid(shape), device="cuda") for _ in shards]
    amax_c = [torch.zeros(_grid(shape), device="cuda") for _ in shards]
    for (local, ro, go), a in zip(shards, amax_r):
        ref.fp8_shard_amax_sm100(local, a, ro, go)
    group = cq.fp8_shard_amax_multi_sm90(
        [s[0] for s in shards], amax_c, [s[1] for s in shards], [s[2] for s in shards]
    )
    group()
    for a, b in zip(amax_r, amax_c):
        assert _same(a, b)


@pytest.mark.parametrize("grid", [(1, 32, 16), (16, 6, 16), (16, 16, 6), (2, 5, 86), (3, 1, 1)])
def test_scales(grid: tuple) -> None:
    amax = _weight(grid, dtype=torch.float32, kind="wide").abs()
    amax.view(-1)[0] = 0.0
    G, R, C = grid
    s_r, st_r = torch.empty(G, R, C, device="cuda"), torch.empty(G, C, R, device="cuda")
    s_c, st_c = torch.empty_like(s_r), torch.empty_like(st_r)
    ref.fp8_scales_sm100(amax, s_r, st_r)
    cq.fp8_scales_sm90(amax, s_c, st_c)
    assert _same(s_r, s_c) and _same(st_r, st_c)
    grids = [amax, amax * 3, amax[:1]]
    outs = [(torch.empty_like(a), torch.empty(a.shape[0], C, R, device="cuda")) for a in grids]
    cq.fp8_scales_multi_sm90(grids, [o[0] for o in outs], [o[1] for o in outs])()
    for a, (s, st) in zip(grids, outs):
        s_r, st_r = torch.empty_like(s), torch.empty_like(st)
        ref.fp8_scales_sm100(a, s_r, st_r)
        assert _same(s, s_r) and _same(st, st_r)


@pytest.mark.parametrize("shape", [(1, 4096, 2048), (1, 512, 2048), (1, 2048, 4096), (16, 768, 2048), (16, 2048, 768), (1, 576, 10944), (5, 384, 192), (3, 100, 260)])  # fmt: skip
def test_pack_transpose(shape: tuple) -> None:
    x = _weight(shape, kind="wide").to(torch.float8_e4m3fn)
    numel = x.numel()
    o_r = torch.zeros(2 * numel, dtype=torch.uint8, device="cuda")
    o_c = torch.ones_like(o_r)
    ref.fp8_pack_transpose_sm100(x, o_r)
    cq.fp8_pack_transpose_sm90(x, o_c)
    assert torch.equal(o_r, o_c)
