"""
Verifier of the ``fp8_weight_quant`` feature against a pure-PyTorch 128x128 blockwise FP8
quantization (E8M0 scale per 128x128 tile from the tile amax).

- ``fp8cast_blockwise_transpose`` / ``_batched``: the blockwise FP8 of W and of W.T (per expert for
  a (G, N, K) weight), scales row-major.
- The FP8 parameter all-gather kernels, driven as ``modules/fp8_param_gather.py`` drives them:
  FSDP dim-0 shards of an fp32 master (rows of a 2D weight, experts of a 3D one) each write their
  per-tile amax of the bf16-rounded shard into a zeroed grid (``fp8_shard_amax``), the grids are
  MAX-reduced, the scales follow (``fp8_scales``), every shard is cast with them
  (``fp8_shard_cast``) and the gathered FP8 weight is packed with its transpose
  (``fp8_pack_transpose``). The result must equal the blockwise quantization of the full bf16
  weight. ``fp8_shard_amax_multi`` / ``fp8_scales_multi`` return one launch for a list of weights
  (a callable on the H200) whose results must equal the per-tensor calls.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import (
    fp8_pack_transpose,
    fp8_scales,
    fp8_scales_multi,
    fp8_shard_amax,
    fp8_shard_amax_multi,
    fp8_shard_cast,
    fp8cast_blockwise_transpose,
    fp8cast_blockwise_transpose_batched,
)

BLOCK = 128


# ---------------------------------------------------------------------------
# PyTorch reference of the recipe
# ---------------------------------------------------------------------------


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(max(amax, 1e-4) / 448)), exact (frexp, no float log2)."""
    v = amax.float().clamp(min=1e-4) / 448.0
    m, e = torch.frexp(v)  # v = m * 2**e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def tiles(x: torch.Tensor) -> torch.Tensor:
    """(G, N, K) -> zero-padded (G, R, 128, C, 128) fp32 tiles."""
    G, N, K = x.shape
    R, C = cdiv(N, BLOCK), cdiv(K, BLOCK)
    xp = F.pad(x.float(), (0, C * BLOCK - K, 0, R * BLOCK - N))
    return xp.view(G, R, BLOCK, C, BLOCK)


def ref_amax(x: torch.Tensor) -> torch.Tensor:
    """(G, R, C) per-tile amax of a (G, N, K) tensor."""
    t = tiles(x)
    if t.numel() == 0:
        return torch.zeros(t.shape[0], t.shape[1], t.shape[3], device=x.device)
    return t.abs().amax(dim=(2, 4))


def ref_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """128x128 blockwise FP8 of a (G, N, K) tensor: (q (G, N, K), scale (G, R, C))."""
    G, N, K = x.shape
    s = e8m0_scale(ref_amax(x))
    q = (tiles(x) / s[:, :, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    q = q.view(G, s.shape[1] * BLOCK, s.shape[2] * BLOCK)[:, :N, :K]
    return q.contiguous(), s


def assert_fp8_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float8_e4m3fn, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    a, b = out.view(torch.uint8), ref.view(torch.uint8)
    assert torch.equal(a, b), f"{name}: {(a != b).sum().item()} mismatching bytes"


def assert_f32_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float32, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    assert torch.equal(out, ref), f"{name}: {(out != ref).sum().item()} mismatching values"


def make_weight(shape: tuple, dtype: torch.dtype = torch.bfloat16, kind: str = "randn") -> torch.Tensor:  # fmt: skip
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(repr((shape, kind)).encode()))
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
    elif kind == "huge":
        x = x * 1e30
    elif kind == "negative":
        x = -x.abs()
    elif kind == "pow2":
        x = torch.exp2(torch.randint(-20, 20, shape, device="cuda", generator=g).float()) * 448.0
    return x.to(dtype)


KINDS = ["randn", "outliers", "zeros", "tiny", "wide", "huge", "negative", "pow2"]


# ---------------------------------------------------------------------------
# fp8cast_blockwise_transpose / fp8cast_blockwise_transpose_batched
# ---------------------------------------------------------------------------


def check_blockwise_transpose(x: torch.Tensor) -> None:
    w_fp8, w_scale, wt_fp8, wt_scale = fp8cast_blockwise_transpose(x)
    q, s = ref_blockwise(x.unsqueeze(0))
    qt, st = ref_blockwise(x.T.unsqueeze(0))
    assert_fp8_equal("w_fp8", w_fp8, q[0])
    assert_f32_equal("w_scale", w_scale, s[0])
    assert_fp8_equal("wt_fp8", wt_fp8, qt[0])
    assert_f32_equal("wt_scale", wt_scale, st[0])
    assert w_scale.is_contiguous() and wt_scale.is_contiguous()  # row-major by contract
    assert torch.equal(w_scale, wt_scale.T)


def check_blockwise_transpose_batched(x: torch.Tensor) -> None:
    w_fp8, w_scale, wt_fp8, wt_scale = fp8cast_blockwise_transpose_batched(x)
    q, s = ref_blockwise(x)
    qt, st = ref_blockwise(x.transpose(1, 2))
    assert_fp8_equal("w_fp8", w_fp8, q)
    assert_f32_equal("w_scale", w_scale, s)
    assert_fp8_equal("wt_fp8", wt_fp8, qt)
    assert_f32_equal("wt_scale", wt_scale, st)
    assert w_scale.is_contiguous() and wt_scale.is_contiguous()
    assert torch.equal(w_scale, wt_scale.transpose(1, 2))


# (N, K) weights of the four models (fused gate_up / down / dense MLP / attention projections),
# tile edges, unaligned and tiny shapes.
SHAPES_2D = [
    (4096, 2048),
    (1536, 2048),
    (2048, 768),
    (2816, 2048),
    (2048, 1408),
    (21888, 2048),
    (2048, 10944),
    (5760, 2880),
    (2880, 2880),
    (512, 2048),
    (576, 10944),
    (128, 128),
    (256, 256),
    (300, 384),
    (100, 300),
    (64, 64),
    (7, 300),
    (130, 7),
    (1, 1),
    (256, 7168),
]
# (G, N, K) expert weights: gate_up (G, 2I, H) and down (G, H, I) of each model, plus odd shapes.
SHAPES_3D = [
    (4, 1536, 2048),
    (4, 2048, 768),
    (4, 2816, 2048),
    (4, 2048, 1408),
    (4, 1024, 2048),
    (4, 2048, 512),
    (2, 5760, 2880),
    (2, 2880, 2880),
    (1, 128, 128),
    (4, 256, 128),
    (8, 256, 512),
    (2, 300, 384),
    (4, 64, 64),
    (5, 384, 192),
    (3, 100, 260),
    (1, 129, 136),
]


@pytest.mark.parametrize("shape", SHAPES_2D, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("kind", KINDS)
def test_blockwise_transpose(shape: tuple, kind: str) -> None:
    check_blockwise_transpose(make_weight(shape, kind=kind))


@pytest.mark.parametrize("shape", [(512, 384), (200, 136), (2048, 768)])
@pytest.mark.parametrize("kind", ["randn", "wide"])
def test_blockwise_transpose_fp32(shape: tuple, kind: str) -> None:
    check_blockwise_transpose(make_weight(shape, dtype=torch.float32, kind=kind))


def test_blockwise_transpose_all_zeros() -> None:
    check_blockwise_transpose(torch.zeros(256, 384, device="cuda", dtype=torch.bfloat16))
    check_blockwise_transpose(torch.zeros(128, 256, device="cuda", dtype=torch.bfloat16))


def test_blockwise_transpose_strided() -> None:
    """A row-strided view (a slice of a wider buffer)."""
    check_blockwise_transpose(make_weight((640, 2304), kind="outliers")[:, :2048])


@pytest.mark.parametrize("shape", SHAPES_3D, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("kind", KINDS)
def test_blockwise_transpose_batched(shape: tuple, kind: str) -> None:
    check_blockwise_transpose_batched(make_weight(shape, kind=kind))


def test_blockwise_transpose_batched_fp32_and_zeros() -> None:
    check_blockwise_transpose_batched(make_weight((3, 260, 384), dtype=torch.float32, kind="wide"))
    check_blockwise_transpose_batched(torch.zeros(4, 128, 256, device="cuda", dtype=torch.bfloat16))


def test_blockwise_transpose_batched_empty() -> None:
    for shape in [(0, 256, 128), (2, 0, 128), (2, 128, 0)]:
        x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        w_fp8, w_scale, wt_fp8, wt_scale = fp8cast_blockwise_transpose_batched(x)
        G, N, K = shape
        assert w_fp8.shape == (G, N, K) and wt_fp8.shape == (G, K, N)
        assert w_scale.shape == (G, cdiv(N, BLOCK), cdiv(K, BLOCK))
        assert wt_scale.shape == (G, cdiv(K, BLOCK), cdiv(N, BLOCK))


# ---------------------------------------------------------------------------
# FP8 parameter all-gather kernels
# ---------------------------------------------------------------------------


def make_shards(w: torch.Tensor, n: int) -> list[tuple[torch.Tensor, int, int]]:
    """
    FSDP dim-0 shards of an fp32 master as fp8_param_gather sees them: (local (g_local, rows, K),
    row_offset, group_offset). A rank past the end holds an empty shard.
    """
    chunk = cdiv(w.shape[0], n)
    out = []
    for r in range(n):
        lo, hi = min(r * chunk, w.shape[0]), min(r * chunk + chunk, w.shape[0])
        local = w[lo:hi].contiguous()
        if w.dim() == 2:
            out.append((local.view(1, *local.shape), lo, 0))
        else:
            out.append((local, 0, lo))
    return out


def grid_of(shape: tuple) -> tuple[int, int, int]:
    G, N, K = (1, *shape) if len(shape) == 2 else shape
    return G, cdiv(N, BLOCK), cdiv(K, BLOCK)


def as_3d(w: torch.Tensor) -> torch.Tensor:
    return w.unsqueeze(0) if w.dim() == 2 else w


def shard_alone(shape: tuple, local: torch.Tensor, ro: int, go: int) -> torch.Tensor:
    """The full (G, N, K) bf16-rounded weight with everything but this shard zeroed."""
    full = torch.zeros(as_3d(torch.empty(shape, device="meta")).shape, device=local.device)
    g, rows, _ = local.shape
    full[go : go + g, ro : ro + rows] = local.bfloat16().float()
    return full


# (weight shape, number of shards): 2D weights shard their rows, 3D weights their experts; tiles
# straddle shards where the shard size is not a multiple of 128, and some ranks hold nothing.
SHARDED = [
    ((4096, 2048), 16),
    ((512, 2048), 16),
    ((2048, 4096), 16),
    ((1536, 2048), 8),
    ((2048, 1408), 8),
    ((1000, 2880), 4),
    ((5760, 2880), 8),
    ((576, 10944), 5),
    ((300, 260), 7),
    ((300, 258), 7),
    ((6, 256), 4),
    ((16, 768, 2048), 2),
    ((16, 2048, 768), 2),
    ((8, 1536, 2048), 4),
    ((4, 2048, 1408), 4),
    ((2, 5760, 2880), 2),
    ((5, 384, 192), 2),
    ((5, 384, 192), 4),
    ((3, 130, 66), 2),
]


@pytest.mark.parametrize("shape,n", SHARDED, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v))  # fmt: skip
@pytest.mark.parametrize("kind", ["wide", "outliers"])
def test_param_gather_pipeline(shape: tuple, n: int, kind: str) -> None:
    master = make_weight(shape, dtype=torch.float32, kind=kind)
    shards = make_shards(master, n)
    G, R, C = grid_of(shape)
    full_q, full_s = ref_blockwise(as_3d(master.bfloat16()))
    full_qt, full_st = ref_blockwise(as_3d(master.bfloat16()).transpose(1, 2))

    # 1. Per-shard amax into a zeroed grid (only the tiles the shard touches), then MAX-reduce.
    amax = torch.zeros(G, R, C, device="cuda")
    for local, ro, go in shards:
        a = torch.zeros(G, R, C, device="cuda")
        fp8_shard_amax(local, a, ro, go)
        assert_f32_equal("shard amax", a, ref_amax(shard_alone(shape, local, ro, go)))
        amax = torch.maximum(amax, a)
    assert_f32_equal("reduced amax", amax, ref_amax(as_3d(master.bfloat16())))

    # 2. Scales and their transpose.
    scale = torch.full((G, R, C), -1.0, device="cuda")
    scale_t = torch.full((G, C, R), -1.0, device="cuda")
    fp8_scales(amax, scale, scale_t)
    assert_f32_equal("scale", scale, full_s)
    assert_f32_equal("scale_t", scale_t, full_st)

    # 3. Each shard cast with the reduced grid; the gathered bytes are the full weight's FP8.
    parts = []
    for local, ro, go in shards:
        out = torch.empty(local.shape, dtype=torch.float8_e4m3fn, device="cuda")
        fp8_shard_cast(local, amax, out, ro, go)
        parts.append(out)
    gathered = torch.cat(parts, dim=1 if len(shape) == 2 else 0)
    assert_fp8_equal("gathered fp8", gathered, full_q)

    # 4. Pack with the transpose into one flat byte buffer.
    numel = gathered.numel()
    packed = torch.full((2 * numel,), 0x55, dtype=torch.uint8, device="cuda")
    fp8_pack_transpose(gathered, packed)
    assert_fp8_equal("packed fp8", packed[:numel].view(torch.float8_e4m3fn).view_as(full_q), full_q)
    packed_t = packed[numel:].view(torch.float8_e4m3fn).view(G, *full_qt.shape[1:])
    assert_fp8_equal("packed fp8_t", packed_t, full_qt)

    # Bit-identical to the blockwise transpose of the bf16 weight (the flag-off path).
    if len(shape) == 2:
        ref = fp8cast_blockwise_transpose(master.bfloat16())
        ref = [t.unsqueeze(0) for t in ref]
    else:
        ref = fp8cast_blockwise_transpose_batched(master.bfloat16())
    assert_fp8_equal("vs blockwise fp8", gathered, ref[0])
    assert_f32_equal("vs blockwise scale", scale, ref[1])
    assert_fp8_equal("vs blockwise fp8_t", packed_t, ref[2])
    assert_f32_equal("vs blockwise scale_t", scale_t, ref[3])


def test_shard_amax_rounds_to_bf16() -> None:
    """The amax and the cast see the bf16-rounded fp32 master."""
    # 448 * (1 + 2**-12) has scale 2 in fp32 but rounds to 448 (scale 1) in bf16.
    w = torch.full((1, 128, 128), 0.5, device="cuda")
    w[0, 3, 5] = 448.0 * (1 + 2**-12)
    amax = torch.zeros(1, 1, 1, device="cuda")
    fp8_shard_amax(w, amax, 0, 0)
    assert amax.item() == 448.0
    scale, scale_t = torch.empty(1, 1, 1, device="cuda"), torch.empty(1, 1, 1, device="cuda")
    fp8_scales(amax, scale, scale_t)
    assert scale.item() == 1.0 and scale_t.item() == 1.0
    out = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device="cuda")
    fp8_shard_cast(w, amax, out, 0, 0)
    assert_fp8_equal("cast", out, w.bfloat16().float().to(torch.float8_e4m3fn))


@pytest.mark.parametrize("grid", [(1, 32, 16), (16, 6, 16), (16, 16, 6), (2, 5, 86), (3, 1, 1), (1, 171, 23)])  # fmt: skip
def test_scales(grid: tuple) -> None:
    amax = make_weight(grid, dtype=torch.float32, kind="wide").abs()
    amax.view(-1)[0] = 0.0
    amax.view(-1)[-1] = 448.0 * 2.0**-5  # exact power of two times 448
    G, R, C = grid
    scale, scale_t = torch.empty(G, R, C, device="cuda"), torch.empty(G, C, R, device="cuda")
    fp8_scales(amax, scale, scale_t)
    assert_f32_equal("scale", scale, e8m0_scale(amax))
    assert_f32_equal("scale_t", scale_t, e8m0_scale(amax).transpose(1, 2))


@pytest.mark.parametrize("shape", [(1, 4096, 2048), (1, 512, 2048), (1, 2048, 4096), (16, 768, 2048), (16, 2048, 768), (1, 576, 10944), (2, 5760, 2880), (5, 384, 192), (3, 100, 260), (1, 129, 136), (2, 24, 40)])  # fmt: skip
def test_pack_transpose(shape: tuple) -> None:
    x = make_weight(shape, kind="wide").to(torch.float8_e4m3fn)
    numel = x.numel()
    out = torch.full((2 * numel,), 0x55, dtype=torch.uint8, device="cuda")
    fp8_pack_transpose(x, out)
    assert torch.equal(out[:numel], x.view(torch.uint8).reshape(-1))
    assert torch.equal(out[numel:], x.transpose(1, 2).contiguous().view(torch.uint8).reshape(-1))


# ---------------------------------------------------------------------------
# One-launch variants over a shard group
# ---------------------------------------------------------------------------

# The FP8 weights of one FSDP shard group: 2D and 3D, tiles straddling shards, an unaligned K
# (258, no 16-byte rows), and a weight whose last rank holds nothing.
GROUP = [(1536, 2048), (2048, 768), (1000, 2880), (4, 1536, 2048), (4, 2048, 768), (300, 258), (6, 256), (3, 130, 66)]  # fmt: skip
RANKS = 4


def packed_grids(device: str = "cuda") -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One zeroed amax buffer holding every weight's grid back to back, as fp8_param_gather packs it."""
    sizes = [grid_of(s) for s in GROUP]
    buffer = torch.zeros(sum(g * r * c for g, r, c in sizes), device=device)
    views, start = [], 0
    for g, r, c in sizes:
        views.append(buffer[start : start + g * r * c].view(g, r, c))
        start += g * r * c
    return buffer, views


def test_multi_launch_param_gather() -> None:
    masters = [make_weight(s, dtype=torch.float32, kind="wide") for s in GROUP]
    per_rank = [[make_shards(w, RANKS)[r] for w in masters] for r in range(RANKS)]

    reduced, reduced_views = packed_grids()
    launches = []
    for r in range(RANKS):
        shards = per_rank[r]
        buffer, amaxes = packed_grids()
        launch = fp8_shard_amax_multi(
            [s[0] for s in shards], amaxes, [s[1] for s in shards], [s[2] for s in shards]
        )
        assert launch is not None and callable(launch), "H200 must have a one-launch amax"
        launch()
        # Equal to the per-tensor kernel, weight by weight.
        for (local, ro, go), a, shape in zip(shards, amaxes, GROUP):
            single = torch.zeros_like(a)
            fp8_shard_amax(local, single, ro, go)
            assert_f32_equal(f"multi amax {shape} rank {r}", a, single)
        launches.append((launch, buffer, amaxes))
        torch.maximum(reduced, buffer, out=reduced)

    for w, a, shape in zip(masters, reduced_views, GROUP):
        assert_f32_equal(f"reduced amax {shape}", a, ref_amax(as_3d(w.bfloat16())))

    scales = [torch.full(grid_of(s), -1.0, device="cuda") for s in GROUP]
    scales_t = [torch.full(grid_of(s), -1.0, device="cuda").transpose(1, 2).contiguous() for s in GROUP]  # fmt: skip
    scales_launch = fp8_scales_multi(reduced_views, scales, scales_t)
    assert scales_launch is not None and callable(scales_launch), (
        "H200 must have a one-launch scales"
    )
    scales_launch()
    for w, a, s, st, shape in zip(masters, reduced_views, scales, scales_t, GROUP):
        single, single_t = torch.empty_like(s), torch.empty_like(st)
        fp8_scales(a, single, single_t)
        assert_f32_equal(f"multi scale {shape}", s, single)
        assert_f32_equal(f"multi scale_t {shape}", st, single_t)
        q, ref_s = ref_blockwise(as_3d(w.bfloat16()))
        assert_f32_equal(f"scale {shape}", s, ref_s)
        assert_f32_equal(f"scale_t {shape}", st, ref_s.transpose(1, 2))
        # The shards cast with the reduced grid gather to the full weight's FP8.
        parts = []
        for r in range(RANKS):
            local, ro, go = per_rank[r][GROUP.index(shape)]
            out = torch.empty(local.shape, dtype=torch.float8_e4m3fn, device="cuda")
            fp8_shard_cast(local, a, out, ro, go)
            parts.append(out)
        assert_fp8_equal(f"gathered {shape}", torch.cat(parts, dim=1 if len(shape) == 2 else 0), q)

    # The launch objects are built once and reused: change the masters in place, re-zero the
    # buffers and launch again (as precompute_fp8_param_scales does after optimizer.step()).
    for w in masters:
        w.mul_(-3.0).add_(0.25)
    per_rank_new = [[make_shards(w, RANKS)[r] for w in masters] for r in range(RANKS)]
    for r, (launch, buffer, amaxes) in enumerate(launches):
        for (old, _, _), (new, _, _) in zip(per_rank[r], per_rank_new[r]):
            old.copy_(new)
        buffer.zero_()
        launch()
        for (local, ro, go), a, shape in zip(per_rank[r], amaxes, GROUP):
            single = torch.zeros_like(a)
            fp8_shard_amax(local, single, ro, go)
            assert_f32_equal(f"relaunch amax {shape} rank {r}", a, single)
    reduced.zero_()
    for _, buffer, _ in launches:
        torch.maximum(reduced, buffer, out=reduced)
    scales_launch()
    for w, s, st, shape in zip(masters, scales, scales_t, GROUP):
        _, ref_s = ref_blockwise(as_3d(w.bfloat16()))
        assert_f32_equal(f"relaunch scale {shape}", s, ref_s)
        assert_f32_equal(f"relaunch scale_t {shape}", st, ref_s.transpose(1, 2))


@pytest.mark.parametrize("grid", [(1, 32, 16), (16, 6, 16), (2, 5, 86), (3, 1, 1)])
def test_scales_multi(grid: tuple) -> None:
    amax = make_weight(grid, dtype=torch.float32, kind="wide").abs()
    amax.view(-1)[0] = 0.0
    G, R, C = grid
    grids = [amax, amax * 3, amax[:1].contiguous()]
    outs = [(torch.empty_like(a), torch.empty(a.shape[0], C, R, device="cuda")) for a in grids]
    launch = fp8_scales_multi(grids, [o[0] for o in outs], [o[1] for o in outs])
    assert launch is not None and callable(launch)
    launch()
    for a, (s, st) in zip(grids, outs):
        assert_f32_equal("scale", s, e8m0_scale(a))
        assert_f32_equal("scale_t", st, e8m0_scale(a).transpose(1, 2))
