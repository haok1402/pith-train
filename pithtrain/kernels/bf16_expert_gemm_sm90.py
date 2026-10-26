"""
SM90 implementations of the bf16_expert_gemm functions: QuACK's SM90 GEMM class with our own
epilogues, compiled once per (kind, lane order, tile config, device, activation) through TVM-FFI.
Tile configs are fixed per shape (CONFIGS); QuACK's autotuner is not used.

Gate and up are two tensors read as one operand: a tensor over the lower address whose size-2 pair
mode strides to the higher one, so they act as one (G, 2I, H) weight with gate and up interleaved
(TMA takes any positive 16-byte-multiple stride). Lane 0 is the lower address, gate or up; the
epilogue variant is picked per call from the pointer order.
"""

from functools import lru_cache
from typing import Callable, NamedTuple

import cutlass
import cutlass.cute as cute
import quack.sm90_utils as quack_sm90_utils
import torch
from cutlass import BFloat16, Float32, Int32
from quack.activation import dswiglu, dswiglu_oai, swiglu, swiglu_oai
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_max_active_clusters
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import pack, unpack
from quack.epilogue.ops import EpiOp, RowVecLoad, TileStore
from quack.rounding import RoundingMode
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments

from pithtrain.kernels.cute_compile_sm90 import fake_stream


def round_bf16(v):
    if isinstance(v, tuple):
        return type(v)(*(round_bf16(x) for x in v))
    return v.to(BFloat16).to(Float32)


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def gated(acc):
    a, b = unpack(acc)
    g, u = round_bf16(a), round_bf16(b)
    return {"D": pack(g, u), "postact": swiglu(g, u)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def gated_swapped(acc):
    a, b = unpack(acc)
    g, u = round_bf16(b), round_bf16(a)
    return {"D": pack(g, u), "postact": swiglu(g, u)}


@gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
def dgated(acc, c):
    g, u = unpack(c)
    dg, du, h = dswiglu(g, u, acc)
    return {"D": pack(dg, du), "postact": h}


@gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
def dgated_swapped(acc, c):
    g, u = unpack(c)
    dg, du, h = dswiglu(g, u, acc)
    return {"D": pack(du, dg), "postact": h}


@gemm_epilogue()
def plain(acc):
    return {"D": acc}


@gemm_epilogue(mode="acc_pair")
def pair_swapped(acc):
    a, b = unpack(acc)
    return {"D": pack(b, a)}


# gpt-oss: the gate / up rows interleave, so gate is always the lower address and there is no
# swapped variant.


@lru_cache(maxsize=None)
def make_gated_oai(alpha: float, limit: float):
    @gemm_epilogue(outputs=("postact",), mode="acc_pair", ops={"bias": RowVecLoad("bias")})
    def gated_oai(acc, bias):
        g, u = unpack(acc + bias)
        g, u = round_bf16(g), round_bf16(u)
        return {"D": pack(g, u), "postact": swiglu_oai(g, u, alpha, limit)}

    return gated_oai


@lru_cache(maxsize=None)
def make_dgated_oai(alpha: float, limit: float):
    @gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
    def dgated_oai(acc, c):
        g, u = unpack(c)
        dg, du, h = dswiglu_oai(g, u, acc, alpha, limit)
        return {"D": pack(dg, du), "postact": h}

    return dgated_oai


class IndexArg(EpiOp):
    """A tensor handed through to the epilogue params (read by ScatterRowStore)."""

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}


class ScatterRowStore(TileStore):
    """
    TileStore of out (slots, N) writing row r of the grouped output to out[idx[r]]: the tile is
    staged in smem as for a TMA store, then all epilogue threads copy its rows with 16-byte stores.
    """

    def __init__(self, name: str, idx_name: str):
        super().__init__(name)
        self.idx_name = idx_name

    def config_key(self):
        return (*super().config_key(), self.idx_name)

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
        assert layout.is_n_major_c()
        setattr(gemm, self._layout_gemm_attr(), layout)
        setattr(gemm, self._dtype_gemm_attr(), tensor.element_type)
        smem_layout = quack_sm90_utils.make_smem_layout_epi(tensor.element_type, layout, gemm.epi_tile, gemm.epi_stage)
        return {self._tma_atom_key(): None, self.name: tensor, self._smem_layout_key(): smem_layout, self._epi_tile_key(): gemm.epi_tile, self._dtype_field(): tensor.element_type}

    def tma_atoms(self, gemm, params):
        return []

    def store_setup(self, gemm, params, smem_tensor, tiled_copy_r2s, tiled_copy_t2r, tile_coord_mnkl, varlen_manager, tidx):
        tiled_copy = self._make_tiled_copy_r2s(gemm, params, tiled_copy_r2s, tiled_copy_t2r)
        batch = tile_coord_mnkl[3]
        self.store_state = dict(smem=smem_tensor, out=getattr(params, self.name), idx=getattr(params, self.idx_name), first=varlen_manager.params.cu_seqlens_m[batch], rows=varlen_manager.len_m(batch), m0=tile_coord_mnkl[0] * gemm.cta_tile_shape_mnk[0], n0=tile_coord_mnkl[1] * gemm.cta_tile_shape_mnk[1], layout=cute.make_ordered_layout(cute.ceil_div(gemm.cta_tile_shape_mnk[:2], gemm.epi_tile), order=(0, 1) if gemm.epi_m_major else (1, 0)), count=0)
        tRS_s = tiled_copy.get_slice(tidx).partition_D(smem_tensor)
        return (tiled_copy, tRS_s, lambda src_idx, dst_idx: None, None)

    @cute.jit
    def store_r2s(self, gemm, tiled_copy, frag_out, tRS_s_stage, tidx):
        super().store_r2s(gemm, tiled_copy, frag_out, tRS_s_stage, tidx)
        st = self.store_state
        # the driver stores the subtiles in epi_idx order within one traced tile, each in stage
        # (tiles before + epi_idx) % epi_stage like its TMA stores
        epi_coord = st["layout"].get_hier_coord(st["count"] % cute.size(st["layout"]))
        st["count"] += 1
        smem = st["smem"]
        thr_s = tiled_copy.get_slice(tidx).partition_D(smem)
        stage_bytes = cute.size(smem[None, None, 0]) * (smem.element_type.width // 8)
        stage = (tRS_s_stage.iterator.toint() - thr_s[None, None, None, 0].iterator.toint()) // stage_bytes
        sub = smem[None, None, stage]
        gemm.epilogue_barrier.arrive_and_wait()
        epi_m, epi_n = cute.size(gemm.epi_tile, mode=[0]), cute.size(gemm.epi_tile, mode=[1])
        out, idx = st["out"], st["idx"]
        cols = cute.size(out, mode=[1])
        nthr = gemm.num_epi_warps * cute.arch.WARP_SIZE
        vecs = epi_n // 8
        t = tidx % nthr
        for it in cutlass.range_constexpr((epi_m * vecs + nthr - 1) // nthr):
            q = it * nthr + t
            r = q // vecs
            c = (q % vecs) * 8
            row, col = st["m0"] + epi_coord[0] * epi_m + r, st["n0"] + epi_coord[1] * epi_n + c
            # rows past the expert's end and columns past N (a partial last tile) stay unwritten
            if (row < st["rows"]) & (col < cols) & (q < epi_m * vecs):
                frag = cute.make_rmem_tensor(cute.make_layout(8), sub.element_type)
                cute.autovec_copy(cute.local_tile(sub, (1, 8), (r, c // 8))[0, None], frag)
                dst = cute.local_tile(out, (1, 8), (idx[st["first"] + row], col // 8))[0, None]
                cute.autovec_copy(frag, dst)


@gemm_epilogue(outputs=(ScatterRowStore("out", "slot"),), extra_ops=(IndexArg("slot"),))
def scatter(acc):
    return {"out": acc}


@gemm_epilogue(outputs=(ScatterRowStore("out", "slot"),), ops={"bias": RowVecLoad("bias")}, extra_ops=(IndexArg("slot"),))
def scatter_bias(acc, bias):
    return {"out": acc + bias}


def mint(mod, *, c=False, paired=False, packed=False, add=False, bias=False) -> type:
    """The QuACK SM90 GEMM class with this epilogue (its operand kinds fixed here)."""
    kind_sig = (("c", "c"),) if c else (("bias", "row"),) if bias else ()
    return mod._mint(kind_sig, 9, paired, packed, (), RoundingMode.RN, (), add)


def paired_tensor(t0: cute.Tensor, t1: cute.Tensor, shape: Callable, stride: Callable) -> cute.Tensor:
    """A tensor over t0 whose size-2 pair mode strides to t1 (same element type)."""
    d = (t1.iterator.toint() - t0.iterator.toint()) // (t0.element_type.width // 8)
    return cute.make_tensor(t0.iterator, cute.make_layout(shape(d), stride=stride(d)))


def transposed(t: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 0]))


class Entry:
    def __init__(self, gemm, cluster_size: int):
        self.gemm = gemm
        self.max_active_clusters = get_max_active_clusters(cluster_size)

    def sched(self) -> TileSchedulerOptions:
        return TileSchedulerOptions(max_active_clusters=Int32(self.max_active_clusters))


class GatedFwd(Entry):
    """preact (M, 2I), h (M, I) = gated(gathered[a_idx] @ [W0|W1]^T), per expert."""

    @cute.jit
    def __call__(self, mX, mW0, mW1, mPre, mH, mCu, mIdx, stream):
        nb, n, k = mW0.shape
        sl, sn, sk = mW0.stride
        mB = paired_tensor(mW0, mW1, lambda d: (nb, (2, n), k), lambda d: (sl, (d, sn), sk))
        epi = self.gemm.EpilogueArguments(postact=mH)
        var = VarlenArguments(mCuSeqlensM=mCu, mAIdx=mIdx)
        self.gemm(mX, mB, mPre, None, epi, self.sched(), var, stream)


class GatedFwdBias(Entry):
    """As GatedFwd plus a (G, 2I) bias row (interleaved like the weight rows) before the gate."""

    @cute.jit
    def __call__(self, mX, mW0, mW1, mBias, mPre, mH, mCu, mIdx, stream):
        nb, n, k = mW0.shape
        sl, sn, sk = mW0.stride
        mB = paired_tensor(mW0, mW1, lambda d: (nb, (2, n), k), lambda d: (sl, (d, sn), sk))
        epi = self.gemm.EpilogueArguments(bias=mBias, postact=mH)
        var = VarlenArguments(mCuSeqlensM=mCu, mAIdx=mIdx)
        self.gemm(mX, mB, mPre, None, epi, self.sched(), var, stream)


class DownScatter(Entry):
    """out[slot_of[r]] = (h @ W_down^T)[r], per expert: rows land in received-slot order."""

    @cute.jit
    def __call__(self, mH, mW, mOut, mSlot, mCu, stream):
        var = VarlenArguments(mCuSeqlensM=mCu)
        epi = self.gemm.EpilogueArguments(out=mOut, slot=mSlot)
        self.gemm(mH, mW, None, None, epi, self.sched(), var, stream)


class DownBiasScatter(Entry):
    """out[slot_of[r]] = (h @ W_down^T + bias)[r], per expert."""

    @cute.jit
    def __call__(self, mH, mW, mBias, mOut, mSlot, mCu, stream):
        var = VarlenArguments(mCuSeqlensM=mCu)
        epi = self.gemm.EpilogueArguments(bias=mBias, out=mOut, slot=mSlot)
        self.gemm(mH, mW, None, None, epi, self.sched(), var, stream)


class Dgated(Entry):
    """dpre (M, 2I), h (M, I) = dgated(dy[slot_of] @ W_down, preact), per expert."""

    @cute.jit
    def __call__(self, mDy, mW, mPre, mDpre, mH, mCu, mIdx, stream):
        epi = self.gemm.EpilogueArguments(postact=mH)
        var = VarlenArguments(mCuSeqlensM=mCu, mAIdx=mIdx)
        self.gemm(mDy, mW, mDpre, mPre, epi, self.sched(), var, stream)


class Dgrad(Entry):
    """dx (M, H) = dpre @ [W0; W1], one GEMM over K = 2I, per expert."""

    @cute.jit
    def __call__(self, mDpre, mW0, mW1, mDx, mCu, stream):
        nb, i, h = mW0.shape
        sl, si, sh = mW0.stride
        mB = paired_tensor(mW0, mW1, lambda d: (nb, h, (2, i)), lambda d: (sl, sh, (d, si)))
        var = VarlenArguments(mCuSeqlensM=mCu)
        self.gemm(mDpre, mB, mDx, None, self.gemm.EpilogueArguments(), self.sched(), var, stream)


class WgradPair(Entry):
    """[G0 | G1] (G, I, H) fp32 += dpre^T @ gathered[a_idx], per expert, as D = (H, (2, I))."""

    @cute.jit
    def __call__(self, mX, mDpre, mG0, mG1, mCu, mIdx, stream):
        nb, i, h = mG0.shape
        sl, si, sh = mG0.stride
        mD = paired_tensor(mG0, mG1, lambda d: (nb, h, (2, i)), lambda d: (sl, sh, (d, si)))
        var = VarlenArguments(mCuSeqlensK=mCu, mAIdx=mIdx)
        epi = self.gemm.EpilogueArguments()
        self.gemm(transposed(mX), transposed(mDpre), mD, None, epi, self.sched(), var, stream)


class WgradDown(Entry):
    """G (G, H, I) fp32 += dy[slot_of]^T @ h, per expert."""

    @cute.jit
    def __call__(self, mDy, mH, mG, mCu, mIdx, stream):
        var = VarlenArguments(mCuSeqlensK=mCu, mAIdx=mIdx)
        epi = self.gemm.EpilogueArguments()
        self.gemm(transposed(mDy), transposed(mH), mG, None, epi, self.sched(), var, stream)


class Config(NamedTuple):
    tile_m: int
    tile_n: int
    cluster_m: int
    pingpong: bool = False


DEFAULT_CONFIG = Config(128, 192, 2)
CONFIGS: dict[tuple, Config] = {
    # qwen3-30b-a3b (H 2048, I 768), best of an offline tile sweep at 1024 rows per expert
    ("gated", 1536, 2048): Config(128, 192, 2, True),
    ("down", 2048, 768): Config(128, 192, 2, True),
    ("dgated", 768, 2048): Config(128, 128, 2, True),
    ("dgrad", 2048, 1536): Config(128, 192, 2, True),
    ("wgrad_pair", 1536, 2048): Config(128, 192, 2, True),
    ("wgrad_down", 768, 2048): Config(128, 256, 1, False),
    # deepseek-v2-lite (H 2048, I 1408), 1536 rows per expert
    ("gated", 2816, 2048): Config(128, 192, 2, True),
    ("down", 2048, 1408): Config(128, 192, 1, True),
    ("dgated", 1408, 2048): Config(128, 128, 2, True),
    ("dgrad", 2048, 2816): Config(128, 192, 2, True),
    ("wgrad_pair", 2816, 2048): Config(128, 192, 2, True),
    ("wgrad_down", 1408, 2048): Config(256, 192, 2, False),
    # qwen3.5-35b-a3b (H 2048, I 512), 512 rows per expert
    ("gated", 1024, 2048): Config(128, 256, 2, False),
    ("down", 2048, 512): Config(128, 192, 1, True),
    ("dgated", 512, 2048): Config(128, 128, 2, True),
    ("dgrad", 2048, 1024): Config(128, 128, 2, True),
    ("wgrad_pair", 1024, 2048): Config(256, 192, 1, False),
    ("wgrad_down", 512, 2048): Config(128, 256, 1, False),
    # gpt-oss-20b (H 2880, I 2880), 2048 rows per expert
    ("gated_oai", 5760, 2880): Config(128, 192, 2, True),
    ("down_bias", 2880, 2880): Config(128, 192, 2, True),
    ("dgated_oai", 2880, 2880): Config(128, 128, 2, True),
    ("dgrad", 2880, 5760): Config(128, 192, 2, True),
    ("wgrad_pair", 5760, 2880): Config(128, 192, 2, True),
    ("wgrad_down", 2880, 2880): Config(128, 192, 1, True),
}
"""(kind, GEMM n, GEMM k) -> Config, fixed offline; shapes not listed take DEFAULT_CONFIG."""


def config(kind: str, n: int, k: int) -> Config:
    return CONFIGS.get((kind, n, k), DEFAULT_CONFIG)


def fake(dtype, rank: int, lead: int, div: int = 8):
    return make_fake_tensor(dtype, tuple(cute.sym_int() for _ in range(rank)), divisibility=div, leading_dim=lead)


def ivec():
    return make_fake_tensor(Int32, (cute.sym_int(),), divisibility=4, leading_dim=0)


KINDS = {
    "gated": (GatedFwd, (gated, gated_swapped), dict(paired=True), True, {},
              lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 3, 2), fake(BFloat16, 3, 2), fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), ivec(), ivec())),
    "dgated": (Dgated, (dgated, dgated_swapped), dict(c=True, packed=True), True, dict(cd_packed="n", implicit_dtype=BFloat16, b_transposed=True),
               lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 3, 2), fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), ivec(), ivec())),
    "dgrad": (Dgrad, (plain, None), {}, False, {},
              lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 3, 2), fake(BFloat16, 3, 2), fake(BFloat16, 2, 1), ivec())),
    "wgrad_pair": (WgradPair, (plain, pair_swapped), dict(add=True), True, {},
                   lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), fake(Float32, 3, 2, 4), fake(Float32, 3, 2, 4), ivec(), ivec())),
    "wgrad_down": (WgradDown, (plain, None), dict(add=True), True, {},
                   lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), fake(Float32, 3, 2, 4), ivec(), ivec())),
    "down_scatter": (DownScatter, (scatter, None), {}, False, {},
                     lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 3, 2), fake(BFloat16, 2, 1), ivec(), ivec())),
    "down_bias_scatter": (DownBiasScatter, (scatter_bias, None), dict(bias=True), False, {},
                          lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 3, 2), fake(BFloat16, 2, 1, 4), fake(BFloat16, 2, 1), ivec(), ivec())),
    # gpt-oss: epilogues built per activation (alpha, limit)
    "gated_oai": (GatedFwdBias, (make_gated_oai, None), dict(paired=True, bias=True), True, {},
                  lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 3, 2), fake(BFloat16, 3, 2), fake(BFloat16, 2, 1, 4), fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), ivec(), ivec())),
    "dgated_oai": (Dgated, (make_dgated_oai, None), dict(c=True, packed=True), True, dict(cd_packed="n", implicit_dtype=BFloat16, b_transposed=True),
                   lambda: (fake(BFloat16, 2, 1), fake(BFloat16, 3, 2), fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), fake(BFloat16, 2, 1), ivec(), ivec())),
}
"""
kind -> (entry, epilogue (lane order kept, swapped), mint kwargs, gather-A, attrs, fake args); an
epilogue that is a function takes the activation parameters (alpha, limit).
"""

compiled: dict = {}


def compiled_gemm(kind: str, swapped: bool, cfg: Config, device: torch.device, act: tuple = ()) -> Callable:
    key = (kind, swapped, cfg, device.index, act)
    fn = compiled.get(key)
    if fn is None:
        entry, mods, mint_kw, gather, attrs, fakes = KINDS[kind]
        mod = mods[1] if swapped else mods[0]
        if act:
            mod = mod(*act)
        if kind == "wgrad_pair" and swapped:
            mint_kw = dict(mint_kw, paired=True)
        cls = mint(mod, **mint_kw)
        gemm = cls(Float32, BFloat16, (cfg.tile_m, cfg.tile_n), (cfg.cluster_m, 1, 1), pingpong=cfg.pingpong, is_persistent=True, gather_A=gather)
        for name, value in attrs.items():
            setattr(gemm, name, value)
        with torch.cuda.device(device):
            fn = cute.compile(entry(gemm, cfg.cluster_m), *fakes(), fake_stream(), options="--enable-tvm-ffi")
        compiled[key] = fn
    return fn


def lower_first(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    (lower, higher, a is the higher) by address; asserts the two can be read as one paired operand.
    """
    assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
    assert a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0
    da = b.data_ptr() - a.data_ptr()
    assert da != 0 and da % 16 == 0, "paired operands must be distinct and 16-byte apart"
    return (a, b, False) if da > 0 else (b, a, True)


def bf16_expert_gated_forward_sm90(x, w_gate, w_up, cu, a_idx, bias=None, act=()):
    M, inter = a_idx.shape[0], w_gate.shape[1]
    w0, w1, swapped = lower_first(w_gate, w_up)
    pre, h = x.new_empty(M, 2 * inter), x.new_empty(M, inter)
    if M == 0:
        return pre, h
    if act:
        assert not swapped and bias is not None
        fn = compiled_gemm("gated_oai", False, config("gated_oai", 2 * inter, x.shape[1]), x.device, act)
        fn(x, w0, w1, bias, pre, h, cu, a_idx)
    else:
        compiled_gemm("gated", swapped, config("gated", 2 * inter, x.shape[1]), x.device)(x, w0, w1, pre, h, cu, a_idx)
    return pre, h


def bf16_expert_down_forward_scattered_sm90(h, w_down, cu, slot_of, bias=None):
    M, H = h.shape[0], w_down.shape[1]
    out = h.new_empty(M, H)
    if M == 0:
        return out
    if bias is not None:
        fn = compiled_gemm("down_bias_scatter", False, config("down_bias", H, h.shape[1]), h.device)
        fn(h, w_down, bias, out, slot_of, cu)
    else:
        compiled_gemm("down_scatter", False, config("down", H, h.shape[1]), h.device)(h, w_down, out, slot_of, cu)
    return out


def bf16_expert_dgated_backward_sm90(dy, w_down, preact, cu, slot_of, w_gate, w_up, act=()):
    M, inter = preact.shape[0], preact.shape[1] // 2
    swapped = w_up.data_ptr() < w_gate.data_ptr()
    dpre = preact.new_empty(M, 2 * inter)
    h = preact.new_empty(M, inter)
    if M:
        if act:
            assert not swapped
            fn = compiled_gemm("dgated_oai", False, config("dgated_oai", inter, dy.shape[1]), dy.device, act)
        else:
            fn = compiled_gemm("dgated", swapped, config("dgated", inter, dy.shape[1]), dy.device)
        fn(dy, w_down, preact, dpre, h, cu, slot_of)
    return dpre, h, swapped


def bf16_expert_dgrad_sm90(dpre, w_gate, w_up, cu, dpre_swapped):
    M, H = dpre.shape[0], w_gate.shape[2]
    w0, w1, swapped = lower_first(w_gate, w_up)
    assert swapped == dpre_swapped, "the weights moved between the dgated backward and the dgrad"
    dx = dpre.new_empty(M, H)
    if M:
        compiled_gemm("dgrad", False, config("dgrad", H, dpre.shape[1]), dpre.device)(dpre, w0, w1, dx, cu)
    return dx


def bf16_expert_wgrad_gate_up_sm90(x, dpre, dpre_swapped, mg_gate, mg_up, cu, a_idx):
    if a_idx.shape[0] == 0:
        return
    m0, m1, mg_swapped = lower_first(mg_gate, mg_up)
    fn = compiled_gemm("wgrad_pair", mg_swapped != dpre_swapped, config("wgrad_pair", dpre.shape[1], x.shape[1]), x.device)
    fn(x, dpre, m0, m1, cu, a_idx)


def bf16_expert_wgrad_down_sm90(dy, h, mg_down, cu, slot_of):
    if slot_of.shape[0] == 0:
        return
    compiled_gemm("wgrad_down", False, config("wgrad_down", h.shape[1], dy.shape[1]), dy.device)(dy, h, mg_down, cu, slot_of)


__all__ = ["bf16_expert_gated_forward_sm90", "bf16_expert_down_forward_scattered_sm90", "bf16_expert_dgated_backward_sm90", "bf16_expert_dgrad_sm90", "bf16_expert_wgrad_gate_up_sm90", "bf16_expert_wgrad_down_sm90"]
