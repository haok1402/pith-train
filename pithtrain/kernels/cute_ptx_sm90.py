"""
Inline-PTX helpers shared by the SM90 CuTe-DSL kernels, each a dsl_user_op emitting one short PTX
sequence. Two calling conventions: asm(res_type, args, ptx, constraints), whose stores return
nothing, and asm_args_last(res_type, ptx, constraints, args), whose stores return a dummy register.
An asm_args_last helper that shares a name with an asm helper carries a suffix: _args_last when only
the emission differs, otherwise one naming the difference (ld_b32_coherent is a coherent ld.global,
not .nc; lds_b32_pred is predicated; sts_b32_nopred is not).
"""

import struct

import cutlass
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = ["abs2", "asm", "asm_args_last", "atom_add_u64", "atom_add_u64_pred", "bf16_hi", "bf16_lo", "bf16_sel", "bf16x2_to_f32", "cp_async16", "cp_async8", "ctz64", "f32_as_u32", "f32_bits", "f32_hex", "f32x2_to_bf16x2", "fence_gpu", "lanemask_lt", "ld_b32", "ld_b32_coherent", "ld_b64", "ld_relaxed", "ld_v4", "lds_b32", "lds_b32_pred", "lds_v2", "lds_v4", "match_any", "max2", "max_u16x2", "pack_bf16", "prmt", "prmt_args_last", "red_add_shared", "red_or_shared", "red_or_u64", "selp", "selp_args_last", "spin_epoch", "spin_until", "st_b32", "st_b64", "st_f32", "st_release", "st_v2", "st_v2_args_last", "st_v4", "st_v4_args_last", "sts_b32", "sts_b32_nopred", "sts_v4", "to_ir", "u32_as_f32", "unpack"]

# ---------------------------------------------------------------------------
# asm(res_type, args, ptx, constraints)
# ---------------------------------------------------------------------------


def f32_hex(v: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08X}"


def asm(res_type, args, ptx: str, cons: str, side_effects: bool = False):
    return llvm.inline_asm(res_type, args, ptx, cons, has_side_effects=side_effects, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def ld_v4(addr: Int64, pred: Int32, *, loc=None, ip=None) -> tuple:
    """
    16-byte global load; zeros where pred == 0. A plain L1-allocating ld.global: measured faster
    than the .nc no-allocate path.
    """
    st = llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()])
    r = asm(
        st, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t"
        "mov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_b32(addr: Int64, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    """Predicated 4-byte global load returning other where pred == 0."""
    return Int32(
        asm(
            T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip), Int32(other).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, $3;\n\t"
            "@p ld.global.nc.b32 $0, [$1];\n\t}",
            "=r,l,r,r",
        )
    )


@dsl_user_op
def st_v2(addr: Int64, a: Uint32, b: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\t@p st.global.v2.b32 [$0], {$1, $2};\n\t}", "l,r,r,r", side_effects=True)


@dsl_user_op
def st_b32(addr: Int64, a: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}", "l,r,r", side_effects=True)


@dsl_user_op
def lds_b32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", side_effects=True))


@dsl_user_op
def sts_b32(addr: Int32, a: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.shared.b32 [$0], $1;\n\t}", "r,r,r", side_effects=True)


@dsl_user_op
def max_u16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Packed max of two halves; equals the bf16 max for non-negative (abs) values."""
    return Uint32(asm(T.i32(), [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)], "max.u16x2 $0, $1, $2;", "=r,r,r"))


@dsl_user_op
def prmt(a: Uint32, b: Uint32, sel: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)], f"prmt.b32 $0, $1, $2, 0x{sel:04X};", "=r,r,r"))


@dsl_user_op
def selp(pred: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """pred ? a : b."""
    return Uint32(asm(T.i32(), [Int32(pred).ir_value(loc=loc, ip=ip), Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\tselp.b32 $0, $2, $3, p;\n\t}", "=r,r,r,r"))


@dsl_user_op
def match_any(v: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [Int32(v).ir_value(loc=loc, ip=ip)], "match.any.sync.b32 $0, $1, -1;", "=r,r"))


@dsl_user_op
def lanemask_lt(*, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [], "mov.u32 $0, %lanemask_lt;", "=r"))


@dsl_user_op
def atom_add_u64(addr: Int64, v: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(asm(T.i64(), [Int64(addr).ir_value(loc=loc, ip=ip), Int64(v).ir_value(loc=loc, ip=ip)], "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;", "=l,l,l", side_effects=True))


@dsl_user_op
def red_add_shared(addr: Int32, v: Int32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p red.shared.add.u32 [$0], $1;\n\t}", "r,r,r", side_effects=True)


@dsl_user_op
def st_release(addr: Int64, v: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)], "st.release.gpu.global.b32 [$0], $1;", "l,r", side_effects=True)


@dsl_user_op
def spin_until(addr: Int64, v: Int32, pred: Int32, *, loc=None, ip=None) -> None:
    """Where pred: spin with acquire loads until *addr == v."""
    asm(
        None, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p, q;\n\t.reg .b32 t;\n\t"
        "setp.eq.b32 q, $2, 0;\n\t"
        "@q bra SQ_DONE_${:uid};\n"
        "SQ_LOOP_${:uid}:\n\t"
        "ld.acquire.gpu.global.b32 t, [$0];\n\t"
        "setp.ne.b32 p, t, $1;\n\t"
        "@p bra SQ_LOOP_${:uid};\n"
        "SQ_DONE_${:uid}:\n\t}",
        "l,r,r", side_effects=True,
    )


@dsl_user_op
def ld_relaxed(addr: Int64, pred: Int32, *, loc=None, ip=None) -> Int32:
    """Predicated gpu-scope relaxed 4-byte load (L2, never a stale L1 line); 0 where not pred."""
    return Int32(
        asm(
            T.i32(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t"
            "@p ld.relaxed.gpu.global.b32 $0, [$1];\n\t}",
            "=r,l,r", side_effects=True,
        )
    )


@dsl_user_op
def fence_gpu(*, loc=None, ip=None) -> None:
    asm(None, [], "fence.acq_rel.gpu;", "", side_effects=True)


@dsl_user_op
def st_b64(addr: Int64, v: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int64(v).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b64 [$0], $1;\n\t}", "l,l,r", side_effects=True)


@dsl_user_op
def red_or_shared(addr: Int32, v: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(v).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p red.shared.or.b32 [$0], $1;\n\t}", "r,r,r", side_effects=True)


@dsl_user_op
def atom_add_u64_pred(addr: Int64, v: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    asm(
        None, [Int64(addr).ir_value(loc=loc, ip=ip), Int64(v).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b64 t;\n\tsetp.ne.b32 p, $2, 0;\n\t"
        "@p atom.relaxed.gpu.global.add.u64 t, [$0], $1;\n\t}",
        "l,l,r", side_effects=True,
    )


@dsl_user_op
def spin_epoch(addr: Int64, epoch: Int32, pred: Int32, *, loc=None, ip=None) -> None:
    """Where pred: spin with acquire loads until the epoch (high word of *addr) moves on."""
    asm(
        None, [Int64(addr).ir_value(loc=loc, ip=ip), Int32(epoch).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p, q;\n\t.reg .b32 lo, hi;\n\t"
        "setp.eq.b32 q, $2, 0;\n\t"
        "@q bra EP_DONE_${:uid};\n"
        "EP_LOOP_${:uid}:\n\t"
        "ld.acquire.gpu.global.v2.b32 {lo, hi}, [$0];\n\t"
        "setp.eq.b32 p, hi, $1;\n\t"
        "@p bra EP_LOOP_${:uid};\n"
        "EP_DONE_${:uid}:\n\t}",
        "l,r,r", side_effects=True,
    )


@dsl_user_op
def red_or_u64(addr: Int64, v: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Int64(v).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p red.relaxed.gpu.global.or.b64 [$0], $1;\n\t}", "l,l,r", side_effects=True)


@dsl_user_op
def ld_b64(addr: Int64, pred: Int32, *, loc=None, ip=None) -> Int64:
    """Predicated 8-byte global load; 0 where pred == 0."""
    return Int64(
        asm(
            T.i64(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, 0;\n\t"
            "@p ld.global.nc.b64 $0, [$1];\n\t}",
            "=l,l,r",
        )
    )


@dsl_user_op
def ctz64(v: Int64, *, loc=None, ip=None) -> Int32:
    """Index of the lowest set bit (64 for 0)."""
    return Int32(asm(T.i32(), [Int64(v).ir_value(loc=loc, ip=ip)], "{\n\t.reg .b64 r;\n\tbrev.b64 r, $1;\n\tclz.b64 $0, r;\n\t}", "=r,l"))


@dsl_user_op
def bf16x2_to_f32(w: Uint32, *, loc=None, ip=None) -> tuple:
    lo = asm(T.f32(), [Uint32(w).ir_value(loc=loc, ip=ip)], "shl.b32 $0, $1, 16;", "=f,r")
    hi = asm(T.f32(), [Uint32(w).ir_value(loc=loc, ip=ip)], "and.b32 $0, $1, 0xFFFF0000;", "=f,r")
    return cutlass.Float32(lo), cutlass.Float32(hi)


@dsl_user_op
def f32x2_to_bf16x2(lo, hi, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [cutlass.Float32(hi).ir_value(loc=loc, ip=ip), cutlass.Float32(lo).ir_value(loc=loc, ip=ip)], "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f"))


@dsl_user_op
def st_v4(addr: Int64, v: tuple, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Uint32(x).ir_value(loc=loc, ip=ip) for x in v] + [Int32(pred).ir_value(loc=loc, ip=ip)], "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n\t}", "l,r,r,r,r,r", side_effects=True)


@dsl_user_op
def f32_bits(v, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), [cutlass.Float32(v).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f"))


# ---------------------------------------------------------------------------
# asm_args_last(res_type, ptx, constraints, args)
# ---------------------------------------------------------------------------


def asm_args_last(res_type, ptx: str, cons: str, args: list, side_effects: bool = False):
    return asm(res_type, args, ptx, cons, side_effects)


def to_ir(x, ty, loc=None, ip=None):
    return ty(x).ir_value(loc=loc, ip=ip)


def unpack(res, n: int, ty, loc=None, ip=None) -> tuple:
    mlir_ty = T.f32() if ty is Float32 else T.i32()
    return tuple(ty(llvm.extractvalue(mlir_ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def ld_b32_coherent(addr: Int64, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    """Predicated coherent 4-byte global load; other when pred is 0."""
    return Int32(asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, $3;\n@p ld.global.b32 $0, [$1];\n}", "=r,l,r,r", [to_ir(addr, Int64, loc, ip), to_ir(pred, Int32, loc, ip), to_ir(other, Int32, loc, ip)]))


@dsl_user_op
def st_v2_args_last(addr: Int64, a: Uint32, b: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm_args_last(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\n@p st.global.v2.b32 [$1], {$2, $3};\n"
        "mov.b32 $0, 0;\n}",
        "=r,l,r,r,r", [to_ir(addr, Int64, loc, ip)] + [to_ir(v, Uint32, loc, ip) for v in (a, b)] + [to_ir(pred, Int32, loc, ip)], side_effects=True,
    )


@dsl_user_op
def st_v4_args_last(addr: Int64, v: tuple, pred: Int32, *, loc=None, ip=None) -> None:
    asm_args_last(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $6, 0;\n@p st.global.v4.b32 [$1], {$2, $3, $4, $5};\n"
        "mov.b32 $0, 0;\n}",
        "=r,l,r,r,r,r,r", [to_ir(addr, Int64, loc, ip)] + [to_ir(x, Uint32, loc, ip) for x in v] + [to_ir(pred, Int32, loc, ip)], side_effects=True,
    )


@dsl_user_op
def st_f32(addr: Int64, v: Float32, pred: Int32, *, loc=None, ip=None) -> None:
    asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\n@p st.global.f32 [$1], $2;\nmov.b32 $0, 0;\n}", "=r,l,f,r", [to_ir(addr, Int64, loc, ip), to_ir(v, Float32, loc, ip), to_ir(pred, Int32, loc, ip)], side_effects=True)


@dsl_user_op
def sts_b32_nopred(addr: Int32, v: Uint32, *, loc=None, ip=None) -> None:
    asm_args_last(T.i32(), "st.shared.b32 [$1], $2;\nmov.b32 $0, 0;", "=r,r,r", [to_ir(addr, Int32, loc, ip), to_ir(v, Uint32, loc, ip)], side_effects=True)


@dsl_user_op
def sts_v4(addr: Int32, v: tuple, *, loc=None, ip=None) -> None:
    asm_args_last(T.i32(), "st.shared.v4.b32 [$1], {$2, $3, $4, $5};\nmov.b32 $0, 0;", "=r,r,r,r,r,r", [to_ir(addr, Int32, loc, ip)] + [to_ir(x, Uint32, loc, ip) for x in v], side_effects=True)


@dsl_user_op
def lds_v4(addr: Int32, *, loc=None, ip=None) -> tuple:
    ty = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = asm_args_last(ty, "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", [to_ir(addr, Int32, loc, ip)], side_effects=True)  # must stay behind the barrier
    return unpack(res, 4, Uint32, loc, ip)


@dsl_user_op
def cp_async16(dst: Int32, src: Int64, src_size: Int32, *, loc=None, ip=None) -> None:
    """16-byte global->shared cp.async.cg; zero-fills when src_size is 0."""
    asm_args_last(T.i32(), "cp.async.cg.shared.global [$1], [$2], 16, $3;\nmov.b32 $0, 0;", "=r,r,l,r", [to_ir(dst, Int32, loc, ip), to_ir(src, Int64, loc, ip), to_ir(src_size, Int32, loc, ip)], side_effects=True)


@dsl_user_op
def lds_b32_pred(addr: Int32, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, $3;\n@p ld.shared.b32 $0, [$1];\n}", "=r,r,r,r", [to_ir(addr, Int32, loc, ip), to_ir(pred, Int32, loc, ip), to_ir(other, Int32, loc, ip)], side_effects=True))


@dsl_user_op
def selp_args_last(pred: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """pred ? a : b."""
    return Uint32(asm_args_last(T.i32(), "{\n.reg .pred p;\nsetp.ne.b32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}", "=r,r,r,r", [to_ir(pred, Int32, loc, ip), to_ir(a, Uint32, loc, ip), to_ir(b, Uint32, loc, ip)]))


@dsl_user_op
def bf16_sel(pred_hi: Int32, x: Uint32, *, loc=None, ip=None) -> Float32:
    """fp32 value of the high (pred_hi) or low BF16 half of x."""
    return Float32(
        asm_args_last(
            T.f32(),
            "{\n.reg .pred p;\n.reg .b32 a, b;\nsetp.ne.b32 p, $1, 0;\n"
            "and.b32 a, $2, 0xFFFF0000;\nshl.b32 b, $2, 16;\nselp.b32 $0, a, b, p;\n}",
            "=f,r,r", [to_ir(pred_hi, Int32, loc, ip), to_ir(x, Uint32, loc, ip)],
        )
    )


@dsl_user_op
def f32_as_u32(x: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm_args_last(T.i32(), "mov.b32 $0, $1;", "=r,f", [to_ir(x, Float32, loc, ip)]))


@dsl_user_op
def u32_as_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "mov.b32 $0, $1;", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def bf16_lo(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "shl.b32 $0, $1, 16;", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def bf16_hi(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(asm_args_last(T.f32(), "and.b32 $0, $1, 0xFFFF0000;", "=f,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def pack_bf16(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Round two fp32 to BF16 (RN, as Triton's .to(tl.bfloat16)) and pack {lo, hi}."""
    return Uint32(asm_args_last(T.i32(), "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", [to_ir(hi, Float32, loc, ip), to_ir(lo, Float32, loc, ip)]))


@dsl_user_op
def abs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Magnitude bits of two packed BF16."""
    return Uint32(asm_args_last(T.i32(), "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r", [to_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def max2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Lane-wise max of two packed BF16 magnitudes (NaN-ignoring, like Triton's max.f32)."""
    return Uint32(asm_args_last(T.i32(), "max.bf16x2 $0, $1, $2;", "=r,r,r", [to_ir(a, Uint32, loc, ip), to_ir(b, Uint32, loc, ip)]))


@dsl_user_op
def prmt_args_last(a: Uint32, b: Uint32, sel: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm_args_last(T.i32(), f"prmt.b32 $0, $1, $2, {sel:#x};", "=r,r,r", [to_ir(a, Uint32, loc, ip), to_ir(b, Uint32, loc, ip)]))


@dsl_user_op
def cp_async8(dst: Int32, src: Int64, src_size: Int32, *, loc=None, ip=None) -> None:
    """8-byte global->shared cp.async.ca; zero-fills when src_size is 0."""
    asm_args_last(T.i32(), "cp.async.ca.shared.global [$1], [$2], 8, $3;\nmov.b32 $0, 0;", "=r,r,l,r", [to_ir(dst, Int32, loc, ip), to_ir(src, Int64, loc, ip), to_ir(src_size, Int32, loc, ip)], side_effects=True)


@dsl_user_op
def lds_v2(addr: Int32, *, loc=None, ip=None) -> tuple:
    ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = asm_args_last(ty, "ld.shared.v2.b32 {$0, $1}, [$2];", "=r,=r,r", [to_ir(addr, Int32, loc, ip)], side_effects=True)  # must stay behind the barrier
    return unpack(res, 2, Uint32, loc, ip)
