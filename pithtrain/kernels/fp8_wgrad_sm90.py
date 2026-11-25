from enum import IntEnum, auto
from math import gcd, lcm
from struct import pack, unpack

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Float8E4M3FN, GridConstant, Int16, Int32, Int64
from cutlass._mlir.dialects import llvm, math
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op, select_
from cutlass.experimental.cuda import TensorMap, TensorMapL2Promotion, TensorMapSwizzle, create_tensor_map_tiled
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, to_ir, unpack as unpack_registers
from pithtrain.kernels.fp8_scale_layout import to_mn_major

__all__ = ["fp8_wgrad_sm90"]

BLOCK_K = 128
SMEM_CAPACITY = 232448
CACHE = {}


def divide_up(size, divisor):
    return (size + divisor - 1) // divisor


def align_up(size, alignment):
    return divide_up(size, alignment) * alignment


def round_float32(value):
    return unpack("f", pack("f", value))[0]


def shared_footprint(block_m, block_n, num_k, rowwise):
    scale_columns = 1 if BLOCK_K % block_n == 0 else 2
    scale_bytes = 0 if rowwise else align_up(4 * num_k * scale_columns, 8)
    reserved = align_up(4 * block_m * block_n, 1024) + 256 + scale_bytes
    stage_bytes = BLOCK_K * (block_m + block_n) + align_up(4 * block_m, 128)
    if rowwise:
        stage_bytes += align_up(4 * block_n, 128)
    return reserved, stage_bytes


def select_layout(M, N, K, num_sms, rowwise=False, block_n_multiple=1):
    block_m_choices = [64, 128]
    if M <= 16:
        block_m_choices.append(16)
    if M <= 32:
        block_m_choices.append(32)

    best_cycles = float("inf")
    best_layout = None
    num_k = divide_up(K, BLOCK_K)
    n_step = lcm(16, block_n_multiple)
    for cluster_m, cluster_n in ((1, 1), (1, 2), (2, 1)):
        if num_sms % (cluster_m * cluster_n):
            continue
        for block_m in block_m_choices:
            for block_n in range(n_step, 193, n_step):
                if block_n > BLOCK_K and BLOCK_K % (block_n - BLOCK_K):
                    continue
                if rowwise and block_n > 128:
                    continue
                out_bytes = 4 * block_m * block_n
                reserved, stage_bytes = shared_footprint(block_m, block_n, num_k, rowwise)
                stages = min(16, (SMEM_CAPACITY - reserved) // stage_bytes)
                min_stages = 3 if block_m * block_n >= 128 * 192 else 4
                if stages < min_stages:
                    continue

                tiles = divide_up(M, block_m) * divide_up(N, block_n)
                waves = divide_up(tiles, num_sms)
                if waves == 1 and cluster_m * cluster_n > 1:
                    continue
                l2_bytes = K * (block_m // cluster_n + block_n // cluster_m) + out_bytes
                l1_bytes = K * (block_m + max(64, block_m) + 2 * block_n) + 2 * out_bytes
                l2_cycles = l2_bytes * tiles // min(64 * num_sms, 6153)
                l1_cycles = l1_bytes * tiles // (128 * num_sms)
                wave_efficiency = round_float32(round_float32(tiles) / round_float32(waves * num_sms))
                cycles = int(round_float32(round_float32(max(l1_cycles, l2_cycles)) / wave_efficiency))
                if cycles < best_cycles:
                    best_cycles = cycles
                    best_layout = (block_m, block_n, stages, cluster_m, cluster_n)

    if best_layout is None:
        raise ValueError("no SM90 FP8 wgrad layout fits the requested dimensions")
    return best_layout


@dsl_user_op
def wait_barrier(pointer, phase, *, loc=None, ip=None):
    asm(
        None,
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
         to_ir(phase, Int32, loc, ip), to_ir(10_000_000, Int32, loc, ip)],
        """{
            .reg .pred complete;
        wait:
            mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1, $2;
            @complete bra done;
            bra wait;
        done:
        }""",
        "r,r,r",
        side_effects=True,
    )


@dsl_user_op
def wait_n16_stage7(pointer, phase, *, loc=None, ip=None):
    asm(
        None,
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
         to_ir(phase, Int32, loc, ip), to_ir(10_000_000, Int32, loc, ip)],
        """{
            .reg .pred complete;
        wait:
            mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1, $2;
            @!complete bra wait;
        }""",
        "r,r,r",
        side_effects=True,
    )


@dsl_user_op
def wgmma_k_block(a_desc_lo, b_desc_lo, block_n, next_scale_ptrs=(), *, loc=None, ip=None):
    check_wgmma_width(block_n)
    # K-major SW128 descriptors have a 1024-byte stride and no leading offset.
    count = block_n // 2
    num_outputs = count + len(next_scale_ptrs)
    outputs = ", ".join(f"${index}" for index in range(count))
    shape = f"m64n{block_n}k32.f32.e4m3.e4m3"
    instructions = [
        "{",
        ".reg .b32 aa, bb, desc_hi;",
        ".reg .b64 a, b;",
        ".reg .pred accumulate;",
        "mov.b32 desc_hi, 0x40000040;",
        "setp.ne.u32 accumulate, 0, 0;",
        "wgmma.fence.sync.aligned;",
        f"mov.b64 a, {{${num_outputs}, desc_hi}};",
        f"mov.b64 b, {{${num_outputs + 1}, desc_hi}};",
    ]
    for k in range(0, BLOCK_K, 32):
        if k:
            instructions += [
                f"add.u32 aa, ${num_outputs}, {k // 16};",
                f"add.u32 bb, ${num_outputs + 1}, {k // 16};",
                "mov.b64 a, {aa, desc_hi};",
                "mov.b64 b, {bb, desc_hi};",
            ]
        instructions.append(f"wgmma.mma_async.sync.aligned.{shape} {{{outputs}}}, a, b, accumulate, 1, 1;")
        if not k:
            instructions.append("setp.ne.u32 accumulate, 1, 0;")
    instructions.append("wgmma.commit_group.sync.aligned;")
    for index in range(len(next_scale_ptrs)):
        instructions.append(f"ld.shared.f32 ${count + index}, [${num_outputs + 2 + index}];")
    instructions += ["wgmma.wait_group.sync.aligned 0;", "}"]
    operands = [to_ir(a_desc_lo, Int32, loc, ip), to_ir(b_desc_lo, Int32, loc, ip)]
    operands += [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip) for ptr in next_scale_ptrs]
    constraints = ",".join(["=f"] * num_outputs + ["r"] * len(operands))
    result = asm(llvm.StructType.get_literal([T.f32()] * num_outputs), operands,
                 "\n".join(instructions), constraints, side_effects=True)
    return unpack_registers(result, num_outputs, Float32, loc, ip)


@dsl_user_op
def load_unicast_triplet(grad, scale, activation, a, a_scale, b, barrier, k, m, n, k_tile, *, loc=None, ip=None):
    values = (
        (a.toint(), Int32), (grad.get_ptr().toint(), Int64), (k, Int32), (m, Int32),
        (barrier.toint(), Int32), (a_scale.toint(), Int32), (scale.get_ptr().toint(), Int64),
        (k_tile, Int32), (b.toint(), Int32), (activation.get_ptr().toint(), Int64),
        (n, Int32), (0x1000000000000000, Int64),
    )
    arguments = [to_ir(value, dtype, loc, ip) for value, dtype in values]
    opcode = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    asm(
        None, arguments,
        f"{{ {opcode} [$0], [$1, {{$2, $3}}], [$4], $11; "
        f"{opcode} [$5], [$6, {{$3, $7}}], [$4], $11; "
        f"{opcode} [$8], [$9, {{$2, $10}}], [$4], $11; }}",
        "r,l,r,r,r,r,l,r,r,l,r,l",
        side_effects=True,
    )



@dsl_user_op
def load_tma_with_hint(descriptor, dst, barrier, coord_lo, coord_hi, multicast, hint, *, loc=None, ip=None):
    values = (
        (dst.toint(), Int32), (descriptor.get_ptr().toint(), Int64), (coord_lo, Int32),
        (coord_hi, Int32), (barrier.toint(), Int32), (hint, Int64),
    )
    args = [to_ir(value, dtype, loc, ip) for value, dtype in values]
    opcode = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    if multicast:
        args.append(to_ir(3, Int16, loc, ip))
        asm(None, args, f"{opcode}.multicast::cluster.L2::cache_hint [$0], [$1, {{$2, $3}}], [$4], $6, $5;",
            "r,l,r,r,r,l,h", side_effects=True)
    else:
        asm(None, args, f"{opcode}.L2::cache_hint [$0], [$1, {{$2, $3}}], [$4], $5;",
            "r,l,r,r,r,l", side_effects=True)


@dsl_user_op
def load_bulk_b_scales(dst, source, size, barrier, *, loc=None, ip=None):
    values = ((dst.toint(), Int32), (source.toint(), Int64), (size, Int32), (barrier.toint(), Int32))
    operands = [to_ir(value, dtype, loc, ip) for value, dtype in values]
    asm(None, operands,
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r", side_effects=True)


@dsl_user_op
def unused_scale(*, loc=None, ip=None):
    return Float32(llvm.UndefOp(T.f32(), loc=loc, ip=ip).result)


def check_wgmma_width(width):
    if not isinstance(width, int) or not 8 <= width <= 256 or width % 8:
        raise ValueError(f"SM90 WGMMA N width {width} must be a multiple of 8 from 8 through 256")


class ProducerRoute(IntEnum):
    RESIDENT_GRADIENT = auto()
    STATIC_RING = auto()
    DYNAMIC_RING = auto()
    RESIDENT_A_PANELS = auto()


class MathRoute(IntEnum):
    ASYNC_PAIRS = auto()
    LONG_A_PAIRS = auto()
    RESIDENT_ACTIVATION = auto()
    SINGLE_N_A = auto()
    PREFIX_RING = auto()
    STATIC_RING = auto()
    DYNAMIC_RING = auto()
    RESIDENT_A_PANELS = auto()


class TileOrder(IntEnum):
    SINGLE_TILE = auto()
    M_LINEAR = auto()
    N_LINEAR = auto()
    N144_B_WIDE = auto()
    SHORT_A = auto()
    SHORT_UNICAST = auto()
    N48_MN_TAIL = auto()
    M_LINEAR_MULTICAST = auto()
    N_GROUPED = auto()
    SINGLE_N80 = auto()
    SHORT_B = auto()
    MN_LINEAR_MULTICAST = auto()
    N_LINEAR_MULTICAST = auto()
    SHORT_N_LINEAR_A = auto()
    SHORT_B_N128 = auto()
    PERSISTENT = auto()
    SHORT_A_TAIL_FIRST = auto()
    N32_A_STAGE4_PARTIAL_M_FIRST = auto()

class ScalePromotion(IntEnum):
    GROUP_PAIRS = auto()
    SELECT_FIRST = auto()
    GROUPS = auto()
    PAIRS_SELECT_FIRST = auto()


def select_route(default, *choices):
    return next((route for route, selected in choices if selected), default)


@dsl_user_op
def prefetch_activation_tile(descriptor, k, n, *, loc=None, ip=None):
    operands = [to_ir(descriptor.get_ptr().toint(), Int64, loc, ip),
                to_ir(k, Int32, loc, ip), to_ir(n, Int32, loc, ip)]
    asm(None, operands, "cp.async.bulk.prefetch.tensor.2d.L2.global [$0, {$1, $2}];",
        "l,r,r", side_effects=True)


class WeightGradient:
    def __init__(self, M, N, K, num_sms, rowwise, block_n_multiple,
                 compact_k_scale=False, aligned_fp8_rows=False, compact_rows=True, compact_output=True,
                 misaligned_fp8_rows=False, aligned_two_sm_operands=False,
                 long_a_storage_ok=False, aligned_stage11_tail=False, fp8_row_alignment=(0, 0), aligned_n160_operands=False,
                 n80_streamed_scale_storage=False, aligned_a128_tail=False,
                 n16_stage10_bulk_storage=False, n16_stage10_pair_storage=False,
                 n160_grad_offset_only=False, aligned_grid_n16_operands=False, aligned_m128_n32_a=False,
                 n48_a_stage9_storage=False, aligned_n192_stage4=False, n16_stage8_local_storage=False,
                 aligned_n16_stage14_operands=False, full_device_sm_budget=False, n112_stage5_l2_storage=False,
                 aligned_n32_stage14_operands=False, aligned_n48_stage8_operands=False,
                 aligned_n96_stage5_narrow_operands=False,
                 aligned_n96_stage4_operands=False, n96_stage4_tail_storage=False,
                 aligned_n32_stage10_operands=False, aligned_m32_n64_stage4_operands=False,
                 n80_b_stage7_aligned_storage=False, aligned_m32_n96_stage4_operands=False,
                 aligned_m16_n64_stage4_prefetch_operands=False, n128_stage4_wide=False,
                 n80_stage6_gate3_storage=False, n80_b_stage9_aligned_storage=False,
                 aligned_stage11_spill_operands=False, n32_a_stage13_aligned_storage=False, n64_a_stage9_aligned_storage=False,
                 n48_a_stage7_aligned_storage=False, aligned_n32_stage12_operands=False,
                 n16_u_stage6_aligned_storage=False, n64_u_stage4_aligned_storage=False,
                 n96_stage15_n88_storage=False, n96_b_stage7_aligned_storage=False):
        self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n = select_layout(M, N, K, num_sms, rowwise, block_n_multiple)
        self.num_m, self.num_n, self.num_k = divide_up(M, self.block_m), divide_up(N, self.block_n), divide_up(K, BLOCK_K)
        self.n64_stage4_skinny = (
            compact_rows and n64_u_stage4_aligned_storage
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 64, 4, 1, 1)
        )
        if self.n64_stage4_skinny:
            self.block_n = 8
            self.num_n = divide_up(N, self.block_n)
        self.n96_stage15_n88 = (
            compact_rows and n96_stage15_n88_storage
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (16, 96, 15, 1, 1)
        )
        if self.n96_stage15_n88:
            self.block_n, self.num_n = 88, divide_up(N, 88)
            reserved, stage_bytes = shared_footprint(self.block_m, self.block_n, self.num_k, rowwise)
            self.stages = min(16, (SMEM_CAPACITY - reserved) // stage_bytes)
        if (not rowwise and compact_rows and block_n_multiple == 32
                and (M, N, K, num_sms) == (64, 136, 1548289, 132)
                and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 13, 1, 1)):
            self.block_m, self.block_n = 32, 16
            self.num_m, self.num_n = divide_up(M, self.block_m), divide_up(N, self.block_n)
            reserved, stage_bytes = shared_footprint(self.block_m, self.block_n, self.num_k, rowwise)
            self.stages = min(16, (SMEM_CAPACITY - reserved) // stage_bytes)
        self.n32_stage4_balanced = (
            not rowwise and compact_rows and (M, N, K, num_sms, block_n_multiple) == (32, 96, 6234112, 2, 1)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (32, 32, 4, 1, 1)
        )
        if self.n32_stage4_balanced:
            self.block_n, self.stages = 48, 8
            self.num_n = divide_up(N, self.block_n)
        self.stream_wide_b_scales = n128_stage4_wide or self.n32_stage4_balanced
        self.n128_stage4_wide = n128_stage4_wide
        if self.n128_stage4_wide:
            assert (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (M, 128, 4, 1, 1)
            self.block_n = 192
            self.num_n = divide_up(N, self.block_n)
        self.n80_b_stage7_one_wave = False
        if (compact_rows and n80_b_stage7_aligned_storage
                and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 80, 7, 2, 1)):
            wide_n = 2 * self.block_n
            wide_panels = divide_up(N, wide_n)
            reserved, stage_bytes = shared_footprint(self.block_m, wide_n, self.num_k, rowwise)
            wide_stages = (SMEM_CAPACITY - reserved) // stage_bytes
            self.n80_b_stage7_one_wave = (
                wide_stages >= 4 and self.num_m * wide_panels <= num_sms < self.num_m * self.num_n
            )
            if self.n80_b_stage7_one_wave:
                self.block_n, self.num_n, self.stages = wide_n, wide_panels, min(16, wide_stages)
        self.m64_n128_single_m_share_a = (
            not rowwise and block_n_multiple == 1 and num_sms == 132
            and self.num_m == 1 and N == 65536 and K == 65536
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 128, 7, 2, 1)
        )
        if self.m64_n128_single_m_share_a:
            # Pair adjacent N tiles so they share the single A tile.
            self.cluster_m, self.cluster_n = 1, 2
        n128_panels = divide_up(N, BLOCK_K)
        self.m128_n80_stage6_balance = (
            not rowwise
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 80, 6, 1, 1)
            and self.num_m * n128_panels <= num_sms < self.num_m * self.num_n
        )
        if self.m128_n80_stage6_balance:
            # Use one wave. The activation and output tensor maps bound the final panel at N.
            self.block_n, self.num_n = BLOCK_K, n128_panels
            reserved, stage_bytes = shared_footprint(self.block_m, self.block_n, self.num_k, rowwise)
            self.stages = min(16, (SMEM_CAPACITY - reserved) // stage_bytes)
        n48_stage7_wide = (
            self.stages == 7 and (M, N, K, num_sms) == (769, 680, 1650689, 98)
        )
        n48_stage11_wide = (
            self.stages == 11 and num_sms == 132
        )
        n48_stage12_wide = (
            self.stages == 12 and (M, N, K, num_sms) == (897, 776, 483329, 132)
        )
        self.m64_n48_a_stage9_wide = False
        if (not rowwise and compact_rows and n48_a_stage9_storage and N % 8 == 0
                and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 48, 1, 2)
                and (self.stages in (8, 9) or n48_stage7_wide or n48_stage11_wide or n48_stage12_wide)):
            # One wider wave replaces two native waves while keeping the A multicast pairs.
            wide_n = 3 * self.block_n
            wide_tiles_n = divide_up(N, wide_n)
            wide_tiles = self.num_m * wide_tiles_n
            reserved, stage_bytes = shared_footprint(self.block_m, wide_n, self.num_k, rowwise)
            wide_stages = (SMEM_CAPACITY - reserved) // stage_bytes
            supported_wide_stages = (5,) if n48_stage11_wide else (6,) if n48_stage12_wide else (3, 4)
            self.m64_n48_a_stage9_wide = (
                wide_stages in supported_wide_stages
                and wide_tiles_n > 1 and 2 * wide_tiles >= num_sms >= wide_tiles
                and num_sms < self.num_m * self.num_n <= 2 * num_sms
                and self.num_n > 8 and self.num_n % 2 == 1
            )
            if self.m64_n48_a_stage9_wide:
                self.block_n, self.num_n, self.stages = wide_n, wide_tiles_n, wide_stages
        self.N, self.num_sms, self.rowwise = N, num_sms, rowwise
        # Aligned FP8 tiles sustain the two-GMMA-group pipeline on this single-CTA route.
        self.n32_u_stage14_one_tile = (
            not rowwise and compact_rows and aligned_n32_stage14_operands
            and num_sms == 2 and self.num_m == self.num_n == 1
            and N == self.block_n
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 14, 1, 1)
        )
        self.n32_u_stage5_one_tile = (
            not rowwise and block_n_multiple == 32 and K == 4759553
            and (M, N, num_sms) in ((33, 8, 132), (64, 32, 2))
            and self.num_m == self.num_n == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 5, 1, 1)
        )
        # The output map still stores a 32-column tile. Only real columns need tensor-core work.
        self.n32_single_tile = (not rowwise and self.num_m == self.num_n == 1 and N < self.block_n and N % 8 == 0
                               and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 4, 1, 1))
        # Amortize the partial ring and keep dynamic staging on misaligned multi-M grids.
        self.n112_unicast_ring = (
            not rowwise and self.num_k >= 3 * self.stages
            and (aligned_fp8_rows or (self.num_m == 1 and self.num_k >= 256))
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 112, 5, 1, 1)
        )
        self.n112_misaligned_unicast = (
            not rowwise and misaligned_fp8_rows and self.num_m > 1 and self.num_n > 1
            and self.num_k >= 3 * self.stages
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 112, 5, 1, 1)
        )
        # Retain the inline WGMMA path on the previously qualified tile grids.
        self.n112_short_k = (
            not rowwise and self.num_k == 128 and aligned_fp8_rows
            and self.num_m == 8
            and ((self.num_n == 5 and (8 <= num_sms <= 12 or num_sms == 22))
                 or (self.num_n == 10 and (40 <= num_sms <= 42 or 64 <= num_sms <= 78)))
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 112, 5, 1, 1)
        )
        self.skinny_short_n144 = (
            not rowwise and N == 8 and self.num_k == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 144, 7, 2, 1)
        )
        self.n160_stage4_single_tile = (
            not rowwise and block_n_multiple == 160 and self.num_m == self.num_n == 1 and self.num_k > self.stages
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 160, 4, 1, 1)
        )
        self.m64_n48_n8_a = (
            not rowwise and num_sms == 98 and N == 8 and self.num_m == 197
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 48, 1, 2)
            and (self.stages, self.num_k) in ((10, 7425), (11, 5601), (12, 3777), (13, 1953))
        )
        self.n48_a_stage8_one_n = (
            not rowwise and compact_rows and 8 <= N < self.block_n and N % 8 == 0 and self.num_n == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 48, 8, 1, 2)
        )
        stage14_one_n = (
            not rowwise and self.num_n == 1 and N % 8 == 0
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 48, 14, 1, 2)
        )
        self.m64_n48_a_stage14_narrow = stage14_one_n and N < self.block_n
        stage14_full_panel = stage14_one_n and N == self.block_n
        stage14_wide_panel = stage14_one_n and N > 8
        self.stage14_full_panel_pairs = stage14_full_panel and self.num_k > self.stages
        self.stage14_grad_l2_128 = stage14_wide_panel and (stage14_full_panel or fp8_row_alignment[0] == 16)
        self.stage14_activation_l2_128 = stage14_wide_panel and (stage14_full_panel or fp8_row_alignment[1] == 16)
        self.m64_n48_narrow_a = self.m64_n48_n8_a or self.m64_n48_a_stage14_narrow
        self.n32_two_sm_narrow = (
            not rowwise and num_sms == 2 and self.num_m == 2 and N == 8 and self.num_k == 1473
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 32, 9, 1, 1)
        )
        local_one_n_ring = (self.stages, num_sms, self.num_m, self.num_k) in (
            (9, 98, 197, 30785), (14, 98, 197, 17665),
            (10, 98, 197, 28161), (11, 98, 197, 25537), (12, 98, 197, 22913),
        )
        # The only N panel has no multicast peer. Use CTA-local release when the one-stage wrap spans three M waves.
        self.n16_a_one_panel_single_wrap = (
            not rowwise and N == self.block_n and 2 * num_sms < self.num_m <= 3 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 16, 16, 1, 2)
            and self.num_k == self.stages + 1
        )
        # There is no A multicast peer when the only N tile is eight columns.
        narrow_a_one_n = (
            not rowwise and compact_rows and n16_stage8_local_storage and N == 8
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 16, 1, 2)
        )
        self.n16_stage8_one_n_local = (
            narrow_a_one_n and self.stages == 8
            and self.num_m == 2 * num_sms + 1 and self.num_k == 33409
        )
        self.n16_stage13_one_n_local = (
            narrow_a_one_n and self.stages == 13 and self.num_m == 2 * num_sms + 1
            and 0 < M % self.block_m <= self.block_m // 2
        )
        self.n16_single_n_local_barriers = (
            self.n16_stage8_one_n_local or self.n16_stage13_one_n_local or self.n16_a_one_panel_single_wrap
        ) or (
            not rowwise and compact_rows and misaligned_fp8_rows and N == 8
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 16, 1, 2)
            and (self.stages == 15 or local_one_n_ring)
            and self.num_m > num_sms and self.num_k > self.stages
        )
        self.n48_a_stage8_grad_l2_128 = (
            self.n48_a_stage8_one_n and N > 8 and fp8_row_alignment[0] == 16
        )
        self.n48_a_stage8_activation_l2_128 = (
            self.n48_a_stage8_one_n and N > 8 and fp8_row_alignment[1] == 16
        )
        self.one_n_a_local_release = self.n16_single_n_local_barriers or self.n48_a_stage8_one_n
        self.n160_single_scale_panel = self.n160_stage4_single_tile and N <= BLOCK_K and N % 8 == 0
        self.narrow_n16_unicast = (
            not rowwise and N == 8 and self.num_m == num_sms
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (128, 16, 1, 1)
            and self.stages in (4, 5, 6, 7, 10)
        )
        self.narrow_n16_default = (
            not rowwise and N == 8 and block_n_multiple == 1
            and self.num_m == self.num_n == 1 and self.num_k > self.stages
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 16, 16, 1, 1)
        )
        self.n96_u_stage5_narrow = (
            aligned_n96_stage5_narrow_operands and self.num_m == self.num_n == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 96, 5, 1, 1)
        )
        narrow_unicast = (
            self.narrow_n16_unicast or self.n160_single_scale_panel or self.narrow_n16_default
            or self.n96_u_stage5_narrow
        )
        use_narrow_math = (self.n32_single_tile or self.n32_two_sm_narrow or self.skinny_short_n144
                           or self.m64_n48_narrow_a or self.n48_a_stage8_one_n or narrow_unicast)
        self.math_block_n = N if use_narrow_math else self.block_n
        self.long_a_n64 = (
            long_a_storage_ok and N % 8 == 0 and self.num_n % 2 == 1 and self.num_m <= num_sms
            and num_sms < self.num_m * self.num_n <= 2 * num_sms
            and 0 < N % self.block_n <= self.block_n // 2
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 64, 10, 1, 2)
        )
        self.long_a_n64_full_second_wave = self.long_a_n64 and self.num_m * (self.num_n - 1) > num_sms
        self.long_a_n64_stage11 = (
            aligned_stage11_tail and self.num_m <= num_sms and self.num_n % 2 == 1
            and self.num_m * (self.num_n - 1) <= num_sms < self.num_m * self.num_n
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 64, 11, 1, 2)
        )
        self.long_b_m16_n128 = (
            not rowwise and self.num_m == 1 and self.num_n == num_sms + 1 and N % 128 == 8
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (16, 128, 2, 1)
            and ((4 <= self.stages <= 10 and self.num_k >= 512)
                 or (self.stages == 11 and (num_sms, self.num_k) == (120, 321)))
        )
        self.n64_a_stage9_spill = (
            compact_rows and n64_a_stage9_aligned_storage
            and self.num_m <= self.num_n and self.num_n % 2 == 1
            and 0 < N % self.block_n <= self.block_n // 2
            and num_sms < self.num_m * (self.num_n - 1)
            and self.num_m * self.num_n <= 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 64, 9, 1, 2)
        )
        self.n48_a_stage7_grad_l2 = (
            compact_rows and n48_a_stage7_aligned_storage
            and self.num_m <= self.num_n and self.num_n % 2 == 1
            and 0 < N % self.block_n <= 16
            and self.num_m * self.num_n == 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 48, 7, 1, 2)
        )
        self.long_a_n64_stage11_spill = (
            aligned_stage11_spill_operands and self.num_m <= self.num_n and self.num_n % 2 == 1
            and num_sms < self.num_m * (self.num_n - 1)
            and self.num_m * self.num_n <= 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 64, 11, 1, 2)
        )
        self.long_a_n64_full_second_wave |= self.long_a_n64_stage11_spill
        self.long_a_n64_tail = self.long_a_n64 or self.long_a_n64_stage11 or self.long_a_n64_stage11_spill
        self.long_b_m32_n128 = (
            not rowwise and self.num_m == 1 and self.num_n == num_sms + 1 and N % 128 == 8
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (32, 128, 2, 1)
            and num_sms in (118, 132) and (self.stages, self.num_k) in (
                (4, 28193), (5, 23041), (6, 17889), (7, 12737), (8, 7585), (9, 2433),
            )
        )
        self.long_b_n128 = self.long_b_m16_n128 or self.long_b_m32_n128
        self.long_a_n128_tail = (
            aligned_a128_tail and self.num_n >= 3 and self.num_n % 2 == 1
            and num_sms < self.num_m * self.num_n <= 2 * num_sms
            and self.num_k > self.stages
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 128, 1, 2)
            and 4 <= self.stages <= 7
            # Restore the parent schedule for the 98-SM Stage 7 tail at this tile grid.
            and (self.stages, num_sms, self.num_m, self.num_n, self.num_k) != (7, 98, 17, 11, 193)
            and (self.stages >= 6 or self.num_m * (self.num_n - 1) <= num_sms)
        )
        self.n96_stage5_mn_tail = (
            not rowwise and not compact_k_scale and compact_rows and align_up(K, 16) % 256 == 16
            and 1 < self.num_m and 1 < self.num_n and self.num_m * self.num_n <= num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 96, 5, 1, 1)
            and N % self.block_n in (8, 16)
        )
        self.n96_stage4_narrow_tail = (
            not rowwise and compact_rows and n96_stage4_tail_storage
            and 1 < self.num_m and 1 < self.num_n and self.num_m * self.num_n <= num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 96, 4, 1, 1)
            and N % self.block_n in (8, 16)
        )
        self.n96_short_n_tail = self.n96_stage5_mn_tail or self.n96_stage4_narrow_tail
        narrow_tail = self.long_b_n128 or self.long_a_n64_tail or self.long_a_n128_tail or self.n96_short_n_tail
        self.long_tail_n = N % self.block_n if narrow_tail else 0
        self.cluster_size = self.cluster_m * self.cluster_n
        self.multicast_a = self.cluster_n > 1
        self.n96_sustained = (
            not rowwise and num_sms == 132 and self.num_m == 4 and self.num_n == 30
            and self.num_k in (16, 64, 128) and K % BLOCK_K == 0
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 96, 6, 1, 1)
        )
        self.n160_sustained = (
            not rowwise and num_sms == 132 and self.num_m == 8 and self.num_n == 32 and self.num_k == 64
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 160, 4, 2, 1)
        )
        # Qualify the six sustained N192 B-multicast call sites.
        self.n192_b_sustained = (
            not rowwise and num_sms == 132
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 192, 3, 2, 1)
            and (M, N, K) in (
                (2048, 4096, 2048),
                (2048, 10944, 16384),
                (2816, 2048, 16384),
                (2880, 4096, 2048),
                (2880, 4096, 8192),
                (5120, 6144, 8192),
            )
        )
        self.sustained_swizzled_output = self.n96_sustained or self.n160_sustained or self.n192_b_sustained
        self.m128_n80_stage6_gate3_mn = (
            compact_rows and n80_stage6_gate3_storage
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 80, 6, 1, 1)
        )
        self.n80_unicast_scale_pairs = (
            not rowwise and not compact_k_scale and compact_rows
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 80, 7, 1)
            and (((M, N, K) == (1024, 1024, 16384) and num_sms >= 64)
                 or ((M, N, K) == (128, 240, 20480) and num_sms <= 4))
        )
        self.n80_sustained_mn = (
            not rowwise and not compact_k_scale and num_sms == 132
            and M == 576 and N == 2048 and K == 16384
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 80, 7, 1, 1)
        )
        self.n128_sustained = (
            not rowwise and compact_k_scale and num_sms == 132
            and M == 2048 and N == 512 and K == 8192
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 128, 8, 1, 1)
        )
        self.n96_stage5_full_n_wave = (
            not rowwise and self.num_m == 1 and self.num_n == num_sms == 2
            and N == self.num_n * self.block_n
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 96, 5, 1)
        )
        self.n96_stage5_block_wave = (
            not rowwise and compact_k_scale and misaligned_fp8_rows and 1 < self.num_m and 1 < self.num_n
            and self.num_m * self.num_n <= num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 96, 5, 1)
        )
        self.n48_one_wave = (
            not rowwise and self.num_m * self.num_n <= num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 48, 8, 1)
        )
        self.n48_mn_tail = (
            self.n48_one_wave and not compact_k_scale and compact_rows
            and num_sms == 122 and block_n_multiple == 48
            and (M, N, K) == (961, 680, 16257)
        )
        use_long_n112_scales = (not rowwise and self.num_k >= 256
                               and (self.block_m, self.block_n, self.cluster_size) == (128, 112, 1))
        use_n80_scales = not rowwise and (self.block_m, self.block_n, self.cluster_size) == (128, 80, 1)
        self.use_linear_k_major_scales = compact_k_scale and (use_n80_scales or use_long_n112_scales)
        primary, secondary = (self.block_n, self.block_m) if self.multicast_a else (self.block_m, self.block_n)
        self.group_tiles = min((8, 16), key=lambda width: width * primary + divide_up(num_sms, width) * secondary)
        spill_pair_tiles = (self.num_m * self.num_n - num_sms) & ~1
        self.n32_a_stage4_partial_m_first = (
            not rowwise and compact_rows
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 4, 1, 2)
            and 0 < M % self.block_m <= self.block_m // 2 and N % self.block_n != 0 and N % 8 == 0
            and self.num_n % 2 == 1 and self.num_n <= self.group_tiles
            and self.num_m * (self.num_n - 1) == num_sms
            and 0 < spill_pair_tiles <= self.num_n - 1
        )
        self.math_threads = 128 if self.block_m <= 64 else 256
        self.first_producer_warp = self.math_threads // 32
        self.producer_warp = self.first_producer_warp + 2
        self.math_registers = 248 if self.math_threads == 128 else 232
        if self.n48_one_wave or (self.n96_sustained and self.num_k == 64 and not compact_k_scale):
            self.math_registers = 224
        if self.n96_sustained and self.num_k == 128:
            self.math_registers = 216
        self.store_threads = min(self.math_threads, self.block_m * 2)
        self.uniform_scale = (
            BLOCK_K % self.block_n == 0 or self.n160_single_scale_panel or self.n48_a_stage8_one_n
            or self.n32_stage4_balanced
        )
        self.scale_columns = 1 if self.uniform_scale else 2
        starts_at_zero = self.num_m * self.num_n <= self.num_sms or self.num_k % (2 * self.stages) == 0
        # Persistent M waves reuse one B panel and restart the four-stage ring at phase zero.
        self.n96_u_stage4_two_sm = (
            not rowwise and compact_rows and aligned_n96_stage4_operands
            and num_sms == 2 and self.num_m > num_sms and self.num_n == 1
            and N == self.block_n and starts_at_zero
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 96, 4, 1)
        )
        self.single_n80_tile = (self.block_m, self.block_n, self.cluster_size) == (64, 80, 1) and self.num_m == self.num_n == 1 and not rowwise
        self.n16_u_stage10 = (
            not rowwise and compact_rows and N == self.block_n and self.num_m == self.num_n == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 16, 10, 1)
        )
        self.n16_u_stage10_pair = self.n16_u_stage10 and n16_stage10_pair_storage
        self.n16_default_full_width_static_math = (
            not rowwise and N == 16 and block_n_multiple == 1
            and self.num_m == self.num_n == 1 and self.num_k >= 512
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 16, 16, 1, 1)
        )
        self.n16_default_grid_static_math = (
            aligned_grid_n16_operands and self.num_m > 1 and self.num_n > 1
            and self.num_m * self.num_n <= num_sms and 1024 <= self.num_k <= 8192
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 16, 16, 1, 1)
        )
        long_n16_static_math = (
            self.stages == 16 and aligned_fp8_rows and self.num_k >= 512
            and (N > 2 * BLOCK_K or self.num_n == 1)
        )
        self.static_n16_stage8_math = (
            self.block_m == 64 and self.block_n == 16
            and (self.stages in (8, 13, 15) or long_n16_static_math)
        )
        self.long_n80_stage4 = not rowwise and self.num_k >= 4096 and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 80, 4, 1)
        self.canonical_n16_scale = (not rowwise and N <= BLOCK_K
                                    and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 16, 15, 1))
        self.m32_n64_stage8_full = (
            not rowwise and compact_rows and (M, N, K, num_sms) == (32, 64, 3989504, 2)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (32, 64, 8, 1)
        )
        self.small_stage8_full_tile = (
            not rowwise and compact_rows and self.stages == 8 and self.cluster_size == 1
            and (
                (M, N, K, num_sms, self.block_m, self.block_n) == (16, 64, 4644864, 2, 16, 64)
                or (M, N, K, num_sms, self.block_m, self.block_n) == (32, 16, 5758976, 132, 32, 16)
            )
        )
        self.m32_n32_stage8_full = (
            not rowwise and compact_rows and (M, N, K, num_sms) == (32, 32, 5169152, 2)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (32, 32, 8, 1)
        )
        self.m16_n32_stage8_full = (
            not rowwise and self.num_m == 1 and N == 32
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (16, 32, 8, 1)
        )
        self.narrow_n160_four_stages = not rowwise and (self.block_m, self.block_n, self.stages, self.cluster_size) == (16, 160, 4, 1)
        self.wide_n192_four_math = (not rowwise
                                   and self.block_m in (16, 32)
                                   and (self.block_n, self.stages, self.cluster_size) == (192, 4, 1))
        # Three full N tiles need two waves, each entering the ring at phase zero.
        self.m32_n64_stage4_two_waves = (
            not rowwise and compact_rows and aligned_m32_n64_stage4_operands
            and num_sms == 2 and self.num_m == 1 and self.num_n == 3 and N == 3 * self.block_n
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (32, 64, 4, 1)
            and self.num_k % (2 * self.stages) == 0
        )
        # Two CTAs cover three N tiles. Both tiles on CTA 0 share one block scale for B.
        self.n64_reuse_grad = (
            not rowwise and aligned_fp8_rows and num_sms == 2
            and (self.num_m, self.num_n, self.num_k) == (1, 3, 37312)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 64, 4, 1)
        )
        self.n32_reuse_grad = (not rowwise and self.num_m == 1 and self.num_n == 3 and num_sms == 2
                               and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 32, 4, 1)
                               and self.num_k % self.stages == 0)
        self.n96_stage9_wave = (
            not rowwise and num_sms == 132 and (M, N, K) == (192, 4096, 16384)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 96, 9, 1)
        )
        self.m64_n112_one_wave = (
            not rowwise and (self.num_m, self.num_n, num_sms) == (2, 64, 132) and self.num_k in (16, 128)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 112, 8, 1)
        )
        self.n16_u_short = (
            not rowwise and num_sms == 132 and (M, N, K) == (621, 24, 283)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 16, 16, 1)
        )
        self.n128_block_output_strips = (
            not rowwise and compact_k_scale and compact_rows and (M, N, K, num_sms) == (512, 2048, 8192, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 128, 8, 1)
        )
        self.n16_u_stage9_one_tile = (
            not rowwise and (M, N, K, num_sms, block_n_multiple) == (64, 16, 4276224, 132, 1)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 16, 9, 1, 1)
        )
        self.n16_u_stage11 = (
            not rowwise and compact_rows and (M, N, K, num_sms) == (64, 16, 3604480, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 16, 11, 1)
        )
        self.n16_u_stage6 = (
            not rowwise and compact_rows and n16_u_stage6_aligned_storage
            and self.num_m == self.num_n == 1 and N == self.block_n
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 16, 6, 1, 1)
        )
        self.n16_u_stage7 = (
            not rowwise and compact_rows and (M, N, K, num_sms) == (64, 16, 4947968, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 16, 7, 1)
        )
        stream_n16 = self.n64_stage4_skinny or (self.block_m, self.block_n) == (64, 16) or (
            compact_rows and (self.block_m, self.block_n, M, N, K, num_sms) == (32, 16, 32, 16, 6561792, 132)
        )
        stream_n48 = self.block_n == N == 48 and num_sms == 2 and (self.block_m, K) in (
            (16, 3133440), (32, 2953216),
        )
        stream_n80 = (
            self.single_n80_tile and self.stages == 9 and compact_rows and n80_streamed_scale_storage
            and N == self.block_n
        )
        stream_stages = 9 if stream_n80 else 8
        self.stream_b_scale_row = False
        if (not rowwise and self.num_m == self.num_n == 1
                and (stream_n80 or ((self.stages, self.cluster_size) == (4, 1) and (stream_n16 or stream_n48)))):
            shared_fixed, shared_per_stage = shared_footprint(self.block_m, self.block_n, 0, False)
            self.stream_b_scale_row = shared_fixed + stream_stages * shared_per_stage <= SMEM_CAPACITY
        self.n80_streamed_scale = self.stream_b_scale_row and stream_n80
        self.stream_n48_b_scales = self.stream_b_scale_row and stream_n48
        if self.stream_b_scale_row:
            # Streaming the single B-scale row frees shared memory for a deeper TMA ring.
            self.stages = stream_stages
            self.uniform_scale = True
            self.scale_columns = 1
        # Restrict B lookahead to FP8 rows aligned to the K-block boundary.
        self.n160_stage4_full_panel = (
            not rowwise and aligned_n160_operands and block_n_multiple == 160 and N == self.block_n
            and self.num_m == self.num_n == 1 and self.num_k > 2 * self.stages
            and ((self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 160, 4, 1)
                 or ((M, K, num_sms) == (32, 1806336, 2)
                     and (self.block_m, self.block_n, self.stages, self.cluster_size) == (32, 160, 4, 1)))
        )
        self.n160_stage4_grad_offset = self.n160_stage4_single_tile and n160_grad_offset_only
        self.m16_n160_stage8 = (
            not rowwise and (M, N, K, num_sms) == (16, 160, 651264, 2)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (16, 160, 8, 1)
        )
        self.n144_unicast_one_wave = (
            not rowwise and num_sms == 132 and (self.num_m, self.num_n, self.num_k) == (32, 4, 56)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 144, 4, 1)
        )
        self.m32_n96_stage4_one_tile = (
            not rowwise and compact_rows and aligned_m32_n96_stage4_operands
            and num_sms == 2 and self.num_m == self.num_n == 1 and N == self.block_n
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (32, 96, 4, 1)
        )
        self.two_sm_single_tile = (
            not rowwise and num_sms == 2 and aligned_two_sm_operands
            and N == self.block_n and self.num_m == self.num_n == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_size)
            == (64, 96, 8, 1)
        )
        if self.two_sm_single_tile and N <= BLOCK_K:
            # A single full-width tile below the scale boundary has one B scale row.
            self.uniform_scale = True
            self.scale_columns = 1
        self.n112_stage4_two_waves = (
            not rowwise and aligned_fp8_rows and num_sms == 2 and (self.num_m, self.num_n, self.num_k) == (1, 3, 6240)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 112, 4, 1)
        )
        self.m16_n128_stage8_single = (
            not rowwise and aligned_fp8_rows and (M, N, K, num_sms) == (16, 128, 2416640, 2)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (16, 128, 8, 1)
        )
        self.m128_n80_short_a = (
            not rowwise and num_sms == 132 and (self.num_m, self.num_n) == (44, 51) and K <= 16
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 80, 7, 1, 2)
        )
        self.m64_n32_long_stage8 = (
            not rowwise and aligned_fp8_rows and num_sms == 2 and (self.num_m, self.num_n, self.num_k) == (1, 1, 30912)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 32, 8, 1)
        )
        self.static_ring = self.n96_stage15_n88 or self.n16_u_stage6 or self.stream_b_scale_row or self.n16_u_stage7 or self.m64_n32_long_stage8 or self.m16_n128_stage8_single or self.n112_stage4_two_waves or self.n16_u_stage11 or self.n144_unicast_one_wave or self.m16_n160_stage8 or self.cluster_size == 1 and not rowwise and starts_at_zero and (self.block_m < 64 and self.uniform_scale or self.block_n == 96 or self.single_n80_tile or self.static_n16_stage8_math or self.long_n80_stage4 or self.narrow_n160_four_stages or self.wide_n192_four_math or self.n32_reuse_grad or self.n64_reuse_grad or self.n32_single_tile or (self.block_m, self.block_n) == (128, 48) or self.m64_n112_one_wave)
        self.dynamic_n96_stage7_producer = (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 96, 7, 1) and self.num_m == self.num_n == 1 and not rowwise
        self.n80_three_tile_mn_scales = self.long_n80_stage4 and self.num_n <= 3 and not compact_k_scale
        self.rolling_n128_producer = (not rowwise and self.num_m == self.num_n == 1
                                      and (self.block_m, self.block_n, self.stages, self.cluster_size) == (32, 128, 8, 1))
        self.m16_n64_stage4_prefetch_b = (
            aligned_m16_n64_stage4_prefetch_operands
            and self.num_m == 1 and num_sms < self.num_n <= 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (16, 64, 4, 1, 1)
            and self.num_k % (2 * self.stages) == 0
        )
        self.inline_small_mma = self.static_ring and (
            self.m16_n128_stage8_single or
            (self.n16_u_stage11 and compact_k_scale) or
            self.m16_n160_stage8 or
            self.m16_n32_stage8_full or self.m32_n32_stage8_full or self.m32_n64_stage8_full or
            self.small_stage8_full_tile or
            self.block_m < 64 and (self.block_n == 16 or (self.block_n == 96 and self.stages == 8))
        )
        self.n192_one_wave = (
            not rowwise and self.num_m == 1 and self.num_n <= num_sms
            and self.num_k == 12 and N % 192 == 0
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 192, 3, 1, 1)
        )
        self.u128_stage4_one_wave = (
            not rowwise and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 128, 4, 1)
            and self.num_k == 128
            and (
                (num_sms, self.num_m, self.num_n) == (132, 32, 4) and N % 128 == 0
                or (num_sms, self.num_m, self.num_n) == (2, 2, 1) and N == 128 and K % BLOCK_K == 0
            )
        )
        if self.u128_stage4_one_wave:
            self.group_tiles = 4
        # The persistent short-K N64 multicast waves fit in one pipeline phase.
        self.short_a_n64 = (
            not rowwise and self.num_k == 1 and self.multicast_a and self.block_n == 64
            and (self.block_m, self.stages, self.num_m, self.num_n, num_sms)
            in ((128, 7, 64, 3, 98), (64, 12, 3, 65, 98), (64, 12, 26, 55, 132), (64, 12, 13, 110, 132))
        )
        # Each A panel gets a fresh stage. The first wave's B panels stay live.
        m_waves = divide_up(self.num_m, num_sms)
        self.n64_a_single_n = (
            not rowwise and N == self.block_n
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 64, 7, 1, 2)
            and m_waves > 1 and m_waves * self.num_k <= self.stages
        )
        # With all stages occupied, padded output rows use the single linear TMA store.
        self.n64_a_swizzled_output = (
            self.n64_a_single_n and (compact_output or m_waves * self.num_k < self.stages)
        )
        self.m64_n128_b_short = (
            not rowwise and (self.num_m, self.num_n, self.num_k, num_sms) == (17, 31, 1, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 128, 8, 2, 1)
        )
        self.m64_n160_b_one_k = (
            not rowwise and compact_rows and block_n_multiple == 1 and 32 < K <= 48
            and (self.num_m, self.num_n, num_sms) == (35, 56, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 160, 6, 2, 1)
        )
        self.swizzle_fp32_output = (
            self.two_sm_single_tile or
            self.m64_n128_b_short or self.m64_n160_b_one_k or
            self.n192_one_wave or (self.u128_stage4_one_wave and N % 128 == 0) or self.short_a_n64 or self.n96_stage9_wave or self.n64_a_swizzled_output or self.n128_block_output_strips
        )
        self.full_a128_stage4 = (
            not rowwise and self.num_n == 4 and self.num_k % 8 == 0
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 128, 4, 1, 2)
        )
        self.a128_stage4_full_groups = (
            not rowwise and self.num_n > self.group_tiles and self.num_n % self.group_tiles == 0
            and self.num_k % 8 == 0
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 128, 4, 1, 2)
        )
        self.short_n144_cluster = (not rowwise and self.num_k <= 8 and self.num_m * self.num_n <= 2 * self.num_sms
                                   and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 144, 4, 2, 1))
        self.reuse_short_a128 = (
            not rowwise and self.num_m == 1 and self.num_k == 2
            and self.num_n % 2 == 0 and self.num_n > 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 128, 5, 1, 2)
        )
        self.n16_stage4_two_phase = (
            not rowwise and compact_rows and self.num_m == 1
            and self.num_k >= 2 * self.stages
            and (self.block_m, self.stages, self.cluster_size) == (16, 4, 1)
            and (
                (self.num_n == 1 and self.block_n == 16)
                or (num_sms == 2 and N == 96 and self.block_n == 32
                    and self.num_k % (2 * self.stages) == 0)
            )
        )
        # Widen short N groups when FP32 stores stream past the 60 MiB H200 L2.
        self.wide_n48_a_one_k = (
            full_device_sm_budget and not rowwise and compact_rows and self.num_k == 1
            and self.num_m > 1
            and self.num_m * self.group_tiles < 2 * num_sms
            # Merge complete N groups and preserve the width of the partial group.
            and self.num_n % 32 < self.group_tiles
            and self.num_m * self.num_n >= 2 * self.stages * num_sms
            and 4 * M * N > (60 << 20)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 48, 9, 1, 2)
        )
        if self.wide_n48_a_one_k:
            self.group_tiles = 32
        self.short_a_n32 = (not rowwise and self.num_k == 1
                            and self.num_m * self.num_n <= (2 * self.stages - 1) * self.num_sms
                            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 32, 10, 1, 2))
        self.short_a_n16 = (not rowwise and self.num_m == self.num_k == 1
                            and self.num_n <= 2 * self.num_sms
                            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (16, 16, 16, 1, 2))
        self.short_b_n128 = (
            not rowwise and self.num_k == 1 and (self.num_m == 1 or self.num_n == 1)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 128, 8, 2, 1)
        )
        self.short_a_n144 = (
            not rowwise and self.num_m == self.num_k == 1
            and self.num_n <= 2 * self.num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 144, 7, 1, 2)
        )
        self.m32_n48_short_a = (
            not rowwise and compact_rows and self.num_m == self.num_k == 1
            and self.num_sms % 2 == 0 and 4 * self.num_sms < self.num_n <= 5 * self.num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (32, 48, 16, 1, 2)
        )
        self.short_a_n96 = (not rowwise and self.num_m == self.num_k == 1
                            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (16, 96, 15, 1, 2))
        self.short_a_n96_no_wrap = self.short_a_n96 and divide_up(self.num_n, self.num_sms) < self.stages
        self.n112_a_full_ring = (
            not rowwise and (self.num_m, self.num_n, self.num_k, num_sms) == (9, 26, 30, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 112, 5, 1, 2)
        )
        self.m32_n112_two_waves = (
            not rowwise and self.num_m == self.num_k == 1 and self.num_n == num_sms + 2
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (32, 112, 11, 1, 2)
        )
        self.m32_n128_two_waves = (
            not rowwise and self.num_m == self.num_k == 1 and self.num_n == num_sms + 2
            and num_sms == 116 and block_n_multiple == 128
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (32, 128, 10, 1, 2)
        )
        paired_n_tiles = self.num_n - 1
        self.n96_a_resident_b = (
            not rowwise and compact_rows and 0 < K < BLOCK_K and self.num_m % 2 == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 96, 6, 1, 2)
            and 1 < self.num_n < self.group_tiles and self.num_n % 2 == 1
            and self.num_m * paired_n_tiles >= 6 * num_sms
            and 2 * (num_sms % paired_n_tiles) == paired_n_tiles
        )
        self.short_a_n112 = (
            not rowwise and self.num_k == 1 and (self.num_m, self.num_n, num_sms) == (5, 37, 98)
            and M % self.block_m > self.block_m // 2
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 112, 5, 1, 2)
        )
        self.short_a_n128 = (
            not rowwise and self.num_m == self.num_k == 1 and self.num_n <= 2 * self.num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 128, 8, 1, 2)
        )
        self.short_b_n80 = (
            not rowwise and self.num_k <= 2 and self.num_m * self.num_n <= 2 * self.num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 80, 11, 2, 1)
        )
        self.reuse_full_b_n144 = (
            not rowwise and N == 144 and K <= BLOCK_K // 2 and self.num_m % 2 == 0
            and self.num_m <= (self.stages - 1) * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 144, 7, 2, 1)
        )
        self.m128_n16_a_stage11 = (
            not rowwise and (self.num_m, self.num_n, self.num_k, num_sms) == (76, 5, 1, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 16, 11, 1, 2)
        )
        self.wide_n144_schedule = (
            not rowwise and compact_rows and (M, N, K, num_sms, block_n_multiple) == (512, 65536, 128, 132, 1)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 144, 4, 2, 1)
        )
        self.n32_a_paired_single_n = (
            not rowwise and self.num_n == 1 and self.num_k > self.stages
            and (
                self.stages in (13, 16)
                or (compact_rows and N == 8 and (num_sms, self.num_m) == (98, 197)
                    and (self.stages, self.num_k) in ((14, 8961), (15, 5825)))
            )
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 32, 1, 2)
        )
        self.n32_a_stage12_multi_n = (
            not rowwise and compact_rows and (M, N, K, num_sms, block_n_multiple) == (513, 904, 1949697, 132, 32)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 12, 1, 2)
        )
        self.n32_a_stage13_multi_n = (
            compact_rows and n32_a_stage13_aligned_storage
            and self.num_n > 1 and self.num_n % 2 == 1
            and num_sms < self.num_m * self.num_n <= 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 13, 1, 2)
        )
        self.n32_a_multi_n_grad_l2_128 = (
            not rowwise and compact_rows and (M, N, num_sms) == (513, 904, 132)
            and (self.stages, K) in ((14, 1146881), (15, 745473))
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 32, 1, 2)
        )
        self.short_n96_b_stage10 = (
            not rowwise and compact_rows and (M, N, K, num_sms) == (1596, 1928, 12, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 96, 10, 2, 1)
        )
        # Four N32 tiles share one B-scale row. Each CTA pair covers one quartet.
        a_panels_per_scale = BLOCK_K // 32
        self.a_panel_quartets = self.num_n // a_panels_per_scale
        self.a_panel_fused_ctas = self.num_m * self.a_panel_quartets * self.cluster_n
        self.resident_a_panels = (
            aligned_m128_n32_a and self.num_m >= 8 and self.num_m % 2 == 0
            and self.num_n > a_panels_per_scale and self.num_n % a_panels_per_scale == 1
            and num_sms < self.num_m * self.num_n <= 2 * num_sms
            and self.a_panel_fused_ctas < num_sms
            and self.num_m <= 2 * (num_sms - self.a_panel_fused_ctas)
            # The ordinary tail can start each output at phase zero.
            and self.num_k % (2 * self.stages) == 0
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 32, 8, 1, 2)
        )
        self.paired_ring_slots = self.stages - self.stages % 2
        self.paired_ring_has_last_stage = self.paired_ring_slots != self.stages
        self.use_linear_stage_index = (
            self.n64_a_single_n or
            self.m128_n16_a_stage11 or
            self.m32_n112_two_waves or self.m32_n128_two_waves
            or self.reuse_full_b_n144 or self.short_a_n64 or self.short_n144_cluster or self.short_a_n16 or self.short_a_n144
            or self.short_a_n96_no_wrap or self.short_a_n128 or self.short_a_n112 or self.short_b_n80
            or (not rowwise and self.multicast_a and self.num_n == 1
                and (self.block_m, self.block_n, self.stages) == (64, 16, 16))
        )
        self.n80_stage7_block = (
            not rowwise and compact_k_scale and (M, N, K, num_sms) == (576, 2048, 16384, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (128, 80, 7, 1)
        )
        self.select_scale_before_product = (self.use_linear_k_major_scales or use_long_n112_scales
                                            or self.static_ring and (self.block_m, self.block_n) == (128, 48))
        self.scale_gap = gcd(BLOCK_K, self.block_n)
        self.specialize_scale = not self.uniform_scale and BLOCK_K // self.scale_gap <= 4
        buffered_scales = 0 if self.stream_b_scale_row or self.stream_wide_b_scales else self.num_k
        reserved, per_stage = shared_footprint(self.block_m, self.block_n, buffered_scales, rowwise)
        self.shared_bytes = reserved + self.stages * per_stage
        self.n80_full_sms_unicast = (
            self.single_n80_tile and self.stages == 10 and num_sms == 132
            and (M, N, K, block_n_multiple) == (64, 80, 98305, 80)
        )
        if self.n80_full_sms_unicast:
            self.producer_warp = self.first_producer_warp + 3
        self.prefetch_n80 = self.n80_full_sms_unicast or (self.single_n80_tile and self.stages == 10 and num_sms == 2)
        self.n80_b_multicast = (
            not rowwise and self.num_m * self.num_n <= 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 80, 10, 2, 1)
        )
        self.n96_b_stage7 = (
            compact_rows and n96_b_stage7_aligned_storage
            and self.num_m % 2 == 1 and 0 < N % self.block_n <= 32
            and num_sms < (self.num_m - 1) * self.num_n
            and self.num_m * self.num_n <= 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 96, 7, 2, 1)
        )
        self.n80_b_stage7_aligned = (
            compact_rows and n80_b_stage7_aligned_storage
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 80, 7, 2, 1)
        )
        self.n80_b_stage9_l2_128 = (
            compact_rows and n80_b_stage9_aligned_storage
            and self.num_m > 1 and self.num_m % 2 == 1
            and num_sms < self.num_m * self.num_n <= 2 * num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 80, 9, 2, 1)
        )
        # The native cluster has one N tile and therefore no A multicast peer.
        self.single_n_a_cluster = (
            self.n64_a_single_n or (
                not rowwise and num_sms == 132 and (M, N, K) == (20428, 16, 1720)
                and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 16, 16, 1, 2)
            )
        )
        self.prefetch_n16 = (not rowwise and
                             (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 16, 16, 1))
        self.narrow_activation = (self.skinny_short_n144 or self.n32_two_sm_narrow
                                  or self.m64_n48_narrow_a or self.n48_a_stage8_one_n or narrow_unicast)
        self.clear_output_padding = (self.n32_single_tile or self.n32_two_sm_narrow
                                     or self.m64_n48_narrow_a or self.n48_a_stage8_one_n or narrow_unicast)
        self.activation_block_n = self.math_block_n if self.narrow_activation else self.block_n
        self.transaction_bytes = (self.block_m + self.activation_block_n) * BLOCK_K + 4 * self.block_m + (4 * self.block_n if rowwise else 0)
        self.b144_three_wave = (
            not rowwise and compact_rows and misaligned_fp8_rows
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (128, 144, 4, 2, 1)
            and self.num_m % 2 == 1 and 2 * num_sms < self.num_m * self.num_n <= 3 * num_sms
            and 2 * self.stages < self.num_k <= 4 * self.stages
        )
        self.m64_n80_input_128b = (
            not rowwise and compact_rows and misaligned_fp8_rows
            and (self.block_m, self.block_n, self.cluster_m, self.cluster_n) == (64, 80, 1, 2)
            and num_sms < self.num_m * self.num_n <= 2 * num_sms
            and self.num_k >= 512
            and (self.stages != 4 or self.num_n % 2 == 0)
        )
        self.m64_n80_grouped_wave = (
            self.m64_n80_input_128b and self.stages in (7, 8)
            and self.num_n > self.group_tiles and self.num_m * self.group_tiles >= num_sms
        )
        self.m64_n112_a_stage7_two_waves = (
            not rowwise and compact_rows and misaligned_fp8_rows
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 112, 7, 1, 2)
            and num_sms < self.num_m * self.num_n <= 2 * num_sms
            and self.num_n > self.group_tiles and self.num_n % 2 == 1
            and self.num_m * self.group_tiles >= num_sms
        )
        self.m64_n48_a_stage10_multi_n = (
            not rowwise and compact_rows and misaligned_fp8_rows
            and (M, N, K, num_sms) == (897, 776, 950273, 132)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 48, 10, 1, 2)
        )
        self.n128_u_stage4_two_sm = (
            not rowwise and compact_rows and (M, N, K, num_sms) == (64, 1152, 3203072, 2)
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 128, 4, 1)
        )
        self.n192_u_stage4_aligned = (
            aligned_n192_stage4 and num_sms == 2 and self.num_m == self.num_n == 1
            and N == self.block_n
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 192, 4, 1)
        )
        self.stage4_ordinal_math = (self.n160_stage4_full_panel or self.n160_stage4_grad_offset
                                    or self.n192_u_stage4_aligned)
        self.m64_n112_a_stage5_l2 = (
            not rowwise and compact_rows and n112_stage5_l2_storage and N % self.block_n == 8
            and (self.num_m, self.num_n, self.num_k, num_sms) == (13, 8, 8353, 98)
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 112, 5, 1, 2)
        )
        n16_unicast_single_tile = (
            aligned_n16_stage14_operands and self.num_m == self.num_n == 1 and N == self.block_n
            and (self.block_m, self.block_n, self.cluster_size) == (64, 16, 1)
        )
        n16_stage14_single_tile = n16_unicast_single_tile and self.stages == 14
        n16_stage5_full_tile = n16_unicast_single_tile and self.stages == 5 and M == self.block_m
        self.n16_u_stage14_aligned = n16_stage14_single_tile and aligned_fp8_rows
        self.n16_u_stage14_shifted = (
            n16_stage14_single_tile and misaligned_fp8_rows and n16_stage10_bulk_storage
        )
        self.n16_u_stage5_aligned = (
            n16_stage5_full_tile and aligned_fp8_rows and self.num_k % 2 == 0
        )
        self.n16_u_stage5_shifted = (
            n16_stage5_full_tile and misaligned_fp8_rows and n16_stage10_bulk_storage
        )
        self.n16_u_bulk_scale = self.n16_u_stage14_shifted or self.n16_u_stage5_shifted or (
            self.n16_u_stage10 and n16_stage10_bulk_storage
        )
        self.n48_u_stage8_single_tile = (
            aligned_n48_stage8_operands and self.num_m == self.num_n == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 48, 8, 1)
            and self.num_k % 2 == 0
        )
        self.n32_u_stage10_single_tile = (
            aligned_n32_stage10_operands and self.num_m == self.num_n == 1
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 10, 1, 1)
            and self.num_k % 2 == 0
        )
        self.m64_n80_stage4_two_m_waves = (
            not rowwise and compact_rows and num_sms == 2
            and self.num_m == 3 and self.num_n == 1 and N == self.block_n
            and self.num_k % (2 * self.stages) == 0
            and (self.block_m, self.block_n, self.stages, self.cluster_size) == (64, 80, 4, 1)
        )
        self.stage4_b_prefetch = (
            self.n96_u_stage4_two_sm or self.m32_n64_stage4_two_waves or self.m32_n96_stage4_one_tile
            or self.m64_n80_stage4_two_m_waves or self.m16_n64_stage4_prefetch_b
        )
        self.n32_u_stage12_one_wave = (
            aligned_n32_stage12_operands and self.num_m == 1 and self.num_n == num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 12, 1, 1)
        )
        self.use_cta_local_stage_wait = (
            self.n16_u_stage6 or
            self.n32_u_stage12_one_wave or self.m32_n96_stage4_one_tile
            or self.m32_n64_stage4_two_waves
            or self.n96_u_stage4_two_sm
            or self.n32_u_stage10_single_tile
            or self.n32_u_stage14_one_tile
            or self.n48_u_stage8_single_tile
            or self.n16_u_stage5_aligned
            or self.n16_u_stage14_aligned
            or self.m32_n64_stage8_full
            or self.m32_n32_stage8_full
            or self.small_stage8_full_tile
            or self.one_n_a_local_release
            or self.n16_u_stage10_pair
            or self.n160_stage4_grad_offset
            or self.n128_u_stage4_two_sm
            or self.n96_short_n_tail
            or self.m64_n48_a_stage9_wide
        )
        self.n32_u_stage10_all_scale_lanes = (
            not rowwise and not compact_k_scale and full_device_sm_budget
            and M == self.block_m and N > BLOCK_K and block_n_multiple == 32
            and self.num_m == 1 and self.num_n <= self.num_sms
            and (self.block_m, self.block_n, self.stages, self.cluster_m, self.cluster_n) == (64, 32, 10, 1, 1)
        )
        self.select_routes()
        for width in self.wgmma_widths():
            check_wgmma_width(width)

    def wgmma_widths(self):
        widths = (self.block_n, self.math_block_n)
        if self.long_tail_n:
            widths += (self.long_tail_n,)
        return widths

    def select_routes(self):
        # Choices in each table are ordered. The first true predicate selects the route.
        dynamic_producer = (
            self.stream_b_scale_row or self.n16_u_stage7 or self.m64_n32_long_stage8
            or self.m16_n128_stage8_single or self.n16_u_stage11 or self.two_sm_single_tile
            or self.m16_n160_stage8 or self.n96_stage9_wave or self.dynamic_n96_stage7_producer
            or self.static_n16_stage8_math or self.rolling_n128_producer
            or self.narrow_n160_four_stages or self.wide_n192_four_math
        )
        self.producer_route = select_route(
            ProducerRoute.DYNAMIC_RING,
            (ProducerRoute.RESIDENT_A_PANELS, self.resident_a_panels),
            (ProducerRoute.RESIDENT_GRADIENT, self.n32_reuse_grad or self.n64_reuse_grad),
            (ProducerRoute.STATIC_RING, self.n32_u_stage12_one_wave or self.n16_u_stage9_one_tile or self.n32_u_stage5_one_tile
             or self.n32_u_stage10_single_tile or self.n32_u_stage14_one_tile
             or self.n48_u_stage8_single_tile or self.n16_u_stage5_shifted or self.n16_u_stage5_aligned
             or self.n16_u_stage14_shifted or self.n16_u_stage14_aligned or self.n16_u_stage10_pair or self.n160_stage4_full_panel
             or self.n160_stage4_grad_offset or self.n128_u_stage4_two_sm
             or self.m64_n80_stage4_two_m_waves or self.n80_stage7_block or self.stream_b_scale_row
             or self.n16_u_stage7 or self.u128_stage4_one_wave or self.n80_sustained_mn
             or (self.static_ring and not dynamic_producer)),
        )
        self.math_route = select_route(
            MathRoute.DYNAMIC_RING,
            (MathRoute.RESIDENT_A_PANELS, self.resident_a_panels),
            (MathRoute.ASYNC_PAIRS, self.n32_u_stage12_one_wave or self.n32_u_stage10_single_tile or self.n32_u_stage14_one_tile
             or self.n48_u_stage8_single_tile or self.n16_u_stage5_aligned or self.n16_u_stage14_aligned
             or self.two_sm_single_tile or self.n16_u_stage10_pair),
            (MathRoute.LONG_A_PAIRS, self.long_a_n64_full_second_wave or self.n32_a_paired_single_n
             or self.stage14_full_panel_pairs),
            (MathRoute.RESIDENT_ACTIVATION, self.n64_a_single_n),
            (MathRoute.SINGLE_N_A, self.single_n_a_cluster),
            (MathRoute.PREFIX_RING, self.n112_unicast_ring or self.n112_a_full_ring or self.long_b_n128),
            (MathRoute.DYNAMIC_RING, self.n96_stage5_full_n_wave),
            (MathRoute.STATIC_RING, self.n16_u_stage9_one_tile or self.n32_u_stage5_one_tile or self.n16_default_full_width_static_math
             or self.n16_default_grid_static_math
             or self.u128_stage4_one_wave or self.static_ring
             or self.full_a128_stage4 or self.a128_stage4_full_groups),
        )
        self.a_tail_first_tiles = 0
        if self.long_a_n64_stage11:
            full_tiles = self.num_m * (self.num_n - 1)
            repeat_tiles = 2 * divide_up(self.num_m * self.num_n - self.num_sms, 2)
            if repeat_tiles <= full_tiles and full_tiles + repeat_tiles <= self.num_sms:
                self.a_tail_first_tiles = repeat_tiles
        self.tile_order = select_route(
            TileOrder.PERSISTENT,
            (TileOrder.SINGLE_TILE, self.n16_u_stage9_one_tile or self.n32_u_stage5_one_tile
             or self.n32_u_stage10_single_tile or self.n32_u_stage14_one_tile
             or self.n48_u_stage8_single_tile or self.n192_u_stage4_aligned or self.n16_u_stage5_aligned
             or self.n16_u_stage14_aligned or self.n16_u_stage10_pair or self.stream_b_scale_row or self.n16_u_stage7
             or self.n16_u_stage6 or self.m64_n32_long_stage8),
            (TileOrder.M_LINEAR, self.m64_n48_narrow_a or self.n48_a_stage8_one_n),
            (TileOrder.SINGLE_TILE, self.m16_n128_stage8_single),
            (TileOrder.N_LINEAR, self.n32_u_stage12_one_wave or self.n112_stage4_two_waves
             or self.n64_reuse_grad or self.n128_u_stage4_two_sm),
            (TileOrder.SINGLE_TILE, self.n16_u_stage11 or self.two_sm_single_tile),
            (TileOrder.M_LINEAR, self.n32_two_sm_narrow),
            (TileOrder.N144_B_WIDE, self.wide_n144_schedule),
            (TileOrder.SINGLE_TILE, self.m16_n160_stage8),
            (TileOrder.SHORT_A, self.m128_n16_a_stage11),
            (TileOrder.SHORT_UNICAST, self.n16_u_short),
            (TileOrder.M_LINEAR, self.single_n_a_cluster or self.one_n_a_local_release),
            (TileOrder.N48_MN_TAIL, self.n48_mn_tail),
            (TileOrder.M_LINEAR_MULTICAST, self.reuse_full_b_n144),
            (TileOrder.N_LINEAR, self.n192_one_wave),
            (TileOrder.N_GROUPED, self.n48_one_wave or self.n128_sustained),
            (TileOrder.SINGLE_N80, self.single_n80_tile),
            (TileOrder.SHORT_B, self.short_n144_cluster or self.short_b_n80 or self.n80_b_multicast
             or self.b144_three_wave),
            (TileOrder.SHORT_A, self.wide_n48_a_one_k or self.short_a_n112 or self.n96_a_resident_b
             or self.n32_a_stage12_multi_n),
            (TileOrder.MN_LINEAR_MULTICAST, self.full_a128_stage4),
            (TileOrder.N_LINEAR_MULTICAST, self.reuse_short_a128 or self.m64_n128_single_m_share_a),
            (TileOrder.N32_A_STAGE4_PARTIAL_M_FIRST, self.n32_a_stage4_partial_m_first),
            (TileOrder.SHORT_A_TAIL_FIRST, self.a_tail_first_tiles > 0),
            (TileOrder.SHORT_A, self.short_a_n32 or self.short_a_n64 or self.long_a_n64_tail
             or self.m64_n80_grouped_wave or self.m64_n112_a_stage7_two_waves
             or self.m64_n48_a_stage9_wide or self.long_a_n128_tail),
            (TileOrder.N_LINEAR_MULTICAST, self.m32_n112_two_waves or self.m32_n128_two_waves),
            (TileOrder.SHORT_N_LINEAR_A, self.short_a_n16 or self.short_a_n144 or self.short_a_n96
             or self.short_a_n128 or self.m32_n48_short_a),
            (TileOrder.SHORT_B, self.m64_n128_b_short or self.short_n96_b_stage10),
            (TileOrder.SHORT_B_N128, self.short_b_n128),
        )
        self.scale_promotion = select_route(
            ScalePromotion.GROUPS,
            (ScalePromotion.PAIRS_SELECT_FIRST,
             self.m64_n48_a_stage9_wide or self.m128_n80_stage6_gate3_mn),
            (ScalePromotion.GROUP_PAIRS, self.b144_three_wave or self.m64_n80_grouped_wave
             or self.m64_n112_a_stage7_two_waves or self.m64_n48_a_stage10_multi_n
             or self.n112_stage4_two_waves
             or self.n80_unicast_scale_pairs
             or self.n144_unicast_one_wave or self.m64_n112_one_wave or self.n112_unicast_ring
             or self.n112_misaligned_unicast or self.n112_a_full_ring),
            (ScalePromotion.SELECT_FIRST, self.select_scale_before_product),
        )
        self.double_producer_phase = (self.n16_stage4_two_phase or self.n96_short_n_tail or self.n160_stage4_grad_offset or self.n112_stage4_two_waves or self.n144_unicast_one_wave
                                      or self.long_n80_stage4 or self.wide_n192_four_math)
        self.double_math_phase = (self.n112_stage4_two_waves or self.n144_unicast_one_wave
                                  or self.u128_stage4_one_wave or self.long_n80_stage4
                                  or self.full_a128_stage4 or self.a128_stage4_full_groups)
        self.swizzle_tma_output = self.swizzle_fp32_output or self.sustained_swizzled_output
        self.input_l2_128b = self.m64_n128_single_m_share_a or self.m64_n48_a_stage9_wide or self.b144_three_wave or self.m64_n80_input_128b or self.m64_n112_a_stage7_two_waves or self.m64_n48_a_stage10_multi_n or self.n64_a_stage9_spill
        self.fixed_scale_split = self.two_sm_single_tile or self.stream_n48_b_scales or self.n192_u_stage4_aligned

    @cute.jit
    def __call__(self, mGrad: cute.Tensor, mGradScale: cute.Tensor, mActivation: cute.Tensor, mActivationScale: cute.Tensor, mOut: cute.Tensor, stream):
        grad_tile, activation_tile = (self.block_m, BLOCK_K), (self.block_n, BLOCK_K)
        grad_scale_tile, activation_scale_tile = (self.block_m, 1), (self.block_n, 1)
        out_tile = (self.block_m, self.block_n)
        atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, Float8E4M3FN)
        grad_layout = cute.tile_to_shape(atom, (*grad_tile, self.stages), (0, 1, 2))
        activation_layout = cute.tile_to_shape(atom, (*activation_tile, self.stages), (0, 1, 2))
        grad_scale_layout = cute.make_layout((*grad_scale_tile, self.stages), stride=(1, self.block_m, align_up(self.block_m, 32)))
        activation_scale_layout = cute.make_layout((*activation_scale_tile, self.stages), stride=(1, self.block_n, align_up(self.block_n, 32)))
        out_layout = cute.make_layout(out_tile, stride=(self.block_n, 1))

        promote = TensorMapL2Promotion.l2_256b
        input_promotion = TensorMapL2Promotion.l2_128b if self.n112_misaligned_unicast else promote
        if cutlass.const_expr(self.m128_n80_short_a and mGrad.stride[0] == 16 and mActivation.stride[0] == 16):
            input_promotion = TensorMapL2Promotion.l2_128b
        if cutlass.const_expr(self.input_l2_128b or self.n80_b_stage7_aligned or self.n96_b_stage7):
            input_promotion = TensorMapL2Promotion.l2_128b
        swizzle = TensorMapSwizzle.s128b
        grad_promotion = input_promotion
        if cutlass.const_expr(self.n32_a_paired_single_n or self.stage14_grad_l2_128 or self.n32_a_multi_n_grad_l2_128 or self.n32_a_stage12_multi_n
                              or self.n32_a_stage13_multi_n):
            grad_promotion = TensorMapL2Promotion.l2_128b
        if cutlass.const_expr(self.n160_stage4_grad_offset or self.m64_n112_a_stage5_l2 or self.n48_a_stage8_grad_l2_128
                              or self.n80_b_stage9_l2_128):
            grad_promotion = TensorMapL2Promotion.l2_128b
        if cutlass.const_expr(self.n48_a_stage7_grad_l2):
            grad_promotion = TensorMapL2Promotion.l2_128b
        grad = create_tensor_map_tiled(mGrad.iterator.toint(), Float8E4M3FN, [mGrad.shape[1], mGrad.shape[0]], [Int64(mGrad.stride[0]) // 16], [BLOCK_K, self.block_m], swizzle=swizzle, l2_promotion=grad_promotion)
        activation_promotion = input_promotion
        stage5_b_promotion = (
            self.n96_stage5_block_wave
            and mGrad.stride[0] % 256 == mActivation.stride[0] % 256 == 16
        )
        if cutlass.const_expr(self.stage14_activation_l2_128 or stage5_b_promotion
                              or self.m64_n112_a_stage5_l2 or self.n48_a_stage8_activation_l2_128
                              or self.n32_a_stage13_multi_n):
            activation_promotion = TensorMapL2Promotion.l2_128b
        if cutlass.const_expr(self.n96_short_n_tail):
            activation_promotion = TensorMapL2Promotion.l2_128b
        activation = create_tensor_map_tiled(mActivation.iterator.toint(), Float8E4M3FN, [mActivation.shape[1], mActivation.shape[0]], [Int64(mActivation.stride[0]) // 16], [BLOCK_K, self.activation_block_n], swizzle=swizzle, l2_promotion=activation_promotion)
        grad_scale = create_tensor_map_tiled(mGradScale.iterator.toint(), Float32, [align_up(mGradScale.shape[0], 4), self.num_k], [Int64(mGradScale.stride[1]) // 4], [self.block_m, 1], l2_promotion=promote)
        if cutlass.const_expr(self.swizzle_tma_output):
            out = create_tensor_map_tiled(mOut.iterator.toint(), Float32, [mOut.shape[1], mOut.shape[0]],
                                          [Int64(mOut.stride[0]) // 4], [32, self.block_m],
                                          swizzle=swizzle, l2_promotion=promote)
        else:
            out = create_tensor_map_tiled(mOut.iterator.toint(), Float32, [mOut.shape[1], mOut.shape[0]], [Int64(mOut.stride[0]) // 4], [self.block_n, self.block_m], l2_promotion=promote)
        activation_scale = grad_scale
        if cutlass.const_expr(self.rowwise):
            activation_scale = create_tensor_map_tiled(mActivationScale.iterator.toint(), Float32, [mActivationScale.shape[0], self.num_k], [Int64(mActivationScale.stride[1]) // 4], [self.block_n, 1], l2_promotion=promote)

        elif cutlass.const_expr(self.long_tail_n):
            activation_scale = create_tensor_map_tiled(
                mActivation.iterator.toint(), Float8E4M3FN,
                [mActivation.shape[1], mActivation.shape[0]], [Int64(mActivation.stride[0]) // 16],
                [BLOCK_K, self.long_tail_n], swizzle=swizzle, l2_promotion=promote)

        operation = warpgroup.MmaF8Op(Float8E4M3FN, Float8E4M3FN, Float32, (64, self.math_block_n, 32), warpgroup.OperandSource.SMEM, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K)
        mma = cute.make_tiled_mma(cute.make_mma_atom(operation), (self.math_threads // 128, 1, 1))
        layouts = (out_layout, grad_layout, activation_layout, grad_scale_layout, activation_scale_layout)
        if cutlass.const_expr(self.canonical_n16_scale):
            scale_layout = cute.make_layout((1, self.num_k), stride=(1, 1))
            mActivationScale = cute.make_tensor(mActivationScale.iterator, scale_layout)
        # Co-locate prefetch with the TMA issuer on the misaligned input path.
        prefetch_warp = self.first_producer_warp
        if cutlass.const_expr((self.prefetch_n16 or self.prefetch_n80)
                              and mGrad.stride[0] % 32 == 16 and mActivation.stride[0] % 32 == 16):
            prefetch_warp = self.producer_warp
        if cutlass.const_expr(self.n16_u_short):
            prefetch_warp = self.producer_warp
        self.kernel(mma, layouts, grad, activation, grad_scale, out, activation_scale, mActivationScale, prefetch_warp).launch(grid=(self.num_sms, 1, 1), block=(self.math_threads + 128, 1, 1), cluster=(self.cluster_size, 1, 1) if self.cluster_size > 1 else None, smem=self.shared_bytes, min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(self, mma: cute.TiledMma, layouts: tuple, grad: GridConstant[TensorMap], activation: GridConstant[TensorMap], grad_scale: GridConstant[TensorMap], out: GridConstant[TensorMap], activation_scale: GridConstant[TensorMap], mActivationScale: cute.Tensor, prefetch_warp: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        first_scale_loader = 32
        # Use the first warpgroup for TMA on the full, misaligned 12-SM wave.
        if cutlass.const_expr(
            self.prefetch_n16 and self.num_sms == self.num_m * self.num_n == 12
            and prefetch_warp == self.producer_warp
        ):
            tidx ^= self.math_threads
            warp ^= self.first_producer_warp
            if cutlass.const_expr((self.num_m, self.num_n) == (4, 3)):
                if cutlass.const_expr(self.num_k == 33):
                    first_scale_loader = 96
                elif cutlass.const_expr(self.num_k == 131):
                    first_scale_loader = 64
            if cutlass.const_expr((self.num_m, self.num_n, self.num_k) == (3, 4, 256)):
                first_scale_loader = 96
            if cutlass.const_expr((self.num_m, self.num_n, self.num_k) == (4, 3, 256)):
                prefetch_warp = self.first_producer_warp
        out_layout, grad_layout, activation_layout, grad_scale_layout, activation_scale_layout = layouts
        smem = cutlass.utils.SmemAllocator()
        s_out = smem.allocate_tensor(Float32, out_layout, byte_alignment=1024)
        s_grad = smem.allocate_tensor(Float8E4M3FN, grad_layout.outer, byte_alignment=1024, swizzle=grad_layout.inner)
        s_activation = smem.allocate_tensor(Float8E4M3FN, activation_layout.outer, byte_alignment=1024, swizzle=activation_layout.inner)
        s_grad_scale = smem.allocate_tensor(Float32, grad_scale_layout, byte_alignment=16)
        if cutlass.const_expr(self.stream_wide_b_scales):
            s_activation_scale = mActivationScale
        elif cutlass.const_expr(self.stream_b_scale_row):
            layout = cute.make_layout((self.num_k, 1), stride=(mActivationScale.stride[1], 1))
            s_activation_scale = cute.make_tensor(mActivationScale.iterator, layout)
        elif cutlass.const_expr(self.rowwise):
            s_activation_scale = smem.allocate_tensor(Float32, activation_scale_layout, byte_alignment=128)
        else:
            scale_alignment = 16 if self.n16_u_bulk_scale else 8
            s_activation_scale = smem.allocate_tensor(
                Float32, cute.make_layout((self.num_k, self.scale_columns)), byte_alignment=scale_alignment
            )
        full = smem.allocate_array(Int64, self.stages)
        empty = smem.allocate_array(Int64, self.stages)
        b_scale_ready = None
        if cutlass.const_expr(self.n16_u_bulk_scale):
            b_scale_ready = smem.allocate_array(Int64, 1)
            if warp == self.first_producer_warp + 1:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(b_scale_ready, 1)

        descriptors = (grad, activation, grad_scale, out, activation_scale)
        self.initialize(warp, descriptors, full, empty, prefetch_warp)
        inputs = (s_grad, s_activation, s_grad_scale, s_activation_scale)
        barriers = (full, empty)
        if warp >= self.first_producer_warp:
            cute.arch.warpgroup_reg_dealloc(40)
            if warp == self.producer_warp:
                with cute.arch.elect_one():
                    if cutlass.const_expr(self.producer_route == ProducerRoute.RESIDENT_A_PANELS):
                        if bx < self.a_panel_fused_ctas:
                            self.load_resident_a_panels(bx, descriptors, inputs, barriers)
                        else:
                            tail_start = self.num_m * (self.num_n - 1) + bx - self.a_panel_fused_ctas
                            self.load_loop(bx, descriptors, inputs, barriers,
                                           tail_start, self.num_sms - self.a_panel_fused_ctas)
                    else:
                        self.load_loop(bx, descriptors, inputs, barriers, bx, self.num_sms)
        else:
            cute.arch.warpgroup_reg_alloc(self.math_registers)
            self.compute_loop(tidx, warp, bx, mma, inputs, barriers, s_out, descriptors[3], mActivationScale, first_scale_loader, b_scale_ready)

    @cute.jit
    def initialize(self, warp, descriptors, full, empty, prefetch_warp: cutlass.Constexpr):
        if warp == prefetch_warp:
            with cute.arch.elect_one():
                for descriptor in cutlass.range_constexpr(4):
                    nvvm.prefetch_tensormap(descriptors[descriptor].get_ptr())
                if cutlass.const_expr(self.rowwise or self.long_tail_n):
                    nvvm.prefetch_tensormap(descriptors[4].get_ptr())
        cute.arch.sync_warp()
        if warp == self.first_producer_warp + 1:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    consumers = 1 if self.single_n_a_cluster or self.one_n_a_local_release else self.cluster_size
                    cute.arch.mbarrier_init(empty + stage, consumers * (self.math_threads // 32))
                cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.cluster_size > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def tile_coords(self, tile_idx):
        if cutlass.const_expr(self.tile_order == TileOrder.SINGLE_TILE):
            return Int32(0), Int32(0), cutlass.Boolean(False)
        if cutlass.const_expr(self.tile_order == TileOrder.M_LINEAR):
            return tile_idx, Int32(0), cutlass.Boolean(False)
        if cutlass.const_expr(self.tile_order == TileOrder.N_LINEAR):
            return Int32(0), tile_idx, cutlass.Boolean(False)
        if cutlass.const_expr(self.tile_order == TileOrder.N144_B_WIDE):
            # Keep both M pairs within each 128-tile N group.
            pair = tile_idx // 2
            group_start = pair // 256 * 128
            group_width = cutlass.min(Int32(128), self.num_n - group_start)
            within_group = pair % 256
            second_m_pair = within_group >= group_width
            ordered_m = tile_idx % 2 + 2 * Int32(second_m_pair)
            ordered_n = group_start + select_(second_m_pair, within_group - group_width, within_group)
            return ordered_m, ordered_n, cutlass.Boolean(True)
        if cutlass.const_expr(self.tile_order == TileOrder.N32_A_STAGE4_PARTIAL_M_FIRST):
            # Give whole CTA pairs on the spill wave the partial M row in the first wave.
            pair_tiles = (self.num_m * self.num_n - self.num_sms) & ~1
            partial_start = (self.num_m - 1) * (self.num_n - 1)
            scheduled_tile = tile_idx
            if tile_idx < pair_tiles:
                scheduled_tile += partial_start
            elif partial_start <= tile_idx < partial_start + pair_tiles:
                scheduled_tile -= partial_start
            tile_n, tile_m, multicast = self.short_cluster_coords(scheduled_tile, True)
            return tile_m, tile_n, multicast
        if cutlass.const_expr(self.tile_order == TileOrder.SHORT_A_TAIL_FIRST):
            full_tiles = self.num_m * (self.num_n - 1)
            scheduled_tile = tile_idx
            # Exchange complete pairs so the CTAs that repeat only handle the narrow strip.
            if tile_idx < self.a_tail_first_tiles:
                scheduled_tile = tile_idx + full_tiles
            elif full_tiles <= tile_idx < full_tiles + self.a_tail_first_tiles:
                scheduled_tile = tile_idx - full_tiles
            tile_n, tile_m, multicast = self.short_cluster_coords(scheduled_tile, True)
            return tile_m, tile_n, multicast
        if cutlass.const_expr(self.tile_order == TileOrder.SHORT_A):
            tile_n, tile_m, multicast = self.short_cluster_coords(tile_idx, True)
            return tile_m, tile_n, multicast
        if cutlass.const_expr(self.tile_order == TileOrder.SHORT_UNICAST):
            tile_m, tile_n, _ = self.short_cluster_coords(tile_idx, False)
            return tile_m, tile_n, cutlass.Boolean(False)
        if cutlass.const_expr(self.tile_order == TileOrder.N48_MN_TAIL):
            # Keep neighboring N tiles together and fold the seven-tile tail at compile time.
            group_n = 8
            group_tiles = self.num_m * group_n
            full_tiles = self.num_n // group_n * group_tiles
            tail_n = self.num_n % group_n
            within_group = tile_idx % group_tiles
            tile_m = within_group // group_n
            tile_n = tile_idx // group_tiles * group_n + within_group % group_n
            if cutlass.const_expr(tail_n):
                if tile_idx >= full_tiles:
                    tail = tile_idx - full_tiles
                    tile_m = tail // tail_n
                    tile_n = self.num_n - tail_n + tail % tail_n
            return tile_m, tile_n, cutlass.Boolean(False)
        if cutlass.const_expr(self.tile_order == TileOrder.M_LINEAR_MULTICAST):
            return tile_idx, Int32(0), cutlass.Boolean(True)
        if cutlass.const_expr(self.tile_order == TileOrder.N_GROUPED):
            # Keep adjacent N tiles together to reuse A while retaining reuse of B.
            n_tiles_per_group = 4 if self.n48_one_wave else 2
            tiles_per_group = self.num_m * n_tiles_per_group
            first_n = tile_idx // tiles_per_group * n_tiles_per_group
            group_width = cutlass.min(n_tiles_per_group, self.num_n - first_n)
            tile_in_group = tile_idx % tiles_per_group
            return tile_in_group // group_width, first_n + tile_in_group % group_width, cutlass.Boolean(False)
        if cutlass.const_expr(self.tile_order == TileOrder.SINGLE_N80):
            return Int32(0), Int32(0), False
        if cutlass.const_expr(self.tile_order == TileOrder.SHORT_B):
            return self.short_cluster_coords(tile_idx, False)
        if cutlass.const_expr(self.tile_order == TileOrder.MN_LINEAR_MULTICAST):
            return tile_idx // self.num_n, tile_idx % self.num_n, cutlass.Boolean(True)
        if cutlass.const_expr(self.tile_order == TileOrder.N_LINEAR_MULTICAST):
            return Int32(0), tile_idx, cutlass.Boolean(True)
        if cutlass.const_expr(self.tile_order == TileOrder.SHORT_N_LINEAR_A):
            return Int32(0), tile_idx, (tile_idx ^ 1) < self.num_n
        if cutlass.const_expr(self.tile_order == TileOrder.SHORT_B_N128):
            if cutlass.const_expr(self.num_m == 1):
                return Int32(0), tile_idx, cutlass.Boolean(False)
            else:
                return tile_idx, Int32(0), (tile_idx ^ 1) < self.num_m
        primary, secondary = self.num_m, self.num_n
        if cutlass.const_expr(self.multicast_a):
            primary, secondary = self.num_n, self.num_m
        tiles_per_group = secondary * self.group_tiles
        first_tile = tile_idx // tiles_per_group * self.group_tiles
        group_width = cutlass.min(self.group_tiles, primary - first_tile)
        in_group = tile_idx % tiles_per_group
        if cutlass.const_expr(self.cluster_size > 1 and (self.num_n if self.multicast_a else self.num_m) % 2 != 0):
            if group_width % 2:
                paired_width = group_width - 1
                paired_tiles = paired_width * secondary
                if in_group < paired_tiles:
                    group_width = paired_width
                else:
                    in_group = in_group - paired_tiles
                    first_tile = first_tile + paired_width
                    group_width = 1
        primary_tile, secondary_tile = first_tile + in_group % group_width, in_group // group_width
        if cutlass.const_expr(self.multicast_a):
            return secondary_tile, primary_tile, group_width > 1
        else:
            return primary_tile, secondary_tile, group_width > 1

    @cute.jit
    def short_cluster_coords(self, tile_idx, multicast_a: cutlass.Constexpr):
        primary, secondary = self.num_m, self.num_n
        if cutlass.const_expr(multicast_a):
            primary, secondary = self.num_n, self.num_m
        block_group = self.group_tiles * secondary
        full_groups, tail_rows = primary // self.group_tiles, primary % self.group_tiles
        full_blocks = full_groups * block_group
        first_tail_row = full_groups * self.group_tiles
        tile_m, tile_n = Int32(0), Int32(0)
        multicast = cutlass.Boolean(False)
        if tile_idx < full_blocks:
            in_group = tile_idx % block_group
            tile_m = tile_idx // block_group * self.group_tiles + in_group % self.group_tiles
            tile_n = in_group // self.group_tiles
            multicast = True
        else:
            paired_rows = tail_rows - tail_rows % 2
            paired_tiles = paired_rows * secondary
            in_tail = tile_idx - full_blocks
            if cutlass.const_expr(paired_rows == 0):
                tile_m, tile_n, multicast = Int32(first_tail_row), in_tail, False
            elif in_tail < paired_tiles:
                tile_m = first_tail_row + in_tail % paired_rows
                tile_n = in_tail // paired_rows
                multicast = True
            else:
                tile_m, tile_n, multicast = Int32(first_tail_row + paired_rows), in_tail - paired_tiles, False
        return tile_m, tile_n, multicast

    @cute.jit
    def load_input(self, descriptor, dst, barrier, coord_lo, coord_hi, multicast: cutlass.Constexpr = False):
        if cutlass.const_expr(self.n80_b_multicast and self.num_sms == 132):
            # Match the reference TMA cache policy for the full-SM B-multicast route.
            load_tma_with_hint(descriptor, dst, barrier, coord_lo, coord_hi, multicast, 0x1000000000000000)
        elif cutlass.const_expr(multicast):
            nvvm.cp_async_bulk_tensor_shared_cluster_global(dst, descriptor.get_ptr(), [coord_lo, coord_hi], barrier, [], multicast_mask=Int16(3))
        else:
            nvvm.cp_async_bulk_tensor_shared_cta_global(dst, descriptor.get_ptr(), [coord_lo, coord_hi], barrier)

    @cute.jit
    def n96_resident_b_stage(self, tile_idx):
        # Each full wave alternates two B tiles. The singleton N strip has its own slot.
        paired_tiles = self.num_m * (self.num_n - 1)
        return select_(tile_idx < paired_tiles, (tile_idx // self.num_sms) & 1, Int32(2))

    @cute.jit
    def load_loop(self, bx, descriptors, inputs, barriers, tile_start, tile_step):
        grad, activation, grad_scale, _, activation_scale = descriptors
        s_grad, s_activation, s_grad_scale, s_activation_scale = inputs
        full, empty = barriers
        stage, phase = Int32(0), Int32(0)
        output_tiles = self.num_m * self.num_n
        if cutlass.const_expr(self.producer_route == ProducerRoute.RESIDENT_GRADIENT):
            output_tiles = self.num_sms
        for tile_idx in cutlass.range(tile_start, output_tiles, tile_step):
            tile_m, tile_n, use_multicast = self.tile_coords(tile_idx)
            m_offset, n_offset = tile_m * self.block_m, tile_n * self.block_n
            if cutlass.const_expr(self.producer_route == ProducerRoute.RESIDENT_GRADIENT):
                if bx == 0:
                    self.load_n32_reused(m_offset, n_offset, descriptors, inputs, barriers)
                else:
                    self.load_static_ring(m_offset, n_offset, descriptors, inputs, barriers)
            elif cutlass.const_expr(self.producer_route == ProducerRoute.STATIC_RING):
                if cutlass.const_expr(self.n96_short_n_tail):
                    if tile_n == self.num_n - 1:
                        tail_descriptors = (grad, activation_scale, grad_scale, descriptors[3], activation_scale)
                        self.load_static_ring(m_offset, n_offset, tail_descriptors, inputs, barriers, self.long_tail_n)
                    else:
                        self.load_static_ring(m_offset, n_offset, descriptors, inputs, barriers)
                else:
                    self.load_static_ring(m_offset, n_offset, descriptors, inputs, barriers)
            else:
                for k_tile in cutlass.range(self.num_k, unroll=self.num_k if self.short_n144_cluster else 1):
                    if cutlass.const_expr(self.use_linear_stage_index):
                        stage, phase = self.linear_ring_position(tile_idx, k_tile)
                    if cutlass.const_expr(not self.n64_a_single_n):
                        self.wait_stage(empty + stage, phase ^ 1)
                    barrier = full + stage
                    k_offset = k_tile * BLOCK_K
                    a = s_grad.iterator + stage * (self.block_m * BLOCK_K)
                    b_stage = stage
                    if cutlass.const_expr(self.n96_a_resident_b):
                        b_stage = self.n96_resident_b_stage(tile_idx)
                    b = s_activation.iterator + b_stage * (self.block_n * BLOCK_K)
                    scale_a = s_grad_scale.iterator + stage * align_up(self.block_m, 32)
                    if cutlass.const_expr(self.reuse_short_a128):
                        if tile_idx < 2 * self.num_sms and bx % 2 == 0:
                            self.load_input(grad, a, barrier, k_offset, m_offset, True)
                            self.load_input(grad_scale, scale_a, barrier, m_offset, k_tile, True)
                    elif cutlass.const_expr(self.multicast_a):
                        if use_multicast:
                            if bx % 2 == 0:
                                self.load_input(grad, a, barrier, k_offset, m_offset, True)
                                self.load_input(grad_scale, scale_a, barrier, m_offset, k_tile, True)
                        else:
                            self.load_input(grad, a, barrier, k_offset, m_offset)
                            self.load_input(grad_scale, scale_a, barrier, m_offset, k_tile)
                    else:
                        self.load_input(grad, a, barrier, k_offset, m_offset)
                        self.load_input(grad_scale, scale_a, barrier, m_offset, k_tile)
                    if cutlass.const_expr(self.n96_a_resident_b):
                        paired_tiles = self.num_m * (self.num_n - 1)
                        load_b = (tile_idx < 2 * self.num_sms or
                                  (tile_idx >= paired_tiles and tile_idx < paired_tiles + self.num_sms))
                        if load_b:
                            self.load_input(activation, b, barrier, k_offset, n_offset)
                    elif cutlass.const_expr(self.long_tail_n):
                        # The final N tile uses the narrower B tensor map.
                        if tile_n == self.num_n - 1:
                            self.load_input(activation_scale, b, barrier, k_offset, n_offset)
                        else:
                            self.load_input(activation, b, barrier, k_offset, n_offset)
                    elif cutlass.const_expr(self.m128_n16_a_stage11):
                        # The first four N tiles retain their position until the odd tail.
                        if tile_idx < self.num_sms or tile_idx >= 4 * self.num_m:
                            self.load_input(activation, b, barrier, k_offset, n_offset)
                    elif cutlass.const_expr(self.single_n_a_cluster):
                        # B stays in the stages allocated to the first M tile.
                        if tile_idx < self.num_sms:
                            self.load_input(activation, b, barrier, k_offset, n_offset)
                    elif cutlass.const_expr(self.reuse_full_b_n144):
                        # Every wave consumes the B tile first loaded into stage zero.
                        if tile_idx < self.num_sms and bx % 2 == 0:
                            self.load_input(activation, b, barrier, k_offset, n_offset, True)
                    elif cutlass.const_expr(self.cluster_m == 2):
                        if use_multicast:
                            if bx % 2 == 0:
                                self.load_input(activation, b, barrier, k_offset, n_offset, True)
                        else:
                            self.load_input(activation, b, barrier, k_offset, n_offset)
                    else:
                        self.load_input(activation, b, barrier, k_offset, n_offset)
                    if cutlass.const_expr(self.rowwise):
                        scale_b = s_activation_scale.iterator + stage * align_up(self.block_n, 32)
                        self.load_input(activation_scale, scale_b, barrier, n_offset, k_tile)
                    transaction_bytes = Int32(self.transaction_bytes)
                    if cutlass.const_expr(self.n96_a_resident_b):
                        transaction_bytes = select_(load_b, transaction_bytes,
                                                    transaction_bytes - self.block_n * BLOCK_K)
                    if cutlass.const_expr(self.m128_n16_a_stage11):
                        retain_b = tile_idx >= self.num_sms and tile_idx < 4 * self.num_m
                        transaction_bytes = select_(retain_b, transaction_bytes - self.block_n * BLOCK_K, transaction_bytes)
                    if cutlass.const_expr(self.reuse_full_b_n144 or self.single_n_a_cluster):
                        transaction_bytes = select_(tile_idx < self.num_sms, transaction_bytes,
                                                    transaction_bytes - self.block_n * BLOCK_K)
                    if cutlass.const_expr(self.long_tail_n):
                        transaction_bytes = select_(tile_n == self.num_n - 1,
                                                    transaction_bytes - (self.block_n - self.long_tail_n) * BLOCK_K,
                                                    transaction_bytes)
                    if cutlass.const_expr(self.reuse_short_a128):
                        transaction_bytes = select_(tile_idx < 2 * self.num_sms, transaction_bytes,
                                                    Int32(self.block_n * BLOCK_K))
                    cute.arch.mbarrier_arrive_and_expect_tx(barrier, transaction_bytes)
                    if cutlass.const_expr(not self.use_linear_stage_index):
                        stage, phase = self.advance_stage(stage, phase)
                if cutlass.const_expr(self.use_linear_stage_index):
                    stage, phase = self.linear_ring_position(tile_idx, self.num_k)
        if cutlass.const_expr(self.cluster_size > 1 and not (self.single_n_a_cluster or self.one_n_a_local_release)):
            live_stages = 4 if self.reuse_short_a128 else self.stages
            for _ in cutlass.range_constexpr(live_stages):
                self.wait_stage(empty + stage, phase ^ 1)
                stage, phase = self.advance_stage(stage, phase)

    @cute.jit
    def resident_a_coords(self, bx):
        pair = bx // self.cluster_n
        m_tile = pair // self.a_panel_quartets
        n_tile = 4 * (pair % self.a_panel_quartets) + bx % self.cluster_n
        return m_tile, n_tile

    @cute.jit
    def load_resident_a_panels(self, bx, descriptors, inputs, barriers):
        grad, activation, grad_scale, _, _ = descriptors
        s_grad, s_activation, s_grad_scale, _ = inputs
        full, empty = barriers
        m_tile, n_tile = self.resident_a_coords(bx)
        m_offset, n_offset = m_tile * self.block_m, n_tile * self.block_n
        for group in cutlass.range(self.num_k // self.stages, unroll=1):
            for slot in cutlass.range_constexpr(self.stages):
                k_tile = group * self.stages + slot
                self.wait_stage(empty + slot, Int32(1))
                barrier = full + slot
                if bx % self.cluster_n == 0:
                    a = s_grad.iterator + slot * self.block_m * BLOCK_K
                    sfa = s_grad_scale.iterator + slot * self.block_m
                    self.load_input(grad, a, barrier, k_tile * BLOCK_K, m_offset, True)
                    self.load_input(grad_scale, sfa, barrier, m_offset, k_tile, True)
                b = s_activation.iterator + slot * self.block_n * BLOCK_K
                self.load_input(activation, b, barrier, k_tile * BLOCK_K, n_offset)
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.transaction_bytes)
            # The second output overwrites B after the first releases each slot.
            for slot in cutlass.range_constexpr(self.stages):
                k_tile = group * self.stages + slot
                self.wait_stage(empty + slot, Int32(0))
                barrier = full + slot
                b = s_activation.iterator + slot * self.block_n * BLOCK_K
                self.load_input(activation, b, barrier, k_tile * BLOCK_K, n_offset + 2 * self.block_n)
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.block_n * BLOCK_K)
        for slot in cutlass.range_constexpr(self.stages):
            self.wait_stage(empty + slot, Int32(1))

    @cute.jit
    def load_n32_reused(self, m_offset, n_offset, descriptors, inputs, barriers):
        _, activation, _, _, _ = descriptors
        _, s_activation, _, _ = inputs
        full, empty = barriers
        other_n_offset = n_offset + self.num_sms * self.block_n
        for group in cutlass.range(self.num_k // self.stages, unroll=1):
            for stage in cutlass.range_constexpr(self.stages):
                k_tile = group * self.stages + stage
                self.load_static_stage(m_offset, n_offset, descriptors, inputs, barriers, stage, Int32(0), k_tile)
            for stage in cutlass.range_constexpr(self.stages):
                k_tile = group * self.stages + stage
                # Overwrite B after the first tile releases the stage. A and its scale stay live.
                self.wait_stage(empty + stage, Int32(0))
                barrier = full + stage
                b = s_activation.iterator + stage * (self.block_n * BLOCK_K)
                self.load_input(activation, b, barrier, k_tile * BLOCK_K, other_n_offset)
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.block_n * BLOCK_K)

    @cute.jit
    def load_static_ring(self, m_offset, n_offset, descriptors, inputs, barriers, tail_n: cutlass.Constexpr = 0):
        if cutlass.const_expr(self.double_producer_phase):
            period = 2 * self.stages
            for group in cutlass.range(self.num_k // period, unroll=1):
                for index in cutlass.range_constexpr(period):
                    slot, phase = index % self.stages, index // self.stages
                    k_tile = group * period + index
                    self.load_static_stage(m_offset, n_offset, descriptors, inputs, barriers, slot, phase, k_tile, tail_n)
            for index in cutlass.range_constexpr(self.num_k % period):
                slot, phase = index % self.stages, index // self.stages
                k_tile = self.num_k // period * period + index
                self.load_static_stage(m_offset, n_offset, descriptors, inputs, barriers, slot, phase, k_tile, tail_n)
        else:
            for group in cutlass.range(self.num_k // self.stages, unroll=1):
                for slot in cutlass.range_constexpr(self.stages):
                    self.load_static_stage(m_offset, n_offset, descriptors, inputs, barriers, slot, group & 1, group * self.stages + slot, tail_n)
            for slot in cutlass.range_constexpr(self.num_k % self.stages):
                group = self.num_k // self.stages
                self.load_static_stage(m_offset, n_offset, descriptors, inputs, barriers, slot, group & 1, group * self.stages + slot, tail_n)

    @cute.jit
    def load_static_stage(self, m_offset, n_offset, descriptors, inputs, barriers, stage, phase, k_tile, tail_n: cutlass.Constexpr = 0):
        grad, activation, grad_scale, _, _ = descriptors
        s_grad, s_activation, s_grad_scale, _ = inputs
        full, empty = barriers
        if cutlass.const_expr(self.stage4_b_prefetch):
            next_block = k_tile + 2 * self.stages
            if next_block < self.num_k:
                prefetch_activation_tile(activation, next_block * BLOCK_K, n_offset)
        if cutlass.const_expr(self.n16_stage4_two_phase):
            next_block = k_tile + 2 * self.stages
            if next_block < self.num_k:
                prefetch_activation_tile(grad, next_block * BLOCK_K, m_offset)
                prefetch_activation_tile(activation, next_block * BLOCK_K, n_offset)
        if cutlass.const_expr(self.n160_stage4_full_panel or self.n128_u_stage4_two_sm):
            next_lookahead = k_tile + 2 * self.stages
            if next_lookahead < self.num_k:
                prefetch_activation_tile(activation, next_lookahead * BLOCK_K, n_offset)
        if cutlass.const_expr(self.n16_u_stage9_one_tile or self.n16_u_stage7 or self.stream_b_scale_row):
            wait_n16_stage7(empty + stage, phase ^ 1)
        else:
            self.wait_stage(empty + stage, phase ^ 1)
        barrier = full + stage
        k_offset = k_tile * BLOCK_K
        a = s_grad.iterator + stage * (self.block_m * BLOCK_K)
        b = s_activation.iterator + stage * (self.block_n * BLOCK_K)
        scale_a = s_grad_scale.iterator + stage * align_up(self.block_m, 32)
        if cutlass.const_expr(self.n80_streamed_scale):
            load_tma_with_hint(grad, a, barrier, k_offset, m_offset, False, 0x12F0000000000000)
            load_tma_with_hint(grad_scale, scale_a, barrier, m_offset, k_tile, False, 0x1000000000000000)
            load_tma_with_hint(activation, b, barrier, k_offset, n_offset, False, 0x12F0000000000000)
        elif cutlass.const_expr(self.u128_stage4_one_wave or (self.single_n80_tile and self.num_sms > 2)):
            load_unicast_triplet(grad, grad_scale, activation, a, scale_a, b, barrier, k_offset, m_offset, n_offset, k_tile)
        elif cutlass.const_expr(self.n160_stage4_grad_offset or self.n16_u_stage14_shifted or self.n16_u_stage5_shifted):
            self.load_input(activation, b, barrier, k_offset, n_offset)
            self.load_input(grad, a, barrier, k_offset, m_offset)
            self.load_input(grad_scale, scale_a, barrier, m_offset, k_tile)
        else:
            self.load_input(grad, a, barrier, k_offset, m_offset)
            self.load_input(grad_scale, scale_a, barrier, m_offset, k_tile)
            self.load_input(activation, b, barrier, k_offset, n_offset)
        live_n = tail_n or self.block_n
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.transaction_bytes - (self.block_n - live_n) * BLOCK_K)

    @cute.jit
    def wait_stage(self, buffer, phase):
        if cutlass.const_expr(self.use_cta_local_stage_wait):
            wait_n16_stage7(buffer, phase)
            return
        if cutlass.const_expr(self.n32_single_tile or self.n80_full_sms_unicast):
            wait_barrier(buffer, phase)
        elif cutlass.const_expr(self.cluster_size == 1 and not self.rowwise):
            if not cute.arch.mbarrier_try_wait(buffer, phase):
                wait_barrier(buffer, phase)
        else:
            cute.arch.mbarrier_wait(buffer, phase)

    @cute.jit
    def linear_ring_position(self, tile_idx, k_tile):
        if cutlass.const_expr(self.n64_a_single_n):
            return tile_idx // self.num_sms * self.num_k + k_tile, Int32(0)
        if cutlass.const_expr(self.m128_n16_a_stage11):
            return tile_idx // self.num_sms + k_tile, Int32(0)
        if cutlass.const_expr(self.m32_n112_two_waves or self.m32_n128_two_waves):
            return Int32(tile_idx >= self.num_sms) + k_tile, Int32(0)
        if cutlass.const_expr(self.single_n_a_cluster):
            return k_tile, (tile_idx // self.num_sms) & 1
        if cutlass.const_expr(self.reuse_full_b_n144):
            return tile_idx // self.num_sms + k_tile, Int32(0)
        if cutlass.const_expr(self.short_a_n64):
            if cutlass.const_expr(self.num_m * self.num_n <= 2 * self.num_sms):
                return Int32(tile_idx >= self.num_sms) + k_tile, Int32(0)
            return tile_idx // self.num_sms + k_tile, Int32(0)
        if cutlass.const_expr(self.short_n144_cluster or self.short_a_n16 or self.short_a_n144
                              or self.short_a_n128 or self.short_a_n112 or self.short_b_n80):
            tile_iteration = Int32(tile_idx >= self.num_sms)
        else:
            tile_iteration = tile_idx // self.num_sms
        ordinal = tile_iteration * self.num_k + k_tile
        if cutlass.const_expr(self.short_a_n144 or self.short_a_n96_no_wrap or self.short_a_n112 or self.short_b_n80):
            return ordinal, Int32(0)
        return ordinal & (self.stages - 1), (ordinal // self.stages) & 1

    @cute.jit
    def advance_stage(self, stage, phase):
        live_stages = 4 if self.reuse_short_a128 else self.stages
        next_stage = select_(stage == live_stages - 1, Int32(0), stage + 1)
        return next_stage, phase ^ Int32(next_stage == 0)

    @cute.jit
    def load_block_scales(self, tidx, tile_n, former_groups, output_groups, mActivationScale, s_activation_scale, first_scale_loader: cutlass.Constexpr):
        if cutlass.const_expr(first_scale_loader != 32):
            # Every tile in these waves uses B scale row zero.
            if tidx >= first_scale_loader:
                if cutlass.const_expr((self.num_m, self.num_n, self.num_k) == (3, 4, 256)):
                    lane = tidx - first_scale_loader
                    for stripe in cutlass.range_constexpr(8):
                        k_tile = lane + stripe * 32
                        s_activation_scale[k_tile, 0] = mActivationScale[0, k_tile]
                else:
                    for k_tile in cutlass.range(tidx - first_scale_loader, self.num_k, self.math_threads - first_scale_loader, unroll=1):
                        s_activation_scale[k_tile, 0] = mActivationScale[0, k_tile]
            cute.arch.barrier(barrier_id=1, number_of_threads=self.math_threads)
            return
        num_groups = select_(former_groups >= output_groups, Int32(1), Int32(2))
        if cutlass.const_expr(self.uniform_scale):
            num_groups = Int32(1)
        if cutlass.const_expr(self.n80_stage7_block or self.n32_u_stage10_all_scale_lanes):
            first_scale_loader = 0
        if tidx >= first_scale_loader:
            first_scale = tile_n * self.block_n // BLOCK_K
            if cutlass.const_expr(self.use_linear_k_major_scales):
                layout = cute.make_layout((self.num_k * self.scale_columns,))
                source = cute.make_tensor(mActivationScale.iterator + first_scale * self.num_k, layout)
                destination = cute.make_tensor(s_activation_scale.iterator, layout)
                for idx in cutlass.range(tidx - first_scale_loader, self.num_k * num_groups, self.math_threads - first_scale_loader):
                    destination[idx] = source[idx]
            else:
                for idx in cutlass.range(tidx - first_scale_loader, self.num_k * num_groups, self.math_threads - first_scale_loader):
                    k_tile, scale_row = idx % self.num_k, idx // self.num_k
                    s_activation_scale[k_tile, scale_row] = mActivationScale[first_scale + scale_row, k_tile]
        cute.arch.barrier(barrier_id=1, number_of_threads=self.math_threads)

    @cute.jit
    def release_stage(self, empty, tile_idx, lane):
        if cutlass.const_expr(self.n64_a_single_n):
            return
        if cutlass.const_expr(self.cluster_size == 1 or self.single_n_a_cluster or self.one_n_a_local_release):
            if lane == 0:
                cute.arch.mbarrier_arrive(empty)
        else:
            if lane < self.cluster_size:
                target = lane
                if cutlass.const_expr(self.num_m % 2 and self.num_n % 2):
                    target = select_((tile_idx ^ 1) < self.num_m * self.num_n, lane, tile_idx % 2)
                cute.arch.mbarrier_arrive(empty, target)

    @cute.jit
    def compute_loop(self, tidx, warp, bx, mma, inputs, barriers, s_out, out, mActivationScale, first_scale_loader: cutlass.Constexpr, b_scale_ready):
        s_grad, s_activation, s_grad_scale, s_activation_scale = inputs
        mma_slice = mma.get_slice(cute.arch.make_warp_uniform(warp // 4 * 128))
        # Small M uses the first lanes of a 64-row WGMMA tile. The descriptor
        # spans 64 rows, while the stage advance follows the actual A tile.
        if cutlass.const_expr(self.block_m < 64):
            grad_layout = cute.make_layout((64, BLOCK_K, self.stages), stride=(BLOCK_K, 1, self.block_m * BLOCK_K))
            s_grad = cute.make_tensor(s_grad.iterator, grad_layout)
        if cutlass.const_expr(self.n32_single_tile or self.narrow_activation):
            activation_layout = cute.make_layout((self.math_block_n, BLOCK_K, self.stages),
                                                  stride=(BLOCK_K, 1, self.block_n * BLOCK_K))
            s_activation = cute.make_tensor(s_activation.iterator, activation_layout)
        if cutlass.const_expr(self.n112_short_k):
            a_warpgroup = s_grad.iterator + tidx // 128 * (64 * BLOCK_K)
            grad_desc = cute.arch.make_warp_uniform((a_warpgroup.toint() >> 4) & 0x3FFF)
            activation_desc = cute.arch.make_warp_uniform((s_activation.iterator.toint() >> 4) & 0x3FFF)
        elif cutlass.const_expr(self.inline_small_mma or self.n80_b_multicast):
            grad_desc = cute.arch.make_warp_uniform((s_grad.iterator.toint() >> 4) & 0x3FFF)
            activation_desc = cute.arch.make_warp_uniform((s_activation.iterator.toint() >> 4) & 0x3FFF)
        else:
            grad_desc = mma_slice.make_fragment_A(mma_slice.partition_A(s_grad))
            activation_desc = mma_slice.make_fragment_B(mma_slice.partition_B(s_activation))
        fragment_shape = (self.math_block_n // 2,)
        tile = cute.make_identity_tensor((self.math_threads // 2, self.math_block_n))
        mma_acc = cute.make_rmem_tensor(cute.shape(mma_slice.partition_C(tile)), Float32)
        partial = cute.make_tensor(mma_acc.iterator, cute.make_layout(fragment_shape))
        accumulator = cute.make_rmem_tensor(fragment_shape, Float32)
        do_store = True
        if cutlass.const_expr(self.block_m < 64):
            do_store = tidx < self.store_threads
        stage, phase = Int32(0), Int32(0)
        output_tiles = self.num_m * self.num_n
        if cutlass.const_expr(self.producer_route == ProducerRoute.RESIDENT_GRADIENT):
            output_tiles = self.num_sms
        if cutlass.const_expr(self.reuse_full_b_n144 or self.single_n_a_cluster or self.m128_n16_a_stage11):
            self.load_block_scales(tidx, Int32(0), Int32(BLOCK_K // 8), Int32(self.block_n // 8), mActivationScale,
                                   s_activation_scale, first_scale_loader)
        tile_start, tile_step = bx, self.num_sms
        if cutlass.const_expr(self.math_route == MathRoute.RESIDENT_A_PANELS):
            tile_step = Int32(self.num_sms)
            if bx < self.a_panel_fused_ctas:
                tile_m, tile_n = self.resident_a_coords(bx)
                groups = Int32(self.block_n // 8)
                self.load_block_scales(tidx, tile_n, groups, groups, mActivationScale, s_activation_scale, first_scale_loader)
                other_accumulator = cute.make_rmem_tensor(fragment_shape, Float32)
                accumulator.fill(0.0)
                other_accumulator.fill(0.0)
                compute_args = (mma, grad_desc, activation_desc, s_grad_scale, s_activation_scale,
                                mma_acc, partial, accumulator)
                other_args = (mma, grad_desc, activation_desc, s_grad_scale, s_activation_scale,
                              mma_acc, partial, other_accumulator)
                self.compute_n32_reused(tidx, bx, barriers, compute_args, other_args, groups)
                self.store_tile(tidx, accumulator, s_out, out, tile_m, tile_n)
                self.store_tile(tidx, other_accumulator, s_out, out, tile_m, tile_n + 2)
                tile_start = Int32(output_tiles)
            else:
                tile_start = self.num_m * (self.num_n - 1) + bx - self.a_panel_fused_ctas
                tile_step = Int32(self.num_sms - self.a_panel_fused_ctas)
        for tile_idx in cutlass.range(tile_start, output_tiles, tile_step):
            tile_m, tile_n, _ = self.tile_coords(tile_idx)
            former_groups = cutlass.min(self.block_n, BLOCK_K - tile_n * self.block_n % BLOCK_K) // 8
            output_groups = cutlass.min(self.N - tile_n * self.block_n, self.block_n) // 8
            tile_activation_scale = s_activation_scale
            if cutlass.const_expr(self.stream_wide_b_scales):
                first_scale = tile_n * self.block_n // BLOCK_K
                layout = cute.make_layout((self.num_k, self.scale_columns),
                                          stride=(mActivationScale.stride[1], mActivationScale.stride[0]))
                tile_activation_scale = cute.make_tensor(
                    mActivationScale.iterator + first_scale * mActivationScale.stride[0], layout)
            elif cutlass.const_expr(self.wide_n48_a_one_k):
                scale_row = tile_n * self.block_n // BLOCK_K
                scale_lo = mActivationScale[scale_row, 0]
                scale_hi = scale_lo
                if output_groups > former_groups:
                    scale_hi = mActivationScale[scale_row + 1, 0]
                tile_activation_scale = (scale_lo, scale_hi)
            elif cutlass.const_expr(self.n16_u_bulk_scale):
                bulk_values = self.num_k // 4 * 4
                if tidx == 0:
                    for tail in cutlass.range_constexpr(bulk_values, self.num_k):
                        s_activation_scale[tail, 0] = mActivationScale[0, tail]
                    cute.arch.mbarrier_arrive_and_expect_tx(b_scale_ready, bulk_values * 4)
                    load_bulk_b_scales(s_activation_scale.iterator, mActivationScale.iterator,
                                       bulk_values * 4, b_scale_ready)
                self.wait_stage(b_scale_ready, Int32(0))
            elif cutlass.const_expr(not (self.stream_b_scale_row or self.rowwise or self.reuse_full_b_n144 or self.single_n_a_cluster or self.m128_n16_a_stage11)):
                self.load_block_scales(tidx, tile_n, former_groups, output_groups, mActivationScale, s_activation_scale, first_scale_loader)
            full_tile = True
            if cutlass.const_expr(self.long_tail_n):
                full_tile = tile_n != self.num_n - 1
            if full_tile:
                accumulator.fill(0.0)
                compute_args = (mma, grad_desc, activation_desc, s_grad_scale, tile_activation_scale, mma_acc, partial, accumulator)
                if cutlass.const_expr(self.producer_route == ProducerRoute.RESIDENT_GRADIENT):
                    if bx == 0:
                        # Each output tile accumulates K blocks in order in its own registers.
                        other_accumulator = cute.make_rmem_tensor(fragment_shape, Float32)
                        other_accumulator.fill(0.0)
                        other_scales = s_activation_scale
                        if cutlass.const_expr(self.n64_reuse_grad):
                            # The second N64 tile uses the next row of B scales.
                            layout = cute.make_layout((self.num_k, 1), stride=(mActivationScale.stride[1], 0))
                            other_scales = cute.make_tensor(mActivationScale.iterator + mActivationScale.stride[0], layout)
                        other_args = (mma, grad_desc, activation_desc, s_grad_scale, other_scales, mma_acc,
                                      partial, other_accumulator)
                        self.compute_n32_reused(tidx, tile_idx, barriers, compute_args, other_args, former_groups)
                        self.store_tile(tidx, other_accumulator, s_out, out, tile_m, tile_n + self.num_sms)
                    else:
                        stage, phase = self.compute_tile(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups)
                elif cutlass.const_expr(self.n80_three_tile_mn_scales):
                    if former_groups == 6:
                        stage, phase = self.compute_tile(tidx, tile_idx, barriers, compute_args, stage, phase, 6)
                    else:
                        stage, phase = self.compute_tile(tidx, tile_idx, barriers, compute_args, stage, phase, 10)
                elif cutlass.const_expr(self.fixed_scale_split):
                    stage, phase = self.compute_tile(tidx, tile_idx, barriers, compute_args, stage, phase,
                                                     min(self.block_n, BLOCK_K) // 8)
                elif cutlass.const_expr(self.specialize_scale and not self.rowwise):
                    stage, phase = self.dispatch_scale_split(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, 0)
                else:
                    stage, phase = self.compute_tile(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups)
                if do_store:
                    self.store_tile(tidx, accumulator, s_out, out, tile_m, tile_n)
            elif cutlass.const_expr(self.long_tail_n):
                tail_stage, tail_phase = self.compute_long_tail(tidx, tile_idx, tile_m, tile_n, inputs, barriers, s_out, out)
                if cutlass.const_expr(self.long_a_n128_tail):
                    stage, phase = tail_stage, tail_phase
            if cutlass.const_expr(not self.n96_sustained):
                cute.arch.sync_warp()

    @cute.jit
    def compute_long_tail(self, tidx, tile_idx, tile_m, tile_n, inputs, barriers, s_out, out):
        s_grad, s_activation, s_grad_scale, s_activation_scale = inputs
        full, empty = barriers
        lane = tidx % 32
        row_lo = tidx // 32 * 16 + lane // 4
        row_hi = row_lo + 8
        do_store = tidx < self.store_threads
        grad_base = s_grad.iterator
        if cutlass.const_expr(self.n96_short_n_tail):
            grad_base += tidx // 128 * (64 * BLOCK_K)
        grad_desc = cute.arch.make_warp_uniform((grad_base.toint() >> 4) & 0x3FFF)
        activation_desc = cute.arch.make_warp_uniform((s_activation.iterator.toint() >> 4) & 0x3FFF)
        accumulator = cute.make_rmem_tensor((self.long_tail_n // 2,), Float32)
        partial = cute.make_rmem_tensor((self.long_tail_n // 2,), Float32)
        accumulator.fill(0.0)
        if cutlass.const_expr(self.n96_short_n_tail):
            stage, phase = Int32(0), Int32(0)
        elif cutlass.const_expr(self.long_a_n64_tail or self.long_a_n128_tail):
            preceding_blocks = tile_idx // self.num_sms * self.num_k
            stage = preceding_blocks % self.stages
            phase = (preceding_blocks // self.stages) & 1
        else:
            stage = Int32(self.num_k % self.stages)
            phase = Int32((self.num_k // self.stages) & 1)
        for k_tile in cutlass.range(self.num_k, unroll=8):
            stage = cute.arch.make_warp_uniform(stage)
            activation = s_activation_scale[k_tile, 0]
            self.wait_stage(full + stage, phase)
            grad_lo, grad_hi = unused_scale(), unused_scale()
            if do_store:
                grad_lo, grad_hi = s_grad_scale[row_lo, 0, stage], s_grad_scale[row_hi, 0, stage]
            a_low = grad_desc + stage * (self.block_m * BLOCK_K // 16)
            b_low = activation_desc + stage * (self.block_n * BLOCK_K // 16)
            output = wgmma_k_block(a_low, b_low, self.long_tail_n)
            for index in cutlass.range_constexpr(self.long_tail_n // 2):
                partial[index] = output[index]
            if cutlass.const_expr(self.long_a_n64_full_second_wave):
                # The last N strip is unicast and no later tile uses the peer's stage.
                if lane < self.cluster_size:
                    cute.arch.mbarrier_arrive(empty + stage)
            else:
                self.release_stage(empty + stage, tile_idx, lane)
            if do_store:
                lo, hi = grad_lo * activation, grad_hi * activation
                for group in cutlass.range_constexpr(self.long_tail_n // 8):
                    self.promote_pair(accumulator, partial, group, (lo, lo, hi, hi))
            stage, phase = self.advance_stage(stage, phase)
        if do_store:
            self.store_tile(tidx, accumulator, s_out, out, tile_m, tile_n, self.long_tail_n)
        return stage, phase

    @cute.jit
    def dispatch_scale_split(self, tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, split: cutlass.Constexpr):
        if former_groups == split:
            stage, phase = self.compute_tile(tidx, tile_idx, barriers, compute_args, stage, phase, split)
        else:
            if cutlass.const_expr(split + self.scale_gap // 8 <= BLOCK_K // 8):
                stage, phase = self.dispatch_scale_split(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, split + self.scale_gap // 8)
        return stage, phase

    @cute.jit
    def compute_tile(self, tidx, tile_idx, barriers, compute_args, stage, phase, former_groups):
        if cutlass.const_expr(self.math_route == MathRoute.ASYNC_PAIRS):
            self.compute_async_pairs(tidx, tile_idx, barriers, compute_args, former_groups)
            return stage, phase
        if cutlass.const_expr(self.math_route == MathRoute.LONG_A_PAIRS):
            return self.compute_n64_a_pairs(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups)
        if cutlass.const_expr(self.math_route == MathRoute.RESIDENT_ACTIVATION):
            first_stage = tile_idx // self.num_sms * self.num_k
            for k_tile in cutlass.range_constexpr(self.num_k):
                self.compute_stage(tidx, tile_idx, barriers, compute_args, first_stage + k_tile,
                                   Int32(0), former_groups, k_tile)
            return first_stage, Int32(0)
        if cutlass.const_expr(self.math_route == MathRoute.SINGLE_N_A):
            phase = (tile_idx // self.num_sms) & 1
            for slot in cutlass.range_constexpr(self.num_k):
                self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, phase, former_groups, slot)
            return Int32(0), phase
        if cutlass.const_expr(self.math_route == MathRoute.PREFIX_RING):
            return self.compute_n112_ring(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups)
        if cutlass.const_expr(self.math_route in (MathRoute.STATIC_RING, MathRoute.RESIDENT_A_PANELS)):
            self.compute_static_ring(tidx, tile_idx, barriers, compute_args, former_groups)
            return stage, phase
        mma, grad_desc, activation_desc, s_grad_scale, s_activation_scale, mma_acc, partial, accumulator = compute_args
        full, empty = barriers
        lane = tidx % 32
        row_lo = tidx // 32 * 16 + lane // 4
        row_hi = row_lo + 8
        do_store = True
        if cutlass.const_expr(self.block_m < 64):
            do_store = tidx < self.store_threads
        if cutlass.const_expr(self.n80_b_multicast):
            b_scale_lo = s_activation_scale[0, 0]
            b_scale_hi = s_activation_scale[0, 1]
        else:
            b_scale_lo, b_scale_hi = Float32(0), Float32(0)
        for k_tile in cutlass.range(self.num_k, unroll=8):
            if cutlass.const_expr(self.use_linear_stage_index):
                stage, phase = self.linear_ring_position(tile_idx, k_tile)
            if cutlass.const_expr(self.stage4_ordinal_math):
                # A single output tile enters the ring at stage and phase zero.
                stage, phase = k_tile % self.stages, (k_tile // self.stages) & 1
            elif cutlass.const_expr(not self.n128_u_stage4_two_sm):
                stage = cute.arch.make_warp_uniform(stage)
            if cutlass.const_expr(self.n80_b_multicast):
                activation_lo, activation_hi = b_scale_lo, b_scale_hi
            elif cutlass.const_expr(self.wide_n48_a_one_k):
                activation_lo, activation_hi = s_activation_scale
            elif cutlass.const_expr(not self.rowwise):
                activation_lo = s_activation_scale[k_tile, 0]
                activation_hi = activation_lo
                if cutlass.const_expr(not self.uniform_scale):
                    activation_hi = s_activation_scale[k_tile, 1]
            self.wait_stage(full + stage, phase)
            grad_lo, grad_hi = Float32(0), Float32(0)
            if cutlass.const_expr(not self.stage4_ordinal_math):
                if do_store:
                    grad_lo, grad_hi = s_grad_scale[row_lo, 0, stage], s_grad_scale[row_hi, 0, stage]

            if cutlass.const_expr(self.n80_b_multicast):
                grad_low = grad_desc + stage * (self.block_m * BLOCK_K // 16)
                activation_low = activation_desc + stage * (self.block_n * BLOCK_K // 16)
                # The tile's B scales are ready in shared memory before the K loop.
                next_k = cutlass.min(k_tile + 1, self.num_k - 1)
                next_lo = s_activation_scale.iterator + next_k
                next_hi = next_lo + self.num_k
                values = wgmma_k_block(grad_low, activation_low, self.block_n, (next_lo, next_hi))
                for index in cutlass.range_constexpr(self.block_n // 2):
                    partial[index] = values[index]
            else:
                b_stage = Int32(0) if self.reuse_full_b_n144 else stage
                if cutlass.const_expr(self.n96_a_resident_b):
                    b_stage = self.n96_resident_b_stage(tile_idx)
                if cutlass.const_expr(self.m128_n16_a_stage11):
                    b_stage = select_(tile_idx < 4 * self.num_m, Int32(0), stage)
                warpgroup.fence()
                for mma_k in cutlass.range_constexpr(BLOCK_K // 32):
                    mma.set(warpgroup.Field.ACCUMULATE, mma_k != 0)
                    cute.gemm(mma, mma_acc, grad_desc[None, None, mma_k, stage], activation_desc[None, None, mma_k, b_stage], mma_acc)
                warpgroup.commit_group()
                if cutlass.const_expr(self.wide_n48_a_one_k):
                    if tidx == 0:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                if cutlass.const_expr(self.stage4_ordinal_math):
                    # Read the current A scales while the WGMMA group is in flight.
                    grad_lo, grad_hi = s_grad_scale[row_lo, 0, stage], s_grad_scale[row_hi, 0, stage]
                warpgroup.wait_group(0)
            if cutlass.const_expr(self.rowwise):
                if do_store:
                    self.promote_rows(accumulator, partial, grad_lo, grad_hi, s_activation_scale, lane % 4 * 2, stage)
            self.release_stage(empty + stage, tile_idx, lane)
            if cutlass.const_expr(not self.rowwise):
                if do_store:
                    self.promote_blocks(accumulator, partial, grad_lo, grad_hi, activation_lo, activation_hi, former_groups)
            if cutlass.const_expr(self.n80_b_multicast):
                b_scale_lo = values[self.block_n // 2]
                b_scale_hi = values[self.block_n // 2 + 1]
            if cutlass.const_expr(not (self.use_linear_stage_index or self.stage4_ordinal_math)):
                stage, phase = self.advance_stage(stage, phase)
        if cutlass.const_expr(self.use_linear_stage_index):
            stage, phase = self.linear_ring_position(tile_idx, self.num_k)
        return stage, phase

    @cute.jit
    def compute_async_pairs(self, tidx, tile_idx, barriers, compute_args, former_groups):
        _, _, _, _, _, first_mma, _, _ = compute_args
        second_mma = cute.make_rmem_tensor(cute.shape(first_mma), Float32)
        second = cute.make_tensor(second_mma.iterator, cute.make_layout((self.math_block_n // 2,)))
        period = 2 * self.stages
        for group in cutlass.range(self.num_k // period, unroll=1):
            for index in cutlass.range_constexpr(0, period, 2):
                k_tile = group * period + index
                self.compute_async_pair(tidx, tile_idx, barriers, compute_args, second_mma, second, k_tile, index)
        for index in cutlass.range_constexpr(0, self.num_k % period // 2 * 2, 2):
            k_tile = (self.num_k // period) * period + index
            self.compute_async_pair(tidx, tile_idx, barriers, compute_args, second_mma, second, k_tile, index)
        if cutlass.const_expr(self.num_k % 2):
            k_tile = self.num_k - 1
            slot, phase = k_tile % self.stages, (k_tile // self.stages) & 1
            self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, phase, former_groups, k_tile)

    @cute.jit
    def compute_n64_a_pairs(self, tidx, tile_idx, barriers, compute_args, stage, phase, former_groups):
        _, _, _, _, _, first_mma, _, _ = compute_args
        second_mma = cute.make_rmem_tensor(cute.shape(first_mma), Float32)
        second = cute.make_tensor(second_mma.iterator, cute.make_layout((self.math_block_n // 2,)))
        prefix = select_(stage == 0, Int32(0), self.stages - stage)
        for k_tile in cutlass.range(prefix, unroll=1):
            self.compute_stage(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, k_tile)
            stage, phase = self.advance_stage(stage, phase)
        remaining = self.num_k - prefix
        groups = remaining // self.stages
        for group in cutlass.range(groups, unroll=1):
            for slot in cutlass.range_constexpr(0, self.paired_ring_slots, 2):
                k_tile = prefix + group * self.stages + slot
                self.compute_async_pair(tidx, tile_idx, barriers, compute_args, second_mma, second,
                                        k_tile, slot, phase)
            if cutlass.const_expr(self.paired_ring_has_last_stage):
                slot = self.paired_ring_slots
                k_tile = prefix + group * self.stages + slot
                self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, phase, former_groups, k_tile)
            phase ^= 1
        stage = Int32(0)
        for k_tile in cutlass.range(prefix + groups * self.stages, self.num_k, unroll=1):
            self.compute_stage(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, k_tile)
            stage, phase = self.advance_stage(stage, phase)
        return stage, phase

    @cute.jit
    def compute_async_pair(self, tidx, tile_idx, barriers, compute_args, second_mma, second, k_tile,
                           index: cutlass.Constexpr, phase_bias=0):
        mma, grad_desc, activation_desc, grad_scale, activation_scale, first_mma, first, accumulator = compute_args
        full, empty = barriers
        row = tidx // 32 * 16 + tidx % 32 // 4
        first_slot, first_phase = index % self.stages, phase_bias ^ (index // self.stages)
        second_slot, second_phase = (index + 1) % self.stages, phase_bias ^ ((index + 1) // self.stages)
        warpgroup.fence()
        first_lo, first_hi = self.issue_async_stage(mma, grad_desc, activation_desc, first_mma,
            full + first_slot, first_phase, first_slot, row, grad_scale, activation_scale[k_tile, 0])
        warpgroup.commit_group()
        second_lo, second_hi = self.issue_async_stage(mma, grad_desc, activation_desc, second_mma,
            full + second_slot, second_phase, second_slot, row, grad_scale, activation_scale[k_tile + 1, 0])
        warpgroup.commit_group()
        # Each stage can be reused only after its WGMMA group retires.
        warpgroup.wait_group(1)
        self.release_stage(empty + first_slot, tile_idx, tidx % 32)
        for column_group in cutlass.range_constexpr(self.math_block_n // 8):
            self.promote_pair(accumulator, first, column_group, (first_lo, first_lo, first_hi, first_hi))
        warpgroup.wait_group(0)
        self.release_stage(empty + second_slot, tile_idx, tidx % 32)
        for column_group in cutlass.range_constexpr(self.math_block_n // 8):
            self.promote_pair(accumulator, second, column_group, (second_lo, second_lo, second_hi, second_hi))

    @cute.jit
    def issue_async_stage(self, mma, grad_desc, activation_desc, accum, barrier, phase, slot,
                          row, grad_scale, activation_scale):
        mma = mma.with_()
        self.wait_stage(barrier, phase)
        lo = grad_scale[row, 0, slot] * activation_scale
        hi = grad_scale[row + 8, 0, slot] * activation_scale
        for mma_k in cutlass.range_constexpr(BLOCK_K // 32):
            mma.set(warpgroup.Field.ACCUMULATE, mma_k != 0)
            cute.gemm(mma, accum, grad_desc[None, None, mma_k, slot], activation_desc[None, None, mma_k, slot], accum)
        return lo, hi

    @cute.jit
    def compute_n112_ring(self, tidx, tile_idx, barriers, compute_args, stage, phase, former_groups):
        # Complete the partial ring before issuing the five literal stage offsets.
        prefix = select_(stage == 0, Int32(0), self.stages - stage)
        for k_tile in cutlass.range(prefix, unroll=1):
            self.compute_stage(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, k_tile)
            stage, phase = self.advance_stage(stage, phase)
        groups = (self.num_k - prefix) // self.stages
        for group in cutlass.range(groups, unroll=1):
            for slot in cutlass.range_constexpr(self.stages):
                k_tile = prefix + group * self.stages + slot
                self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, phase, former_groups, k_tile)
            phase ^= 1
        stage = Int32(0)
        for k_tile in cutlass.range(prefix + groups * self.stages, self.num_k, unroll=1):
            self.compute_stage(tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, k_tile)
            stage, phase = self.advance_stage(stage, phase)
        return stage, phase

    @cute.jit
    def compute_static_ring(self, tidx, tile_idx, barriers, compute_args, former_groups):
        if cutlass.const_expr(self.double_math_phase):
            period = 2 * self.stages
            for group in cutlass.range(self.num_k // period, unroll=1):
                for index in cutlass.range_constexpr(period):
                    slot, phase = index % self.stages, index // self.stages
                    k_tile = group * period + index
                    self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, phase, former_groups, k_tile)
            for index in cutlass.range_constexpr(self.num_k % period):
                slot, phase = index % self.stages, index // self.stages
                k_tile = self.num_k // period * period + index
                self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, phase, former_groups, k_tile)
        else:
            for group in cutlass.range(self.num_k // self.stages, unroll=1):
                for slot in cutlass.range_constexpr(self.stages):
                    self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, group & 1, former_groups, group * self.stages + slot)
            for slot in cutlass.range_constexpr(self.num_k % self.stages):
                group = self.num_k // self.stages
                self.compute_stage(tidx, tile_idx, barriers, compute_args, slot, group & 1, former_groups, group * self.stages + slot)

    @cute.jit
    def compute_n32_reused(self, tidx, tile_idx, barriers, compute_args, other_args, former_groups):
        for group in cutlass.range(self.num_k // self.stages, unroll=1):
            for stage in cutlass.range_constexpr(self.stages):
                k_tile = group * self.stages + stage
                self.compute_stage(tidx, tile_idx, barriers, compute_args, stage, Int32(0), former_groups, k_tile)
            for stage in cutlass.range_constexpr(self.stages):
                k_tile = group * self.stages + stage
                self.compute_stage(tidx, tile_idx, barriers, other_args, stage, Int32(1), former_groups, k_tile)

    @cute.jit
    def compute_stage(self, tidx, tile_idx, barriers, compute_args, stage, phase, former_groups, k_tile):
        mma, grad_desc, activation_desc, s_grad_scale, s_activation_scale, mma_acc, partial, accumulator = compute_args
        mma = mma.with_()
        full, empty = barriers
        lane = tidx % 32
        row_lo = tidx // 32 * 16 + lane // 4
        row_hi = row_lo + 8
        do_store = True
        if cutlass.const_expr(self.block_m < 64):
            do_store = tidx < self.store_threads
        if cutlass.const_expr(not self.rowwise):
            activation_lo = s_activation_scale[k_tile, 0]
            activation_hi = activation_lo
            if cutlass.const_expr(not self.uniform_scale):
                activation_hi = s_activation_scale[k_tile, 1]
        activation_stage = k_tile if cutlass.const_expr(self.n64_a_single_n) else stage
        if cutlass.const_expr(self.m64_n32_long_stage8):
            cute.arch.mbarrier_wait(full + stage, phase)
        elif cutlass.const_expr(self.n16_u_stage9_one_tile or self.n16_u_stage7 or self.stream_b_scale_row):
            wait_n16_stage7(full + stage, phase)
        else:
            self.wait_stage(full + stage, phase)
        grad_lo, grad_hi = Float32(0), Float32(0)
        if cutlass.const_expr((self.inline_small_mma and self.block_n == 96 and self.num_n == 1)
                              or self.m16_n128_stage8_single or self.rolling_n128_producer or self.m16_n32_stage8_full or self.m32_n32_stage8_full or self.m32_n64_stage8_full
                              or self.small_stage8_full_tile):
            grad_lo, grad_hi = unused_scale(), unused_scale()
        if do_store:
            grad_lo, grad_hi = s_grad_scale[row_lo, 0, stage], s_grad_scale[row_hi, 0, stage]

        if cutlass.const_expr(self.inline_small_mma or self.n112_short_k):
            grad_low = grad_desc + stage * (self.block_m * BLOCK_K // 16)
            activation_low = activation_desc + stage * (self.block_n * BLOCK_K // 16)
            values = wgmma_k_block(grad_low, activation_low, self.block_n)
            for index in cutlass.range_constexpr(self.block_n // 2):
                partial[index] = values[index]
        else:
            warpgroup.fence()
            for mma_k in cutlass.range_constexpr(BLOCK_K // 32):
                mma.set(warpgroup.Field.ACCUMULATE, mma_k != 0)
                cute.gemm(mma, mma_acc, grad_desc[None, None, mma_k, stage], activation_desc[None, None, mma_k, activation_stage], mma_acc)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
        if cutlass.const_expr(self.rowwise):
            if do_store:
                self.promote_rows(accumulator, partial, grad_lo, grad_hi, s_activation_scale, lane % 4 * 2, stage)
        self.release_stage(empty + stage, tile_idx, lane)
        if cutlass.const_expr(not self.rowwise):
            if do_store:
                self.promote_blocks(accumulator, partial, grad_lo, grad_hi, activation_lo, activation_hi, former_groups)

    @cute.jit
    def promote_blocks(self, accumulator, partial, grad_lo, grad_hi, activation_lo, activation_hi, former_groups):
        if cutlass.const_expr(self.scale_promotion == ScalePromotion.PAIRS_SELECT_FIRST):
            for group in cutlass.range_constexpr(0, self.math_block_n // 8, 2):
                activation = select_(group < former_groups, activation_lo, activation_hi)
                lo, hi = grad_lo * activation, grad_hi * activation
                self.promote_pair(accumulator, partial, group, (lo, lo, hi, hi))
                self.promote_pair(accumulator, partial, group + 1, (lo, lo, hi, hi))
        elif cutlass.const_expr(self.scale_promotion == ScalePromotion.GROUP_PAIRS):
            lo_a, hi_a = grad_lo * activation_lo, grad_hi * activation_lo
            lo_b, hi_b = grad_lo * activation_hi, grad_hi * activation_hi
            # These tile widths cross scale boundaries in pairs of eight-column groups.
            for group in cutlass.range_constexpr(0, self.math_block_n // 8, 2):
                lo = select_(group < former_groups, lo_a, lo_b)
                hi = select_(group < former_groups, hi_a, hi_b)
                self.promote_pair(accumulator, partial, group, (lo, lo, hi, hi))
                self.promote_pair(accumulator, partial, group + 1, (lo, lo, hi, hi))
        elif cutlass.const_expr(self.scale_promotion == ScalePromotion.SELECT_FIRST):
            for group in cutlass.range_constexpr(self.math_block_n // 8):
                activation = select_(group < former_groups, activation_lo, activation_hi)
                lo, hi = grad_lo * activation, grad_hi * activation
                self.promote_pair(accumulator, partial, group, (lo, lo, hi, hi))
        else:
            scale_lo, scale_hi = grad_lo * activation_lo, grad_hi * activation_lo
            scale_next_lo, scale_next_hi = grad_lo * activation_hi, grad_hi * activation_hi
            for group in cutlass.range_constexpr(self.math_block_n // 8):
                lo, hi = scale_lo, scale_hi
                if cutlass.const_expr((self.specialize_scale or self.n80_three_tile_mn_scales) and not self.uniform_scale):
                    if cutlass.const_expr(group >= former_groups):
                        lo, hi = scale_next_lo, scale_next_hi
                elif cutlass.const_expr(not self.uniform_scale):
                    lo = select_(group < former_groups, scale_lo, scale_next_lo)
                    hi = select_(group < former_groups, scale_hi, scale_next_hi)
                self.promote_pair(accumulator, partial, group, (lo, lo, hi, hi))

    @cute.jit
    def promote_rows(self, accumulator, partial, grad_lo, grad_hi, s_activation_scale, col_pair, stage):
        for group in cutlass.range_constexpr(self.block_n // 8):
            col = group * 8 + col_pair
            activation_lo, activation_hi = s_activation_scale[col, 0, stage], s_activation_scale[col + 1, 0, stage]
            self.promote_pair(accumulator, partial, group, (grad_lo * activation_lo, grad_lo * activation_hi, grad_hi * activation_lo, grad_hi * activation_hi))

    @cute.jit
    def promote_pair(self, accumulator, partial, group, scales):
        for entry in cutlass.range_constexpr(4):
            idx = group * 4 + entry
            accumulator[idx] = Float32(math.fma(partial[idx].ir_value(), scales[entry].ir_value(), accumulator[idx].ir_value()))

    @cute.jit
    def store_tile(self, tidx, accumulator, s_out, out, tile_m, tile_n, math_n: cutlass.Constexpr = 0):
        if cutlass.const_expr(self.m64_n128_b_short or self.m64_n160_b_one_k
                             or self.short_a_n64 or self.n64_a_swizzled_output):
            # Each issuing lane must finish reading before the CTA reuses the shared tile.
            if tidx < self.block_n // 32:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(barrier_id=2, number_of_threads=self.store_threads)
        elif cutlass.const_expr(not (self.swizzle_fp32_output or self.n96_sustained or self.n16_u_short)):
            if cutlass.const_expr(self.n160_sustained or self.n192_b_sustained):
                # Each issuing lane waits before any warp reuses the shared tile.
                if cute.arch.warp_idx() < self.block_n // 32 and tidx % 32 == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            elif cutlass.const_expr(not self.wide_n48_a_one_k):
                if tidx == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(barrier_id=2, number_of_threads=self.store_threads)
        row_lo = tidx // 32 * 16 + tidx % 32 // 4
        col_pair = tidx % 4 * 2
        for group in cutlass.range_constexpr((math_n or self.math_block_n) // 8):
            if cutlass.const_expr(self.swizzle_tma_output):
                # The two rows share a swizzle phase because they are eight rows apart.
                strip = group // 4 * (self.block_m * 32)
                col = col_pair + group % 4 * 8
                col = ((col // 4) ^ (row_lo % 8)) * 4 + col % 4
                tile = cute.make_tensor(s_out.iterator + strip + row_lo * 32 + col,
                                        cute.make_layout((2, 2), stride=(1, 8 * 32)))
            else:
                col = col_pair + group * 8
                tile = cute.make_tensor(s_out.iterator + row_lo * self.block_n + col,
                                        cute.make_layout((2, 2), stride=(1, 8 * self.block_n)))
            values = cute.make_tensor(accumulator.iterator + group * 4, cute.make_layout((2, 2)))
            values.store(values.load() + Float32(0))
            cute.autovec_copy(values, tile)
        if cutlass.const_expr(self.clear_output_padding):
            for group in cutlass.range_constexpr(self.math_block_n // 8, self.block_n // 8):
                col = col_pair + group * 8
                tile = cute.make_tensor(s_out.iterator + row_lo * self.block_n + col,
                                        cute.make_layout((2, 2), stride=(1, 8 * self.block_n)))
                tile.fill(Float32(0))
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=2, number_of_threads=self.store_threads)
        if cutlass.const_expr(self.swizzle_fp32_output):
            if tidx < self.block_n // 32:
                source = s_out.iterator + tidx * self.block_m * 32
                col = tile_n * self.block_n + tidx * 32
                nvvm.cp_async_bulk_tensor_global_shared_cta(out.get_ptr(), source, [col, tile_m * self.block_m])
                cute.arch.cp_async_bulk_commit_group()
        elif cutlass.const_expr(self.sustained_swizzled_output):
            strip = cute.arch.warp_idx()
            if strip < self.block_n // 32 and tidx % 32 == 0:
                source = s_out.iterator + strip * self.block_m * 32
                col = tile_n * self.block_n + strip * 32
                nvvm.cp_async_bulk_tensor_global_shared_cta(out.get_ptr(), source, [col, tile_m * self.block_m])
                cute.arch.cp_async_bulk_commit_group()
        elif tidx == 0:
            nvvm.cp_async_bulk_tensor_global_shared_cta(out.get_ptr(), s_out.iterator, [tile_n * self.block_n, tile_m * self.block_m])
            cute.arch.cp_async_bulk_commit_group()


def fp8_wgrad_sm90(grad, grad_scale, activation, activation_scale, out, *, num_sms=None, block_n_multiple=1):
    """
    Write the fresh FP32 Hopper weight gradient with one scale product per K block.
    """
    M, K = grad.shape
    N = activation.shape[0]
    if activation.shape != (N, K) or out.shape != (M, N):
        raise ValueError("expected matching (M, K), (N, K) and (M, N) shapes")
    if grad.dtype != torch.float8_e4m3fn or activation.dtype != grad.dtype or out.dtype != torch.float32:
        raise TypeError("expected FP8 E4M3 operands and FP32 output")
    if not out.is_cuda or any(tensor.device != out.device for tensor in (grad, activation)):
        raise ValueError("expected operands and output on the same CUDA device")
    if any(tensor.stride(1) != 1 for tensor in (grad, activation, out)):
        raise ValueError("expected unit inner strides")
    if not M or not N:
        return
    if not K:
        out.zero_()
        return

    if N % 8:
        raise ValueError("positive work requires N aligned to 8")
    if any((tensor.stride(0) * tensor.element_size()) % 16 or tensor.data_ptr() % 16 for tensor in (grad, activation, out)):
        raise ValueError("expected 16-byte TMA alignment")
    num_k = divide_up(K, BLOCK_K)
    rowwise = activation_scale.shape == (N, num_k)
    if grad_scale.shape != (M, num_k) or not (rowwise or activation_scale.shape == (divide_up(N, BLOCK_K), num_k)):
        raise ValueError("expected one scale per input row or scale tile for every K block")
    if any(scale.dtype != torch.float32 for scale in (grad_scale, activation_scale)):
        raise TypeError("expected FP32 scales")
    if any(scale.device != out.device for scale in (grad_scale, activation_scale)):
        raise ValueError("expected scales on the operand CUDA device")

    grad_scale = to_mn_major(grad_scale)
    if rowwise:
        activation_scale = to_mn_major(activation_scale)
    elif not (activation_scale.is_contiguous() or activation_scale.T.is_contiguous()):
        activation_scale = activation_scale.contiguous()
    if not rowwise and M <= 64 and N <= 80 and block_n_multiple == 80 and activation_scale.shape[0] == 1:
        activation_scale = activation_scale.as_strided(activation_scale.shape, (num_k, activation_scale.stride(1)))
    device_sms = torch.cuda.get_device_properties(out.device).multi_processor_count
    num_sms = device_sms if num_sms is None else num_sms
    full_device_sm_budget = num_sms == device_sms
    if not 2 <= num_sms <= device_sms or num_sms % 2:
        raise ValueError("expected an even SM count from two through the device SM count")
    if not 1 <= block_n_multiple <= 192:
        raise ValueError("expected a positive block-N multiple at most 192")
    # The widened streamed-scale route reads consecutive K scales in each row.
    n128_stage4_wide = (
        not rowwise and (M, K) in ((16, 4792320), (32, 4268032))
        and (N, num_sms, block_n_multiple) == (1152, 2, 1)
        and (grad.stride(0), activation.stride(0), out.stride(0)) == (K, K, N)
    )
    if n128_stage4_wide:
        activation_scale = activation_scale.contiguous()
    tensors = (grad, grad_scale, activation, activation_scale, out)
    long_a_storage_ok = False
    # Eleven 64x64 stages fit through 8192 K blocks. When full-width tiles
    # extend into the second wave, qualify only partial K blocks and FP8 rows
    # whose pitches both miss the 128-byte TMA swizzle alignment.
    if not rowwise and num_k > 8192 and 0 < N % 64 <= 32:
        m_tiles, n_tiles = divide_up(M, 64), divide_up(N, 64)
        narrow_second_wave = m_tiles * (n_tiles - 1) <= num_sms
        partial_k_rows = K % BLOCK_K and all(
            tensor.stride(0) * tensor.element_size() % BLOCK_K for tensor in (grad, activation)
        )
        long_a_storage_ok = (
            (narrow_second_wave or partial_k_rows)
            and all(tensor.data_ptr() % 32 == 0 for tensor in tensors)
        )
    # Offset views can share the shape and strides of a separately compiled aligned launch.
    aligned_two_sm_operands = (
        num_sms == 2 and M <= 64 and N == 96 and not rowwise
        and all(tensor.data_ptr() % 32 == 0 for tensor in tensors)
        and all(tensor.stride(0) * tensor.element_size() % 32 == 0 for tensor in (grad, activation))
    )
    # The 64x64 block-scaled tile fits eleven stages with 4,033 to 8,192 K blocks.
    aligned_stage11_tail = (
        not rowwise and 4032 < num_k <= 8192 and 0 < N % 64 <= 16
        and all(tensor.data_ptr() % 32 == 0 for tensor in tensors)
    )
    # Offset views can share the strides of an aligned launch and need a separate cached kernel.
    aligned_n160_operands = (
        not rowwise and M <= 64 and N == 160 and block_n_multiple == 160
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    # Pointer shifts can change this route with identical shapes and strides.
    n80_streamed_scale_storage = (
        not rowwise and M <= 64 and N == 80
        and all(tensor.data_ptr() % 32 == 0 and tensor.stride(0) * tensor.element_size() % 32 == 16
                for tensor in (grad, activation))
    )
    aligned_a128_tail = (
        not rowwise and block_n_multiple == 128 and 0 < N % 128 <= 16
        and all(tensor.data_ptr() % 32 == 0 for tensor in tensors)
        and all(tensor.stride(0) * tensor.element_size() % BLOCK_K for tensor in (grad, activation))
        and out.stride(0) * out.element_size() % 32 == 0
    )
    aligned_grid_n16_operands = (
        not rowwise and block_n_multiple == 1 and M > 64 and N > 16 and 1024 <= num_k <= 8192
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    # Identical strides can describe an aligned allocation or an offset FP8 view.
    n16_stage10_pair_storage = (
        not rowwise and M <= 64 and N == 16
        and all(tensor.data_ptr() % BLOCK_K == 0 and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    n16_stage10_bulk_storage = (
        not rowwise and M <= 64 and N == 16 and activation_scale.stride(1) == 1
        and activation_scale.data_ptr() % 16 == 0
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K in (0, 16)
                for tensor in (grad, activation))
    )
    # This route requires a compact native tile with only grad shifted by 16 bytes.
    n160_grad_offset_only = (
        not rowwise and num_sms == 2 and M <= 64 and N == 160 and block_n_multiple == 160
        and K % BLOCK_K == 0 and out.stride(0) == N
        and grad.data_ptr() % BLOCK_K == 16 and activation.data_ptr() % BLOCK_K == 0
        and all(tensor.stride(0) * tensor.element_size() == K for tensor in (grad, activation))
    )
    aligned_m128_n32_a = (
        not rowwise and block_n_multiple == 32 and M > 128 and N > 32
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    # Separate same-stride views when either FP8 pointer crosses the alignment boundary.
    n48_a_stage9_storage = (
        not rowwise and block_n_multiple == 48
        and all(tensor.data_ptr() % 32 == 0 and tensor.stride(0) * tensor.element_size() % 32 == 16
                for tensor in (grad, activation))
    )
    aligned_n192_stage4 = (
        not rowwise and num_sms == 2 and block_n_multiple == 192 and M <= 64 and N == 192
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    n16_stage8_local_storage = (
        not rowwise and N == 8 and block_n_multiple == 16
        and all(tensor.data_ptr() % 32 == 0 and tensor.stride(0) * tensor.element_size() % 32 == 16
                for tensor in (grad, activation))
    )
    # The Stage 5 and Stage 14 N16 routes use the same FP8 address classes.
    aligned_n16_stage14_operands = False
    if not rowwise and M <= 64 and N == 16 and block_n_multiple in (1, 16):
        fp8_pitches = tuple(tensor.stride(0) * tensor.element_size() % BLOCK_K for tensor in (grad, activation))
        aligned_n16_stage14_operands = (
            all(tensor.data_ptr() % BLOCK_K == 0 for tensor in (grad, activation))
            and (fp8_pitches == (0, 0) or (M == 64 and num_sms == 2 and fp8_pitches == (16, 16)))
        )
    n112_stage5_l2_storage = (
        not rowwise and block_n_multiple == 112
        and all(tensor.data_ptr() % 32 == 0 and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    # The native 64x32 tile fits fourteen stages for 8,961 to 12,096 K blocks.
    # Gate async pairs on the actual 128-byte TMA tile bases and row pitches.
    aligned_n32_stage14_operands = (
        not rowwise and num_sms == 2 and M <= 64 and N == 32 and 8960 < num_k <= 12096
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    aligned_n48_stage8_operands = (
        not rowwise and num_sms == 2 and M == 64 and N == block_n_multiple == 48
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    aligned_n32_stage10_operands = (
        not rowwise and num_sms == 2 and M == 64 and N == block_n_multiple == 32
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    aligned_n32_stage12_operands = (
        not rowwise and num_sms == 2 and M <= 64 and 32 < N <= 64 and 15232 < num_k <= 18368
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    aligned_n96_stage5_narrow_operands = False
    if full_device_sm_budget and not rowwise and M <= 64 and 0 < N <= 32 and N % 8 == 0 and block_n_multiple == 96:
        fp8_pitches = tuple(tensor.stride(0) * tensor.element_size() % BLOCK_K for tensor in (grad, activation))
        aligned_n96_stage5_narrow_operands = (
            fp8_pitches in ((0, 0), (16, 16))
            and all(tensor.data_ptr() % BLOCK_K == 0 for tensor in (grad, activation))
        )
    # The native 128x96 tile fits four stages for 4,641 to 8,288 K blocks.
    n96_stage4 = not rowwise and 4640 < num_k <= 8288
    aligned_n96_stage4_operands = (
        n96_stage4 and num_sms == 2 and M > 256 and N == 96
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    # Offset FP8 views need a separate cached fallback.
    n96_stage4_tail_storage = (
        n96_stage4 and num_sms > 2 and M > 128 and N > 96 and N % 96 in (8, 16)
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % 256 == 16
                for tensor in (grad, activation))
    )
    # The native 32x64 tile fits four stages for 40,481 to 43,584 K blocks.
    aligned_m32_n64_stage4_operands = (
        not rowwise and num_sms == 2 and M <= 32 and N == 192 and 40480 < num_k <= 43584
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    n80_b_stage7_aligned_storage = (
        not rowwise and (M, N, K, num_sms, block_n_multiple) in (
            (1025, 1128, 995329, 132, 80),
            (1025, 1048, 995329, 124, 80),
        )
        and all(tensor.data_ptr() % 32 == 0 and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    # The native 64x80 footprint fits nine stages for 3,105 to 5,440 K blocks.
    n80_b_stage9_aligned_storage = (
        not rowwise and 3104 < num_k <= 5440 and lcm(16, block_n_multiple) == 80
        and all(tensor.data_ptr() % 32 == 0 and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    # The native 32x96 tile fits four stages for 17,169 to 19,232 K blocks.
    aligned_m32_n96_stage4_operands = (
        not rowwise and num_sms == 2 and M <= 32 and N == 96 and 17168 < num_k <= 19232
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    aligned_m16_n64_stage4_prefetch_operands = (
        not rowwise and num_sms == 2 and M == 16 and N % 64 == 0 and 128 < N <= 256
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                for tensor in (grad, activation))
    )
    aligned_stage11_spill_operands = (
        aligned_stage11_tail and block_n_multiple == 64 and num_k * 4 <= 16 * 1024
        and all(tensor.stride(0) * tensor.element_size() % BLOCK_K == 16 for tensor in (grad, activation))
    )
    # The native 64x32 footprint fits thirteen stages for 12,097 to 15,232 K blocks.
    n32_a_stage13_aligned_storage = (
        full_device_sm_budget and not rowwise and 12096 < num_k <= 15232
        and lcm(16, block_n_multiple) == 32
        and all(tensor.data_ptr() % 32 == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    n80_stage6_gate3_storage = (
        not rowwise and K == 131072 and block_n_multiple == 1
        and (M, N, num_sms) in (
            (128, 4096, 52), (128, 4096, 56),
            (256, 4096, 104), (256, 4096, 114), (256, 512, 14),
        )
        and activation_scale.stride() != (num_k, 1)
        and all(tensor.data_ptr() % BLOCK_K == 0 and tensor.stride(0) * tensor.element_size() == K
                for tensor in (grad, activation))
    )
    # The native 64x64 footprint fits nine stages for 12,353 to 16,512 K blocks.
    n64_a_stage9_aligned_storage = (
        not rowwise and block_n_multiple == 64 and 12352 < num_k <= 16512
        and all(tensor.data_ptr() % 32 == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    # The native 64x16 footprint fits six stages for 38,657 to 41,280 K blocks.
    n16_u_stage6_aligned_storage = (
        not rowwise and 32 < M <= 64 and N == 16 and 38656 < num_k <= 41280
        and (
            (num_sms, block_n_multiple) == (2, 16)
            and all(tensor.data_ptr() % 32 == 0
                    and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                    for tensor in (grad, activation))
            or (num_sms, block_n_multiple) == (132, 1)
            and all(tensor.data_ptr() % BLOCK_K == 0
                    and tensor.stride(0) * tensor.element_size() % BLOCK_K == 0
                    for tensor in (grad, activation))
        )
    )
    # The native 128x48 footprint fits seven stages for 2,913 to 5,792 K blocks.
    n48_a_stage7_aligned_storage = (
        full_device_sm_budget and not rowwise and block_n_multiple == 48 and 2912 < num_k <= 5792
        and all(tensor.data_ptr() % 32 == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    # The native 64x64 footprint fits four stages for 33,153 to 37,312 K blocks.
    n64_u_stage4_aligned_storage = (
        full_device_sm_budget and not rowwise and block_n_multiple == 64
        and 32 < M <= 64 and N == 8 and 33152 < num_k <= 37312
        and all(tensor.data_ptr() % 32 == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    n96_stage15_n88_storage = (
        not rowwise and (M, N, K, num_sms, block_n_multiple) == (4, 11488, 36415, 132, 1)
        and all(tensor.data_ptr() % BLOCK_K == 0
                and tensor.stride(0) * tensor.element_size() == align_up(K, 16)
                for tensor in (grad, activation))
    )
    # The native 64x96 footprint fits seven stages for 5,217 to 7,808 K blocks.
    n96_b_stage7_aligned_storage = (
        not rowwise and block_n_multiple == 96 and 5216 < num_k <= 7808
        and all(tensor.data_ptr() % 32 == 0
                and tensor.stride(0) * tensor.element_size() % BLOCK_K == 16
                for tensor in (grad, activation))
    )
    key = ("fp8_wgrad", M, N, K, num_sms, rowwise, block_n_multiple,
           tuple(tensor.stride() for tensor in tensors), aligned_two_sm_operands, long_a_storage_ok,
           aligned_stage11_tail, aligned_n160_operands, n80_streamed_scale_storage, aligned_a128_tail,
           n16_stage10_bulk_storage, n16_stage10_pair_storage, n160_grad_offset_only, aligned_grid_n16_operands,
           aligned_m128_n32_a, n48_a_stage9_storage, aligned_n192_stage4, n16_stage8_local_storage,
           aligned_n16_stage14_operands, full_device_sm_budget, n112_stage5_l2_storage, aligned_n32_stage14_operands,
           aligned_n48_stage8_operands, aligned_n96_stage5_narrow_operands,
           aligned_n96_stage4_operands, n96_stage4_tail_storage, aligned_n32_stage10_operands,
           aligned_m32_n64_stage4_operands, n80_b_stage7_aligned_storage,
           aligned_m32_n96_stage4_operands, aligned_m16_n64_stage4_prefetch_operands,
           n80_stage6_gate3_storage, n80_b_stage9_aligned_storage,
           aligned_stage11_spill_operands, n32_a_stage13_aligned_storage,
           n64_a_stage9_aligned_storage, n48_a_stage7_aligned_storage,
           aligned_n32_stage12_operands, n16_u_stage6_aligned_storage, n64_u_stage4_aligned_storage,
           n96_stage15_n88_storage, n96_b_stage7_aligned_storage)
    compiled = CACHE.get(key)
    if compiled is None:
        dtypes = (Float8E4M3FN, Float32, Float8E4M3FN, Float32, Float32)
        operands = []
        for idx, (dtype, tensor) in enumerate(zip(dtypes, tensors)):
            if idx == 3 and not rowwise:
                operands.append(fake(dtype, tuple(tensor.shape), tensor.stride(), 4))
            else:
                # Keep a two-dimensional TMA box when a global extent is one.
                shape = tuple(cute.sym_int32() if extent == 1 else extent for extent in tensor.shape)
                operands.append(fake(dtype, shape, tensor.stride()))
        fp8_row_alignment = (grad.stride(0) % 32, activation.stride(0) % 32)
        aligned_fp8_rows = fp8_row_alignment == (0, 0)
        misaligned_fp8_rows = fp8_row_alignment == (16, 16)
        natural_k_stride = align_up(K, 16)
        compact_output = out.stride(0) == N
        compact_rows = grad.stride(0) == activation.stride(0) == natural_k_stride and compact_output
        operation = WeightGradient(M, N, K, num_sms, rowwise, block_n_multiple,
                                   activation_scale.stride() == (num_k, 1), aligned_fp8_rows, compact_rows,
                                   compact_output, misaligned_fp8_rows, aligned_two_sm_operands,
                                   long_a_storage_ok, aligned_stage11_tail, fp8_row_alignment, aligned_n160_operands,
                                   n80_streamed_scale_storage=n80_streamed_scale_storage,
                                   aligned_a128_tail=aligned_a128_tail,
                                   n16_stage10_bulk_storage=n16_stage10_bulk_storage,
                                   n16_stage10_pair_storage=n16_stage10_pair_storage,
                                   n160_grad_offset_only=n160_grad_offset_only,
                                   aligned_grid_n16_operands=aligned_grid_n16_operands,
                                   aligned_m128_n32_a=aligned_m128_n32_a,
                                   n48_a_stage9_storage=n48_a_stage9_storage,
                                   aligned_n192_stage4=aligned_n192_stage4,
                                   n16_stage8_local_storage=n16_stage8_local_storage,
                                   aligned_n16_stage14_operands=aligned_n16_stage14_operands,
                                   full_device_sm_budget=full_device_sm_budget,
                                   n112_stage5_l2_storage=n112_stage5_l2_storage,
                                   aligned_n32_stage14_operands=aligned_n32_stage14_operands,
                                   aligned_n48_stage8_operands=aligned_n48_stage8_operands,
                                   aligned_n96_stage5_narrow_operands=aligned_n96_stage5_narrow_operands,
                                   aligned_n96_stage4_operands=aligned_n96_stage4_operands,
                                   n96_stage4_tail_storage=n96_stage4_tail_storage,
                                   aligned_n32_stage10_operands=aligned_n32_stage10_operands,
                                   aligned_m32_n64_stage4_operands=aligned_m32_n64_stage4_operands,
                                   n80_b_stage7_aligned_storage=n80_b_stage7_aligned_storage,
                                   aligned_m32_n96_stage4_operands=aligned_m32_n96_stage4_operands,
                                   aligned_m16_n64_stage4_prefetch_operands=aligned_m16_n64_stage4_prefetch_operands,
                                   n128_stage4_wide=n128_stage4_wide,
                                   n80_stage6_gate3_storage=n80_stage6_gate3_storage,
                                   n80_b_stage9_aligned_storage=n80_b_stage9_aligned_storage,
                                   aligned_stage11_spill_operands=aligned_stage11_spill_operands,
                                   n32_a_stage13_aligned_storage=n32_a_stage13_aligned_storage,
                                   n64_a_stage9_aligned_storage=n64_a_stage9_aligned_storage,
                                   n48_a_stage7_aligned_storage=n48_a_stage7_aligned_storage,
                                   aligned_n32_stage12_operands=aligned_n32_stage12_operands,
                                   n16_u_stage6_aligned_storage=n16_u_stage6_aligned_storage,
                                   n64_u_stage4_aligned_storage=n64_u_stage4_aligned_storage,
                                   n96_stage15_n88_storage=n96_stage15_n88_storage,
                                   n96_b_stage7_aligned_storage=n96_b_stage7_aligned_storage)
        compiled = cute.compile(operation, *operands, fake_stream(), options="--enable-tvm-ffi")
        CACHE[key] = compiled
    compiled(*tensors)
