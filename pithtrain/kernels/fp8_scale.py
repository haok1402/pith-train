"""
The E8M0 (power-of-2) FP8 scale every Triton quantization kernel computes, as a ``@triton.jit``
device function.

Both architectures use E8M0 scaling factors: computed natively via PTX
``cvt.rp.satfinite.ue8m0x2.f32`` on SM100 and emulated with IEEE-754 bit manipulation on SM90
(``kernels.arch.NATIVE_E8M0``). Block granularity is 128 elements. The SM90 CuTe-DSL kernels
(``kernels/*_sm90.py``) reproduce the SM90 branch bit for bit.
"""

import triton
import triton.language as tl


@triton.jit
def compute_fp8_scale(amax, NATIVE_E8M0: tl.constexpr):
    """Compute the float32 E8M0 (power-of-2) scale from per-group amax values.

    The scale is always a power of 2, so quantize and dequantize
    multiplications are exact (IEEE-754 exponent shift, no mantissa rounding).
    ``NATIVE_E8M0`` selects how it is produced: the SM100+ native PTX
    ``cvt.rp.satfinite.ue8m0x2.f32`` when True, or a software-emulated
    power-of-2 ceil (IEEE-754 bit manipulation) on SM90 when False.

    Returns (scale, reciprocal_scale), both exact powers of 2.
    """
    FP8_MAX_RCP: tl.constexpr = 1.0 / 448.0

    amax_clamped = tl.maximum(amax.to(tl.float32), 1e-4)
    scale_input = amax_clamped * FP8_MAX_RCP

    if NATIVE_E8M0:
        # SM100: native PTX cvt.rp (round-positive = ceil) to E8M0
        scale_e8m0_biased = tl.inline_asm_elementwise(
            asm="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            constraints="=h,r",
            args=[scale_input],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)

        scale_fp = (scale_e8m0_biased.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    else:
        # SM90: emulate the same E8M0 power-of-2 ceil via IEEE-754 bit
        # manipulation -- clear mantissa bits (floor to pow2), then increment
        # the exponent if the value wasn't already an exact power of 2.
        bits = scale_input.to(tl.int32, bitcast=True)
        mantissa = bits & 0x007FFFFF
        scale_fp = ((bits & 0x7F800000) + tl.where(mantissa != 0, 0x00800000, 0)).to(
            tl.float32, bitcast=True
        )

    # Common tail for both modes: clamp + exact reciprocal
    fp32_min_normal = tl.exp2(-126.0)
    scale_fp = tl.maximum(scale_fp, fp32_min_normal)

    # Exact reciprocal via exponent flip: 2^(e-127) -> 2^(127-e)
    rcp_bits = (254 << 23) - scale_fp.to(tl.int32, bitcast=True)
    rcp_scale = rcp_bits.to(tl.float32, bitcast=True)

    return scale_fp, rcp_scale
