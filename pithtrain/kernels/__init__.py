"""
PithTrain's kernel library. __all__ is its whole interface: code outside this package imports
names from pithtrain.kernels, never a submodule. Each name's docstring is its contract, verified
against PyTorch in tests/kernels/test_<feature>.py. The library picks the SM90 or SM100
implementation itself; callers that need to know whether a fused path exists use the capability
queries fp8_fused_experts_supported, bf16_experts_supported and gated_delta_rule_pinned_supported.
Importing the package never loads cutlass or QuACK.
"""

from pithtrain.kernels.bf16_expert_gemm import bf16_expert_dgated_backward, bf16_expert_dgrad, bf16_expert_down_forward_scattered, bf16_expert_gated_forward, bf16_expert_wgrad_down, bf16_expert_wgrad_gate_up
from pithtrain.kernels.bf16_expert_slots import bf16_expert_gather_rows, bf16_expert_gather_sum, bf16_expert_group_slots, expert_slot_segment_sum
from pithtrain.kernels.bf16_expert_support import bf16_experts_supported
from pithtrain.kernels.causal_conv1d import causal_conv1d_bwd, causal_conv1d_fwd, causal_conv1d_out_like
from pithtrain.kernels.clamped_swiglu import clamped_swiglu_bwd, clamped_swiglu_fwd
from pithtrain.kernels.cross_entropy import cross_entropy_fwd_inplace
from pithtrain.kernels.ep_dispatch import expert_and_expand_idx, fused_dedup_prepare_dispatch
from pithtrain.kernels.flash_attn_memoize import memoized_launch
from pithtrain.kernels.fp8_clamped_swiglu_quant import clamped_swiglu_bwd_quant, clamped_swiglu_quant_kmajor, clamped_swiglu_quant_rowwise
from pithtrain.kernels.fp8_gemm import fp8_act_weight_gemm, fp8_dgrad_wgrad, fp8_fused_experts_supported, fp8_gemm_nt, fp8_grouped_wgrad_buffer, fp8_m_grouped_gemm, k_grouped_fp8_gemm, m_grouped_group_indices, quantize_grouped_input
from pithtrain.kernels.fp8_rowwise_blockwise_transpose import fp8cast_rowwise_blockwise_transpose
from pithtrain.kernels.fp8_rowwise_transpose import fp8cast_rowwise_transpose
from pithtrain.kernels.fp8_scale_layout import act_scale_like
from pithtrain.kernels.fp8_scatter_quant import scatter_quant
from pithtrain.kernels.fp8_silu_mul_quant import silu_mul_bwd_quant, silu_mul_bwd_quant_gu, silu_mul_quant_fwd, silu_mul_quant_gu_fwd
from pithtrain.kernels.fp8_weight_quant import fp8_pack_transpose, fp8_scales, fp8_scales_multi, fp8_shard_amax, fp8_shard_amax_multi, fp8_shard_cast, fp8cast_blockwise_transpose, fp8cast_blockwise_transpose_batched
from pithtrain.kernels.gated_delta_rule_pinned_sm90 import chunk_gdr_bwd_pinned, chunk_gdr_fwd_pinned, gated_delta_rule_pinned_supported
from pithtrain.kernels.group_bias_add import group_bias_add_
from pithtrain.kernels.segment_sum import segment_sum_along_rows
from pithtrain.kernels.silu_mul import silu_mul_bwd, silu_mul_fwd
from pithtrain.kernels.token_scatter import scatter_by_expert

__all__ = []

# FP8 dense GEMMs and activation quantization
__all__.append("act_scale_like")
__all__.append("fp8_act_weight_gemm")
__all__.append("fp8_dgrad_wgrad")
__all__.append("fp8_gemm_nt")
__all__.append("fp8cast_rowwise_blockwise_transpose")
__all__.append("fp8cast_rowwise_transpose")

# FP8 grouped (expert) GEMMs
__all__.append("fp8_fused_experts_supported")
__all__.append("fp8_grouped_wgrad_buffer")
__all__.append("fp8_m_grouped_gemm")
__all__.append("k_grouped_fp8_gemm")
__all__.append("m_grouped_group_indices")
__all__.append("quantize_grouped_input")

# FP8 weight quantization and parameter all-gather
__all__.append("fp8_pack_transpose")
__all__.append("fp8_scales")
__all__.append("fp8_scales_multi")
__all__.append("fp8_shard_amax")
__all__.append("fp8_shard_amax_multi")
__all__.append("fp8_shard_cast")
__all__.append("fp8cast_blockwise_transpose")
__all__.append("fp8cast_blockwise_transpose_batched")

# Fused activation + FP8 quantization of the experts
__all__.append("clamped_swiglu_bwd_quant")
__all__.append("clamped_swiglu_quant_kmajor")
__all__.append("clamped_swiglu_quant_rowwise")
__all__.append("scatter_quant")
__all__.append("silu_mul_bwd_quant")
__all__.append("silu_mul_bwd_quant_gu")
__all__.append("silu_mul_quant_fwd")
__all__.append("silu_mul_quant_gu_fwd")

# Elementwise, loss and reduction kernels
__all__.append("causal_conv1d_bwd")
__all__.append("causal_conv1d_fwd")
__all__.append("causal_conv1d_out_like")
__all__.append("clamped_swiglu_bwd")
__all__.append("clamped_swiglu_fwd")
__all__.append("cross_entropy_fwd_inplace")
__all__.append("group_bias_add_")
__all__.append("segment_sum_along_rows")
__all__.append("silu_mul_bwd")
__all__.append("silu_mul_fwd")

# MoE routing and token movement
__all__.append("expert_and_expand_idx")
__all__.append("fused_dedup_prepare_dispatch")
__all__.append("scatter_by_expert")

# FlashAttention-4 launch memoization
__all__.append("memoized_launch")

# BF16 CuTe experts (SM90 only)
__all__.append("bf16_expert_dgated_backward")
__all__.append("bf16_expert_dgrad")
__all__.append("bf16_expert_down_forward_scattered")
__all__.append("bf16_expert_gated_forward")
__all__.append("bf16_expert_gather_rows")
__all__.append("bf16_expert_gather_sum")
__all__.append("bf16_expert_group_slots")
__all__.append("bf16_expert_wgrad_down")
__all__.append("bf16_expert_wgrad_gate_up")
__all__.append("bf16_experts_supported")
__all__.append("expert_slot_segment_sum")

# Gated DeltaNet pinned launches (SM90 only)
__all__.append("chunk_gdr_bwd_pinned")
__all__.append("chunk_gdr_fwd_pinned")
__all__.append("gated_delta_rule_pinned_supported")
