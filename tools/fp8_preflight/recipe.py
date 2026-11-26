"""The dense Qwen3.5 H200 recipe and its offline launch geometry."""

from contextlib import contextmanager, nullcontext
from itertools import batched
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ARCH = "sm_90a"
SMS = 132
WORLD_SIZE = 8
SEQUENCE = 4096
MICROBATCH = 8
ROWS = SEQUENCE * MICROBATCH

# The first number is the projection's output width. Its transpose is the dgrad GEMM.
PROJECTIONS = (
    (1024, 5120),
    (5120, 6144),
    (5120, 17408),
    (6144, 5120),
    (10240, 5120),
    (12288, 5120),
    (17408, 5120),
)


@contextmanager
def offline_target():
    import sys

    import torch

    arch = ModuleType("pithtrain.kernels.arch")
    arch.ARCH_MAJOR, arch.SM_COUNT, arch.NATIVE_E8M0 = 9, SMS, False
    properties = SimpleNamespace(major=9, minor=0, multi_processor_count=SMS, name="NVIDIA H200")
    previous = sys.modules.get(arch.__name__)
    sys.modules[arch.__name__] = arch
    try:
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_properties", return_value=properties),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            yield
    finally:
        if previous is None:
            sys.modules.pop(arch.__name__)
        else:
            sys.modules[arch.__name__] = previous


def compile_cute(compiler, config, model):
    import torch
    from cutlass import cute

    from pithtrain.kernels import cross_entropy_fwd_bwd

    stream = SimpleNamespace(cuda_stream=0, wait_stream=lambda _: None)
    with (
        patch.object(cute, "compile", compiler),
        patch.object(
            torch.Tensor, "is_cuda", property(lambda tensor: tensor.device.type == "meta")
        ),
        patch.object(torch.cuda, "device", return_value=nullcontext()),
        patch.object(torch.cuda, "current_stream", return_value=stream),
        patch.object(torch.cuda, "Stream", return_value=stream),
        patch.object(torch.cuda, "stream", return_value=nullcontext()),
        torch.device("meta"),
    ):
        fp8_projections(config)
        fp8_weights()
        mixer_kernels(config)
        cross_entropy_fwd_bwd(
            torch.empty((ROWS, config.vocab_size), dtype=torch.bfloat16),
            torch.empty(ROWS, dtype=torch.int64),
            -100,
            sum_grad=True,
        )
        reduce_scatter(model)


def fp8_projections(config):
    import torch

    from pithtrain.kernels import (
        fp8_quant_act,
        fp8_quant_delta_gate,
        fp8_quant_delta_grad,
        fp8_quant_grad,
        fp8_quant_swiglu,
        fp8_silu_mul_bwd,
        fp8_wgrad,
        silu_mul_bwd,
    )

    bf16, fp8, fp32 = torch.bfloat16, torch.float8_e4m3fn, torch.float32
    widths = {width for shape in PROJECTIONS for width in shape}
    for width in sorted(widths):
        x = torch.empty((ROWS, width), dtype=bf16)
        fp8_quant_act(x)
        fp8_quant_grad(x)
    gate = torch.empty((ROWS, config.intermediate_size), dtype=bf16)
    fp8_quant_swiglu(gate, gate)
    fp8_silu_mul_bwd(gate, gate, gate, torch.full((1,), ROWS, dtype=torch.int32))
    silu_mul_bwd(gate, gate, gate)
    delta = torch.empty(
        (ROWS, config.linear_num_value_heads * config.linear_value_head_dim), dtype=bf16
    )
    norm = torch.empty(config.linear_value_head_dim, dtype=fp32)
    inverse_rms = torch.empty(ROWS * config.linear_num_value_heads, dtype=fp32)
    fp8_quant_delta_gate(delta, norm, inverse_rms, delta)
    fp8_quant_delta_grad(delta, delta, delta, norm, inverse_rms)
    for m, n in PROJECTIONS:
        fp8_wgrad(
            torch.empty((m, ROWS), dtype=fp8),
            torch.empty_strided((m, ROWS // 128), (1, m), dtype=fp32),
            torch.empty((n, ROWS), dtype=fp8),
            torch.empty((n // 128, ROWS // 128), dtype=fp32),
            torch.empty((m, n), dtype=fp32),
        )


def fp8_weights():
    import torch

    from pithtrain.kernels import (
        fp8_pack_transpose,
        fp8_scales,
        fp8_scales_multi,
        fp8_shard_amax,
        fp8_shard_amax_multi,
        fp8_shard_cast,
    )

    fp8 = torch.float8_e4m3fn
    master = torch.empty((1, 128, 128), dtype=torch.float32)
    scale = torch.empty((1, 1, 1), dtype=torch.float32)
    fp8_shard_amax_multi([master], [scale], [0], [0])
    fp8_scales_multi([scale], [scale], [scale])
    fp8_scales(scale, scale, scale)
    for n, k in PROJECTIONS:
        for rank in range(WORLD_SIZE):
            shard = torch.empty((1, n // WORLD_SIZE, k), dtype=torch.float32)
            shard_scale = torch.empty((1, n // 128, k // 128), dtype=torch.float32)
            fp8_shard_amax(shard, shard_scale, rank * n // WORLD_SIZE, 0)
            codes = torch.empty_like(shard, dtype=fp8)
            fp8_shard_cast(shard, shard_scale, codes, rank * n // WORLD_SIZE, 0)
        codes = torch.empty((1, n, k), dtype=fp8)
        fp8_pack_transpose(codes, torch.empty(2 * codes.numel(), dtype=torch.uint8))
    gathered_packs()


def gathered_packs():
    import torch

    from pithtrain.kernels import fp8_pack_gathered_transpose

    # FSDP's picker needs full 128-row tiles and permits four-byte-aligned peer strides.
    tile = 128
    for padding in (0, 4, 8):
        shards = torch.empty_strided(
            (WORLD_SIZE, tile, tile), (tile * tile + padding, tile, 1), dtype=torch.uint8
        )
        fp8_pack_gathered_transpose(shards, torch.empty(2 * shards.numel(), dtype=torch.uint8))


def mixer_kernels(config):
    import torch

    from pithtrain.kernels import (
        causal_conv1d_bwd,
        causal_conv1d_fwd,
        chunk_gdr_bwd_pinned,
        chunk_gdr_fwd_pinned,
        delta_matching_bwd,
        delta_matching_bwd_prepare,
        delta_matching_fwd,
    )

    kh, vh = config.linear_num_key_heads, config.linear_num_value_heads
    channels = 2 * kh * config.linear_key_head_dim + vh * config.linear_value_head_dim
    bf16 = torch.bfloat16
    mixed = torch.empty((MICROBATCH, SEQUENCE, channels), dtype=bf16)
    for weight_dtype in (bf16, torch.float32):
        weight = torch.empty((channels, 1, config.linear_conv_kernel_dim), dtype=weight_dtype)
        conv = causal_conv1d_fwd(mixed, weight, 0)
        for grad in (mixed, conv):
            causal_conv1d_bwd(grad, mixed, weight, 0)

    # The GDR operator folds batch into heads, then passes contiguous tensors to the picker.
    q = torch.empty((1, SEQUENCE, MICROBATCH * kh, config.linear_key_head_dim), dtype=bf16)
    v = torch.empty((1, SEQUENCE, MICROBATCH * vh, config.linear_value_head_dim), dtype=bf16)
    gate = torch.empty(v.shape[:-1], dtype=torch.float32)
    beta = torch.empty_like(gate, dtype=bf16)
    out, cumulative, a = chunk_gdr_fwd_pinned(q, q, v, gate, beta)
    chunk_gdr_bwd_pinned(q, q, v, cumulative, beta, a, out)

    def attention_operands(heads):
        d = config.head_dim
        return (
            torch.empty((MICROBATCH, SEQUENCE, heads, d), dtype=bf16),
            torch.empty((MICROBATCH, heads, SEQUENCE, d), dtype=bf16).transpose(1, 2),
        )

    scale = config.head_dim**-0.5
    queries = attention_operands(config.num_attention_heads)
    keys = attention_operands(config.num_key_value_heads)
    for q, k in zip(queries, keys, strict=True):
        _, *state = delta_matching_fwd(q, k, k, scale)
        state = tuple(state)
        for gradient in queries:
            prepared = delta_matching_bwd_prepare(gradient, state)
            delta_matching_bwd(gradient, state, prepared, scale)


def reduce_scatter(model):
    import torch
    from torch.distributed.fsdp._fully_shard._fsdp_init import (
        _get_managed_modules,
        _get_managed_states,
    )

    from pithtrain.kernels import pack_reduce_scatter

    # MixedPrecisionPolicy gives every unsharded gradient the FP32 reduction dtype.
    groups = list(batched(model.layers, 2))
    groups.extend([(model.embed_tokens,), (model.lm_head,), (model.norm,)])
    for group in groups:
        params, _ = _get_managed_states(_get_managed_modules(group))
        grads = [torch.empty_like(param, dtype=torch.float32) for param in params]
        output = torch.empty(sum(grad.numel() for grad in grads), dtype=torch.float32)
        pack_reduce_scatter(grads, output, WORLD_SIZE)
