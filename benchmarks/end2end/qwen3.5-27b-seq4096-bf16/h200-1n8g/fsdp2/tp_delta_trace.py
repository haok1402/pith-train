"""Trace compiled DeltaNet inputs and adjoints on the reduced fixture."""

from collections import defaultdict
from unittest.mock import patch

import torch
import torch.distributed as dist

GQA_TRACE_CASES = ("trace-gqa-compiled", "restore-gqa-k-compiled", "restore-gqa-adjoint-compiled")
DECODER_TRACE_CASES = ("trace-decoder-compiled",)
TRACE_CASES = (
    "trace-gdn-compiled",
    "restore-gdn-adjoint-compiled",
    "restore-gdn-core-compiled",
    "qk-round-silu-compiled",
    "qk-fp32-silu-compiled",
    "qk-fp32-silu-rounded-loss-compiled",
    "delta-stock-checkpoint-compiled",
    *GQA_TRACE_CASES,
    *DECODER_TRACE_CASES,
)
PROJECTIONS = ("q", "k", "v", "decay", "beta")
SEQUENCE_FIELDS = ("gdn_row", "mlp_input", "mlp_output")
TENSORS = defaultdict(dict)
MICROBATCH = 0


def begin_microbatch(index):
    global MICROBATCH
    MICROBATCH = index


def stock_shard(direction, name, rank, size):
    dim = 0 if name in SEQUENCE_FIELDS else 2
    return TENSORS["stock", direction, MICROBATCH][name].chunk(size, dim=dim)[rank]


@torch.library.custom_op("pithtrain::tp_audit_capture", mutates_args=())
def capture(x: torch.Tensor, arm: str, name: str, rank: int, size: int) -> torch.Tensor:
    TENSORS[arm, "forward", MICROBATCH][name] = x.detach().clone()
    if (
        arm == "restore-gdn-core-compiled"
        and name in PROJECTIONS
        or arm == "restore-gqa-k-compiled"
        and name == "gqa_k"
    ):
        x = stock_shard("forward", name, rank, size)
    return x.contiguous().clone()


@capture.register_fake
def fake_capture(x, arm, name, rank, size):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("pithtrain::tp_audit_capture_backward", mutates_args=())
def capture_backward(grad: torch.Tensor, arm: str, name: str, rank: int, size: int) -> torch.Tensor:
    TENSORS[arm, "backward", MICROBATCH][name] = grad.detach().clone()
    if (
        arm.startswith("restore-gdn-")
        and name == "output"
        or arm == "restore-gqa-adjoint-compiled"
        and name == "gqa_output"
    ):
        grad = stock_shard("backward", name, rank, size)
    return grad.contiguous().clone()


@capture_backward.register_fake
def fake_capture_backward(grad, arm, name, rank, size):
    return torch.empty_like(grad, memory_format=torch.contiguous_format)


def setup_capture(ctx, inputs, output):
    _, *ctx.settings = inputs


def backward_capture(ctx, grad):
    return capture_backward(grad, *ctx.settings), None, None, None, None


capture.register_autograd(backward_capture, setup_context=setup_capture)


def trace_core(fn, arm, rank=0, size=1):
    def wrapped(q, k, v, decay, beta, *args):
        traced = tuple(
            capture(x, arm, name, rank, size)
            for x, name in zip((q, k, v, decay, beta), PROJECTIONS, strict=True)
        )
        return capture(fn(*traced, *args), arm, "output", rank, size)

    return wrapped


def trace_gqa(fn, arm, rank, size):
    def wrapped(q, k, v, *args, **kwargs):
        traced = tuple(
            capture(x, arm, name, rank, size)
            for x, name in zip((q, k, v), ("gqa_q", "gqa_k", "gqa_v"), strict=True)
        )
        return capture(fn(*traced, *args, **kwargs), arm, "gqa_output", rank, size)

    return wrapped


def trace_attention(module, arm, rank=0, size=1, *, include_gqa=False):
    target = dict(gated_delta_rule=trace_core(module.gated_delta_rule, arm, rank, size))
    if include_gqa:
        target["flash_attn_func"] = trace_gqa(module.flash_attn_func, arm, rank, size)
    return patch.multiple(module, **target)


def trace_stock_decoder(model):
    layer = model.layers[0]
    attn, mlp = layer.linear_attn.block, layer.mlp.forward

    def attn_with_trace(*args):
        x = attn(*args)
        return capture(x.flatten(0, 1), "stock", "gdn_row", 0, 1).view_as(x)

    def mlp_with_trace(x):
        x = capture(x.flatten(0, 1), "stock", "mlp_input", 0, 1).view_as(x)
        output = mlp(x)
        return capture(output.flatten(0, 1), "stock", "mlp_output", 0, 1).view_as(output)

    layer.linear_attn.block = attn_with_trace
    layer.mlp.forward = mlp_with_trace


def report(arm, rank, size, relative_error):
    for mb in (0, 1):
        begin_microbatch(mb)
        for direction in ("forward", "backward"):
            for name, value in TENSORS[arm, direction, mb].items():
                expected = stock_shard(direction, name, rank, size)
                print(
                    f"core_trace case={arm} rank={dist.get_rank()} mb={mb} "
                    f"{direction} {name} tp_to_stock={relative_error(value, expected):.8f}",
                    flush=True,
                )
