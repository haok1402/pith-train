"""Trace compiled DeltaNet inputs and adjoints on the reduced fixture."""

from collections import defaultdict

import torch
import torch.distributed as dist

TRACE_CASES = ("trace-gdn-compiled", "restore-gdn-adjoint-compiled", "restore-gdn-core-compiled")
PROJECTIONS = ("q", "k", "v", "decay", "beta")
TENSORS = defaultdict(dict)
MICROBATCH = 0


def begin_microbatch(index):
    global MICROBATCH
    MICROBATCH = index


def stock_shard(direction, name, rank, size):
    return TENSORS["stock", direction, MICROBATCH][name].chunk(size, dim=2)[rank]


@torch.library.custom_op("pithtrain::tp_audit_capture", mutates_args=())
def capture(x: torch.Tensor, arm: str, name: str, rank: int, size: int) -> torch.Tensor:
    TENSORS[arm, "forward", MICROBATCH][name] = x.detach().clone()
    if arm == "restore-gdn-core-compiled" and name in PROJECTIONS:
        x = stock_shard("forward", name, rank, size)
    return x.contiguous().clone()


@capture.register_fake
def fake_capture(x, arm, name, rank, size):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("pithtrain::tp_audit_capture_backward", mutates_args=())
def capture_backward(grad: torch.Tensor, arm: str, name: str, rank: int, size: int) -> torch.Tensor:
    TENSORS[arm, "backward", MICROBATCH][name] = grad.detach().clone()
    if arm.startswith("restore-gdn-") and name == "output":
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
