"""
Sliding-window attention under context parallelism against FlashAttention on the whole sequence.

Every case runs the same window (and sinks) at cp_size 1 on the full sequence as the reference and
compares the output and the q, k, v and sink gradients of this rank's tokens. The dense cases put
the window inside one neighbouring block (the halo path, block == window_left + 1 the tightest) and
across several blocks (the gathered path); the packed cases add documents shorter than the window,
empty documents and documents shorter than 2 * cp_size blocks of the window.
"""

from dataclasses import dataclass, fields
from itertools import accumulate

import pytest
import torch

from pithtrain.contexts import distributed
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func
from pithtrain.operators.sliding_window_attention import (
    sliding_window_attention_func,
    sliding_window_attention_varlen_func,
)
from tests.utilities import launch


@dataclass
class Request:
    B: int
    S: int
    window: int
    HQ: int = 64
    HK: int = 8
    D: int = 64
    sink: bool = True
    seqlens: tuple[int, ...] | None = None
    atol: float = 1e-5


@dataclass
class Result:
    out: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    dsink: torch.Tensor | None


def zigzag(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    chunks = x.chunk(2 * cp_size, dim=1)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]], dim=1).contiguous()


def zigzag_packed(x: torch.Tensor, bounds: list[int], cp_rank: int, cp_size: int) -> torch.Tensor:
    docs = [x[None, start:end] for start, end in zip(bounds[:-1], bounds[1:]) if end > start]
    return torch.cat([zigzag(doc, cp_rank, cp_size)[0] for doc in docs]).contiguous()


def relative_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected, actual = expected.double(), actual.double()
    num = (expected - actual).square().sum()
    return float(num / (expected.square() + actual.square()).sum().clamp_min(1e-12))


def record(req: Request) -> tuple[Result, Result]:
    cp_group = distributed.cp_group
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()
    scale, window_left = req.D**-0.5, req.window - 1
    packed = req.seqlens is not None
    T = sum(req.seqlens) if packed else req.S
    shape = (T,) if packed else (req.B, req.S)

    torch.manual_seed(42)
    q, grad_out = (torch.randn(*shape, req.HQ, req.D, device=device, dtype=torch.bfloat16) for _ in range(2))  # fmt: skip
    k, v = (torch.randn(*shape, req.HK, req.D, device=device, dtype=torch.bfloat16) for _ in range(2))  # fmt: skip
    sink = torch.randn(req.HQ, device=device, dtype=torch.bfloat16) if req.sink else None
    if packed:
        bounds = [0, *accumulate(req.seqlens)]
        cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device=device)
        shard = lambda x: zigzag_packed(x, bounds, cp_rank, cp_size)
    else:
        shard = lambda x: zigzag(x, cp_rank, cp_size)

    leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
    sink_ref = sink.clone().requires_grad_(True) if req.sink else None
    window = (window_left, 0)
    if packed:
        out = flash_attn_varlen_func(*leaves, cu_seqlens, T, scale, causal=True, window_size=window, learnable_sink=sink_ref)  # fmt: skip
    else:
        out = flash_attn_func(*leaves, scale, causal=True, window_size=window, learnable_sink=sink_ref)  # fmt: skip
    out.backward(grad_out)
    dsink = sink_ref.grad.float() if req.sink else None
    ref = Result(shard(out), *(shard(x.grad) for x in leaves), dsink)

    leaves = [shard(x).requires_grad_(True) for x in (q, k, v)]
    sink_imp = sink.clone().requires_grad_(True) if req.sink else None
    if packed:
        out = sliding_window_attention_varlen_func(*leaves, cu_seqlens, scale, window_left, cp_group, sink_imp)  # fmt: skip
    else:
        out = sliding_window_attention_func(*leaves, scale, window_left, cp_group, sink_imp)
    out.backward(shard(grad_out))
    dsink = None
    if req.sink:
        dsink = sink_imp.grad.float()
        torch.distributed.all_reduce(dsink, group=cp_group)
    imp = Result(out, *(x.grad for x in leaves), dsink)
    return ref, imp


def verify(req: Request) -> None:
    ref, imp = record(req)
    for f in fields(ref):
        expected, actual = getattr(ref, f.name), getattr(imp, f.name)
        if expected is None:
            continue
        error = relative_error(expected, actual)
        if not error < req.atol:
            raise AssertionError(f"{f.name} diverged: {error=:.2e} >= {req.atol=}")


REQUESTS = []
# Dense, halo path: the window crosses into the neighbouring block only.
REQUESTS.append(pytest.param(2, Request(B=1, S=2048, window=128), id="CP2-S2048-W128"))
REQUESTS.append(pytest.param(4, Request(B=2, S=4096, window=128), id="CP4-B2-S4096-W128"))
REQUESTS.append(pytest.param(4, Request(B=1, S=1024, window=128), id="CP4-S1024-W128-tight"))
REQUESTS.append(pytest.param(8, Request(B=1, S=2048, window=128), id="CP8-S2048-W128-tight"))
REQUESTS.append(pytest.param(2, Request(B=1, S=2048, window=64, HQ=8, HK=8, sink=False), id="CP2-MHA-nosink"))  # fmt: skip
# Dense, gathered path: the window spans several blocks.
REQUESTS.append(pytest.param(4, Request(B=2, S=512, window=128), id="CP4-B2-S512-W128-gather"))
REQUESTS.append(pytest.param(2, Request(B=1, S=256, window=200), id="CP2-S256-W200-gather"))
# Packed: documents shorter than the window, empty ones, and long ones.
PACKED = [
    (2, (4, 60, 0, 512, 196, 252), "CP2-6docs"),
    (4, (8, 120, 1024, 392, 504), "CP4-5docs"),
    (4, (8,) * 32, "CP4-32x8"),
    (4, (0, 4096, 0), "CP4-1doc"),
    (8, (16, 2048, 0, 528, 112), "CP8-5docs"),
]
for cp, seqlens, name in PACKED:
    REQUESTS.append(pytest.param(cp, Request(B=1, S=0, window=128, seqlens=seqlens), id=f"{name}-W128"))  # fmt: skip


@pytest.mark.parametrize("cp_size,req", REQUESTS)
def test_sliding_window_attention_vs_full(cp_size: int, req: Request) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify, req)
