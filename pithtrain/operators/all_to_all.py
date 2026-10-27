"""PyTorch implementation of all-to-all communication operator."""

import functools
from typing import List, Tuple

import torch
import torch.autograd
import torch.distributed

# Cache the compiler-disabled wrapper once at import time.
_all_to_all_single = torch.compiler.disable(torch.distributed.all_to_all_single)

# The asynchronous all-to-all goes straight to the process group's binding with prebuilt options,
# skipping torch.distributed.all_to_all_single's Python layers (exception logger, torch-function
# dispatch, group membership and tensor checks), which cost more host time than the enqueue
# itself on this host-bound step. The NCCL call, its arguments and the Work it returns are the
# same.
_ASYNC_OPTIONS = torch.distributed.AllToAllOptions()
_ASYNC_OPTIONS.asyncOp = True
_SYNC_OPTIONS = torch.distributed.AllToAllOptions()
_SYNC_OPTIONS.asyncOp = False


def _no_grad(fn):
    """
    torch.no_grad() as a decorator without its per-call context object (clone, __enter__,
    __exit__): the same grad-mode switch around the call and restore after it. Traced code keeps
    torch.no_grad().
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.compiler.is_compiling():
            with torch.no_grad():
                return fn(*args, **kwargs)
        was_grad_enabled = torch.is_grad_enabled()
        torch._C._set_grad_enabled(False)
        try:
            return fn(*args, **kwargs)
        finally:
            torch._C._set_grad_enabled(was_grad_enabled)

    return wrapper


# Pad all-to-all output allocations to multiples of this many rows.
# Reduces CUDA memory fragmentation by collapsing varying allocation sizes
# into the same caching-allocator bucket for reuse across micro-batches.
_ALLTOALL_PAD_ALIGNMENT = 1024


@_no_grad
def all_to_all_async(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: torch.distributed.ProcessGroup,
) -> Tuple[torch.Tensor, torch.distributed.Work]:
    """
    Async all-to-all of x on the current stream: (output, work). The caller waits on work before
    reading output and implements the backward itself. The split sizes and group are those of
    torch.distributed.all_to_all_single.
    """
    valid_size = sum(output_split_sizes)
    # NCCL reads x by these splits; a mismatch would read past the tensor.
    assert sum(input_split_sizes) == x.shape[0], "input splits do not cover the input rows"
    padded_size = (
        (valid_size + _ALLTOALL_PAD_ALIGNMENT - 1)
        // _ALLTOALL_PAD_ALIGNMENT
        * _ALLTOALL_PAD_ALIGNMENT
    )
    buf = x.new_empty(padded_size, *x.shape[1:])
    output = buf[:valid_size]
    if group is not None and not torch.compiler.is_compiling():
        work = group.all_to_all_single(
            output, x, output_split_sizes, input_split_sizes, _ASYNC_OPTIONS
        )
    else:
        work = _all_to_all_single(
            output, x, output_split_sizes, input_split_sizes, group, async_op=True
        )
    return output, work


@_no_grad
def all_to_all_sync(
    output: torch.Tensor,
    x: torch.Tensor,
    output_split_sizes: List[int] | None = None,
    input_split_sizes: List[int] | None = None,
    group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """
    torch.distributed.all_to_all_single(output, x, ..., async_op=False) through the lean path: the
    current stream waits for the collective, as it does there.
    """
    if group is None or torch.compiler.is_compiling():
        _all_to_all_single(output, x, output_split_sizes, input_split_sizes, group)
        return
    if input_split_sizes:
        assert sum(input_split_sizes) == x.shape[0], "input splits do not cover the input rows"
        assert sum(output_split_sizes) == output.shape[0], "output splits do not cover the output"
    work = group.all_to_all_single(
        output, x, output_split_sizes or [], input_split_sizes or [], _SYNC_OPTIONS
    )
    if work is not None:
        work.wait()


@_no_grad
def direct_all_to_all(
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """
    Async all-to-all without torch.autograd.Function overhead. The returned tensor carries its
    torch.distributed.Work handle as output.comm_work; the caller waits on it before reading the
    tensor and implements the backward itself. The arguments are those of
    torch.distributed.all_to_all_single.
    """
    output, comm_work = all_to_all_async(input, output_split_sizes, input_split_sizes, group)
    output.comm_work = comm_work
    return output
