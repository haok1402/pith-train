"""
Shared helpers for the CuTe-DSL expert-parallel dispatch kernels.

On SM90 the CuTe kernels are the only path of every entry point that has one
(``fused_dedup_prepare_dispatch``, ``expert_and_expand_idx``, the BF16 stage-3 scatter and the FP8
experts' ``precompute_group_indices``); elsewhere, and for inputs a kernel does not cover, the
entry points run the original Triton / PyTorch path, which the tests also call directly.

Like ``token_scatter``'s pinned buffers, the kernels' device scratch assumes one launching thread
and one stream at a time.
"""

import functools

import cutlass.cute as cute
import torch


@functools.cache
def use_cute() -> bool:
    """Whether the CuTe kernels run (an SM90 GPU)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def fake(dtype: type, shape: tuple, stride: tuple, align: int = 16) -> object:
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=align)


def fake_stream() -> object:
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
