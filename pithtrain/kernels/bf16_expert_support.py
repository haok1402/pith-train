"""
Whether the BF16 routed experts can take the fused SM90 CuTe-DSL path (``operators/bf16_experts.py``
over ``kernels/bf16_expert_gemm_sm90.py`` and ``kernels/bf16_expert_slots_sm90.py``), built on
QuACK's SM90 GEMM classes.

QuACK is imported only by those modules, and only once :func:`bf16_experts_supported` said yes:
its import patches CuTe-DSL globally, and its GEMM classes are not a stable API, so the path is
taken only with the exact version the kernels were validated against. This module never imports
quack or cutlass, so the models call it on every architecture (it returns False off SM90).

BF16 training takes the fused path wherever it can run (SM90, the validated QuACK version, at most
``MAX_GROUPS`` local experts); elsewhere it keeps ``F.grouped_mm`` with the separate scatter /
gather / SwiGLU kernels. FP8 training never takes it: callers check ``training.fp8`` first.
"""

import importlib.metadata
import warnings
from functools import lru_cache

import torch

from pithtrain.kernels.arch import is_sm90_device

__all__ = ["MAX_GROUPS", "QUACK_VERSION", "bf16_experts_supported", "quack_version"]

MAX_GROUPS = 64
"""Most local experts the fused path takes (one int64 bitmask per token in its backward)."""

QUACK_VERSION = "0.6.5"
"""quack-kernels version the kernels were validated against; any other version keeps torch."""


def quack_version() -> str | None:
    try:
        return importlib.metadata.version("quack-kernels")
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=None)
def _supported_on(device_index: int) -> bool:
    """Whether this process can run the CuTe experts on the device, warning once when it cannot."""
    if not is_sm90_device(device_index):
        reason = f"needs SM90, device {device_index} is not SM90"
    elif (version := quack_version()) != QUACK_VERSION:
        reason = f"needs quack-kernels=={QUACK_VERSION}, found {version}"
    else:
        return True
    warnings.warn(f"BF16 CuTe experts disabled ({reason}); using F.grouped_mm", stacklevel=3)
    return False


def bf16_experts_supported(num_groups: int | None = None, device: torch.device | None = None) -> bool:
    """
    Whether BF16 routed experts with ``num_groups`` local experts (any count when None) take the
    fused CuTe path (the ``bf16_expert_*`` kernels) on ``device``, the current device by default:
    SM90, quack-kernels exactly ``QUACK_VERSION``, and at most ``MAX_GROUPS`` (64) local experts.
    Warns once per device when the device or the QuACK version rules it out. Never imports QuACK or
    cutlass, so it is safe to call on every architecture (False off SM90).
    """
    device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _supported_on(index) and (num_groups is None or num_groups <= MAX_GROUPS)
