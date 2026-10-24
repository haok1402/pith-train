"""
Device capability of this process: the one place in pithtrain that probes it.

SM90 (Hopper) and SM100 (Blackwell) are the supported targets. Each kernel feature with an
SM90-specific implementation has a picker ``kernels/<feature>.py`` that calls :func:`is_sm90` and
runs ``<feature>_sm90`` or ``<feature>_sm100``; nothing else in pithtrain reads the capability.
"""

import functools

import torch

ARCH_MAJOR, _ = torch.cuda.get_device_capability()
"""Compute-capability major of the current device at import: 9 on SM90, 10 on SM100."""

NATIVE_E8M0 = ARCH_MAJOR >= 10
"""
Whether the E8M0 FP8 scale has a native PTX op (``cvt.rp.satfinite.ue8m0x2.f32``, SM100+); SM90
emulates the same power-of-2 scale in software. Passed to the Triton kernels as a constexpr.
"""


@functools.cache
def is_sm90() -> bool:
    """Whether the current device is SM90, which selects the ``_sm90`` kernels. Cutlass-free."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def is_sm90_device(device_index: int) -> bool:
    """
    Whether device ``device_index`` is SM90. Uncached, and reads ``torch.cuda.get_device_capability``
    at call time, so callers keep their own per-device cache and a test can patch the probe.
    """
    return torch.cuda.get_device_capability(device_index)[0] == 9


@functools.cache
def sm_count(device: torch.device | int | None = None) -> int:
    """Streaming-multiprocessor count of ``device`` (the current device by default)."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).multi_processor_count
