"""
Pinned host buffers for the asynchronous device-to-host copies of the MoE ops.

Used by the token scatter (its padded group sizes), the EP dispatch (its split metadata) and the
fused scatter-quant to avoid the per-call cudaStreamSynchronize of ``.tolist()`` / ``.item()``.

These are process-global, not per-thread, which is fine because the MoE ops only ever run
on one thread at a time. DualPipeV launches forwards from a single thread and runs every
backward through run_backward(), which disables autograd multithreading (pipeline/execution.py),
and nothing recomputes the MoE forward on a separate thread. If that ever changes (two
models stepping in one process, or a plain multithreaded .backward() overlapping a forward)
the shared buffers would race and corrupt the ks / EP-splits read-back, so key the buffer
by threading.get_ident() and drop the shared copy stream/event at that point.
"""

import torch

_pinned_buffers: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}


def get_pinned_buffer(name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
    # Cache pinned-memory buffers to avoid per-call allocation.
    # Freeing a pinned tensor triggers cudaEventRecordWithFlags (~10 us)
    # in PyTorch's CachingHostAllocator.
    key = (name, dtype, numel)
    buf = _pinned_buffers.get(key)
    if buf is None:
        buf = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
        _pinned_buffers[key] = buf
    return buf
