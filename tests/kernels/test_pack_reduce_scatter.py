import pytest
import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives

from pithtrain.kernels import pack_reduce_scatter


def test_mixed_bucket_packs_exactly() -> None:
    generator = torch.Generator(device="cuda").manual_seed(192)
    shapes = ((48,), (48,), (8192, 2048), (64, 5120))
    dtypes = (torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16)
    grads = [
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        for shape, dtype in zip(shapes, dtypes, strict=True)
    ]
    expected = torch.empty(sum(grad.numel() for grad in grads), device="cuda", dtype=torch.float32)
    actual = torch.empty_like(expected)
    if not pack_reduce_scatter(grads, actual, 8):
        pytest.skip("FP32 bucket packing requires SM90")
    _fsdp_collectives.foreach_reduce_scatter_copy_in(grads, expected, 8)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
