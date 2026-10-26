"""
Compile-time helpers shared by the SM90 CuTe-DSL kernels: the fake tensors and stream cute.compile
traces a kernel against (launched later through TVM-FFI), and the torch-to-CuTe element types.
"""

import cutlass
import cutlass.cute as cute
import torch

__all__ = ["TORCH2CUTE", "fake", "fake_stream"]


def fake(dtype: type, shape: tuple, stride: tuple, align: int = 16) -> object:
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=align)


def fake_stream() -> object:
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


TORCH2CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16, torch.float32: cutlass.Float32, torch.float8_e4m3fn: cutlass.Float8E4M3FN, torch.uint8: cutlass.Uint8}
"""
CuTe-DSL element type of each torch dtype the SM90 kernels compile for.
"""
