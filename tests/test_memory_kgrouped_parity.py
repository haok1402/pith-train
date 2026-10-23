"""
_k_grouped_fp8_gemm_nt on shapes whose DeepGEMM SM90 k-grouped call deadlocks when made directly
(an odd tile count: ceil(M / 128) * ceil(N / block_n) odd, M the first operand's channels). The
guard in _k_grouped_call must make every one finish and equal the product of the dequantized FP8
operands. Each case runs in a subprocess under a timeout, so a regression fails instead of hanging
the test session (a hung kernel cannot be interrupted from Python). Hopper only.
"""

import os
import subprocess
import sys

import pytest

from pithtrain.operators.linear import ARCH_MAJOR

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="Hopper k-grouped path")

CASE = r"""
import sys, torch
from pithtrain.operators.deepgemm_quantize import fp8cast_rowwise_kmajor
from pithtrain.operators.grouped_linear import _k_grouped_fp8_gemm_nt
N, K = int(sys.argv[1]), int(sys.argv[2])
ks = [int(v) for v in sys.argv[3].split(",")]
torch.manual_seed(0)
M = sum(ks)
offs = torch.tensor(ks, device="cuda").cumsum(0).to(torch.int32)
dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
a = fp8cast_rowwise_kmajor(dy, offs)[2:]
b = fp8cast_rowwise_kmajor(x, offs)[2:]
c = torch.randn(len(ks), N, K, device="cuda", dtype=torch.float32)  # accumulated into
c0 = c.clone()
_k_grouped_fp8_gemm_nt(a, b, c, ks, torch.tensor(ks, device="cuda", dtype=torch.int32), c=c)
torch.cuda.synchronize()

def deq(q, n):
    data, scale = q
    out, row = [], 0
    for k in ks:
        blk = data[row * n : (row + k) * n].view(n, k).double()
        s = scale[:, row // 128 : (row + k) // 128].double().repeat_interleave(128, dim=1)
        out.append(blk * s)
        row += k
    return out

A, B = deq(a, N), deq(b, K)
ref = torch.stack([c0[g].double() + A[g] @ B[g].T for g in range(len(ks))])
err = ((c.double() - ref).norm() / ref.norm()).item()
assert err < 1e-3, err  # DeepGEMM accumulates WGMMA partials at ~fp22, ~1e-4 relative
print("OK", err)
"""

SHAPES = [
    (2880, 2880, "512,256,384,128"),  # gpt-oss: both widths 23 blocks (pad path); hung before
    (2880, 1024, "256,128,640"),  # odd M, even N: transposed product; hung before
    (5760, 2880, "384,128"),  # gpt-oss gate_up wgrad; hung before
    (1408, 2048, "1024,0,512,256"),  # DeepSeek-V2-Lite gate/up wgrad (11 blocks), empty group
    (2048, 1408, "256,384"),  # DeepSeek-V2-Lite down wgrad (already safe)
    (768, 2048, "640,128,0,256"),  # Qwen3 (already safe, unchanged path)
]


@pytest.mark.parametrize("n,k,ks", SHAPES)
def test_k_grouped_parity_guard(n: int, k: int, ks: str) -> None:
    env = dict(os.environ)
    try:
        r = subprocess.run(
            [sys.executable, "-c", CASE, str(n), str(k), ks],
            env=env, capture_output=True, text=True, timeout=240,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"k-grouped FP8 GEMM hung for (N, K) = ({n}, {k}), ks = {ks}")
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
    print(n, k, ks, r.stdout.strip())
