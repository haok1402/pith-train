"""
FLA's dense chunked gated delta rule (chunk size 64) launched directly on SM90 with the launch
configs FLA's autotuner picks on H200, skipping FLA's per-launch Python wrappers.

The launch order, allocations, in-place accumulations and dtypes are those of FLA 0.5.2's
chunk_gated_delta_rule_fwd / _bwd on the dense path (no cu_seqlens, no initial or final state), so
the outputs are bit-identical while the pinned configs are FLA's picks (pinned_launch_supported).
The ops stay opaque to torch.compile: an Inductor-launched copy of a Triton kernel is not
bit-identical. Grouped value heads run on unrepeated q / k, which the kernels index per group.
"""

import fla
import torch
import triton
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64, chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg, chunk_bwd_kernel_dv_local, chunk_fwd_kernel_o
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd_kernel, recompute_w_u_fwd_kernel
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
from triton import knobs

from pithtrain.kernels.arch import is_sm90_device

__all__ = ["BT", "chunk_gdr_bwd_pinned", "chunk_gdr_fwd_pinned", "gated_delta_rule_pinned_supported", "pinned_launch_supported", "pinned_shapes"]

BT = 64  # chunk size
BC = 16  # sub-chunk of the fused kkt + triangular solve, fixed by FLA at BT = 64

# Launch configs FLA's autotuner selects on H200 at H = HV = 32, K = V = 128 (bf16 q/k/v/beta,
# fp32 gate). The BK / BV the launches below pass explicitly are FLA's own constants on Hopper.
VERSIONS = {"fla": "0.5.2", "triton": "3.7.1"}
CONFIGS = {"cumsum_fwd": dict(num_warps=1, num_stages=3), "cumsum_rev": dict(num_warps=1, num_stages=3), "kkt_solve": dict(BK=64, num_warps=1, num_stages=3), "w_u": dict(num_warps=4, num_stages=4), "fwd_h": dict(BV=32, num_warps=4, num_stages=3), "fwd_o": dict(BK=64, BV=64, num_warps=4, num_stages=3), "dv_local": dict(num_warps=2, num_stages=3), "bwd_dhu": dict(BV=32, num_warps=4, num_stages=3), "dqkwg": dict(num_warps=4, num_stages=3), "wy_bwd": dict(num_warps=4, num_stages=4)}

KERNELS = {"cumsum": chunk_local_cumsum_scalar_kernel, "kkt_solve": chunk_gated_delta_rule_fwd_kkt_solve_kernel, "w_u": recompute_w_u_fwd_kernel, "fwd_h": chunk_gated_delta_rule_fwd_kernel_h_blockdim64, "fwd_o": chunk_fwd_kernel_o, "dv_local": chunk_bwd_kernel_dv_local, "bwd_dhu": chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64, "dqkwg": chunk_bwd_kernel_dqkwg, "wy_bwd": prepare_wy_repr_bwd_kernel}

# The parameter lists the launches below are written against.
SIGNATURES = {
    "cumsum": "s o scale cu_seqlens chunk_indices T B H BT REVERSE HAS_SCALE IS_VARLEN HEAD_FIRST",
    "kkt_solve": "k g beta A cu_seqlens chunk_indices T H HV K BT BC BK USE_G IS_VARLEN",
    "w_u": "k v beta w u A g cu_seqlens chunk_indices T H HV K V BT BK BV USE_G IS_VARLEN",
    "fwd_h": "k v w v_new g gk h h0 ht cu_seqlens chunk_offsets T H HV K V BT BV USE_G USE_GK "
    "USE_INITIAL_STATE STORE_FINAL_STATE SAVE_NEW_VALUE STATE_V_FIRST IS_VARLEN",
    "fwd_o": "q k v h g g_gamma o cu_seqlens chunk_indices scale T H HV K V BT BK BV USE_G "
    "USE_G_GAMMA STATE_V_FIRST IS_VARLEN",
    "dv_local": "q k g g_gamma A do dv cu_seqlens chunk_indices scale T H HV K V BT BK BV USE_G "
    "USE_G_GAMMA USE_A IS_VARLEN",
    "bwd_dhu": "q k w g gk dht dh0 do dh dv dv2 cu_seqlens chunk_offsets scale T H HV K V BT BV "
    "USE_G USE_GK USE_INITIAL_STATE USE_FINAL_STATE_GRADIENT STATE_V_FIRST IS_VARLEN",
    "dqkwg": "q k v g g_gamma h do dh dq dk dw dv dg cu_seqlens chunk_indices scale B T H HV K V "
    "BT BK BV USE_G USE_G_GAMMA USE_DW STATE_V_FIRST IS_VARLEN",
    "wy_bwd": "k v beta g A dw du dk dv db dg cu_seqlens chunk_indices T H HV K V BT BK BV USE_G "
    "IS_VARLEN",
}


def _jit(name: str) -> triton.JITFunction:
    """The raw JITFunction under FLA's Heuristics(CachedAutotuner(...)) wrappers."""
    return KERNELS[name].fn.fn


_compiled: dict[tuple, object] = {}
_keyed: dict[str, tuple[list[int], list[int]]] = {}


def _launch(name: str, grid: tuple, **kwargs) -> None:
    """
    jit[grid](**kwargs), with the compiled kernel cached and launched straight through its C
    launcher on the device's current stream. The cache key is what Triton specializes these kernels
    on: each tensor's dtype and 16-byte alignment, the value of every other argument except T
    (do_not_specialize) and float scalars (never specialized), plus the device and launch config.
    """
    jit = _jit(name)
    args = [kwargs[n] for n in jit.arg_names]
    if name not in _keyed:
        tensors = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        values = [i for i, n in enumerate(jit.arg_names) if i not in tensors and n != "T"]
        _keyed[name] = tensors, values
    tensors, values = _keyed[name]
    # A float scalar keys on its type only (None, e.g. an absent scale, is a constexpr).
    device = args[tensors[0]].device.index
    key = (name, device, kwargs["num_warps"], kwargs["num_stages"], *(float if args[i].__class__ is float else args[i] for i in values), *((args[i].dtype, args[i].data_ptr() & 15) for i in tensors))
    kernel = _compiled.get(key)
    if kernel is None:
        _compiled[key] = jit[grid](**kwargs)
        return
    grid = (*grid, 1, 1)
    stream = torch._C._cuda_getCurrentRawStream(device)
    enter, exit_ = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    metadata = None if enter is None else kernel.launch_metadata(grid, stream, *args)
    kernel.run(grid[0], grid[1], grid[2], stream, kernel.function, kernel.packed_metadata, metadata, enter, exit_, *args)


_supported: dict[int, bool] = {}


@torch.compiler.assume_constant_result
def pinned_launch_supported(device_index: int) -> bool:
    """
    Whether the pinned configs apply on this device: the recorded FLA and Triton versions, SM90,
    and FLA kernels whose wrappers and parameter lists match the launches here. Cached per device;
    a constant to torch.compile.
    """
    if device_index not in _supported:
        _supported[device_index] = _check_supported(device_index)
    return _supported[device_index]


def _check_supported(device_index: int) -> bool:
    if fla.__version__ != VERSIONS["fla"] or triton.__version__ != VERSIONS["triton"]:
        return False
    if not is_sm90_device(device_index):
        return False
    for name, kernel in KERNELS.items():
        inner = getattr(getattr(kernel, "fn", None), "fn", None)
        if not isinstance(inner, triton.JITFunction) or inner.arg_names != SIGNATURES[name].split():
            return False
    return True


def _cumsum(s: torch.Tensor, reverse: bool) -> torch.Tensor:
    """chunk_local_cumsum (scalar, fp32 out): scaled by 1/ln 2 forward, unscaled in reverse."""
    B, T, H = s.shape
    o = torch.empty_like(s, dtype=torch.float)
    config = CONFIGS["cumsum_rev" if reverse else "cumsum_fwd"]
    _launch("cumsum", (triton.cdiv(T, BT), B * H), s=s, o=o, scale=None if reverse else RCP_LN2, cu_seqlens=None, chunk_indices=None, T=T, B=B, H=H, BT=BT, REVERSE=reverse, HAS_SCALE=not reverse, IS_VARLEN=False, HEAD_FIRST=False, **config)
    return o


def _w_u(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """recompute_w_u_fwd."""
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    w, u = k.new_empty(B, T, HV, K), torch.empty_like(v)
    _launch("w_u", (triton.cdiv(T, BT), B * HV), k=k, v=v, beta=beta, w=w, u=u, A=A, g=g, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=64, BV=64, USE_G=True, IS_VARLEN=False, **CONFIGS["w_u"])
    return w, u


def _fwd_h(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """chunk_gated_delta_rule_fwd_h: per-chunk states h and the transformed values."""
    B, T, H, K, HV, V = *k.shape, *u.shape[2:]
    h, v_new = k.new_empty(B, triton.cdiv(T, BT), HV, K, V), torch.empty_like(u)
    config = CONFIGS["fwd_h"]
    _launch("fwd_h", (triton.cdiv(V, config["BV"]), B * HV), k=k, v=u, w=w, v_new=v_new, g=g, gk=None, h=h, h0=None, ht=None, cu_seqlens=None, chunk_offsets=None, T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_G=True, USE_GK=False, USE_INITIAL_STATE=False, STORE_FINAL_STATE=False, SAVE_NEW_VALUE=True, STATE_V_FIRST=False, IS_VARLEN=False, **config)
    return h, v_new


def chunk_gdr_fwd_pinned(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense chunked gated delta rule forward (chunk size 64, scale K ** -0.5, no initial or final
    state): fresh (o, g_cum, A).

    q, k: bf16 (B, T, H, K) contiguous, unrepeated key heads. v: bf16 (B, T, HV, V). g: fp32
    (B, T, HV), the raw log gate. beta: bf16 (B, T, HV). HV is a multiple of H; value head j reads
    key head j // (HV / H). Inputs are not modified.

    o: bf16 (B, T, HV, V). g_cum: fp32 (B, T, HV), the chunk-local cumulative gate. A: bf16
    (B, T, HV, 64), the solved WY matrix. g_cum and A are inputs of chunk_gdr_bwd_pinned.
    Bit-identical to FLA 0.5.2's chunk_gated_delta_rule_fwd on q / k repeat-interleaved to HV
    heads; deterministic. Only where gated_delta_rule_pinned_supported is True.
    """
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    NT = triton.cdiv(T, BT)
    g_cum = _cumsum(g, reverse=False)
    # The fused kkt + solve stores only the triangular blocks: A must start zeroed.
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    _launch("kkt_solve", (NT, B * HV), k=k, g=g_cum, beta=beta, A=A, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, USE_G=True, IS_VARLEN=False, **CONFIGS["kkt_solve"])
    w, u = _w_u(k, v, beta, A, g_cum)
    h, v_new = _fwd_h(k, w, u, g_cum)
    o = torch.empty_like(v_new)
    config = CONFIGS["fwd_o"]
    _launch("fwd_o", (triton.cdiv(V, config["BV"]), NT, B * HV), q=q, k=k, v=v_new, h=h, g=g_cum, g_gamma=None, o=o, cu_seqlens=None, chunk_indices=None, scale=K**-0.5, T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_G=True, USE_G_GAMMA=False, STATE_V_FIRST=False, IS_VARLEN=False, **config)
    return o, g_cum, A


def chunk_gdr_bwd_pinned(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward of chunk_gdr_fwd_pinned: fresh (dq, dk, dv, dg, dbeta).

    q, k, v, beta: the forward inputs; g_cum, A: the forward outputs; do: bf16 (B, T, HV, V), the
    gradient of o. Inputs are not modified.

    dq, dk: (B, T, HV, K) in q's / k's dtype, per value head (the caller sums each group of HV / H
    heads, in repeat_interleave order, for the key-head gradients). dv like v; dg: fp32 (B, T, HV),
    with respect to the raw gate g; dbeta like beta. Bit-identical to FLA 0.5.2's
    chunk_gated_delta_rule_bwd on repeat-interleaved q / k; deterministic. Only where
    gated_delta_rule_pinned_supported is True.
    """
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    NT, scale = triton.cdiv(T, BT), K**-0.5
    w, u = _w_u(k, v, beta, A, g_cum)
    h, v_new = _fwd_h(k, w, u, g_cum)
    dv_local = torch.empty_like(do)
    _launch("dv_local", (NT, B * HV), q=q, k=k, g=g_cum, g_gamma=None, A=None, do=do, dv=dv_local, cu_seqlens=None, chunk_indices=None, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=128, BV=128, USE_G=True, USE_G_GAMMA=False, USE_A=False, IS_VARLEN=False, **CONFIGS["dv_local"])
    dh, du = q.new_empty(B, NT, HV, K, V), torch.empty_like(dv_local)
    config = CONFIGS["bwd_dhu"]
    _launch("bwd_dhu", (triton.cdiv(V, config["BV"]), B * HV), q=q, k=k, w=w, g=g_cum, gk=None, dht=None, dh0=None, do=do, dh=dh, dv=dv_local, dv2=du, cu_seqlens=None, chunk_offsets=None, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_G=True, USE_GK=False, USE_INITIAL_STATE=False, USE_FINAL_STATE_GRADIENT=False, STATE_V_FIRST=False, IS_VARLEN=False, **config)
    dq, dk, dw = q.new_empty(B, T, HV, K), k.new_empty(B, T, HV, K), torch.empty_like(w)
    dg_parts = torch.empty(1, *g_cum.shape, dtype=torch.float32, device=g_cum.device)
    _launch("dqkwg", (1, NT, B * HV), q=q, k=k, v=v_new, g=g_cum, g_gamma=None, h=h, do=do, dh=dh, dq=dq, dk=dk, dw=dw, dv=du, dg=dg_parts, cu_seqlens=None, chunk_indices=None, scale=scale, B=B, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=128, BV=128, USE_G=True, USE_G_GAMMA=False, USE_DW=True, STATE_V_FIRST=False, IS_VARLEN=False, **CONFIGS["dqkwg"])
    dg = dg_parts.sum(0)
    dk2, dv, db, dg2 = k.new_empty(B, T, HV, K), torch.empty_like(v), torch.empty_like(beta), torch.empty_like(g_cum)
    _launch("wy_bwd", (NT, B * HV), k=k, v=v, beta=beta, g=g_cum, A=A, dw=dw, du=du, dk=dk2, dv=dv, db=db, dg=dg2, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=64, BV=64, USE_G=True, IS_VARLEN=False, **CONFIGS["wy_bwd"])
    dk.add_(dk2)
    dg.add_(dg2)
    dg = _cumsum(dg, reverse=True)
    return dq, dk, dv, dg, db


def pinned_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether a call has the geometry the configs were recorded at: B == 1, HV == 32, HV a multiple
    of H, K == V == 128, bf16 q / k / v / beta and an fp32 gate. FLA's autotune keys hold B (cumsum),
    H, HV, K and V, and the backward's single K block needs K <= 128.
    """
    B, _, H, K = k.shape
    HV, V = v.shape[2:]
    bf = torch.bfloat16
    return B == 1 and HV == 32 and HV % H == 0 and K == 128 and V == 128 and q.shape == k.shape and q.dtype == k.dtype == v.dtype == beta.dtype == bf and g.dtype == torch.float32


def gated_delta_rule_pinned_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:
    """
    Whether chunk_gdr_fwd_pinned / chunk_gdr_bwd_pinned apply to this call: the recorded geometry
    (pinned_shapes) on SM90 with the recorded FLA and Triton versions (pinned_launch_supported).
    Otherwise the caller uses FLA's autotuned entry points. Constant per device and geometry, so
    torch.compile evaluates it rather than tracing into it.
    """
    return pinned_shapes(q, k, v, g, beta) and pinned_launch_supported(q.device.index)
