"""
The dense chunked gated delta rule as a custom op that launches FLA's kernels directly.

FLA wraps every kernel as Heuristics(CachedAutotuner(JITFunction)) behind a backend-dispatch
wrapper, and that Python layer costs ~60-90 us per launch (autotune key build, heuristics,
dispatch), 16 launches per forward + backward. Here the same JITFunctions are launched with the
heuristic constexprs and the autotuned launch config passed explicitly (~20 us per launch). The op
stays opaque to torch.compile on purpose: an Inductor-launched copy of a user kernel is compiled
differently and is not bit-identical, and it launches no cheaper.

The launch order, allocations, in-place accumulations and dtypes reproduce FLA 0.5.2's
chunk_gated_delta_rule_fwd / chunk_gated_delta_rule_bwd (fla/ops/gated_delta_rule/chunk.py) on its
dense path: cu_seqlens, initial and final state, CP context and the in-kernel gate all absent,
chunk size 64. The output is therefore bit-identical as long as the pinned configs are the ones
FLA's autotuner picks at these shapes; none of the autotune keys holds the sequence length, so one
config per kernel covers every length. The configs were recorded on H200
(kda-task/profile/fla-config-manifest.json); any other FLA or Triton version, GPU architecture or
kernel signature leaves the dense path to the autotuned custom op instead
(pinned_launch_supported).

Grouped value heads (more value than key heads) run on unrepeated q / k, which the kernels index
per group; the backward then sums the q / k gradients over each group in the order a
repeat_interleave of q / k would, so the result matches repeating them bit for bit.
"""

import fla
import torch
import triton
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from fla.ops.common.chunk_o import (
    chunk_bwd_kernel_dqkwg,
    chunk_bwd_kernel_dv_local,
    chunk_fwd_kernel_o,
)
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd_kernel, recompute_w_u_fwd_kernel
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
from triton import knobs

__all__ = ["dense_gated_delta_rule", "pinned_launch_supported", "pinned_shapes"]

BT = 64  # chunk size
BC = 16  # sub-chunk of the fused kkt + triangular solve, fixed by FLA at BT = 64

# Launch configs FLA's autotuner selects on H200 at H = HV = 32, K = V = 128 (bf16 q/k/v/beta,
# fp32 gate), recorded with FLA 0.5.2 / Triton 3.7.1. Wrapper-fixed block sizes (BK/BV of the
# w/u, dv_local, dqkwg and WY-backward kernels) are FLA's own constants on Hopper.
VERSIONS = {"fla": "0.5.2", "triton": "3.7.1"}
CONFIGS = {
    "cumsum_fwd": dict(num_warps=1, num_stages=3),
    "cumsum_rev": dict(num_warps=1, num_stages=3),
    "kkt_solve": dict(BK=64, num_warps=1, num_stages=3),
    "w_u": dict(num_warps=4, num_stages=4),
    "fwd_h": dict(BV=32, num_warps=4, num_stages=3),
    "fwd_o": dict(BK=64, BV=64, num_warps=4, num_stages=3),
    "dv_local": dict(num_warps=2, num_stages=3),
    "bwd_dhu": dict(BV=32, num_warps=4, num_stages=3),
    "dqkwg": dict(num_warps=4, num_stages=3),
    "wy_bwd": dict(num_warps=4, num_stages=4),
}

KERNELS = {
    "cumsum": chunk_local_cumsum_scalar_kernel,
    "kkt_solve": chunk_gated_delta_rule_fwd_kkt_solve_kernel,
    "w_u": recompute_w_u_fwd_kernel,
    "fwd_h": chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    "fwd_o": chunk_fwd_kernel_o,
    "dv_local": chunk_bwd_kernel_dv_local,
    "bwd_dhu": chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    "dqkwg": chunk_bwd_kernel_dqkwg,
    "wy_bwd": prepare_wy_repr_bwd_kernel,
}

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
    jit[grid](**kwargs) with the compiled kernel cached per launch site. Triton's own run binds
    the arguments and rebuilds the specialization key in Python on every launch; here the key is
    what it specializes on for these kernels (the 16-byte alignment of each tensor and the value of
    every other argument but the sequence length T, which is do_not_specialize, and float scalars,
    which are never specialized), plus each tensor's dtype and the device, and a cached kernel goes
    straight to its C launcher, on that device's current stream, with the arguments Triton passes.
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
    key = (name, device, kwargs["num_warps"], kwargs["num_stages"], *(float if args[i].__class__ is float else args[i] for i in values), *((args[i].dtype, args[i].data_ptr() & 15) for i in tensors))  # fmt: skip
    kernel = _compiled.get(key)
    if kernel is None:
        _compiled[key] = jit[grid](**kwargs)
        return
    grid = (*grid, 1, 1)
    stream = torch._C._cuda_getCurrentRawStream(device)
    enter, exit_ = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    metadata = None if enter is None else kernel.launch_metadata(grid, stream, *args)
    kernel.run(grid[0], grid[1], grid[2], stream, kernel.function, kernel.packed_metadata, metadata, enter, exit_, *args)  # fmt: skip


_supported: dict[int, bool] = {}


@torch.compiler.assume_constant_result
def pinned_launch_supported(device_index: int) -> bool:
    """
    Whether the pinned configs apply: the recorded FLA and Triton versions, SM90, and kernels
    whose wrapper structure and parameter lists match the launches written here. Evaluated once
    per device, and a constant to torch.compile, which calls it instead of tracing into the
    kernel objects.
    """
    if device_index not in _supported:
        _supported[device_index] = _check_supported(device_index)
    return _supported[device_index]


def _check_supported(device_index: int) -> bool:
    if fla.__version__ != VERSIONS["fla"] or triton.__version__ != VERSIONS["triton"]:
        return False
    if torch.cuda.get_device_capability(device_index) != (9, 0):
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
    _launch("cumsum", (triton.cdiv(T, BT), B * H),
        s=s, o=o, scale=None if reverse else RCP_LN2, cu_seqlens=None, chunk_indices=None, T=T,
        B=B, H=H, BT=BT, REVERSE=reverse, HAS_SCALE=not reverse, IS_VARLEN=False,
        HEAD_FIRST=False, **config,
    )  # fmt: skip
    return o


def _w_u(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """recompute_w_u_fwd."""
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    w, u = k.new_empty(B, T, HV, K), torch.empty_like(v)
    _launch("w_u", (triton.cdiv(T, BT), B * HV),
        k=k, v=v, beta=beta, w=w, u=u, A=A, g=g, cu_seqlens=None, chunk_indices=None, T=T, H=H,
        HV=HV, K=K, V=V, BT=BT, BK=64, BV=64, USE_G=True, IS_VARLEN=False, **CONFIGS["w_u"],
    )  # fmt: skip
    return w, u


def _fwd_h(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """chunk_gated_delta_rule_fwd_h: per-chunk states h and the transformed values."""
    B, T, H, K, HV, V = *k.shape, *u.shape[2:]
    h, v_new = k.new_empty(B, triton.cdiv(T, BT), HV, K, V), torch.empty_like(u)
    config = CONFIGS["fwd_h"]
    _launch("fwd_h", (triton.cdiv(V, config["BV"]), B * HV),
        k=k, v=u, w=w, v_new=v_new, g=g, gk=None, h=h, h0=None, ht=None, cu_seqlens=None,
        chunk_offsets=None, T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_G=True, USE_GK=False,
        USE_INITIAL_STATE=False, STORE_FINAL_STATE=False, SAVE_NEW_VALUE=True,
        STATE_V_FIRST=False, IS_VARLEN=False, **config,
    )  # fmt: skip
    return h, v_new


def _key_heads(t: torch.Tensor) -> torch.Tensor:
    """[b, s, heads, dim], from either that or its [b, s, heads, group, dim] expand over a group."""
    if t.dim() == 5:
        t = t[:, :, :, 0]
    return t if t.is_contiguous() else t.contiguous()


@torch.library.custom_op("pithtrain::gated_delta_rule_dense_fwd", mutates_args=())
def _dense_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """
    chunk_gated_delta_rule_fwd, dense: returns the output, the chunk-cumulated gate and the solved
    WY matrix A, the two forward intermediates the backward reads. v, g and beta have HV heads;
    q and k have HV too, or H with HV a multiple of H, given as their [b, s, H, HV // H, dim]
    expand (grouped value heads share a key head).
    """
    q, k = _key_heads(q), _key_heads(k)
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    NT = triton.cdiv(T, BT)
    g_cum = _cumsum(g, reverse=False)
    # The fused kkt + solve stores only the triangular blocks: A must start zeroed.
    A = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    _launch("kkt_solve", (NT, B * HV),
        k=k, g=g_cum, beta=beta, A=A, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K,
        BT=BT, BC=BC, USE_G=True, IS_VARLEN=False, **CONFIGS["kkt_solve"],
    )  # fmt: skip
    w, u = _w_u(k, v, beta, A, g_cum)
    h, v_new = _fwd_h(k, w, u, g_cum)
    o = torch.empty_like(v_new)
    config = CONFIGS["fwd_o"]
    _launch("fwd_o", (triton.cdiv(V, config["BV"]), NT, B * HV),
        q=q, k=k, v=v_new, h=h, g=g_cum, g_gamma=None, o=o, cu_seqlens=None, chunk_indices=None,
        scale=K**-0.5, T=T, H=H, HV=HV, K=K, V=V, BT=BT, USE_G=True, USE_G_GAMMA=False,
        STATE_V_FIRST=False, IS_VARLEN=False, **config,
    )  # fmt: skip
    return o, g_cum, A


@_dense_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
    B, T, HV = g.shape
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    return torch.empty_like(v), torch.empty_like(g, dtype=torch.float), A


@torch.library.custom_op("pithtrain::gated_delta_rule_dense_bwd", mutates_args=())
def _dense_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """
    chunk_gated_delta_rule_bwd, dense: (dq, dk, dv, dg, dbeta), dg w.r.t. the raw gate.

    With grouped value heads the q / k gradients are returned per value head, shaped like the
    [b, s, H, HV // H, dim] expands q and k came in as, and autograd's backward of the expand sums
    them over each group: the same reduction, at the same point, as the backward of a
    repeat_interleave of q / k ahead of an H == HV call. (FLA's own wrappers instead reduce each of
    dk's two terms over the group before adding them, a different rounding.)
    """
    q_in, k_in = q, k
    q, k = _key_heads(q), _key_heads(k)
    B, T, H, K, HV, V = *k.shape, *v.shape[2:]
    NT, scale = triton.cdiv(T, BT), K**-0.5
    w, u = _w_u(k, v, beta, A, g_cum)
    h, v_new = _fwd_h(k, w, u, g_cum)
    dv_local = torch.empty_like(do)
    _launch("dv_local", (NT, B * HV),
        q=q, k=k, g=g_cum, g_gamma=None, A=None, do=do, dv=dv_local, cu_seqlens=None,
        chunk_indices=None, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=128, BV=128,
        USE_G=True, USE_G_GAMMA=False, USE_A=False, IS_VARLEN=False, **CONFIGS["dv_local"],
    )  # fmt: skip
    dh, du = q.new_empty(B, NT, HV, K, V), torch.empty_like(dv_local)
    config = CONFIGS["bwd_dhu"]
    _launch("bwd_dhu", (triton.cdiv(V, config["BV"]), B * HV),
        q=q, k=k, w=w, g=g_cum, gk=None, dht=None, dh0=None, do=do, dh=dh, dv=dv_local, dv2=du,
        cu_seqlens=None, chunk_offsets=None, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT,
        USE_G=True, USE_GK=False, USE_INITIAL_STATE=False, USE_FINAL_STATE_GRADIENT=False,
        STATE_V_FIRST=False, IS_VARLEN=False, **config,
    )  # fmt: skip
    dq, dk, dw = q.new_empty(B, T, HV, K), k.new_empty(B, T, HV, K), torch.empty_like(w)
    dg_parts = torch.empty(1, *g_cum.shape, dtype=torch.float32, device=g_cum.device)
    _launch("dqkwg", (1, NT, B * HV),
        q=q, k=k, v=v_new, g=g_cum, g_gamma=None, h=h, do=do, dh=dh, dq=dq, dk=dk, dw=dw, dv=du,
        dg=dg_parts, cu_seqlens=None, chunk_indices=None, scale=scale, B=B, T=T, H=H, HV=HV, K=K,
        V=V, BT=BT, BK=128, BV=128, USE_G=True, USE_G_GAMMA=False, USE_DW=True,
        STATE_V_FIRST=False, IS_VARLEN=False, **CONFIGS["dqkwg"],
    )  # fmt: skip
    dg = dg_parts.sum(0)
    dk2, dv, db, dg2 = k.new_empty(B, T, HV, K), torch.empty_like(v), torch.empty_like(beta), torch.empty_like(g_cum)  # fmt: skip
    _launch("wy_bwd", (NT, B * HV),
        k=k, v=v, beta=beta, g=g_cum, A=A, dw=dw, du=du, dk=dk2, dv=dv, db=db, dg=dg2,
        cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=64, BV=64,
        USE_G=True, IS_VARLEN=False, **CONFIGS["wy_bwd"],
    )  # fmt: skip
    dk.add_(dk2)
    dg.add_(dg2)
    dg = _cumsum(dg, reverse=True)
    return dq.view(q_in.shape), dk.view(k_in.shape), dv, dg, db


@_dense_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor):  # fmt: skip
    cf = torch.contiguous_format
    return tuple(torch.empty_like(t, memory_format=cf) for t in (q, k, v, g_cum, beta))


def _setup_context(ctx, inputs, output) -> None:
    q, k, v, _, beta = inputs
    _, g_cum, A = output
    ctx.save_for_backward(q, k, v, g_cum, beta, A)


def _backward(ctx, do: torch.Tensor, *_unused) -> tuple:
    q, k, v, g_cum, beta, A = ctx.saved_tensors
    return _dense_bwd(q, k, v, g_cum, beta, A, do.contiguous())


_dense_fwd.register_autograd(_backward, setup_context=_setup_context)


def pinned_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> bool:  # fmt: skip
    """
    Whether a call has the geometry the configs were recorded at (batch 1, 32 value heads, key and
    value head dim 128, bf16 q / k / v / beta and an fp32 gate, value heads a multiple of key
    heads): only there are they the picks of FLA's autotuner (whose keys hold B for the cumsum, and
    H, HV, K and V), and the backward's single K block assumes K <= 128.
    """
    B, _, H, K = k.shape
    HV, V = v.shape[2:]
    bf = torch.bfloat16
    return (
        B == 1 and HV == 32 and HV % H == 0 and K == 128 and V == 128 and q.shape == k.shape
        and q.dtype == k.dtype == v.dtype == beta.dtype == bf and g.dtype == torch.float32
    )  # fmt: skip


def dense_gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """
    Chunked gated delta rule: q / k [b, s, heads, dim], v [b, s, value_heads, value_dim], g and
    beta [b, s, value_heads], value_heads a multiple of heads, through the pinned FLA launches;
    see the module docstring.
    """
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    B, S, H, K = k.shape
    group = v.shape[2] // H
    if group > 1:
        # A view, not a copy: the kernels read each key head once per group by index.
        q, k = (t.unsqueeze(3).expand(B, S, H, group, K) for t in (q, k))
    o, _, _ = _dense_fwd(q, k, v, g, beta)
    return o
