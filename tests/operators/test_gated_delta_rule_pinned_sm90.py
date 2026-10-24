"""
The pinned-launch dense delta rule (kernels/gated_delta_rule_pinned_sm90.py, through operators.gated_delta_rule) against FLA's autotuned path.

FLA's autotuners are restricted to the pinned configs, so both sides run the same kernels with the
same configs, and everything must match bit for bit: output, every input gradient, dtypes and
strides, eager and compiled. That the pinned configs are the ones FLA's own autotuning picks on
H200 is recorded separately (kda-task/profile/fla-config-manifest.json).
"""

import json
from pathlib import Path

import fla
import pytest
import torch
import triton

from pithtrain.kernels import gated_delta_rule_pinned_sm90 as pinned
from pithtrain.kernels.arch import is_sm90
from pithtrain.operators.gated_delta_rule import _gdr_fwd, dense_gated_delta_rule, gated_delta_rule

pytestmark = pytest.mark.skipif(
    not is_sm90(), reason="the pinned launch configs are recorded on SM90"
)

LAUNCH = ("num_warps", "num_stages")


@pytest.fixture
def fla_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each FLA autotuner the single config the pinned path launches with."""
    picks = {
        "cumsum": pinned.CONFIGS["cumsum_fwd"], "kkt_solve": pinned.CONFIGS["kkt_solve"],
        "w_u": pinned.CONFIGS["w_u"], "fwd_h": pinned.CONFIGS["fwd_h"],
        "fwd_o": pinned.CONFIGS["fwd_o"], "dv_local": pinned.CONFIGS["dv_local"],
        "bwd_dhu": pinned.CONFIGS["bwd_dhu"], "dqkwg": pinned.CONFIGS["dqkwg"],
        "wy_bwd": pinned.CONFIGS["wy_bwd"],
    }  # fmt: skip
    # One config per tuner: the forward and reverse cumsum must agree for this to cover both.
    assert pinned.CONFIGS["cumsum_fwd"] == pinned.CONFIGS["cumsum_rev"]
    for name, pick in picks.items():
        tuner = pinned.KERNELS[name].fn
        meta = {k: v for k, v in pick.items() if k not in LAUNCH}
        monkeypatch.setattr(tuner, "configs", [triton.Config(meta, **{k: pick[k] for k in LAUNCH})])
        monkeypatch.setattr(tuner, "cache", {})


def inputs(S: int, packed: bool, seed: int = 0, B: int = 1, H: int = 32) -> tuple[torch.Tensor, ...]:  # fmt: skip
    """q / k with H heads, v / g / beta with 32 (H = 16: Qwen3.5's grouped value heads)."""
    torch.manual_seed(seed)
    HV, K, V, dev, bf = 32, 128, 128, "cuda", torch.bfloat16
    q = torch.nn.functional.normalize(torch.randn(B, S, H, K, device=dev), dim=-1).to(bf)
    k = torch.nn.functional.normalize(torch.randn(B, S, H, K, device=dev), dim=-1).to(bf)
    v = torch.randn(B, S, HV, V, device=dev, dtype=bf)
    g = -torch.rand(B, S, HV, device=dev) * 0.1
    if packed:
        # Document starts reset the state the way the packed model path does (g = -128).
        g[:, [0, 300, 1777, 2048, S - 5]] = -128.0
    beta = torch.rand(B, S, HV, device=dev, dtype=bf)
    do = torch.randn(B, S, HV, V, device=dev, dtype=bf)
    return q, k, v, g, beta, do


def run(fn, args, compile: bool = False):
    *xs, do = args
    xs = [x.clone().requires_grad_() for x in xs]
    f = torch.compile(fn, fullgraph=True) if compile else fn
    o = f(*xs)
    o.backward(do)
    return [o.detach()] + [x.grad for x in xs]


def reference(q, k, v, g, beta):
    """The r5-base path: q / k repeated to the value heads, then the autotuned custom op."""
    repeats = v.shape[2] // q.shape[2]
    q, k = q.repeat_interleave(repeats, dim=2), k.repeat_interleave(repeats, dim=2)
    return _gdr_fwd(q, k, v, g, beta, None)[0]


def assert_identical(got: list, ref: list) -> None:
    for name, a, b in zip(("o", "dq", "dk", "dv", "dg", "dbeta"), got, ref):
        assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride(), name
        assert torch.equal(a, b), name


@pytest.mark.parametrize("H", [32, 16])
@pytest.mark.parametrize("S,packed,B", [(4096, False, 1), (1000, False, 1), (4096, True, 1), (8192, False, 1), (1000, False, 3)])  # fmt: skip
def test_pinned_matches_fla(fla_pinned, S: int, packed: bool, B: int, H: int) -> None:
    """Eager and compiled; H = 16 runs grouped value heads on unrepeated q / k."""
    assert pinned.pinned_launch_supported(torch.cuda.current_device())
    args = inputs(S, packed, B=B, H=H)
    assert_identical(run(dense_gated_delta_rule, args), run(reference, args))
    pinned_path = lambda *x: gated_delta_rule(*x, pinned=True)  # noqa: E731
    assert_identical(run(pinned_path, args, compile=True), run(reference, args, compile=True))


def test_config_drift_detected(fla_pinned, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A launch config other than FLA's breaks the bitwise match: the test sees config drift.
    (num_stages and the forward kernels' num_warps turn out bit-neutral; dqkwg's num_warps
    changes the gate gradient.)
    """
    args = inputs(4096, False)
    ref = run(reference, args)
    warps = 2 if pinned.CONFIGS["dqkwg"]["num_warps"] == 4 else 4
    monkeypatch.setitem(pinned.CONFIGS, "dqkwg", dict(pinned.CONFIGS["dqkwg"], num_warps=warps))
    got = run(dense_gated_delta_rule, args)
    assert not torch.equal(got[4], ref[4])


def test_version_guard_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another FLA version disables the pinned path; gated_delta_rule takes the autotuned op."""
    pinned._supported.clear()
    monkeypatch.setattr(fla, "__version__", "0.0.0")
    try:
        assert not pinned.pinned_launch_supported(torch.cuda.current_device())
        called = []
        monkeypatch.setattr("pithtrain.operators.gated_delta_rule.dense_gated_delta_rule", lambda *a: called.append(a))  # fmt: skip
        q, k, v, g, beta, _ = inputs(1000, False)
        o = gated_delta_rule(q, k, v, g, beta, pinned=True)
        assert not called and o.shape == v.shape
    finally:
        pinned._supported.clear()


def test_pinned_configs_are_the_recorded_picks() -> None:
    """
    The pinned configs, checked against their record rather than against themselves: the
    per-kernel majority picks of r5-base's own autotuning on H200 (8 ranks of a training run), as
    committed in kda-task/profile/fla-config-manifest.json. The parity tests above force FLA onto
    CONFIGS, so a change here would otherwise go unnoticed.
    """
    recorded = {
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
    assert pinned.CONFIGS == recorded
    assert pinned.VERSIONS == {"fla": "0.5.2", "triton": "3.7.1"}
    manifest = Path(__file__).parents[1] / "kda-task/profile/fla-config-manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["pinned"] == recorded


def test_pinned_only_at_the_recorded_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off the recorded geometry (batch, heads, head dim, dtypes) gated_delta_rule falls back."""
    q, k, v, g, beta, _ = inputs(1000, False, H=16)
    assert pinned.pinned_shapes(q, k, v, g, beta)
    q3, k3, v3, g3, b3, _ = inputs(1000, False, B=3, H=16)
    assert not pinned.pinned_shapes(q3, k3, v3, g3, b3)
    assert not pinned.pinned_shapes(q, k, v, g.to(torch.bfloat16), beta)
    assert not pinned.pinned_shapes(q[..., :64], k[..., :64], v, g, beta)
    called = []
    monkeypatch.setattr("pithtrain.operators.gated_delta_rule.dense_gated_delta_rule", lambda *a: called.append(a))  # fmt: skip
    gated_delta_rule(q3, k3, v3, g3, b3, pinned=True)
    assert not called
