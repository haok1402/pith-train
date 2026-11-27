"""Verify and stage persistent Inductor graphs for the dense FP8 recipe."""

import hashlib
import pickle
import pickletools
import re
from pathlib import Path

from tools.fp8_preflight.artifacts import read_json, write_json
from tools.fp8_preflight.misses import record_miss
from tools.fp8_preflight.recipe import MICROBATCH, SEQUENCE, WORLD_SIZE

ROUTES = {
    ("gated_delta", False),
    ("gated_delta", True),
    ("full_attention", False),
    ("full_attention", True),
    ("residual_stage", False),
}


def graph_metadata(raw, key, hidden_size, system_key):
    """Read the cache record's producer metadata without unpickling it."""
    strings = [item for _, item, _ in pickletools.genops(raw) if isinstance(item, str)]
    if key not in strings:
        raise ValueError(f"FX cache record does not contain its key {key}")
    if not any(item.endswith(f"system_info[hash]: {system_key}") for item in strings):
        raise ValueError(f"FX cache record {key} targets another compiler or accelerator")
    source = next(item for item in strings if item.startswith("class GraphModule("))
    inputs = next(line for line in source.splitlines() if "def forward(" in line)
    devices = set(re.findall(r"cuda:([0-7])", inputs))
    if len(devices) != 1 or f"bf16[{MICROBATCH}, {SEQUENCE}, {hidden_size}]" not in inputs:
        raise ValueError(f"FX cache graph {key} has an unexpected input signature")
    trace = next(item for item in strings if "fx_kwargs[is_backward]:" in item)
    backward = trace.endswith("True")
    if "pithtrain.gated_delta_rule_dense_" in source:
        route = "gated_delta"
    elif "pithtrain.delta_matching_attn_" in source:
        route = "full_attention"
    elif any("in forward_stage5" in item for item in strings):
        route = "residual_stage"
    else:
        raise ValueError(f"FX cache graph {key} has an unexpected decoder route")
    return int(devices.pop()), route, backward


def seal_cache(bundle, seed, manifest, model_config_hash, hidden_size):
    """Copy verified FX entries into the bundle so the source cache can be removed."""
    from torch._inductor.codecache import CacheBase, torch_key, triton_key

    receipt = read_json(manifest)
    if receipt["model_config"] != model_config_hash:
        raise ValueError("Inductor manifest targets another model configuration")
    if receipt["torch_key"] != torch_key().hex() or receipt["triton_key"] != triton_key():
        raise ValueError("Inductor manifest does not match the active PyTorch and Triton")
    system_key = CacheBase.get_system()["hash"]
    root = bundle / "inductor"
    keys, coverage = set(), set()
    for name, expected in receipt["files"].items():
        path = Path(name)
        if (
            len(path.parts) != 4
            or path.parts[0] != "fxgraph"
            or not all(re.fullmatch(r"[a-z2-7]+", part) for part in path.parts[1:])
            or path.parts[1] != path.parts[2][1:3]
        ):
            raise ValueError(f"Unexpected FX cache path {path}")
        raw = (seed / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Inductor seed content changed for {path}")
        key = path.parts[2]
        coverage.add(graph_metadata(raw, key, hidden_size, system_key))
        keys.add(key)
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    wanted = {(rank, route, backward) for rank in range(WORLD_SIZE) for route, backward in ROUTES}
    if coverage != wanted:
        raise ValueError(f"Inductor coverage mismatch, missing {sorted(wanted - coverage)}")
    write_json(root / "seal.json", {**receipt, "keys": sorted(keys)})
    print(f"PREFLIGHT inductor_graphs={len(keys)} covered_routes={len(coverage)}", flush=True)


def require_cache(bundle):
    """Record sealed hits and let Inductor compile graphs that miss."""
    from torch._inductor import compile_fx
    from torch._inductor.codecache import FxGraphCache, torch_key, triton_key

    try:
        seal = read_json(bundle / "inductor" / "seal.json")
        if seal["torch_key"] != torch_key().hex() or seal["triton_key"] != triton_key():
            raise ValueError("Live PyTorch and Triton differ from the Inductor seal")
        allowed = set(seal["keys"])
    except (OSError, ValueError, KeyError) as error:
        record_miss("inductor", "seal", str(error))
        allowed = set()
    lookup = FxGraphCache.load_with_key
    codegen = compile_fx.fx_codegen_and_compile
    pending = set()

    def load_with_key(cls, key, *args, **kwargs):
        try:
            graph, info = lookup(key, *args, **kwargs)
        except (OSError, RuntimeError, pickle.UnpicklingError) as error:
            graph, info = None, {
                "cache_state": "miss",
                "key": key,
                "components": kwargs.get("debug_lines", args[0] if args else []),
            }
            record_miss("inductor", key, str(error).splitlines()[0])
            pending.add(key)
            return graph, info
        if graph is None:
            record_miss("inductor", key, "no usable compiled graph", backward=kwargs.get("is_backward"))
            pending.add(key)
        elif key in allowed:
            print(f"PREFLIGHT inductor_cache_hit={key} backward={kwargs.get('is_backward')}", flush=True)
        return graph, info

    def recorded_codegen(gm, *args, **kwargs):
        if pending:
            pending.clear()
        else:
            # Inductor can bypass FX caching before assigning a key.
            source = str(gm.graph)
            key = hashlib.sha256(source.encode()).hexdigest()
            record_miss("inductor", key, "FX cache bypass", graph=source)
        return codegen(gm, *args, **kwargs)

    FxGraphCache.load_with_key = classmethod(load_with_key)
    compile_fx.fx_codegen_and_compile = recorded_codegen
