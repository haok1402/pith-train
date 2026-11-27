"""Prepare the initial SM90 block scales from the exact rank-weight payloads."""

import mmap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.fp8_preflight.artifacts import digest, read_json
from tools.fp8_preflight.recipe import WORLD_SIZE

BLOCK = 128


def shard_amax(rows):
    import numpy as np

    n, k = rows.shape
    if n % BLOCK or k % BLOCK:
        raise ValueError(f"Unaligned scale geometry {(n, k)}")
    bits = rows & np.uint16(0x7FFF)
    maxima = bits.reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK).max(axis=(1, 3))
    if np.any(maxima >= 0x7F80):
        raise ValueError("Initial FP8 scale preparation requires finite weights")
    return (maxima.astype(np.uint32) << 16).view(np.float32)


def scale_from_amax(amax):
    import numpy as np

    # Match max.f32, mul.rn.f32 and the exponent rounding in compute_scale_bits.
    bits = (np.maximum(amax, np.float32(1e-4)) * np.float32(1.0 / 448.0)).view(np.uint32)
    exponent = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).astype(np.uint32) * 0x00800000
    return np.maximum(exponent, 0x00800000).view(np.float32)


def prepare_initial_scales(root, weights, model):
    import numpy as np
    import torch
    from safetensors.numpy import save_file

    from pithtrain.operators.linear import FP8Linear

    shapes = {
        f"{name}.weight": tuple(module.weight.shape)
        for name, module in model.named_modules()
        if isinstance(module, FP8Linear)
    }
    merged = {
        name: np.empty((n // BLOCK, k // BLOCK), dtype=np.float32)
        for name, (n, k) in shapes.items()
    }

    def read_rank(rank):
        records = read_json(weights / f"rank-{rank}.json")["records"]
        seen = set()
        with (weights / f"rank-{rank}.bin").open("rb") as stream, mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_COPY
        ) as packed:
            packed.madvise(mmap.MADV_SEQUENTIAL)
            for item in records:
                name = item["name"]
                if name not in shapes:
                    continue
                n, k = shapes[name]
                start, end = item["row"]
                if (
                    name in seen or n % (WORLD_SIZE * BLOCK) or k % BLOCK
                    or start != rank * n // WORLD_SIZE or end != (rank + 1) * n // WORLD_SIZE
                    or item["shape"] != [end - start, k]
                ):
                    raise ValueError(f"Invalid FP8 scale shard rank={rank} parameter={name}")
                count = (end - start) * k
                if item["dtype"] == "BF16":
                    rows = np.frombuffer(packed, dtype=np.uint16, count=count, offset=item["offset"])
                else:
                    fp32 = torch.frombuffer(packed, dtype=torch.float32, count=count, offset=item["offset"])
                    rows = fp32.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
                    del fp32
                merged[name][start // BLOCK : end // BLOCK] = shard_amax(rows.reshape(-1, k))
                del rows
                seen.add(name)
        if seen != shapes.keys():
            raise ValueError(f"Incomplete initial FP8 scales for rank {rank}")
        print(f"PREFLIGHT fp8_amax_rank={rank} weights={len(seen)}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(read_rank, range(WORLD_SIZE)))
    tensors = {}
    for name, amax in merged.items():
        tensors[f"amax:{name}"] = amax
        tensors[f"scale:{name}"] = scale_from_amax(amax)
    path = root / "initial-scales.safetensors"
    save_file(tensors, str(path))
    return digest(path)


def load_initial_scales(path: Path, model):
    from safetensors.torch import load_file

    from pithtrain.modules.fp8_param_gather import fp8_groups, scale_grids

    values = load_file(str(path))
    names = {id(module): f"{name}.weight" for name, module in model.named_modules()}
    states = []
    for _, _, group_states in fp8_groups:
        for state in group_states:
            name = names[id(state.module)]
            amax = values[f"amax:{name}"].unsqueeze(0)
            scale = values[f"scale:{name}"].unsqueeze(0)
            scale_dst, scale_t_dst = scale_grids(state.module)
            if amax.shape != state.amax.shape or scale.shape != scale_dst.shape:
                raise ValueError(f"Initial FP8 scale geometry mismatch at {name}")
            states.append((state, amax, scale, scale_dst, scale_t_dst))
    if len(states) * 2 != len(values):
        raise ValueError("Initial FP8 scales do not match the FSDP model")
    for state, amax, scale, scale_dst, scale_t_dst in states:
        state.amax.copy_(amax, non_blocking=True)
        scale_dst.copy_(scale, non_blocking=True)
        scale_t_dst.copy_(scale.transpose(-2, -1), non_blocking=True)
        state.cast_shard()
        state.version = state.fsdp_param._sharded_local_tensor._version
