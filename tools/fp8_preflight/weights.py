"""Prepare dense rank-local safetensor slices using the training converter."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from tools.fp8_preflight.artifacts import digest, read_json, write_json
from tools.fp8_preflight.recipe import SEQUENCE, WORLD_SIZE


def model_geometry(model_path):
    import torch
    from transformers import AutoConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models.qwen35 import Qwen35TextModel
    from pithtrain.operators.linear import FP8Linear

    config = AutoConfig.from_pretrained(model_path, local_files_only=True).text_config
    config.max_position_embeddings = SEQUENCE
    distributed.cp_size = distributed.ep_size = 1
    training.fp8 = training.fp8_attention = True
    training.Linear = FP8Linear
    with torch.device("meta"):
        model = Qwen35TextModel(config)
    shapes = {tuple(m.weight.shape) for m in model.modules() if isinstance(m, FP8Linear)}
    from tools.fp8_preflight.recipe import PROJECTIONS

    if shapes != set(PROJECTIONS):
        raise ValueError(f"Unsupported dense FP8 projection geometry: {sorted(shapes)}")
    return config, model


def source_index(source):
    return read_json(source / "model.safetensors.index.json")["weight_map"]


def plans(source, model, rank):
    import torch

    from pithtrain.modules.hf_converters.qwen35 import Qwen35Converter

    converter = Qwen35Converter()
    if not converter.detect_hf(source):
        raise ValueError(f"Unsupported dense HF checkpoint at {source}")
    keys = source_index(source)
    full_keys = {key.removeprefix("model."): key for key in keys}
    files = {key: keys[original] for key, original in full_keys.items()}
    modules = dict(model.named_modules())
    mesh = SimpleNamespace(ndim=1, get_local_rank=lambda _: rank, size=lambda _: WORLD_SIZE)
    result = []
    used = set()
    for name, parameter in model.named_parameters():
        chunk = (parameter.shape[0] + WORLD_SIZE - 1) // WORLD_SIZE
        length = max(0, min(chunk, parameter.shape[0] - rank * chunk))
        local = torch.empty((length, *parameter.shape[1:]), dtype=parameter.dtype, device="meta")
        sharded = SimpleNamespace(_local_tensor=local, device_mesh=mesh, shape=parameter.shape)
        (op,) = converter.plan_param(name, sharded, files, modules)
        if op.transform is not None or len(op.shard_files) != 1 or len(op.hf_keys) != 1:
            raise ValueError(f"Packed weights require direct dense slices, got {name}")
        used.add(op.hf_keys[0])
        result.append(
            {
                "name": op.local_fqn,
                "source_key": full_keys[op.hf_keys[0]],
                "shard": op.shard_files[0],
                "row": [op.row_slice.start, op.row_slice.stop],
                "shape": list(op.dst.shape),
            }
        )
    unused = set(files) - used
    unexpected = unused - converter.expected_unmapped(unused) - {"lm_head.weight"}
    if unexpected:
        raise ValueError(f"Unmapped checkpoint tensors: {sorted(unexpected)}")
    return sorted(result, key=lambda item: item["shard"])


def verify_records(records, expected):
    layout = [
        {key: record[key] for key in plan} for record, plan in zip(records, expected, strict=True)
    ]
    if layout != expected:
        raise ValueError("Packed weights disagree with the checkpoint converter")
    end = 0
    for record in records:
        if record["dtype"] not in ("BF16", "F32") or record["offset"] != end + (-end % 4):
            raise ValueError(f"Invalid packed weight record {record['name']}")
        from math import prod

        width = 2 if record["dtype"] == "BF16" else 4
        if record["bytes"] != prod(record["shape"]) * width or record["offset"] % width:
            raise ValueError(f"Invalid packed weight slice {record['name']}")
        end = record["offset"] + record["bytes"]
    return end


def prepare_rank(root, source, model, rank, seed=None):
    import hashlib

    import torch
    from safetensors import safe_open

    data = root / f"rank-{rank}.bin"
    expected = plans(source, model, rank)
    hashes = hashlib.sha256()
    if seed is None:
        records = []
        with ExitStack() as stack, data.open("xb") as output:
            files = {
                shard: stack.enter_context(safe_open(source / shard, framework="pt", device="cpu"))
                for shard in {plan["shard"] for plan in expected}
            }
            for plan in expected:
                item = files[plan["shard"]].get_slice(plan["source_key"])
                array = item[slice(*plan["row"])].contiguous()
                dtype = {torch.bfloat16: "BF16", torch.float32: "F32"}[array.dtype]
                payload = memoryview(array.view(torch.uint8).numpy()).cast("B")
                pad = bytes(-output.tell() % 4)
                output.write(pad)
                hashes.update(pad)
                records.append(
                    {**plan, "offset": output.tell(), "bytes": len(payload), "dtype": dtype}
                )
                output.write(payload)
                hashes.update(payload)
        record = {
            "rank": rank,
            "bytes": data.stat().st_size,
            "sha256": hashes.hexdigest(),
            "records": records,
        }
    else:
        seed = Path(seed)
        old = read_json(seed / f"rank-{rank}.json")
        record = {key: old[key] for key in ("rank", "bytes", "sha256", "records")}
        verify_records(record["records"], expected)
        src = seed / data.name
        # Prove a seed belongs to the live checkpoint, including its payload bytes.
        with ExitStack() as stack, src.open("rb") as packed:
            files = {
                shard: stack.enter_context(safe_open(source / shard, framework="pt", device="cpu"))
                for shard in {plan["shard"] for plan in expected}
            }
            for item in record["records"]:
                array = files[item["shard"]].get_slice(item["source_key"])[slice(*item["row"])]
                payload = memoryview(array.contiguous().view(torch.uint8).numpy()).cast("B")
                hashes.update(packed.read(item["offset"] - packed.tell()))
                raw = packed.read(item["bytes"])
                if payload != raw:
                    raise ValueError(f"Weight seed rank {rank} disagrees at {item['name']}")
                hashes.update(raw)
        if hashes.hexdigest() != record["sha256"]:
            raise ValueError(f"Invalid weight seed digest for rank {rank}")
        data.hardlink_to(src)
    end = verify_records(record["records"], expected)
    if record["rank"] != rank or end != record["bytes"] or data.stat().st_size != end:
        raise ValueError(f"Wrong packed weight length for rank {rank}")
    write_json(root / f"rank-{rank}.json", record)
    print(f"PREFLIGHT weights rank={rank} tensors={len(expected)} bytes={end}", flush=True)


def load_rank(root, source, model, rank):
    import mmap

    import torch

    from pithtrain.modules import hf_loader
    from pithtrain.modules.hf_converters.qwen35 import Qwen35Converter

    record = read_json(root / f"rank-{rank}.json")
    source_keys = source_index(source)
    keys = {key.removeprefix("model."): key for key in source_keys}
    converter = Qwen35Converter()
    if not converter.detect_hf(source):
        raise ValueError(f"Unsupported HF checkpoint at {source}")
    plan, _ = hf_loader._plan_copies(
        model, {key: source_keys[name] for key, name in keys.items()}, converter
    )
    destinations = {op.local_fqn: op for op in plan}
    if len(destinations) != len(plan) or len(plan) != len(record["records"]):
        raise ValueError("Packed weights do not match the live FSDP model")
    for item in record["records"]:
        op = destinations[item["name"]]
        if (
            op.transform is not None
            or not op.dst.is_cuda
            or op.dst.dtype != torch.float32
            or op.shard_files != (item["shard"],)
            or keys[op.hf_keys[0]] != item["source_key"]
            or list(op.dst.shape) != item["shape"]
            or [op.row_slice.start, op.row_slice.stop] != item["row"]
        ):
            raise ValueError(f"Packed weights disagree with live FSDP slice {op.local_fqn}")

    path = root / f"rank-{rank}.bin"
    if path.stat().st_size != record["bytes"] or record["rank"] != rank:
        raise ValueError(f"Incomplete packed weights for rank {rank}")
    dtype = {"BF16": torch.bfloat16, "F32": torch.float32}
    records = record["records"]
    with (
        torch.no_grad(),
        path.open("rb") as file,
        mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY) as packed,
    ):
        packed.madvise(mmap.MADV_SEQUENTIAL)
        first = 0
        while first < len(records):
            start = records[first]["offset"]
            end = first + 1
            while (
                end < len(records)
                and records[end]["offset"] + records[end]["bytes"] - start <= 128 << 20
            ):
                end += 1
            stop = records[end - 1]["offset"] + records[end - 1]["bytes"]
            host = torch.frombuffer(packed, dtype=torch.uint8, count=stop - start, offset=start)
            slab = host.to(device=rank, non_blocking=True)
            for item in records[first:end]:
                offset = item["offset"] - start
                value = slab[offset : offset + item["bytes"]].view(dtype[item["dtype"]])
                destinations[item["name"]].dst.copy_(value.view(item["shape"]), non_blocking=True)
            del host, slab, value
            first = end
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.distributed.barrier()


def verify_rank(root, rank, *, payload=True):
    record = read_json(root / f"rank-{rank}.json")
    data = root / f"rank-{rank}.bin"
    if (
        record["rank"] != rank
        or data.stat().st_size != record["bytes"]
        or (payload and digest(data) != record["sha256"])
    ):
        raise ValueError(f"Invalid weight bundle for rank {rank}")
