"""Prepare and verify a dense FP8 startup bundle in a CPU Slurm step."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

from tools.fp8_preflight.artifacts import (
    CuteArtifacts,
    digest,
    read_json,
    signature,
    verify_wrapper,
    write_json,
)
from tools.fp8_preflight.deepgemm import compile_deepgemm
from tools.fp8_preflight.imports import build_import_archive
from tools.fp8_preflight.recipe import (
    ARCH,
    PROJECTIONS,
    ROWS,
    SMEM_PER_SM,
    SMS,
    WORLD_SIZE,
    compile_cute,
    offline_target,
)
from tools.fp8_preflight.weights import (
    model_geometry,
    plans,
    prepare_rank,
    source_index,
    verify_rank,
    verify_records,
)

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "benchmarks/end2end/qwen3.5-27b-seq4096-bf16/h200-1n8g/fsdp2/script.py"


def fingerprint(files, root):
    result = hashlib.sha256()
    for path in sorted(files):
        result.update(str(path.relative_to(root)).encode() + b"\0")
        result.update(bytes.fromhex(digest(path)))
    return result.hexdigest()


def source_fingerprint():
    source_dirs = [REPO / "pithtrain", REPO / "tools/fp8_preflight"]
    files = [BENCHMARK, REPO / "examples/pretrain_lm/launch.sh"]
    for directory in source_dirs:
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    return fingerprint(files, REPO)


def kernel_identity(args, package):
    cuda = Path(os.environ["CUDA_HOME"])
    (extension,) = package.glob("_C*.so")
    deep_jit = Path(os.environ["DEEP_JIT_INCLUDE"]).resolve()
    distributions = (
        "torch",
        "nvidia-cutlass-dsl",
        "apache-tvm-ffi",
        "deep-gemm",
        "quack-kernels",
        "flash-attn-4",
        "triton",
        "transformers",
    )
    deps = {name: metadata.version(name) for name in distributions}
    return {
        "version": 1,
        "source": source_fingerprint(),
        "arch": ARCH,
        "sm_count": SMS,
        "shared_memory_per_sm": SMEM_PER_SM,
        "world_size": WORLD_SIZE,
        "rows": ROWS,
        "python": sys.version,
        "dependencies": deps,
        "deepgemm_extension": digest(extension),
        "deep_jit_headers": fingerprint(
            (p for p in deep_jit.rglob("*.hpp") if p.is_file()), deep_jit
        ),
        "deepgemm_headers": fingerprint(
            (p for p in (package / "csrc").rglob("*") if p.is_file()), package
        ),
        "cuda": str(cuda.resolve()),
        "nvcc": digest(cuda / "bin/nvcc"),
        "ptxas": digest(cuda / "bin/ptxas"),
        "cxx": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "model_config": digest(args.model / "config.json"),
    }


def weight_identity(args):
    shards = sorted(set(source_index(args.weights).values()))
    checkpoint = {}
    for name in shards:
        st = (args.weights / name).stat()
        checkpoint[name] = [st.st_size, st.st_mtime_ns, st.st_ino]
    return {
        "schema": 1,
        "model_config": digest(args.model / "config.json"),
        "weights": str(args.weights),
        "weights_index": digest(args.weights / "model.safetensors.index.json"),
        "checkpoint": checkpoint,
        "converter": fingerprint(
            [
                REPO / "pithtrain/modules/hf_loader.py",
                REPO / "pithtrain/modules/hf_converters/qwen35.py",
                REPO / "tools/fp8_preflight/weights.py",
            ],
            REPO,
        ),
    }


def plan_cute(config, model):
    wanted = {}
    entered = set()

    def record(operation, *args, **kwargs):
        key, description = signature(operation, args, kwargs)
        wanted[key] = (description, operation, args, kwargs)
        return verify_wrapper(operation, args, kwargs, lambda: entered.add(key))

    compile_cute(record, config, model)
    if set(wanted) != entered:
        raise ValueError(f"CuTe wrappers did not call all planned artifacts: {sorted(set(wanted) - entered)}")
    names = Counter(plan[0][0][1].rsplit(".", 1)[-1] for plan in wanted.values())
    required = {
        "WeightGradient": len(PROJECTIONS),
        "ShardAmaxMulti": 1,
        "ScalesMulti": 1,
        "WyRepresentation": 1,
        "ChunkStateOutput": 1,
        "launch_forward_attention": 1,
        "CrossEntropy": 1,
    }
    missing = {name: count for name, count in required.items() if names[name] < count}
    gathered = {
        operation.load_bytes
        for _, operation, _, _ in wanted.values()
        if type(operation).__name__ == "PackTranspose" and operation.mode == "gathered"
    }
    if missing or gathered != {4, 8, 16}:
        raise ValueError(f"Incomplete CuTe fixture coverage, {missing=}, {gathered=}")
    print(f"PREFLIGHT cute_plan={len(wanted)} gathered={sorted(gathered)}", flush=True)
    return wanted


def verify_cute(root, entries, wanted):
    if set(wanted) != set(entries):
        raise ValueError(
            f"CuTe coverage mismatch, missing {sorted(set(wanted) - set(entries))}, "
            f"extra {sorted(set(entries) - set(wanted))}"
        )
    for key, entry in entries.items():
        path = root / entry["path"]
        if entry["signature"] != wanted[key][0] or path.resolve().parent != root / "cute":
            raise ValueError(f"Invalid CuTe entry {key}")
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Corrupt CuTe artifact {path}")


def verify_kernels(root, seal, recipe, wanted):
    if seal["identity"] != recipe or digest(root / "imports.zip") != seal["imports"]:
        raise ValueError("Preflight identity or import archive mismatch")
    verify_cute(root, seal["cute"], wanted)
    verify_deepgemm(root, seal["deepgemm"])


def verify_deepgemm(root, entries):
    shapes = set(PROJECTIONS) | {(k, n) for n, k in PROJECTIONS}
    if set(entries) != {f"{n},{k}" for n, k in shapes}:
        raise ValueError("Incomplete DeepGEMM coverage")
    for entry in entries.values():
        folder = root / "deepgemm/cache" / entry["key"]
        if not (folder / ".committed").is_file():
            raise ValueError(f"Uncommitted DeepGEMM artifact at {folder}")
        for name in ("kernel.cu", "kernel.cubin"):
            if digest(folder / name) != entry[name]:
                raise ValueError(f"Corrupt DeepGEMM artifact at {folder / name}")


def build_kernels(root, args, recipe, wanted, package):
    from torch._inductor import codecache, compile_fx  # noqa: F401

    from pithtrain.tasks import pretrain_lm  # noqa: F401

    cute = publish_stage(
        args.cache / "cute",
        recipe,
        lambda stage: build_cute(stage, wanted),
        lambda stage, entries, fresh: verify_cute(stage, entries, wanted),
    )
    deepgemm = prepare_deepgemm(args, recipe, package)
    shutil.copytree(cute / "cute", root / "cute", copy_function=os.link)
    shutil.copytree(deepgemm / "deepgemm", root / "deepgemm")
    build_import_archive(root / "imports.zip")
    return {
        "identity": recipe,
        "cute": read_json(cute / "seal.json"),
        "deepgemm": read_json(deepgemm / "seal.json"),
        "imports": digest(root / "imports.zip"),
    }


def build_cute(root, wanted):
    entries = {}
    artifacts = CuteArtifacts(root, entries, build=True)
    for key, (description, operation, operands, options) in wanted.items():
        artifacts.export(key, description, operation, operands, options)
    return entries


def prepare_deepgemm(args, recipe, package):
    key_fields = (
        "arch",
        "sm_count",
        "rows",
        "cuda",
        "nvcc",
        "cxx",
        "python",
        "deepgemm_extension",
        "deepgemm_headers",
        "deep_jit_headers",
    )
    identity = {field: recipe[field] for field in key_fields}
    identity["projections"] = PROJECTIONS
    identity["builder"] = fingerprint(
        [Path(__file__).with_name(name) for name in ("deepgemm.cpp", "deepgemm.py")],
        REPO,
    )
    return publish_stage(
        args.cache / "deepgemm",
        identity,
        lambda root: compile_deepgemm(root, package, args.deepgemm_seed),
        lambda root, entries, fresh: verify_deepgemm(root, entries),
    )


def build_weights(root, args, recipe, model):
    def prepare(rank):
        prepare_rank(root, args.weights, model, rank, args.weight_seed)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare, range(WORLD_SIZE)))
    return {
        "identity": recipe,
        "weights": {str(rank): digest(root / f"rank-{rank}.json") for rank in range(WORLD_SIZE)},
    }


def verify_weights(root, seal, recipe, source, model, *, payload):
    if seal["identity"] != recipe:
        raise ValueError("Preflight weight identity mismatch")

    def verify(rank):
        if digest(root / f"rank-{rank}.json") != seal["weights"][str(rank)]:
            raise ValueError(f"Rank {rank} weight plan mismatch")
        record = read_json(root / f"rank-{rank}.json")
        if verify_records(record["records"], plans(source, model, rank)) != record["bytes"]:
            raise ValueError(f"Incomplete packed slices for rank {rank}")
        verify_rank(root, rank, payload=payload)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, range(WORLD_SIZE)))


def publish_stage(cache, identity, build, verify):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    bundle = cache / key
    with (cache / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if bundle.exists():
            verify(bundle, read_json(bundle / "seal.json"), fresh=False)
            print(f"PREFLIGHT reused stage={bundle}", flush=True)
        else:
            stage = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=cache))
            try:
                seal = build(stage)
                verify(stage, seal, fresh=True)
                write_json(stage / "seal.json", seal)
                stage.rename(bundle)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            print(f"PREFLIGHT sealed stage={bundle}", flush=True)
    return bundle


def prepare_kernels(args):
    import deep_gemm

    package = Path(deep_gemm.__file__).resolve().parent
    recipe = kernel_identity(args, package)
    config, model = model_geometry(args.model)
    wanted = plan_cute(config, model)
    return publish_stage(
        args.cache / "kernels",
        recipe,
        lambda root: build_kernels(root, args, recipe, wanted, package),
        lambda root, seal, fresh: verify_kernels(root, seal, recipe, wanted),
    )


def prepare_weights_and_seal(args):
    kernel = Path(args.kernels_ready.read_text().strip())
    kernel_seal = read_json(kernel / "seal.json")
    recipe = weight_identity(args)
    target = kernel_seal["identity"]
    if target["model_config"] != recipe["model_config"] or target["source"] != source_fingerprint():
        raise ValueError("The model or source changed after the kernel seal")
    config, model = model_geometry(args.model)
    weights = publish_stage(
        args.cache / "weights",
        recipe,
        lambda root: build_weights(root, args, recipe, model),
        lambda root, seal, fresh: verify_weights(
            root, seal, recipe, args.weights, model, payload=not fresh or args.weight_seed is None
        ),
    )
    prepare_dataset(args)
    bundle = args.ready.parent / "bundle"
    bundle.mkdir()
    config_path = bundle / "model-config.json"
    write_json(config_path, config.to_dict())
    for name in ("cute", "imports.zip"):
        (bundle / name).symlink_to(kernel / name)
    shutil.copytree(kernel / "deepgemm", bundle / "deepgemm")
    for rank in range(WORLD_SIZE):
        for ext in ("bin", "json"):
            name = f"rank-{rank}.{ext}"
            (bundle / name).symlink_to(weights / name)
    write_json(
        bundle / "seal.json",
        {
            **kernel_seal,
            "identity": {**kernel_seal["identity"], **recipe},
            "prepared_model_config": digest(config_path),
            "weights": read_json(weights / "seal.json")["weights"],
        },
    )
    return bundle


def prepare_dataset(args):
    import numpy as np

    from tools.fp8_preflight.recipe import SEQUENCE

    files = sorted(args.dataset.rglob("*.bin"))
    lengths = [max(0, (len(np.load(file, mmap_mode="r")) - 1) // SEQUENCE) for file in files]
    if sum(lengths) < args.steps * WORLD_SIZE * (ROWS // SEQUENCE):
        raise ValueError("Dataset does not contain enough samples for the requested steps")
    args.staged_dataset.mkdir()
    for file in files:
        target = args.staged_dataset / file.relative_to(args.dataset)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(file.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("kernels", "weights"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--staged-dataset", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--kernels-ready", type=Path)
    parser.add_argument("--deepgemm-seed", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--weight-seed", type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    args = parser.parse_args()
    args.model, args.weights, args.cache = (
        path.resolve() for path in (args.model, args.weights, args.cache)
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("CUTE_DSL_ARCH") != ARCH:
        parser.error(f"CPU preflight requires CUDA_VISIBLE_DEVICES='' and CUTE_DSL_ARCH={ARCH}")
    if args.phase == "weights" and not all(
        (args.kernels_ready, args.dataset, args.staged_dataset, args.steps)
    ):
        parser.error("weights requires --kernels-ready, --dataset, --staged-dataset and --steps")
    with offline_target():
        bundle = (
            prepare_kernels(args) if args.phase == "kernels" else prepare_weights_and_seal(args)
        )
    temporary = args.ready.with_suffix(".pending")
    temporary.write_text(str(bundle) + "\n")
    temporary.replace(args.ready)
    print(f"PREFLIGHT verified bundle={bundle}", flush=True)


if __name__ == "__main__":
    main()
