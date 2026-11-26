"""Verify the sealed dense recipe's ABI on CPU and its actual launches on one H200."""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

from tools.fp8_preflight.artifacts import (
    CuteArtifacts,
    digest,
    read_json,
    signature,
    verify_wrapper,
)
from tools.fp8_preflight.imports import ImportArchive
from tools.fp8_preflight.recipe import (
    PROJECTIONS,
    ROWS,
    SMEM_PER_SM,
    SMS,
    compile_cute,
    exercise_cute,
    offline_target,
    shared_memory_attribute,
)
from tools.fp8_preflight.weights import model_geometry


@contextmanager
def measure(label):
    start = time.monotonic()
    yield
    print(f"ARTIFACT_CHECK {label} seconds={time.monotonic() - start:.3f}", flush=True)


@contextmanager
def sample_gpu(path):
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,uuid,utilization.gpu,power.draw,memory.used",
        "--format=csv,nounits",
        "--loop-ms=200",
    ]
    devices = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
    if devices:
        command.append(f"--id={devices}")
    with path.open("x") as output:
        sampler = subprocess.Popen(command, stdout=output, env={**os.environ, "TZ": "UTC"})
        try:
            yield
        finally:
            if sampler.poll() is None:
                sampler.terminate()
            sampler.wait()
    with path.open() as output:
        if not output.readline() or not output.readline():
            raise RuntimeError(f"GPU utilization sampler produced no samples in {path}")


def coverage(found, wanted):
    if set(found) != set(wanted):
        raise RuntimeError(
            f"Artifact coverage mismatch, missing {sorted(set(wanted) - set(found))}, "
            f"extra {sorted(set(found) - set(wanted))}"
        )
    if any(found[key] == 0 for key in wanted):
        raise RuntimeError("Some artifacts were selected without being called")


def check_abi(root, seal, config, model):
    from cutlass.runtime import load_module

    calls = Counter()

    def require(operation, *args, **kwargs):
        key, description = signature(operation, args, kwargs)
        entry = seal["cute"][key]
        if entry["signature"] != description or digest(root / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"CuTe binary or compile signature mismatch for {key}")
        module = load_module(str(root / entry["path"]), enable_tvm_ffi=True)
        if not callable(module.preflight_entry):
            raise RuntimeError(f"Missing exported function in {key}")
        def launched():
            calls[key] += 1

        return verify_wrapper(operation, args, kwargs, launched)

    compile_cute(require, config, model)
    coverage(calls, seal["cute"])
    print(
        f"ARTIFACT_CHECK abi entries={len(calls)} real_wrapper_calls={sum(calls.values())}",
        flush=True,
    )


def check_cuda(root, seal, config, model):
    import torch
    from cutlass import cute
    from cutlass.base_dsl.runtime import cuda as cuda_helpers

    from pithtrain.kernels import fp8_gemm_nt

    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor, props.multi_processor_count) != (9, 0, SMS):
        raise RuntimeError(f"Expected one H200, got {props}")
    if cuda_helpers.get_device_attribute(shared_memory_attribute(cuda_helpers), 0) != SMEM_PER_SM:
        raise RuntimeError("Live device does not match the sealed shared-memory target")
    artifacts = CuteArtifacts(root, seal["cute"], build=False)
    calls = Counter()

    def record(operation, *args, **kwargs):
        key, _ = signature(operation, args, kwargs)
        fn = artifacts(operation, *args, **kwargs)

        def launch(*runtime_args, **runtime_kwargs):
            result = fn(*runtime_args, **runtime_kwargs)
            calls[key] += 1
            return result

        return launch

    cute.compile = record
    with torch.device("cuda:0"), measure("cute_launches"):
        exercise_cute(config, model)
        torch.cuda.synchronize()
    coverage(calls, seal["cute"])
    print(
        f"ARTIFACT_CHECK cute launched={len(calls)} real_wrapper_calls={sum(calls.values())}",
        flush=True,
    )

    shapes = set(PROJECTIONS) | {(k, n) for n, k in PROJECTIONS}
    if set(seal["deepgemm"]) != {f"{n},{k}" for n, k in shapes}:
        raise RuntimeError("DeepGEMM launch fixtures do not match the seal")
    with torch.device("cuda:0"), measure("deepgemm_launches"):
        for n, k in sorted(shapes):
            a = torch.zeros((ROWS, k), dtype=torch.uint8).view(torch.float8_e4m3fn)
            b = torch.zeros((n, k), dtype=torch.uint8).view(torch.float8_e4m3fn)
            a_scale = torch.empty_strided((ROWS, k // 128), (1, ROWS), dtype=torch.float32).fill_(1)
            b_scale = torch.ones((n // 128, k // 128), dtype=torch.float32)
            fp8_gemm_nt(a, a_scale, b, b_scale, ROWS, n, a.device, torch.bfloat16)
            print(f"ARTIFACT_CHECK deepgemm launched n={n} k={k}", flush=True)
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("abi", "cuda"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--preload-imports", action="store_true")
    parser.add_argument("--gpu-utilization", type=Path)
    args = parser.parse_args()
    if args.mode == "cuda" and args.gpu_utilization is None:
        parser.error("cuda mode requires --gpu-utilization")
    sampler = sample_gpu(args.gpu_utilization) if args.mode == "cuda" else nullcontext()
    with sampler:
        run(args)


def run(args):
    seal = read_json(args.bundle / "seal.json")
    with measure("import_archive"):
        sys.meta_path.insert(
            0, ImportArchive(args.bundle / "imports.zip", preload=args.preload_imports)
        )
    load_library = ctypes.CDLL.__init__

    def timed_library(library, name, *args, **kwargs):
        started = time.monotonic()
        load_library(library, name, *args, **kwargs)
        elapsed = time.monotonic() - started
        if elapsed >= 0.25:
            print(f"ARTIFACT_CHECK dlopen name={name} seconds={elapsed:.3f}", flush=True)

    with measure("torch_import"), patch.object(ctypes.CDLL, "__init__", timed_library):
        import torch

    if args.mode == "abi":
        with offline_target(), measure("abi"):
            config, model = model_geometry(args.model)
            check_abi(args.bundle, seal, config, model)
    else:
        with measure("cuda_device"):
            torch.cuda.set_device(0)
        with measure("pithtrain_import"):
            from cutlass import cute  # noqa: F401
            from torch._inductor import compile_fx  # noqa: F401

            from pithtrain.modules import training  # noqa: F401
            from pithtrain.tasks import pretrain_lm  # noqa: F401

        with measure("model_geometry"):
            config, model = model_geometry(args.model)
        check_cuda(args.bundle, seal, config, model)


if __name__ == "__main__":
    main()
