#!/usr/bin/env python3
"""Build, run and adjudicate the SM90 FP8 wgrad publication gate."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import runpy
import shlex
import socket
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from functools import partial
from importlib import import_module, metadata
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

from tools.fp8_preflight.artifacts import CuteArtifacts, digest, read_json, write_json
from tools.fp8_preflight.deepgemm import build_host_compiler
from tools.fp8_preflight.recipe import SMS, offline_target

REPO = Path(__file__).resolve().parents[2]
ENTRY = Path(__file__).resolve()
SEED = 2315
SMOKE = (128, 128, 128)
PATCHED_REFERENCE = "d7c91af681dc6b366a2ebc9190c525b93e9b5991a0a6af0f62b3677e9d4d8f57"
PAIRS = (("ref", "pin"), ("repeat", "ref"), ("pin", "ref"), ("ref", "repeat"))
EDGES = [
    {"shape": [3, 40, 144]},
    {"shape": [71, 280, 384]},
    {"shape": [1, 8, 145], "pad": True},
    {"shape": [17, 136, 145], "pad": True},
    {"shape": [17, 136, 145]},
    {"shape": [3, 13, 144]},
    {"shape": [128, 128, 0]},
    {"shape": [0, 128, 128]},
    {"shape": [17, 136, 144], "offset": "scales"},
    {"shape": [17, 24, 144], "offset": "fp8"},
    {"shape": [17, 24, 144], "offset": "inner"},
    {"shape": [17, 24, 144], "offset": "output"},
    {"shape": [17, 24, 144], "offset": "pitch"},
    {"shape": [17, 24, 144], "offset": "aligned"},
]
HOST_MAIN = r"""
int main(int argc, char** argv) {
    using namespace deep_gemm;
    if (argc != 3) return 2;
    init_jit(argv[1]);
    std::ifstream requests(argv[2]);
    if (!requests) return 2;
    std::string tag, template_path, mode;
    int m, n, k;
    while (requests >> tag >> template_path >> mode >> m >> n >> k) {
        const bool accumulate = mode == "stock";
        const bool one_d = accumulate || mode == "row";
        const bool bf16 = mode == "bf16" || mode == "bf16-mnk";
        const int cm = mode == "bf16-mnk" ? m : 0;
        const GemmDesc desc{
            .gemm_type = GemmType::Normal,
            .kernel_type = one_d ? KernelType::Kernel1D1D : KernelType::Kernel1D2D,
            .m = m, .n = n, .k = k, .num_groups = 1,
            .a_dtype = torch::kFloat8_e4m3fn, .b_dtype = torch::kFloat8_e4m3fn,
            .cd_dtype = bf16 ? torch::kBFloat16 : torch::kFloat,
            .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
            .with_accumulation = one_d, .num_sms = 132, .tc_util = 100,
            .compiled_dims = cm ? "mnk" : "nk",
        };
        const auto config = get_best_config<SM90ArchSpec>(desc);
        std::ifstream input(template_path);
        if (!input) return 2;
        const std::string format{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const auto& l = config.layout;
        const auto& s = config.storage_config;
        const auto& p = config.pipeline_config;
        const auto& t = config.launch_config;
        const auto render = [&](auto&&... args) {
            return std::vformat(format, std::make_format_args(args...));
        };
        const char* dtype = bf16 ? "cutlass::bfloat16_t" : "float";
        const auto code = one_d
            ? render(cm,n,k,1,l.block_m,l.block_n,l.block_k,s.swizzle_a_mode,s.swizzle_b_mode,
                     p.num_stages,t.num_tma_threads,t.num_math_threads,
                     l.get_cluster_size(),l.cluster_n > 1,t.num_sms,"GemmType::Normal",dtype,accumulate)
            : render("cute::UMMA::Major::K",cm,n,k,1,l.block_m,l.block_n,l.block_k,
                     s.swizzle_a_mode,s.swizzle_b_mode,s.swizzle_cd_mode,p.num_stages,
                     t.num_tma_threads,t.num_math_threads,l.get_cluster_size(),l.cluster_n > 1,
                     t.num_sms,"GemmType::Normal",dtype);
        auto path = jit->compile_without_load(tag, code);
        std::cout << "GATE " << l.block_m << " " << l.block_n << " " << l.block_k << " "
            << p.num_stages << " " << l.cluster_m << " " << l.cluster_n << " "
            << s.swizzle_a_mode << " " << s.swizzle_b_mode << " " << s.swizzle_cd_mode << " "
            << t.num_sms << " " << path.string() << std::endl;
    }
}
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def shape(row):
    return tuple(row[axis] for axis in "MNK")


def timing_scope(rows, changed):
    """Cover every nominated witness and its closest other production geometry."""
    lookup = {row["id"]: row for row in rows}
    if changed == "dispatch":
        # The eligibility check wraps every selector. Include the tuned dense
        # specializations that share a DeepGEMM selector with other geometries.
        representatives = {}
        for row in rows:
            representatives.setdefault(tuple(row["expected_signature"]), row["id"])
        affected = set(representatives.values()) | {4, 7, 9, 13, 20, 21, 35}
    else:
        affected = set(map(int, changed.split(",")))
    require(affected and affected <= lookup.keys(), f"Invalid timing case IDs: {affected}")
    selected = affected.copy()
    for case in affected:
        target = shape(lookup[case])
        neighbors = [row for row in rows if row["id"] != case]
        neighbor = min(
            neighbors,
            key=lambda row: (
                sum(abs(math.log2(a / b)) for a, b in zip(shape(row), target, strict=True)),
                row["id"],
            ),
        )
        selected.add(neighbor["id"])
    return {"affected": sorted(affected), "cases": sorted(selected)}


def ceildiv(x, divisor):
    return (x + divisor - 1) // divisor


def edge_cases():
    return [{**edge, "rowwise": rowwise} for edge in EDGES for rowwise in (False, True)] + [
        {"shape": [17, 24, 144], "sms": sms} for sms in range(1, SMS + 1)
    ]


def scratch_tensor(size, pitch, offset, device, dtype):
    import torch

    rows, cols = size
    host = torch.empty(rows * pitch + offset + 16, device=device, dtype=dtype)
    return host[offset:].as_strided(size, (pitch, 1))


def geometry(case, device, *, populate):
    import torch

    m, n, k = case["shape"]
    rowwise = case.get("rowwise", False)
    offset = case.get("offset")
    blocks = ceildiv(k, 128)
    pitch = max(1, ceildiv(k, 16) * 16 if case.get("pad") else k)
    gen = torch.Generator(device=device).manual_seed(17) if populate else None
    codes = []
    for idx, rows in enumerate((m, n)):
        if offset == "inner" and idx == 0:
            tensor = torch.empty((rows, 2 * k), device=device, dtype=torch.float8_e4m3fn)[:, ::2]
        else:
            shift = 16 if offset == "aligned" else int(offset == "fp8" and idx == 0)
            tensor = scratch_tensor((rows, k), pitch, shift, device, torch.float8_e4m3fn)
        if populate:
            values = torch.randint(-8, 9, (rows, k), device=device, generator=gen)
            tensor.copy_(values.to(tensor.dtype))
        codes.append(tensor)
    scales = []
    for idx, rows in enumerate((m, n if rowwise else ceildiv(n, 128))):
        if offset == "scales" and (idx == 0 or rowwise):
            stride = ceildiv(rows, 4) * 4
            storage = torch.empty(blocks * stride + 1, device=device)
            scale = storage[1:].as_strided((rows, blocks), (1, stride))
        else:
            scale = torch.empty((rows, blocks), device=device)
        if populate:
            exponents = torch.randint(-4, 5, scale.shape, device=device, generator=gen)
            scale.copy_(exponents.float().exp2())
        scales.append(scale)
    stride = n + {"output": 4, "pitch": 1}.get(offset, 0)
    out = scratch_tensor((m, n), max(1, stride), int(offset == "output"), device, torch.float32)
    return (codes[0], scales[0], codes[1], scales[1]), out


@contextmanager
def cpu_geometry():
    import torch

    stream = SimpleNamespace(cuda_stream=0)
    with (
        patch.object(torch.Tensor, "is_cuda", property(lambda t: t.device.type == "meta")),
        patch.object(torch.Tensor, "data_ptr", lambda t: t.storage_offset() * t.element_size()),
        patch.object(torch.cuda, "device", return_value=nullcontext()),
        patch.object(torch.cuda, "current_stream", return_value=stream),
        torch.device("meta"),
    ):
        yield


def dense_geometry(row):
    return {"shape": shape(row)}


def dense_recipe_ops(row):
    import torch

    from pithtrain.kernels import fp8_quant_act, fp8_quant_grad

    m, n, k = shape(row)
    fp8_quant_act(torch.empty((k, n), dtype=torch.bfloat16))
    fp8_quant_grad(torch.empty((k, m), dtype=torch.bfloat16))


def compile_cute(root, rows):
    import torch
    from cutlass import cute

    from pithtrain.kernels import (
        fp8_quant_act,
        fp8_quant_delta_gate,
        fp8_quant_grad,
        fp8_quant_swiglu,
        fp8_wgrad,
    )

    entries = {}
    with patch.object(cute, "compile", CuteArtifacts(root, entries, build=True)), cpu_geometry():
        for row in rows:
            dense_recipe_ops(row)
            data, dest = geometry(dense_geometry(row), "meta", populate=False)
            fp8_wgrad(*data, dest)
        for case in edge_cases():
            sm = case.get("sms", SMS)
            data, dest = geometry(case, "meta", populate=False)
            # The fallback contains no CuTe launch.
            with patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=sm),
            ):
                fp8_wgrad(*data, dest)
        x = torch.empty((128, 128), dtype=torch.bfloat16)
        quant = import_module("pithtrain.kernels.fp8_weight_quant_sm90")
        for dtype in (torch.bfloat16, torch.float32):
            quant.resolve_blockwise_kernel(dtype, x.shape, x.stride(), 0)
        fp8_quant_act(x)
        fp8_quant_grad(x)
        data, dest = geometry({"shape": SMOKE}, "meta", populate=False)
        fp8_wgrad(*data, dest)
        wide = torch.empty((256, 256), dtype=torch.bfloat16)
        fp8_quant_act(wide)
        fp8_quant_swiglu(wide, wide)
        fp8_quant_delta_gate(wide, wide.new_empty(128), wide.new_empty((256, 2)), wide)
    return entries


def native_requests(rows, kind, stock):
    if kind == "stock":
        return [(stock, shape(row)) for row in rows] + [
            (
                "bf16-mnk" if k == 16384 and (n, m) in ((17408, 5120), (5120, 17408)) else "bf16",
                (k, n, m),
            )
            for m, n, k in map(shape, rows)
        ]
    requests = [("block", shape(row)) for row in rows]
    for edge in edge_cases():
        m, n, k = edge["shape"]
        if m and n and k:
            requests.append(("row" if edge.get("rowwise") else "block", (m, ceildiv(n, 8) * 8, k)))
    return requests + [("block", SMOKE), ("bf16", SMOKE)]


def build_native(root, package, requests):
    root.mkdir()
    cpu_source = root / "host.cpp"
    template = (REPO / "tools/fp8_preflight/deepgemm.cpp").read_text().split("int main(")[0]
    cpu_source.write_text(template + HOST_MAIN)
    compiler = build_host_compiler(root, package, cpu_source)
    request_file = root / "requests.txt"
    templates = {}
    original_format_args = {}
    lines = []
    unique = list(dict.fromkeys(requests))
    for mode, (m, n, k) in unique:
        flavor = "1d1d" if mode in ("row", "stock") else "1d2d"
        tag = f"sm90_fp8_gemm_{flavor}"
        if flavor not in templates:
            header = package / f"csrc/jit_kernels/impls/{tag}.hpp"
            match = re.search(
                r'jit->compile\(tag, std::format\(R"\((.*?)\n\)",(.*?)\)\);',
                header.read_text(),
                re.S,
            )
            if match is None:
                raise ValueError(f"Unsupported DeepGEMM JIT template {header}")
            path = root / f"{flavor}.txt"
            path.write_text(match[1] + "\n")
            templates[flavor] = path
            original_format_args[flavor] = match[2]
        lines.append(f"{tag} {templates[flavor]} {mode} {m} {n} {k}\n")
    # Some pre-port versions hard-code accumulation in the kernel template.
    if "1d1d" in templates and "args.accumulate" not in original_format_args["1d1d"]:
        require(
            not any(mode == "row" for mode, _ in unique),
            "The patched reference lacks a fresh FP32 row-scale arm",
        )
    request_file.write_text("".join(lines))
    cache = root / "jit"
    env = {
        **os.environ,
        "DG_JIT_CACHE_DIR": str(cache),
        "DJ_JIT_CACHE_DIR": str(cache),
        "DG_JIT_NVCC_COMPILER": str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"),
    }
    process = subprocess.Popen(
        [str(compiler), str(package), str(request_file)], env=env, stdout=subprocess.PIPE, text=True
    )
    records = {}
    with process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("GATE "):
                mode, dims = unique[len(records)]
                *signature, directory = line.split()[1:]
                folder = Path(directory)
                require(
                    folder.parent == cache / "cache" and (folder / ".committed").is_file(),
                    f"Incomplete DeepGEMM cache entry {folder}",
                )
                records[(mode, dims)] = {
                    "signature": list(map(int, signature)),
                    "path": str(folder.relative_to(root)),
                    "sha256": digest(folder / "kernel.cubin"),
                }
    require(process.wait() == 0 and len(records) == len(unique), "CPU DeepGEMM compilation failed")
    return records


def prepare(args, plan):
    import pytest
    import torch

    root = args.output
    runtime = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "dependencies": {
            name: metadata.version(name)
            for name in (
                "torch",
                "nvidia-cutlass-dsl",
                "apache-tvm-ffi",
                "triton",
            )
        },
    }
    with offline_target():
        test_status = pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_import_layers.py",
                "tests/test_dense_fp8_wgrad_recipe.py",
            ]
        )
        require(not test_status, f"CPU branch tests failed ({test_status})")
        cute = compile_cute(root, plan["rows"])
    native = {}
    for kind, package in (("patched", args.patched), ("stock", args.stock)):
        if kind == "stock" and package is None:
            continue
        (extension,) = package.glob("_C*.so")
        runtime[kind] = {"extension_sha256": digest(extension)}
        if kind == "patched":
            sources = [*package.glob("*.py"), extension]
            sources.extend(path for path in (package / "include").rglob("*") if path.is_file())
            contents = [(str(path.relative_to(package)), digest(path)) for path in sorted(sources)]
            reference_hash = hashlib.sha256(json.dumps(contents).encode()).hexdigest()
            require(reference_hash == PATCHED_REFERENCE, "Patched DeepGEMM reference changed")
            runtime[kind]["reference_sha256"] = reference_hash
        records = build_native(root / kind, package, native_requests(plan["rows"], kind, "stock"))
        if kind == "patched":
            for row in plan["rows"]:
                require(
                    records["block", shape(row)]["signature"] == row["expected_signature"],
                    f"Frozen DeepGEMM selector changed for matrix case {row['id']}",
                )
        native[kind] = {
            f"{mode}:{','.join(map(str, dims))}": rec for (mode, dims), rec in records.items()
        }
    gen = torch.Generator().manual_seed(42)
    samples = [torch.randn(SMOKE[:2], generator=gen).bfloat16() for _ in range(5)]
    samples[0] /= 16
    torch.save(samples, root / "smoke-input.pt")
    nvcc = root / "version-only-nvcc"
    nvcc.write_text(
        "#!/bin/sh\n"
        'if [ "$#" = 1 ] && [ "$1" = --version ]; then exec "$CUDA_HOME/bin/nvcc" --version; fi\n'
        'echo "Unexpected DeepGEMM compile on a GPU worker" >&2\nexit 86\n'
    )
    nvcc.chmod(0o755)
    write_json(
        root / "seal.json",
        {
            "plan_sha256": digest(root / "plan.json"),
            "cute": cute,
            "native": native,
            "weights_sha256": digest(root / "smoke-input.pt"),
            "compiler_guard_sha256": digest(nvcc),
            "source_sha256": digest(ENTRY),
            "runtime": runtime,
            "inductor": {"mode": "eager", "graphs": []},
        },
    )


def setup_gpu(args, kind):
    root = args.output
    seal = read_json(root / "seal.json")
    for name, checksum in (
        ("plan.json", seal["plan_sha256"]),
        ("smoke-input.pt", seal["weights_sha256"]),
        ("version-only-nvcc", seal["compiler_guard_sha256"]),
    ):
        require(digest(root / name) == checksum, f"Preflight input changed: {name}")
    require(digest(ENTRY) == seal["source_sha256"], "Gate source changed after CPU preflight")
    cache = root / kind / "jit"
    os.environ.update(
        DG_JIT_CACHE_DIR=str(cache),
        DJ_JIT_CACHE_DIR=str(cache),
        DG_JIT_NVCC_COMPILER=str(root / "version-only-nvcc"),
        DG_JIT_USE_NVRTC="0",
        DG_JIT_PRINT_LOAD_TIME="1",
        PITHTRAIN_FP8_PREFLIGHT_STRICT="1",
    )
    import deep_gemm
    import torch
    from cutlass import cute

    package = args.stock if kind == "stock" else args.patched
    require(
        Path(deep_gemm.__file__).resolve().parent == package,
        f"Loaded unexpected DeepGEMM package: {deep_gemm.__file__}",
    )
    runtime = seal["runtime"]
    require(
        sys.version == runtime["python"]
        and str(Path(sys.executable).resolve()) == runtime["executable"]
        and all(
            metadata.version(name) == version for name, version in runtime["dependencies"].items()
        )
        and digest(Path(deep_gemm._C.__file__)) == runtime[kind]["extension_sha256"],
        "Worker runtime differs from CPU preflight",
    )
    prop = torch.cuda.get_device_properties(0)
    require(
        torch.cuda.device_count() == 1
        and prop.multi_processor_count == SMS
        and "H200" in prop.name,
        "Gate requires exactly one full H200",
    )
    cute.compile = CuteArtifacts(root, seal["cute"], build=False)
    deep_gemm.set_num_sms(SMS)
    deep_gemm.set_block_size_multiple_of((1, 1))
    return torch, deep_gemm


def fixture_module(args, torch, dg):
    from pithtrain.kernels import act_scale_like, fp8_quant_act, fp8_quant_grad

    def to_mn_major(scale):
        return act_scale_like(*scale.shape, scale.device).copy_(scale)

    path = args.evaluator
    sys.path.insert(0, str(path.parent))
    fixture = SimpleNamespace(**runpy.run_path(str(path), run_name="fp8_gate_fixture"))
    for function in (fixture.fixture, fixture.operands):
        function.__globals__.update(
            torch=torch,
            deep_gemm=dg,
            to_mn_major=to_mn_major,
            fp8_quant_act=fp8_quant_act,
            fp8_quant_grad=fp8_quant_grad,
        )
    return fixture


def exact(actual, expected, label):
    import torch

    require(
        torch.equal(actual.view(torch.int32), expected.view(torch.int32)),
        f"FP32 bits differ: {label}",
    )


def reference_launch(dg, data, dest, rowwise=False):
    grad, grad_scale, activation, activation_scale = data
    dg.fp8_fp4_gemm_nt(
        (grad, grad_scale),
        (activation, activation_scale),
        dest,
        recipe=(1, 1 if rowwise else 128, 128),
    )


def dense_launches(torch, dg, fixture, row, *, edge=False):
    from pithtrain.kernels import fp8_wgrad

    data = fixture.operands(
        fixture.fixture(shape(row), SEED, edge, False, "recipe"), "block", SEED, edge=edge
    )
    dest = torch.empty(shape(row)[:2], device="cuda", dtype=torch.float32)
    return data, dest, partial(reference_launch, dg, data, dest), partial(fp8_wgrad, *data, dest)


def check_pair(torch, dest, reference, candidate, label):
    dest.fill_(float("nan"))
    reference()
    expect = dest.clone()
    require(torch.isfinite(expect).all().item(), f"Invalid reference output for {label}")
    dest.fill_(float("nan"))
    candidate()
    exact(dest, expect, label)


def dense_exact(args, plan, torch, dg):
    fixture = fixture_module(args, torch, dg)
    results = []
    for row in plan["rows"]:
        for edge in (False, True):
            _, dest, ref, pin = dense_launches(torch, dg, fixture, row, edge=edge)
            check_pair(torch, dest, ref, pin, f"case {row['id']} edge={edge}")
        results.append(row["id"])
        print(f"Exact dense case {row['id']}", flush=True)
    edges = check_edges(torch, dg, edge_cases())
    return {"dense": results, "edges": edges}


def legal_reference(torch, dg, data, rowwise, shape):
    m, n, k = shape
    if not m or not n or not k:
        return torch.zeros((m, n), device="cuda", dtype=torch.float32)
    grad, gs, act, acs = data
    padded_n = ceildiv(n, 8) * 8
    # DeepGEMM requires aligned physical rows and an N divisible by eight.
    # Extend only physical pitch and unused output columns.
    pitch = ceildiv(k, 16) * 16
    aligned_grad = scratch_tensor((m, k), pitch, 0, "cuda", grad.dtype)
    aligned_grad.copy_(grad)
    aligned_act = scratch_tensor((padded_n, k), pitch, 0, "cuda", act.dtype)
    aligned_act.zero_()
    aligned_act[:n].copy_(act)
    if rowwise and padded_n != n:
        aligned_scale = acs.new_zeros((padded_n, acs.shape[1]))
        aligned_scale[:n].copy_(acs)
    else:
        aligned_scale = acs.contiguous()
    output = torch.empty((m, padded_n), device="cuda", dtype=torch.float32)
    reference_launch(
        dg, (aligned_grad, gs.contiguous(), aligned_act, aligned_scale), output, rowwise
    )
    return output[:, :n]


def check_edges(torch, dg, cases, *, caller_stream=False):
    from pithtrain.kernels import fp8_wgrad

    results = []
    for case in cases:
        data, dest = geometry(case, "cuda", populate=True)
        expected = legal_reference(torch, dg, data, case.get("rowwise", False), case["shape"])
        check_guards = case.get("offset") in ("output", "pitch")
        backing = dest._base if check_guards else dest
        backing.fill_(float("nan"))
        sm = case.get("sms", SMS)
        stream = torch.cuda.Stream() if caller_stream else torch.cuda.current_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=sm),
            ):
                fp8_wgrad(*data, dest)
            exact(dest, expected, case)
            if check_guards:
                mask = torch.ones_like(backing, dtype=torch.bool)
                mask.as_strided(dest.shape, dest.stride(), dest.storage_offset()).fill_(False)
                require(backing[mask].isnan().all().item(), f"Output guard changed: {case}")
        torch.cuda.current_stream().wait_stream(stream)
        results.append(case)
    return results


def timing(args, plan, torch, dg, order):
    fixture = fixture_module(args, torch, dg)
    protocol = SimpleNamespace(**runpy.run_path(str(args.protocol), run_name="fp8_gate_protocol"))
    uuid = protocol.device_uuid()
    limits = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=enforced.power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    require(float(limits.strip()) == 700, f"Expected enforced 700 W cap for {uuid}: {limits}")
    results = []
    telemetry = args.output / f"{order}.csv"
    with protocol.sample_gpu(telemetry):
        for row in plan["rows"]:
            if row["id"] not in plan["scope"]["cases"]:
                continue
            _, dest, ref, pin = dense_launches(torch, dg, fixture, row)
            check_pair(torch, dest, ref, pin, row["id"])
            _, dest_edge, ref_edge, pin_edge = dense_launches(torch, dg, fixture, row, edge=True)
            check_pair(torch, dest_edge, ref_edge, pin_edge, f"{row['id']} subnormals")
            graph_ref, graph_pin = [fixture.make_graph(call, 32) for call in (ref, pin)]
            graphs = {"ref": graph_ref, "pin": graph_pin, "repeat": graph_ref}
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

            def epoch(arm, batch, seconds):
                ms, count = protocol.run_epoch(graphs[arm], batch, seconds, torch, start, end)
                return ms, count

            graph_ms = max(epoch(arm, 1, 0)[0] for arm in ("ref", "pin"))
            batch = max(1, min(32, int(30 / max(graph_ms, 0.001))))
            begin = protocol.utc()
            pre_ms, pre_count = epoch("ref", batch, 2.0)
            pre = {"begin": begin, "end": protocol.utc(), "ms": pre_ms, "graphs": pre_count}
            require(pre_ms >= 2000, f"Short preconditioning for case {row['id']}")
            ab, aa, epochs = [], [], []
            for pair in PAIRS:
                times = {}
                for arm in reversed(pair) if order == "reverse" else pair:
                    epoch(arm, batch, 0.25)
                    begin = protocol.utc()
                    elapsed, count = epoch(arm, batch, 0.760)
                    require(elapsed >= 750, f"Short epoch for case {row['id']}")
                    times[arm] = elapsed * 1e3 / (count * 32)
                    epochs.append(
                        {
                            "arm": arm,
                            "begin": begin,
                            "end": protocol.utc(),
                            "ms": elapsed,
                            "graphs": count,
                            "us": times[arm],
                        }
                    )
                (ab if "pin" in times else aa).append(
                    times.get("pin", times.get("repeat")) / times["ref"]
                )
            result = {
                "case": row["id"],
                "selector": row["expected_signature"],
                "precondition": pre,
                "epochs": epochs,
                "ab": ab,
                "aa": aa,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    with telemetry.open() as stream:
        samples = [
            sample
            for sample in csv.DictReader(stream, skipinitialspace=True)
            if uuid in sample.values()
        ]
    require(
        samples and all(float(sample["enforced.power.limit [W]"]) == 700 for sample in samples),
        f"Missing or invalid telemetry for {uuid}",
    )
    return {
        "uuid": uuid,
        "node": socket.gethostname(),
        "cases": results,
        "telemetry_samples": len(samples),
        "job": os.environ.get("SLURM_JOB_ID"),
        "qos": os.environ.get("WGRAD_GATE_QOS"),
    }


def sanitize(args, plan, torch, dg):
    fixture = fixture_module(args, torch, dg)
    checked = check_edges(torch, dg, edge_cases()[: len(EDGES) * 2], caller_stream=True)
    stream, outer = torch.cuda.Stream(), torch.cuda.current_stream()
    stream.wait_stream(outer)
    with torch.cuda.stream(stream):
        row = plan["rows"][0]
        data, dest, ref, pin = dense_launches(torch, dg, fixture, row, edge=True)
        check_pair(torch, dest, ref, pin, "non-default caller stream")
        saved = data[1][0, 0].clone()
        expected = dest[0, 0].clone()
        consumed = torch.empty_like(expected)

        def dependent_launch():
            data[1][0, 0].copy_(saved)
            dest.fill_(float("nan"))
            pin()
            consumed.copy_(dest[0, 0])

        graph = fixture.make_graph(dependent_launch, 1, stream=stream)
        data[1][0, 0].zero_()
        graph.replay()
        exact(consumed, expected, "caller-stream write and consume")
    outer.wait_stream(stream)
    return {"cases": len(checked) + 1, "caller_stream": True}


def smoke(args, torch, dg):
    import pytest
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from pithtrain.kernels import fp8_gemm_fwd, fp8_quant_grad, fp8_quant_weight
    from pithtrain.modules.wgrad_accumulation import install_wgrad_accumulation
    from pithtrain.operators.linear import FP8Linear

    require(
        os.environ.get("PITHTRAIN_PREFLIGHT_SEAL") == str(args.output / "seal.json"),
        "Integration must run through the preflight launcher",
    )
    weight, x1, x2, dy1, dy2 = torch.load(args.output / "smoke-input.pt", weights_only=True)
    # CPU preflight generates and seals the tensors. Moving parameters and inputs
    # to the device is part of the training step.
    weight = weight.cuda()
    layer = FP8Linear(128, 128, bias=False, device="cuda")
    with torch.no_grad():
        layer.weight.copy_(weight)
    dist.init_process_group(
        "nccl", init_method=(args.output / "smoke-rendezvous").as_uri(), rank=0, world_size=1
    )
    try:
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fully_shard(
            layer, mesh=init_device_mesh("cuda", (1,)), mp_policy=policy, reshard_after_forward=True
        )
        layer.set_gradient_divide_factor(1.0)
        install_wgrad_accumulation(layer, overwrite=True)
        quantized, scale, *_ = fp8_quant_weight(weight.bfloat16())
        reference_grad = None
        losses = []
        for index, (x, dy) in enumerate(((x1, dy1), (x2, dy2))):
            x, dy = x.cuda().requires_grad_(), dy.cuda()
            layer.set_requires_gradient_sync(index == 1)
            actual = (layer(x).float() * dy.float()).sum()
            with torch.no_grad():
                output, xt, xs = fp8_gemm_fwd(x, quantized, scale)
                expected = (output.float() * dy.float()).sum()
                _, _, dyt, dys = fp8_quant_grad(dy)
                gradient = torch.empty_like(weight, dtype=torch.float32)
                reference_launch(dg, (dyt, dys, xt, xs), gradient)
                reference_grad = (
                    gradient if reference_grad is None else reference_grad.add_(gradient)
                )
            exact(actual, expected, f"smoke loss microbatch {index + 1}")
            actual.backward()
            if index == 0:
                require(layer.weight.grad is None, "FSDP reduced before the accumulation boundary")
            losses.append(actual.item())
        shard = layer.weight.grad.to_local()
        require(shard.dtype == torch.float32, "The FSDP gradient shard lost FP32 precision")
        exact(shard, reference_grad, "accumulated FSDP weight-gradient shard")
    finally:
        dist.destroy_process_group()

    test_status = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={args.output / 'pytest-gpu'}",
            "tests/operators/test_dense_fp8_wgrad.py",
            "tests/test_fsdp2_main_grad.py",
            "-k",
            "not rank-skip and not compiled",
        ]
    )
    require(not test_status, f"GPU branch regression tests failed ({test_status})")
    return {"microbatches": 2, "ranks": 1, "losses": losses, "gradient_shard_bits": "exact"}


def stock_measurement(plan, torch, dg):
    from pithtrain.kernels import fp8_gemm_bwd, fp8_gemm_nt, fp8_quant_act, fp8_quant_grad

    results = []
    for row in plan["rows"]:
        m, n, k = shape(row)
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        dy = torch.randn((k, m), device="cuda", dtype=torch.bfloat16, generator=generator)
        x = torch.randn((k, n), device="cuda", dtype=torch.bfloat16, generator=generator)
        _, _, xt, xs = fp8_quant_act(x)
        # W.T is already quantized when the backward starts.
        wt = torch.ones((n, m), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
        ws = torch.ones((ceildiv(n, 128), ceildiv(m, 128)), device="cuda")
        grad = torch.empty((m, n), device="cuda")

        def pin():
            return fp8_gemm_bwd(dy, wt, ws, xt, xs, n, grad)

        def stock():
            codes, scale, dyt, dys = fp8_quant_grad(dy)
            xgrad = fp8_gemm_nt(codes, scale, wt, ws, k, n, dy.device, dy.dtype)
            grad.zero_()
            channels = xs.repeat_interleave(128, dim=0)[:n]
            dg.fp8_fp4_gemm_nt((dyt, dys), (xt, channels), grad, c=grad, recipe=(1, 1, 128))
            return xgrad

        pin()
        stock()
        graphs = {}
        for name, call in (("pin", pin), ("stock", stock)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            graphs[name] = graph
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        ratios, timings = [], []
        for pair in [("stock", "pin"), ("pin", "stock")] * 4:
            times = {}
            for arm in pair:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graphs[arm].replay()
                stop.record()
                stop.synchronize()
                times[arm] = start.elapsed_time(stop) * 50
            timings.append(times)
            ratios.append(times["pin"] / times["stock"])
        result = {"case": row["id"], "shape": [m, n, k], "ratio": median(ratios), "us": timings}
        print(json.dumps(result), flush=True)
        results.append(result)
    return {
        "cases": results,
        "median_ratio": median(row["ratio"] for row in results),
        "job": os.environ.get("SLURM_JOB_ID"),
    }


def worker(args):
    plan = read_json(args.output / "plan.json")
    if args.step == "prepare":
        prepare(args, plan)
        return
    kind = "stock" if args.step == "stock" else "patched"
    torch, dg = setup_gpu(args, kind)
    if args.step == "exact":
        result = dense_exact(args, plan, torch, dg)
    elif args.step in ("forward", "reverse"):
        result = timing(args, plan, torch, dg, args.step)
    elif args.step == "sanitize":
        result = sanitize(args, plan, torch, dg)
    elif args.step == "smoke":
        result = smoke(args, torch, dg)
    elif args.step == "stock":
        result = stock_measurement(plan, torch, dg)
    else:
        raise ValueError(args.step)
    result["seal_sha256"] = digest(args.output / "seal.json")
    write_json(args.output / f"{args.step}.json", result)


def launch(args, step, command, *, exclude=None):
    root = args.output
    env = os.environ.copy()
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        env.setdefault(name, "")
    package = args.stock if step == "stock" else args.patched
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(package.parent), str(REPO), env["PYTHONPATH"]))
    )
    cuda = env["CUDA_HOME"]
    env.update(
        PATH=f"{cuda}/bin:{env['PATH']}",
        LD_LIBRARY_PATH=f"{cuda}/compat:{cuda}/lib64:{env['LD_LIBRARY_PATH']}",
        CUTE_DSL_ARCH="sm_90a",
        CUTE_DSL_DISABLE_FILE_CACHING="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        TORCHINDUCTOR_COMPILE_THREADS="2",
        TRITON_PTXAS_PATH=f"{cuda}/bin/ptxas",
        CUDA_CACHE_DISABLE="1",
        DG_JIT_USE_NVRTC="0",
        WGRAD_GATE_STEP=step,
        WGRAD_GATE_OUTPUT=str(root),
        WGRAD_GATE_EXCLUDE_NODE=exclude or "",
    )
    for name in (
        "DG_EXPERIMENT_BLOCK_M",
        "DG_EXPERIMENT_BLOCK_N",
        "DG_EXPERIMENT_CLUSTER",
        "PYTORCH_ALLOC_CONF",
        "CUDA_LAUNCH_BLOCKING",
        "PYTORCH_NO_CUDA_MEMORY_CACHING",
    ):
        env.pop(name, None)
    for name in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "PITHTRAIN_TUNING_DIR",
        "CUTE_DSL_CACHE_DIR",
        "QUACK_CACHE_DIR",
    ):
        target = root / "cache" / name.lower()
        target.mkdir(parents=True, exist_ok=True)
        env[name] = str(target)
    cli = [
        sys.executable,
        "-B",
        str(ENTRY),
        "_worker",
        "--step",
        step,
        "--output",
        str(root),
        "--matrix",
        str(args.matrix),
        "--patched",
        str(args.patched),
        "--evaluator",
        str(args.evaluator),
        "--protocol",
        str(args.protocol),
    ]
    if args.stock is not None:
        cli += ["--stock", str(args.stock)]
    if step == "prepare":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if step == "sanitize":
        cli = [
            f"{cuda}/bin/compute-sanitizer",
            "--tool",
            "memcheck",
            "--error-exitcode",
            "99",
            "--log-file",
            str(root / "memcheck.log"),
            *cli,
        ]
    if step == "smoke":
        cli = [
            "bash",
            str(REPO / "tools/fp8_preflight/launch.sh"),
            "--verified-smoke",
            str(root / "seal.json"),
            *shlex.split(command),
            "--",
            *cli,
        ]
    else:
        cli = [*shlex.split(command), *cli]
    print(f"Gate step {step}: {root / (step + '.log')}", flush=True)
    with (root / f"{step}.log").open("w") as stream:
        result = subprocess.run(cli, env=env, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def timing_verdict(root, plan):
    left, right = (read_json(root / f"{direction}.json") for direction in ("forward", "reverse"))
    require(left["uuid"] != right["uuid"], "Timing orders ran on the same GPU UUID")
    require(left["job"] != right["job"], "Timing orders require independent allocations")
    results = []
    for forward, reverse in zip(left["cases"], right["cases"], strict=True):
        require(forward["case"] == reverse["case"], "Mismatched timing cases")
        gaps = [100 * (median(order["ab"]) - 1) for order in (forward, reverse)]
        aa = [100 * (median(order["aa"]) - 1) for order in (forward, reverse)]
        results.append(
            {
                "case": forward["case"],
                "gaps_pct": gaps,
                "aa_gaps_pct": aa,
                "lag": all(median(order["ab"]) > 1.007 for order in (forward, reverse)),
            }
        )
    require(
        [row["case"] for row in results] == plan["scope"]["cases"], "Incomplete timing coverage"
    )
    return {
        "cases": results,
        "lags": [row["case"] for row in results if row["lag"]],
        "gpu_uuids": [left["uuid"], right["uuid"]],
    }


def run(args):
    require(args.changed is not None, "Supply --changed with affected matrix cases or dispatch")
    require(
        args.cpu_command and args.gpu_command and args.smoke_command,
        "Supply CPU, one-GPU and short-QoS smoke launch commands",
    )
    require(
        subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode == 0,
        "Commit the candidate before preparing publication artifacts",
    )
    rows = sorted(read_json(args.matrix)["cases"], key=lambda row: row["id"])
    require([row["id"] for row in rows] == list(range(1, 56)), "Expected 55 dense matrix cases")
    require(
        all(
            (row["layout"], row["sms"], row["block_n_multiple"]) == ("block", SMS, 1)
            for row in rows
        ),
        "The dense matrix must use block scales, full SMs and N multiple 1",
    )
    root = args.output
    root.mkdir(parents=True)
    plan = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "rows": rows,
        "matrix_sha256": digest(args.matrix),
        "evaluator_sha256": digest(args.evaluator),
        "protocol_sha256": digest(args.protocol),
        "scope": timing_scope(rows, args.changed),
    }
    write_json(root / "plan.json", plan)
    result = {"head": plan["head"], "scope": plan["scope"], "steps": {}}
    steps = result["steps"]
    steps["prepare"] = launch(args, "prepare", args.cpu_command)
    if steps["prepare"]:
        write_json(root / "result.json", result)
        return 1
    for step in ("exact", "forward", "reverse", "sanitize", "smoke", "stock"):
        if step == "stock" and args.stock is None:
            continue
        command = args.smoke_command if step == "smoke" else args.gpu_command
        exclude = None
        if step == "reverse" and not steps.get("forward"):
            exclude = read_json(root / "forward.json")["node"]
        steps[step] = launch(args, step, command, exclude=exclude)
        if step == "sanitize" and steps[step] == 0:
            text = (root / "memcheck.log").read_text()
            if not re.search(r"ERROR SUMMARY: 0 errors\b", text):
                steps[step] = 1
        write_json(root / "result.json", result)
    if not (steps["forward"] or steps["reverse"]):
        try:
            result["timing"] = timing_verdict(root, plan)
        except RuntimeError as error:
            result["timing_error"] = str(error)
    result["pass"] = (
        all(steps[step] == 0 for step in ("prepare", "exact", "sanitize", "smoke"))
        and "timing" in result
        and not result["timing"]["lags"]
    )
    write_json(root / "result.json", result)
    return 0 if result["pass"] and steps.get("stock", 0) == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "_worker"])
    parser.add_argument(
        "--step", choices=["prepare", "exact", "forward", "reverse", "sanitize", "smoke", "stock"]
    )
    for name in ("matrix", "output", "evaluator", "protocol", "patched"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--changed")
    parser.add_argument("--cpu-command")
    parser.add_argument("--gpu-command")
    parser.add_argument("--smoke-command")
    args = parser.parse_args()
    for name in ("matrix", "output", "evaluator", "protocol", "patched", "stock"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.action == "run":
        return run(args)
    worker(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
