"""Precompile the dense interop GEMMs using DeepGEMM's own JIT heuristics and key."""

import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path

from tools.fp8_preflight.artifacts import digest, read_json
from tools.fp8_preflight.recipe import PROJECTIONS


def compile_deepgemm(root, package, seed=None):
    from torch.utils.cpp_extension import include_paths

    cuda = Path(os.environ["CUDA_HOME"])
    source = Path(__file__).with_suffix(".cpp")
    include = Path(os.environ["DEEP_JIT_INCLUDE"])
    elf = os.environ.get("ELFUTILS_INCLUDE")
    torchlib = package.parent / "torch/lib"
    python = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    compiler = root / "compile-deepgemm"
    includes = [
        package,
        package / "include",
        include,
        cuda / "include",
        cuda / "include/cccl",
        *include_paths(),
        sysconfig.get_path("include"),
    ]
    if elf:
        includes.append(elf)
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            *(f"-I{path}" for path in includes),
            str(source),
            "-Wl,--gc-sections",
            "-o",
            str(compiler),
            str(python),
            f"-L{cuda / 'lib64'}",
            f"-L{cuda / 'lib64/stubs'}",
            "-lcudart",
            "-lcuda",
            "-lnvrtc",
            "-ldl",
            f"-L{torchlib}",
            f"-Wl,-rpath,{torchlib}",
            "-lc10",
            "-lcublasLt",
            "-lcublas",
        ],
        check=True,
    )
    header = package / "csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
    template = re.search(
        r'jit->compile\(tag, std::format\(R"\((.*?)\n\)",', header.read_text(), re.S
    )
    if template is None:
        raise ValueError(f"Unsupported DeepGEMM JIT generator: {header}")
    cpp = root / "deepgemm-template.txt"
    cpp.write_text(template[1] + "\n")
    shapes = set(PROJECTIONS) | {(k, n) for n, k in PROJECTIONS}
    geometry = root / "deepgemm-shapes.txt"
    geometry.write_text("".join(f"{n} {k}\n" for n, k in sorted(shapes)))
    cache = root / "deepgemm"
    if seed is not None:
        seed = seed.resolve()
        proof = read_json(seed / "deepgemm.json")
        (extension,) = package.glob("_C*.so")
        if proof["module_sha256"] != digest(extension) or (proof["arch"], proof["cuda"]) != (
            "sm_90a",
            "13.2",
        ):
            raise ValueError("DeepGEMM seed targets a different extension or toolkit")
        seed_entries = {
            shape: entry
            for shape, entry in proof["entries"].items()
            if shape in {f"{n},{k}" for n, k in shapes}
        }
        for entry in seed_entries.values():
            key = entry["key"]
            if Path(key).name != key:
                raise ValueError(f"Invalid DeepGEMM seed key {key}")
            folder = seed / "deepgemm/cache" / key
            if not (folder / ".committed").is_file() or any(
                digest(folder / file) != entry[file] for file in ("kernel.cu", "kernel.cubin")
            ):
                raise ValueError(f"DeepGEMM seed artifact disagrees at {folder}")
            local = cache / "cache" / folder.name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                shutil.copytree(folder, local)
        print(f"PREFLIGHT deepgemm_seed_entries={len(seed_entries)}", flush=True)
    result = subprocess.check_output(
        [str(compiler), str(package), str(cpp), str(geometry)],
        text=True,
        env={**os.environ, "DG_JIT_CACHE_DIR": str(cache), "DJ_JIT_CACHE_DIR": str(cache)},
    )
    print(result, end="", flush=True)
    entries = {}
    for n, k, directory in re.findall(r"^DG_TRAIN (\d+) (\d+) (.+)$", result, re.M):
        folder = Path(directory)
        if not (folder / ".committed").is_file() or folder.parent != cache / "cache":
            raise ValueError(f"DeepGEMM did not commit {folder}")
        entries[f"{n},{k}"] = {
            "key": folder.name,
            **{file: digest(folder / file) for file in ("kernel.cu", "kernel.cubin")},
        }
    if set(entries) != {f"{n},{k}" for n, k in shapes}:
        raise ValueError("DeepGEMM coverage does not match the recipe projections")
    for path in (compiler, cpp, geometry):
        path.unlink()
    return entries
