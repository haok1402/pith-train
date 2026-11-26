"""Content-addressed CuTe artifacts and preflight receipts."""

import hashlib
import inspect
import json
import subprocess
from enum import Enum
from pathlib import Path
from types import FunctionType, MethodType

from tools.fp8_preflight.recipe import offline_cute_hardware


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(path.read_bytes())


def qualified(value):
    return value.__module__ + "." + value.__qualname__


def signature(operation, args, kwargs):
    """Record the launcher state and its complete fake-tensor ABI."""
    import torch
    from cutlass.cute.runtime import _FakeTensor
    from cutlass.cute.typing import SymInt
    from cutlass.runtime import _FakeStream

    symbols = {}

    def encode(value):
        if isinstance(value, Enum):
            return ["enum", qualified(type(value)), value.name]
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, torch.dtype):
            return ["dtype", str(value)]
        if isinstance(value, SymInt):
            return [
                "symbol",
                symbols.setdefault(id(value), len(symbols)),
                value.width,
                value.divisibility,
                value.symbol,
            ]
        if isinstance(value, _FakeTensor):
            return [
                "tensor",
                qualified(value.element_type),
                encode(value.memspace),
                value._assumed_align,
                value._use_32bit_stride,
                encode(value.shape),
                encode(value.stride),
            ]
        if isinstance(value, _FakeStream):
            return ["stream", value.use_tvm_ffi_env_stream]
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [encode(part) for part in value]]
        if isinstance(value, dict):
            return ["dict", [[encode(key), encode(item)] for key, item in value.items()]]
        if isinstance(value, (FunctionType, type)):
            return ["name", qualified(value)]
        if isinstance(value, MethodType):
            return ["method", qualified(value.__func__), encode(value.__self__)]
        cls = type(value)
        if cls.__module__.startswith("pithtrain.kernels.") and hasattr(value, "__dict__"):
            return ["launcher", qualified(cls), encode(vars(value))]
        if cls.__module__.startswith("cutlass.") and isinstance(
            getattr(value, "value", None), (int, float)
        ):
            return ["scalar", qualified(cls), value.value]
        raise TypeError(f"Cannot seal CuTe compile argument {qualified(cls)}")

    description = [encode(operation), encode(args), encode(kwargs)]
    key = hashlib.sha256(json.dumps(description, separators=(",", ":")).encode()).hexdigest()
    return key, description


def ffi_parameters(operation, args, kwargs):
    from cutlass.cute._tvm_ffi_args_spec_converter import _tvm_ffi_args_spec_converter

    signature = inspect.signature(operation)
    operands = {name: value for name, value in kwargs.items() if name != "options"}
    _, spec, dataclasses = _tvm_ffi_args_spec_converter(
        qualified(operation if isinstance(operation, FunctionType) else type(operation)),
        signature,
        args,
        operands,
    )
    return spec, dataclasses


def ffi_call(entry, operation, args, kwargs):
    """Apply the same runtime signature adapter as the CuTe JIT function."""
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    spec, dataclasses = ffi_parameters(operation, args, kwargs)
    if spec.arg_defaults or spec.kwonly_names or dataclasses:
        return make_kwargs_wrapper(
            entry,
            arg_names=spec.arg_names,
            arg_defaults=spec.arg_defaults,
            kwonly_names=spec.kwonly_names,
            kwonly_defaults=spec.kwonly_defaults,
            map_dataclass_to_tuple=dataclasses,
        )
    return entry


def verify_wrapper(operation, args, kwargs, launched):
    """Check that the real Python wrapper provides every exported argument."""
    spec, _ = ffi_parameters(operation, args, kwargs)
    expected = len(spec.arg_names) + len(spec.kwonly_names)

    def accept(*runtime_args):
        if len(runtime_args) != expected:
            raise RuntimeError(
                f"CuTe exported arity mismatch for {operation}: expected {expected}, "
                f"got {len(runtime_args)}"
            )
        launched()

    return ffi_call(accept, operation, args, kwargs)


class CuteArtifacts:
    """Export the normal compiler's executables, or load them with compilation disabled."""

    def __init__(self, root: Path, entries: dict, *, build: bool):
        from cutlass import cute
        from cutlass.runtime import find_runtime_libraries, load_module

        self.root = root
        self.entries = entries
        self.build = build
        self.compiler = cute.compile
        self.load_module = load_module
        self.libraries = find_runtime_libraries(enable_tvm_ffi=True) if build else ()
        self.loaded = {}

    def __call__(self, operation, *args, **kwargs):
        key, description = signature(operation, args, kwargs)
        if key in self.loaded:
            return self.loaded[key]
        if key not in self.entries:
            if not self.build:
                raise RuntimeError(f"Missing sealed CuTe artifact {key} for {description}")
            self.export(key, description, operation, args, kwargs)
        if self.build:
            return self.skip_launch
        path = self.root / self.entries[key]["path"]
        entry = self.load_module(str(path), enable_tvm_ffi=True).preflight_entry
        adapted = ffi_call(entry, operation, args, kwargs)
        self.loaded[key] = adapted
        print(f"PREFLIGHT cute_load key={key} executable={description[0][:2]}", flush=True)
        return adapted

    def export(self, key, description, operation, args, kwargs):
        binary = self.root / "cute" / f"{key}.so"
        binary.parent.mkdir(exist_ok=True)
        obj = binary.with_suffix(".o")
        with offline_cute_hardware():
            entry = self.compiler(operation, *args, **kwargs)
            entry.export_to_c(str(obj), "preflight_entry", export_only_tvm_ffi_symbols=True)
        rpath = ":".join(dict.fromkeys(str(Path(lib).parent) for lib in self.libraries))
        subprocess.run(
            [
                "c++",
                "-shared",
                "-Wl,--no-undefined",
                f"-Wl,-rpath,{rpath}",
                str(obj),
                *self.libraries,
                "-o",
                str(binary),
            ],
            check=True,
        )
        obj.unlink()
        self.entries[key] = {
            "path": str(binary.relative_to(self.root)),
            "sha256": digest(binary),
            "signature": description,
        }
        print(f"PREFLIGHT cute_build key={key} executable={description[0][:2]}", flush=True)

    @staticmethod
    def skip_launch(*args):
        pass
