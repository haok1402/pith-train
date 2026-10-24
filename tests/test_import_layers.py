"""
Import layering and kernel naming rules, checked statically (CPU only, nothing is imported).

An AST walk over ``pithtrain/`` and ``tests/`` that enforces the package layering (AGENTS.md,
"Kernels") and the ``_sm90`` / ``_sm100`` naming convention of ``pithtrain/kernels/``:

- ``kernels/`` is flat and its ``__init__.py`` is empty, as is ``operators/__init__.py``;
- ``kernels/`` imports only third-party code and other ``kernels/`` modules; ``contexts/`` imports
  nothing from pithtrain at runtime; ``operators/`` imports only ``kernels/``, ``contexts/`` and
  sibling operators; ``pipeline/``, ``models/`` and ``tasks/`` import only what the layering allows
  (models import ``pipeline/`` for ``layer_partition`` / ``model_forward``, a package-level 2-cycle
  that is acyclic per module because ``pipeline/`` imports only ``models/interface.py``);
- ``_sm90`` / ``_sm100`` files exist only in ``kernels/``, and every ``_sm90`` file has its
  ``_sm100`` sibling unless it is SM90-only or a shared SM90 helper;
- no ``_sm100`` file imports an ``_sm90`` file or a picker, an ``_sm90`` file imports another
  feature's ``_smXX`` file only along the named cross-feature edges, and no picker imports another
  picker;
- outside ``kernels/`` and ``tests/``, only the two SM90-only operators import an ``_smXX`` file;
- ``get_device_capability`` appears in ``pithtrain/`` only in ``kernels/arch.py``;
- every ``tests/**/test_*_sm90.py`` skips unless SM90;
- the ``modules/`` files the models import do not import the models back;
- the module import graph of ``pithtrain/`` has no cycle.

Imports inside functions count; imports under ``if TYPE_CHECKING:`` do not.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "pithtrain"
KERNELS = PKG / "kernels"

# SM90 kernels with no SM100 kernel to pair with: SM100 runs a non-kernel path.
SM90_ONLY = {
    "group_indices_sm90",
    "bf16_expert_gemm_sm90",
    "bf16_expert_slots_sm90",
    "gated_delta_rule_pinned_sm90",
}
# Shared CuTe helpers that only SM90 code uses (rule 6).
SM90_HELPERS = {"cute_ptx_sm90", "cute_compile_sm90", "fp8_e8m0_scale_sm90"}
# Cross-feature edges from an _sm90 file into another feature's _smXX file (F4-F6), plus the
# SM90-only features' internal edges.
SM90_CROSS_EDGES = {
    ("fp8_silu_mul_quant_sm90", "fp8_rowwise_kmajor_sm90"),
    ("fp8_clamped_swiglu_quant_sm90", "fp8_rowwise_kmajor_sm90"),
    ("fp8_scatter_quant_sm90", "fp8_rowwise_kmajor_sm90"),
}
# The only modules outside kernels/ that may import an _smXX file (both SM90-only, imported lazily
# by their callers after the gate).
SMXX_IMPORTERS = {"pithtrain.operators.bf16_experts", "pithtrain.operators.gated_delta_rule"}
# What models/ may import from kernels/ and modules/.
MODELS_KERNELS = {"arch", "fp8_gemm", "fp8_weight_quant", "bf16_expert_support"}
MODELS_MODULES = {"load_balance", "recompute", "fp8_param_gather"}
PIPELINE_OPERATORS = {"all_to_all", "routing"}

# Violations of the current tree, as (rule, module, detail). Shrinks as the restructure lands; the
# test also fails on an entry that no longer occurs, so the list stays exact.
ALLOWED: set[tuple[str, str, str]] = set()


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _modules(base: Path) -> dict[str, Path]:
    return {_module_name(p): p for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts}


MODULES = _modules(PKG)
TEST_MODULES = _modules(ROOT / "tests")


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _resolve(name: str) -> str:
    """The longest prefix of ``name`` that is a pithtrain module, else ``name`` itself."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in MODULES:
            return cand
    return name


def _imports(path: Path) -> list[str]:
    """Every module ``path`` imports at runtime (lazy imports included, TYPE_CHECKING excluded)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    mod = _module_name(path)
    pkg = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    out: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            out.extend(_resolve(a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                base = ".".join(anchor + ([base] if base else []))
            for a in node.names:
                full = f"{base}.{a.name}"
                out.append(full if full in MODULES else _resolve(base))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out


IMPORTS = {m: [t for t in _imports(p) if t.startswith("pithtrain")] for m, p in MODULES.items()}


def _layer(mod: str) -> str:
    parts = mod.split(".")
    return parts[1] if len(parts) > 1 else ""


def _stem(mod: str) -> str:
    return mod.rpartition(".")[2]


def _smxx(mod: str) -> bool:
    return _stem(mod).endswith(("_sm90", "_sm100"))


def _pickers() -> set[str]:
    stems = {_stem(m) for m in MODULES if _layer(m) == "kernels"}
    return {
        f"pithtrain.kernels.{s}" for s in stems if f"{s}_sm90" in stems and f"{s}_sm100" in stems
    }


PICKERS = _pickers()


def _violations() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()

    # Package shape.
    if KERNELS.exists():
        for p in KERNELS.iterdir():
            if p.is_dir() and p.name != "__pycache__":
                found.add(("kernels-flat", "pithtrain.kernels", p.name))
    for init in (KERNELS / "__init__.py", PKG / "operators" / "__init__.py"):
        if init.exists() and init.read_text().strip():
            found.add(("empty-init", _module_name(init), ""))

    for mod, path in MODULES.items():
        layer, stem = _layer(mod), _stem(mod)
        targets = IMPORTS[mod]

        # Naming: _smXX files live only in kernels/, and an _sm90 file has its _sm100 sibling.
        if _smxx(mod) and layer != "kernels":
            found.add(("smxx-outside-kernels", mod, ""))
        if layer == "kernels" and stem.endswith("_sm90"):
            sibling = stem[: -len("_sm90")] + "_sm100"
            if (
                stem not in SM90_ONLY | SM90_HELPERS
                and f"pithtrain.kernels.{sibling}" not in MODULES
            ):
                found.add(("sm90-without-sm100", mod, sibling))

        # Capability checks only in kernels/arch.py.
        if mod != "pithtrain.kernels.arch" and "get_device_capability" in path.read_text():
            found.add(("capability-outside-arch", mod, ""))

        for t in targets:
            tl = _layer(t)
            # Package layering.
            if layer == "kernels" and tl != "kernels":
                found.add(("kernels-imports", mod, t))
            elif layer == "contexts" and t != mod:
                found.add(("contexts-imports", mod, t))
            elif layer == "operators" and tl not in ("kernels", "contexts", "operators"):
                found.add(("operators-imports", mod, t))
            elif layer == "pipeline" and not (
                tl in ("contexts", "pipeline")
                or t == "pithtrain.models.interface"
                or (tl == "operators" and _stem(t) in PIPELINE_OPERATORS)
            ):
                found.add(("pipeline-imports", mod, t))
            elif layer == "models" and (
                tl == "tasks"
                or (tl == "kernels" and _stem(t) not in MODELS_KERNELS)
                or (tl == "modules" and _stem(t) not in MODELS_MODULES)
            ):
                found.add(("models-imports", mod, t))
            elif layer == "modules" and tl in ("models", "pipeline", "tasks"):
                if mod != "pithtrain.modules.training" or tl == "tasks":
                    found.add(("modules-imports", mod, t))
            # Models import these modules files, so they must not import the models back.
            if mod in {f"pithtrain.modules.{s}" for s in MODELS_MODULES} and (
                tl == "models" or t == "pithtrain.modules.training"
            ):
                found.add(("models-modules-cycle", mod, t))

            # _smXX edges.
            if _smxx(t):
                if layer != "kernels" and mod not in SMXX_IMPORTERS:
                    found.add(("smxx-importer", mod, t))
                if stem.endswith("_sm100") and _stem(t).endswith("_sm90"):
                    found.add(("sm100-imports-sm90", mod, t))
                if stem.endswith("_sm90") and stem not in SM90_HELPERS:
                    own = stem[: -len("_sm90")]
                    ts = _stem(t)
                    ok = (
                        ts == own + "_sm100" or ts in SM90_HELPERS or (stem, ts) in SM90_CROSS_EDGES
                    )
                    if not ok:
                        found.add(("sm90-cross-feature", mod, t))
            if t in PICKERS:
                if stem.endswith("_sm100"):
                    found.add(("sm100-imports-picker", mod, t))
                if mod in PICKERS and t != mod:
                    found.add(("picker-imports-picker", mod, t))

    # SM90 tests skip unless SM90.
    for mod, path in TEST_MODULES.items():
        if _stem(mod).startswith("test_") and _stem(mod).endswith("_sm90"):
            src = path.read_text()
            if "pytestmark" not in src or "is_sm90" not in src:
                found.add(("sm90-test-skip", mod, ""))

    # The module import graph has no cycle.
    for cycle in _cycles():
        found.add(("cycle", cycle[0], " -> ".join(cycle)))
    return found


def _cycles() -> list[list[str]]:
    """One representative per strongly connected component of more than one module."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    sccs: list[list[str]] = []
    counter = [0]

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in IMPORTS.get(v, []):
            if w not in MODULES or w == v:
                continue
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                sccs.append(sorted(comp))

    for v in MODULES:
        if v not in index:
            strong(v)
    return sccs


VIOLATIONS = _violations()


def test_no_new_violations() -> None:
    new = sorted(VIOLATIONS - ALLOWED)
    assert not new, "import-layer violations:\n" + "\n".join(map(str, new))


def test_allowlist_is_exact() -> None:
    stale = sorted(ALLOWED - VIOLATIONS)
    assert not stale, "allowlisted violations that no longer occur:\n" + "\n".join(map(str, stale))
