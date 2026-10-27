"""
Import layering, kernel naming and the kernel library's interface, checked statically (CPU only,
nothing is imported).

An AST walk over ``pithtrain/``, ``tests/``, ``benchmarks/``, ``tools/`` and ``examples/`` that
enforces the package layering (AGENTS.md, "Kernels"), the ``_sm90`` / ``_sm100`` naming convention
of ``pithtrain/kernels/`` and its black-box interface:

- ``kernels/`` is flat; its ``__init__.py`` holds only ``from pithtrain.kernels.<module> import
  <names>`` lines and ``__all__`` (a list built by ``__all__.append`` calls), which lists exactly
  the imported names, each defined at the top
  level of its module (no ``_sm100`` module, no private name); ``operators/__init__.py`` is empty;
- outside ``kernels/`` the only import of the kernel library is ``from pithtrain.kernels import
  <name>`` with ``<name>`` in ``__all__`` (no submodule, no ``import pithtrain.kernels``);
- outside ``kernels/`` nothing in ``pithtrain/`` branches on the architecture: no
  ``get_device_capability``, ``ARCH_MAJOR``, ``NATIVE_E8M0``, ``is_sm90`` or ``is_sm90_device``
  (capability queries named for a feature are exported instead), and neither does
  ``tests/kernels/``;
- every name in ``__all__`` is exercised (imported and used) by ``tests/kernels/test_<feature>.py``,
  ``<feature>`` its module's stem without ``_sm90`` / ``_sm100``, and ``tests/kernels/`` imports
  nothing from pithtrain but ``pithtrain.kernels``;
- inside ``kernels/`` every module lists what it exports in a literal ``__all__``, defines no
  ``_name`` at any level (functions, classes, methods, attributes, locals, parameters, import
  aliases; ``__all__`` marks what is internal, not an underscore), and imports from a
  sibling only names in that sibling's ``__all__`` (``from pithtrain.kernels.x import n`` or ``x.n``
  through a module alias);
- importing ``pithtrain.kernels`` loads neither cutlass, QuACK nor FA4: no module it imports at
  module level, transitively, imports them at module level (the ``_sm90`` CuTe kernels and FA4
  are imported inside the functions that run them);
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
- ``get_device_capability`` appears in ``pithtrain/`` only in ``kernels/arch.py``;
- no test file is named for an architecture (``test_*_sm90.py`` / ``test_*_sm100.py``);
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
# What models/ may import from modules/ (from kernels/: only the package, like everyone else).
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
# Every Python file outside pithtrain/kernels/ that may import the kernel library.
OUTSIDE_FILES = [
    p
    for base in ("pithtrain", "tests", "benchmarks", "tools", "examples")
    for p in sorted((ROOT / base).rglob("*.py"))
    if "__pycache__" not in p.parts and KERNELS not in p.parents
]
# Names that read the device architecture; only kernels/ may use them.
ARCH_NAMES = {"get_device_capability", "ARCH_MAJOR", "NATIVE_E8M0", "is_sm90", "is_sm90_device"}


def _exports() -> tuple[dict[str, list[str]], list[str], list[str]]:
    """
    From kernels/__init__.py: the imported names by defining module stem, ``__all__``, and every
    statement that is neither such an import nor ``__all__`` (the docstring aside).
    """
    tree = ast.parse((KERNELS / "__init__.py").read_text())
    exports: dict[str, list[str]] = {}
    all_names: list[str] = []
    other: list[str] = []
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "pithtrain.kernels."
        ):
            stem = node.module.rpartition(".")[2]
            exports.setdefault(stem, []).extend(a.name for a in node.names if a.asname is None)
            other.extend(f"alias {a.name}" for a in node.names if a.asname is not None)
        elif isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["__all__"]:  # fmt: skip
            all_names = list(ast.literal_eval(node.value))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "__all__.append"
        ):
            all_names.append(ast.literal_eval(node.value.args[0]))
        elif not (i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            other.append(ast.dump(node)[:60])
    return exports, all_names, other


EXPORTS, ALL_LIST, INIT_OTHER = _exports()
ALL = {name for names in EXPORTS.values() for name in names}


# Third-party packages importing pithtrain.kernels must not load: cutlass and QuACK (CuTe-DSL; QuACK
# patches it globally), and flash_attn (FA4 imports both).
HEAVY = {"cutlass", "quack", "flash_attn"}


def _top_level_imports(path: Path) -> list[str]:
    """Modules ``path`` imports at module level (not inside a function), resolved to pithtrain
    modules where they are one."""
    out: list[str] = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            out.extend(_resolve(a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for a in node.names:
                full = f"{node.module}.{a.name}"
                out.append(full if full in MODULES else _resolve(node.module))
    return out


def _eager_closure(root: str) -> set[str]:
    """The pithtrain modules importing ``root`` loads: module-level imports, transitively."""
    seen, stack = {root}, [root]
    while stack:
        for t in _top_level_imports(MODULES[stack.pop()]):
            if t in MODULES and t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


VERIFIER_OF = {"tuning": "bf16_expert_gemm"}
"""
Modules verified through another feature's verifier: launch-config tuning by the GEMMs it tunes.
"""


def _feature(stem: str) -> str:
    if stem in VERIFIER_OF:
        return VERIFIER_OF[stem]
    for suffix in ("_sm90", "_sm100"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _defined_names(tree: ast.Module) -> set[str]:
    """Every name a module defines at any level: functions, classes, methods, assignment targets
    (attributes such as self.x included), parameters and import aliases. Reads of other code's
    attributes (a library's _hooks) are not definitions."""
    names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
            names.add(n.attr)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {a.asname for a in n.names if a.asname}
    return names


def _kernel_imports(path: Path) -> list[tuple[int, str, str]]:
    """(line, form, detail) of every import of pithtrain.kernels in ``path``, TYPE_CHECKING too."""
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "pithtrain.kernels" or a.name.startswith("pithtrain.kernels."):
                    out.append((node.lineno, "import", a.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("pithtrain.kernels."):
                out.append((node.lineno, "submodule", node.module))
            elif node.module == "pithtrain.kernels":
                for a in node.names:
                    out.append((node.lineno, "name", a.name))
            elif node.module == "pithtrain" and any(a.name == "kernels" for a in node.names):
                out.append((node.lineno, "import", "pithtrain.kernels"))
    return out


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
    init = PKG / "operators" / "__init__.py"
    if init.exists() and init.read_text().strip():
        found.add(("empty-init", _module_name(init), ""))

    # The kernel library's interface: __all__ is exactly the imported names, each defined in the
    # module it is imported from, nothing private, nothing else in __init__.py.
    if not EXPORTS:
        found.add(("exports-missing", "pithtrain.kernels", "imports"))
    if sorted(ALL_LIST) != sorted(ALL) or len(ALL_LIST) != len(set(ALL_LIST)):
        diff = sorted(set(ALL_LIST) ^ ALL)
        found.add(("exports-all-mismatch", "pithtrain.kernels", ", ".join(diff)))
    for detail in INIT_OTHER:
        found.add(("init-extra", "pithtrain.kernels", detail))
    for module, names in EXPORTS.items():
        path = KERNELS / f"{module}.py"
        if not path.exists():
            found.add(("exports-module", "pithtrain.kernels", module))
            continue
        if module.endswith("_sm100"):
            found.add(("exports-sm100", "pithtrain.kernels", module))
        defined = _top_level_names(path)
        for name in names:
            if name not in defined:
                found.add(("exports-undefined", f"pithtrain.kernels.{module}", name))
            if name.startswith("_"):
                found.add(("exports-private", f"pithtrain.kernels.{module}", name))
    if len(ALL) != sum(len(v) for v in EXPORTS.values()):
        found.add(("exports-duplicate", "pithtrain.kernels", ""))

    # Outside kernels/: only `from pithtrain.kernels import <name in __all__>`, no arch branches.
    for path in OUTSIDE_FILES:
        rel = str(path.relative_to(ROOT))
        for line, form, detail in _kernel_imports(path):
            if form != "name" or detail not in ALL:
                found.add(("kernels-interface", rel, f"{line}: {form} {detail}"))
        in_pkg = PKG in path.parents
        in_kernel_tests = (ROOT / "tests" / "kernels") in path.parents
        if in_pkg or in_kernel_tests:
            for node in ast.walk(ast.parse(path.read_text())):
                name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None  # fmt: skip
                if name in ARCH_NAMES:
                    found.add(("arch-outside-kernels", rel, name))

    # Every exported name is exercised by its feature's verifier, which imports only the library.
    kernel_tests = ROOT / "tests" / "kernels"
    for module, names in EXPORTS.items():
        test = kernel_tests / f"test_{_feature(module)}.py"
        if not test.exists():
            found.add(("verifier-missing", str(test.relative_to(ROOT)), module))
            continue
        tree = ast.parse(test.read_text())
        imported = {d for _, form, d in _kernel_imports(test) if form == "name"}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for name in names:
            if name not in imported or name not in used:
                found.add(("verifier-coverage", str(test.relative_to(ROOT)), name))
    if kernel_tests.exists():
        for path in sorted(kernel_tests.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""] if node.level == 0 else ["." * node.level]
                for m in mods:
                    if m.startswith(".") or (
                        (m == "pithtrain" or m.startswith(("pithtrain.", "tests")))
                        and m != "pithtrain.kernels"
                    ):
                        found.add(("verifier-imports", str(path.relative_to(ROOT)), m))

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
                or (tl == "kernels" and t != "pithtrain.kernels")
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
                if layer != "kernels":
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

    # Importing pithtrain.kernels never loads cutlass, QuACK or FA4 (which loads both): no module
    # the package imports at module level, transitively, imports them at module level.
    for mod in sorted(_eager_closure("pithtrain.kernels")):
        heavy = [t for t in _top_level_imports(MODULES[mod]) if t.split(".")[0] in HEAVY]
        if heavy:
            found.add(("kernels-eager-heavy", mod, heavy[0]))

    # Inside kernels/: a literal __all__ per module, no module-level _names, sibling imports only
    # of names in the sibling's __all__.
    module_all: dict[str, set[str]] = {}
    for path in sorted(KERNELS.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        decl = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and [getattr(t, "id", None) for t in n.targets] == ["__all__"]
        ]
        try:
            module_all[path.stem] = set(ast.literal_eval(decl[0].value)) if decl else set()
        except ValueError:
            module_all[path.stem] = set()
        if not decl:
            found.add(("kernels-module-all", f"pithtrain.kernels.{path.stem}", "missing __all__"))
        for name in sorted(_defined_names(tree)):
            if (
                name.startswith("_")
                and name != "_"
                and not (name.startswith("__") and name.endswith("__"))
            ):
                found.add(("kernels-underscore-name", f"pithtrain.kernels.{path.stem}", name))
    for path in sorted(KERNELS.glob("*.py")):
        tree = ast.parse(path.read_text())
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.startswith("pithtrain.kernels."):
                    target = node.module.rpartition(".")[2]
                    for a in node.names:
                        if a.name not in module_all.get(target, set()):
                            found.add(
                                (
                                    "kernels-sibling-import",
                                    f"pithtrain.kernels.{path.stem}",
                                    f"{target}.{a.name}",
                                )
                            )
                elif node.module == "pithtrain.kernels":
                    aliases.update(
                        {a.asname or a.name: a.name for a in node.names if a.name in module_all}
                    )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
            ):
                target = aliases[node.value.id]
                if node.attr not in module_all[target]:
                    found.add(
                        (
                            "kernels-sibling-import",
                            f"pithtrain.kernels.{path.stem}",
                            f"{target}.{node.attr}",
                        )
                    )

    # Tests are named for the feature, never for an architecture.
    for mod in TEST_MODULES:
        if _stem(mod).startswith("test_") and _smxx(mod):
            found.add(("arch-named-test", mod, ""))

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
