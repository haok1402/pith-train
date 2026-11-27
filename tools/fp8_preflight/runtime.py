"""Verify the native runtime shards used by the training step."""

import re
import sys
import sysconfig
from pathlib import Path

from tools.fp8_preflight.artifacts import digest


def seal_runtime(bundle, seed):
    base_python = Path((seed / "python").read_text().strip())
    site = Path((seed / "site").read_text().strip())
    source_site = Path(sysconfig.get_path("platlib"))
    if (
        base_python != Path(sys._base_executable).resolve()
        or site != source_site.relative_to(Path(sys.prefix))
    ):
        raise ValueError("Runtime shards target another Python environment")
    root = bundle / "runtime"
    root.mkdir()
    names = set()
    for line in (seed / "sha256").read_text().splitlines():
        checksum, name = line.split()
        if name in names or not re.fullmatch(r"venv-[0-9]+\.tar\.zst", name):
            raise ValueError(f"Invalid runtime shard {name}")
        path = seed / name
        if digest(path) != checksum:
            raise ValueError(f"Runtime shard content changed for {name}")
        (root / name).symlink_to(path.resolve())
        names.add(name)
    if not names:
        raise ValueError("No runtime shards were sealed")
    (root / "python").write_text(f"{base_python}\n")
    (root / "site").write_text(f"{site}\n")
    (root / "source-site").write_text(f"{source_site}\n")
    (root / "sha256").write_text((seed / "sha256").read_text())
    print(f"PREFLIGHT runtime_shards={len(names)}", flush=True)
