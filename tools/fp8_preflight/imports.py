"""Bundle Python bytecode ahead of allocation while keeping native modules at their paths."""

import importlib.abc
import importlib.machinery
import importlib.util
import json
import marshal
import sys
import zipfile
from pathlib import Path


def build_import_archive(path):
    entries = {}
    with zipfile.ZipFile(path, "w") as archive:
        for name, module in sorted(tuple(sys.modules.items())):
            spec = getattr(module, "__spec__", None)
            if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                continue
            origin = Path(spec.origin)
            try:
                source = origin.read_bytes()
            except FileNotFoundError:
                continue
            code = compile(source, str(origin), "exec", dont_inherit=True)
            entries[name] = [str(origin), spec.submodule_search_locations is not None]
            archive.writestr(f"{name}.pyc", marshal.dumps(code))
            archive.writestr(f"{name}.py", source)
        archive.writestr("manifest.json", json.dumps(entries))
    print(f"PREFLIGHT import_modules={len(entries)}", flush=True)


class ArchiveLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, archive):
        super().__init__(name, path)
        self.archive = archive

    def get_code(self, fullname):
        return marshal.loads(self.archive.read(f"{fullname}.pyc"))

    def get_source(self, fullname):
        return importlib.util.decode_source(self.archive.read(f"{fullname}.py"))


class ImportArchive(importlib.abc.MetaPathFinder):
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        self.entries = json.loads(self.archive.read("manifest.json"))

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.entries:
            return None
        origin, package = self.entries[fullname]
        loader = ArchiveLoader(fullname, origin, self.archive)
        locations = [str(Path(origin).parent)] if package else None
        return importlib.util.spec_from_file_location(
            fullname, origin, loader=loader, submodule_search_locations=locations
        )
