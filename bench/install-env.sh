#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
elf_deps="$XDG_CACHE_HOME/elfutils"
mkdir -p "$elf_deps/debs"
if [[ ! -f "$elf_deps/usr/include/elfutils/libdwfl.h" || ! -f "$elf_deps/usr/include/libelf.h" ]]; then
    cd "$elf_deps/debs"
    ubuntu_archive=https://archive.ubuntu.com/ubuntu/pool/main/e/elfutils
    while read -r file sha512; do
        curl -fLsS --retry 3 -o "$file" "$ubuntu_archive/$file"
        printf '%s  %s\n' "$sha512" "$file" | sha512sum --check
        dpkg-deb -x "$file" "$elf_deps"
    done <<'PACKAGES'
libdw-dev_0.190-1.1ubuntu0.1_amd64.deb 356a8aeb1d84acb185e4e753339ccf3f6e62b8c4eeb2348881ad7fa123176d399cf71e8a3c30bd0dab4de0ae139434feedaa593ee21d64dc92c0022f97999c6b
libelf-dev_0.190-1.1ubuntu0.1_amd64.deb 1da482551f79e7c3bd3704521fdf0267471d003e94e02d1949a1e5894456357f55a31d8025635dc409f1fd1155e7e7984d88444cf02eedae20ac7d5b1ee7c2cf
PACKAGES
fi
export CPLUS_INCLUDE_PATH="$elf_deps/usr/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export MAX_JOBS=12 UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=8
cd "$E2EFP8_ROOT/pith-train"
if [[ ! -f uv.lock ]]; then
    cp "$E2EFP8_ROOT/bench/pith-train.uv.lock" uv.lock
fi
uv sync --locked --python /usr/bin/python3.12
.venv/bin/python -c 'import deep_gemm, torch, transformers; print(f"torch {torch.__version__}, transformers {transformers.__version__}, DeepGEMM {deep_gemm.__version__}")'
