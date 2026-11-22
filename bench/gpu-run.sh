#!/usr/bin/env bash
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$bench_dir/env.sh"
exec /usr/bin/python3 "$bench_dir/gpu_gate.py" "$@"
