#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/bench/env.sh"
out="$root/inputs/Qwen3.5-27B"
mkdir -p "$out"
revision=81a287ca484a5e71148f5beb7f367bb5286d70f9
base="https://huggingface.co/Qwen/Qwen3.5-27B/resolve/$revision"
export out base

fetch() {
    local file=$1
    if [[ ! -s "$out/$file" ]]; then
        curl -fLsS --retry 4 --connect-timeout 30 --continue-at - \
            -o "$out/$file.partial" "$base/$file"
        mv "$out/$file.partial" "$out/$file"
    fi
    printf '%s %s bytes\n' "$file" "$(stat -c %s "$out/$file")"
}
export -f fetch

for file in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
    fetch "$file"
done
/usr/bin/python3 - "$out/model.safetensors.index.json" <<'PY' | xargs -r -P 4 -n 1 bash -c 'fetch "$0"'
import json
import sys

with open(sys.argv[1]) as source:
    print(*sorted(set(json.load(source)["weight_map"].values())), sep="\n")
PY
