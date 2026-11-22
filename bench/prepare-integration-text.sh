#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

data="$E2EFP8_ROOT/inputs/integration-wikitext-train"
model="$E2EFP8_ROOT/inputs/Qwen3.5-27B"
mkdir -p "$data"
shard="$data/train-00000-of-00002.parquet"
url=https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/train-00000-of-00002.parquet
if [[ ! -f "$shard" ]]; then
    curl -fLsS --retry 3 --continue-at - -o "$shard.partial" "$url"
    mv "$shard.partial" "$shard"
fi
printf '%s  %s\n' 74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212 "$shard" | sha256sum --check

text_python=${E2EFP8_TEXT_PYTHON:-"$E2EFP8_ROOT/wt-fsdp2/.venv/bin/python"}
"$text_python" - "$model/tokenizer.json" "$shard" "$data/wikitext-train.bin" <<'PY'
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as parquet
from tokenizers import Tokenizer

tokenizer_path, source, destination = sys.argv[1:]
tokenizer = Tokenizer.from_file(tokenizer_path)
target = 8 * 256 * 4096 + 1
tokens = np.empty(target, dtype=np.int32)
filled = 0
for batch in parquet.ParquetFile(source).iter_batches(batch_size=4096, columns=["text"]):
    text = "\n\n".join(batch.column(0).to_pylist())
    if filled:
        text = "\n\n" + text
    ids = tokenizer.encode(text, add_special_tokens=False).ids[: target - filled]
    tokens[filled : filled + len(ids)] = ids
    filled += len(ids)
    if filled == target:
        break
if filled != target:
    raise ValueError(f"WikiText shard yielded {filled:,} tokens, expected {target:,}")
destination = Path(destination)
partial = destination.with_suffix(".partial")
with partial.open("wb") as output:
    np.save(output, tokens)
partial.replace(destination)
print(f"{filled:,} Qwen tokens from WikiText-103 train, {(filled - 1) // 4096:,} complete 4K sequences")
PY
