"""Record recoverable cache misses for the next CPU preflight."""

import json
import os
import sys
from pathlib import Path


def record_miss(cache, key, reason, **details):
    record = {"cache": cache, "key": key, "reason": reason, **details}
    print(f"WARNING: FP8 preflight {cache} miss key={key}: {reason}", file=sys.stderr, flush=True)
    directory = os.environ.get("PITHTRAIN_MISS_DIR")
    if directory:
        rank = int(os.environ.get("RANK", "0"))
        try:
            with (Path(directory) / f"rank-{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
        except OSError as error:
            print(f"WARNING: Could not write cache miss receipt: {error}", file=sys.stderr)


def write_deepgemm_receipt(output):
    """DeepGEMM names the exact cache key only after publishing its cubin."""
    output = Path(output)
    cache = output / "cache" / "deepgemm" / "cache"
    entries = [
        {"cache": "deepgemm", "key": path.name, "path": str(path)}
        for path in sorted(cache.glob("*"))
        if (path / ".committed").is_file() and (path / "kernel.cubin").is_file()
    ]
    if entries:
        (output / "misses" / "deepgemm-keys.json").write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    write_deepgemm_receipt(sys.argv[1])
