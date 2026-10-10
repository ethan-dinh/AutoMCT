"""Copy raw data into a sample's raw/ folder, reporting progress by bytes."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from studio.runners._common import progress, read_spec, setup_logging

CHUNK = 16 * 1024 * 1024


def _files(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return [p for p in sorted(src.rglob("*")) if p.is_file() and not p.name.startswith("._")]


def main() -> int:
    setup_logging()
    spec = read_spec("Copy raw data into the workspace")
    src, dst = Path(spec["src"]), Path(spec["dst"])
    files = _files(src)
    total = sum(f.stat().st_size for f in files) or 1
    done = 0
    print(f"Copying {len(files)} file(s), {total / 1e9:.2f} GB, from {src}", flush=True)
    base = dst if src.is_file() else dst / src.name
    for f in files:
        target = base / (f.name if src.is_file() else f.relative_to(src))
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "rb") as fin, open(target, "wb") as fout:
            while True:
                block = fin.read(CHUNK)
                if not block:
                    break
                fout.write(block)
                done += len(block)
                progress(done / total, f"Copied {done / 1e9:.2f} of {total / 1e9:.2f} GB")
        shutil.copystat(f, target)
    result = dst / src.name
    print("@@result " + json.dumps({"path": str(result)}), flush=True)
    return 0 if result.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
