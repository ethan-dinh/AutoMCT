"""Build one view's preview files for the studio viewer (see studio/volumes.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from studio.runners._common import read_spec, setup_logging
from studio.volumes import build_view


def main() -> int:
    setup_logging()
    spec = read_spec("Build a viewer preview")
    meta = build_view(
        Path(spec["out"]),
        view=spec["view"],
        base=spec["base"],
        overlays=spec.get("overlays") or [],
        budget=int(spec.get("budget") or 24_000_000),
        masked=bool(spec.get("masked")),
        z_first=bool(spec.get("z_first")),
        unwrap=bool(spec.get("unwrap")),
        crop_to_tissue=bool(spec.get("crop_to_tissue", True)),
        source_kind=spec.get("source_kind"),
        voxel_mm=spec.get("voxel_mm"),
        log=lambda message: print(message, flush=True),
    )
    print("@@result " + json.dumps({"view": meta["view"], "shape": meta["shape"],
                                   "factors": meta["factors"], "counts": meta["counts"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
