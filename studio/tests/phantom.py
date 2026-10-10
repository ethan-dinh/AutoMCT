"""
A small synthetic hemimandible, written in the segmentation pipeline's layout.

Air, a bone bar, a curved incisor tube with a brighter labial enamel band,
and three molars (M1 > M2 > M3) with enamel caps -- enough structure for every
studio code path to have real work to do, in a few MB. The intensities are in
the range of raw 16-bit scanner counts, not normalized.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from studio.paths import SEGMENT

AIR, BONE, DENTIN, ENAMEL = 60.0, 520.0, 640.0, 930.0
SPACING = (0.012, 0.012, 0.012)


def _noise(rng, volume, sigma):
    volume += rng.normal(0.0, sigma, volume.shape).astype(np.float32)


def mandible(shape=(180, 90, 72), seed=0):
    """Return (scan, masks) where masks = {"incisor", "bone", "molar", "enamel"}."""
    depth, height, width = shape
    rng = np.random.default_rng(seed)
    scan = np.full(shape, AIR, dtype=np.float32)
    z, y, x = np.indices(shape, dtype=np.float32)

    bone = (y > 34) & (y < 78) & (x > 10) & (x < 62) & (z > 8) & (z < depth - 8)

    # Incisor: a tube whose centre bows in y along z.
    cy = 56 + 12 * np.sin(np.pi * z / depth)
    cx = 36.0
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    incisor = (r < 10) & (z > 4) & (z < depth - 20)
    labial = incisor & (r > 7) & (y > cy)

    molar = np.zeros(shape, dtype=bool)
    enamel = np.zeros(shape, dtype=bool)
    for (z0, z1), half in (((96, 128), 11), ((131, 152), 9), ((155, 168), 7)):
        box = (z >= z0) & (z < z1) & (np.abs(x - 36) < half) & (y > 16) & (y < 40)
        cap = box & (y < 20)
        molar |= box
        enamel |= cap

    bone &= ~incisor & ~molar
    scan[bone] = BONE
    scan[incisor] = DENTIN
    scan[labial] = ENAMEL
    scan[molar] = DENTIN - 20
    scan[enamel] = ENAMEL + 20
    _noise(rng, scan, 12.0)
    return scan, {"incisor": incisor, "bone": bone, "molar": molar, "enamel": enamel}


def write_segmentation(root: Path, name: str, seed: int = 0, fmt: str = "nrrd") -> Path:
    """Write <root>/<name>/ exactly as segment_mandible's main.py lays it out."""
    if str(SEGMENT) not in sys.path:
        sys.path.insert(0, str(SEGMENT))
    from data_io import save_mask, save_volume, write_spacing_sidecar  # noqa: PLC0415

    scan, masks = mandible(seed=seed)
    folder = root / name
    save_volume(scan, name, f"{name}_reoriented_volume", base_dir=str(root), output_format=fmt, spacing=SPACING)
    for part in ("incisor", "bone", "molar"):
        save_volume(np.where(masks[part], scan, 0).astype(np.float32), os.path.join(name, "volumes"),
                    f"{name}_{part}_volume", base_dir=str(root), output_format=fmt, spacing=SPACING)
        save_mask(masks[part], os.path.join(name, "masks"), f"{name}_{part}_mask",
                  base_dir=str(root), output_format=fmt, spacing=SPACING)
    write_spacing_sidecar(name, SPACING, base_dir=str(root), name=f"{name}_voxel_size")
    return folder


def write_isq(path: Path, shape=(180, 90, 72), seed=0) -> Path:
    """A minimal Scanco ISQ: the 512-byte pre-header convert_ISQ_NRRD.py parses, then int16 voxels."""
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import struct  # noqa: PLC0415
    from convert_ISQ_NRRD import _FIELDS, _HEADER_SIZE, _MAGIC  # noqa: PLC0415

    scan, _ = mandible(shape, seed=seed)
    depth, height, width = shape
    header = bytearray(_HEADER_SIZE)
    header[:len(_MAGIC)] = _MAGIC
    um = [s * 1000 for s in SPACING]
    values = {
        "dim_x_pixels": width, "dim_y_pixels": height, "dim_z_pixels": depth,
        "dim_x_um": int(round(width * um[2])), "dim_y_um": int(round(height * um[1])),
        "dim_z_um": int(round(depth * um[0])), "slice_increment_um": int(round(um[0])),
        "header_data_offset_blocks": 0, "mu_scaling": 4096,
    }
    for field_name, offset, fmt in _FIELDS:
        if field_name in values and not fmt.endswith("s"):
            struct.pack_into("<" + fmt, header, offset, values[field_name])
    with open(path, "wb") as handle:
        handle.write(bytes(header))
        handle.write(np.clip(np.rint(scan), -32768, 32767).astype("<i2").tobytes())
    return path


__all__ = ["mandible", "write_segmentation", "write_isq", "SPACING", "math"]
