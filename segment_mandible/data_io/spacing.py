"""
Voxel spacing: reading it off an input volume, tracking it through the
lossless reorientation, and writing it into output files.

The segmentation pipeline works purely in voxel indices, so spacing is
carried alongside the array rather than baked into it. It is read once at
load time, permuted to match ``reorient_mandible``'s axis shuffle, and
written into whichever container the CLI was asked for.

Both output formats are ``(Z, Y, X)`` in memory:

  - NRRD stores a full "space directions" matrix, so spacing round-trips
    exactly and 3D Slicer / ITK-SNAP open the result at true physical size.
  - TIFF has no 3-D geometry field. ImageJ's convention (XResolution /
    YResolution plus a ``spacing=`` entry in the ImageDescription) is the
    closest thing, and is what Fiji reads back as voxel size.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Fallback when an input carries no spacing at all (BMP stacks, TIFFs
# written without resolution tags). 1 mm isotropic keeps the array
# geometrically meaningful without inventing a scanner-specific number.
DEFAULT_SPACING_MM = (1.0, 1.0, 1.0)


def spacing_from_nrrd_header(header: dict) -> tuple[float, float, float] | None:
    """
    Per-axis voxel spacing in mm, in the pipeline's (Z, Y, X) order.

    ``load_nrrd`` transposes the on-disk (X, Y, Z) array to (Z, Y, X), so the
    spacing derived here is reversed to match. Row norms are used rather than
    the matrix diagonal so oblique/gantry-tilted acquisitions -- where an
    array axis is not aligned with any single L/P/S axis -- still give the
    correct physical step along that axis.

    Returns None when the header carries no usable geometry, so the caller
    can fall back rather than silently assuming 1 mm.
    """
    directions = header.get("space directions")
    if directions is not None:
        rows = []
        for row in np.asarray(directions, dtype=object):
            # A "none" row marks a non-spatial axis (e.g. a colour channel);
            # NRRD writes it as None, which asarray leaves as an object.
            if row is None or (isinstance(row, float) and np.isnan(row)):
                continue
            row = np.asarray(row, dtype=float)
            if row.size == 3 and not np.isnan(row).any():
                rows.append(row)
        if len(rows) == 3:
            spacing_xyz = np.linalg.norm(np.asarray(rows, dtype=float), axis=1)
            if np.all(spacing_xyz > 0):
                # On-disk rows are (X, Y, Z); the pipeline wants (Z, Y, X).
                return (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

    spacings = header.get("spacings")
    if spacings is not None:
        values = np.asarray(spacings, dtype=float)
        if values.size == 3 and np.all(np.isfinite(values)) and np.all(values > 0):
            return (float(values[2]), float(values[1]), float(values[0]))

    return None


def _parse_description(page) -> dict:
    """
    Recover key/value metadata from a TIFF ImageDescription tag.

    Handles both the newline-separated ``key=value`` form ImageJ uses and the
    JSON object tifffile writes when handed a ``metadata=`` dict.
    """
    tag = page.tags.get("ImageDescription")
    if tag is None or not isinstance(tag.value, str):
        return {}

    text = tag.value.strip()
    if text.startswith("{"):
        import json
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}

    fields = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    return fields


def spacing_from_tiff(path: str) -> tuple[float, float, float] | None:
    """
    Recover (Z, Y, X) spacing in mm from an ImageJ-style TIFF.

    Mirrors what ``save_volume`` writes: XResolution/YResolution as
    pixels-per-unit, and the Z step plus the unit name in the ImageJ
    ImageDescription. Returns None if the file carries no resolution tags.
    """
    import tifffile as tiff

    try:
        with tiff.TiffFile(path) as handle:
            page = handle.pages.first
            meta = dict(handle.imagej_metadata or {})

            if not meta:
                # BigTIFF cannot carry tifffile's imagej=True flag, so the
                # ImageJ key=value description written by _write_tiff is not
                # surfaced as imagej_metadata. Parse it directly; also covers
                # tifffile's JSON-style description from older outputs.
                meta = _parse_description(page)

            unit = str(meta.get("unit", "")).lower()
            # ImageJ writes "micron"/"um" for micrometres; everything the
            # converters produce is mm, which is also NRRD's unit.
            to_mm = 1e-3 if unit in ("micron", "microns", "um", "\xb5m") else 1.0

            def _resolution(tag_name: str) -> float | None:
                tag = page.tags.get(tag_name)
                if tag is None:
                    return None
                value = tag.value
                # Stored as a (numerator, denominator) rational = px per unit.
                if isinstance(value, tuple) and len(value) == 2 and value[1] and value[0]:
                    px_per_unit = value[0] / value[1]
                elif isinstance(value, (int, float)) and value:
                    px_per_unit = float(value)
                else:
                    return None
                return (1.0 / px_per_unit) * to_mm if px_per_unit > 0 else None

            dx = _resolution("XResolution")
            dy = _resolution("YResolution")
            dz_raw = meta.get("spacing")
            try:
                dz = float(dz_raw) * to_mm if dz_raw else None
            except (TypeError, ValueError):
                dz = None

            if dx is None and dy is None and dz is None:
                return None

            # A TIFF with only in-plane resolution is common; assume the Z
            # step matches Y rather than dropping the spacing entirely.
            dy = dy if dy is not None else dx
            dx = dx if dx is not None else dy
            dz = dz if dz is not None else dy
            if dz is None or dy is None or dx is None or not (dz > 0 and dy > 0 and dx > 0):
                return None
            return (float(dz), float(dy), float(dx))
    except Exception as e:  # pragma: no cover - malformed/unreadable tags
        logger.warning("Could not read voxel spacing from %s: %s", path, e)
        return None


def permute_spacing_for_record(
    spacing: tuple[float, float, float],
    record: dict | None,
) -> tuple[float, float, float]:
    """
    Map spacing through the axis-permuting steps of ``reorient_mandible``.

    Only ``perm`` and ``swap12`` move data between axes; the crop and the
    flips keep each axis where it was, so they leave spacing untouched. Skipping
    this would attach the wrong physical size to each axis of an anisotropic
    scan -- the array would look right and measure wrong.
    """
    if not record:
        return spacing

    result = list(spacing)

    perm = record.get("perm")
    if perm is not None:
        result = [result[int(axis)] for axis in perm]

    if record.get("swap12"):
        result[1], result[2] = result[2], result[1]

    return (float(result[0]), float(result[1]), float(result[2]))
