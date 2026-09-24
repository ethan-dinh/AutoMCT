"""
DICOM series loading: assembling a directory of slices into a (Z, Y, X) volume
with voxel values and voxel spacing taken from the series' own headers.

Scanner output reaches this pipeline in several forms. NRRD and TIFF carry
their geometry in a single file header, but a DICOM series spreads it across
one header per slice, so three things have to be recovered here rather than
read off a single tag:

  - **Slice order.** Files are not reliably named in acquisition order, so
    slices are sorted by their physical position (``ImagePositionPatient``'s
    slice-axis component), falling back to ``InstanceNumber``.
  - **Voxel values.** Pixel data is stored as integers with a per-series
    affine map back to physical units (``RescaleSlope`` / ``RescaleIntercept``,
    and the ``DoseGridScaling``-style scaling some microCT exporters write).
    That map is *not* applied by default -- raw stored values are preserved,
    matching ``tools/convert_DICOM_NRRD.py``, which applies the rescale only
    behind its ``--apply-rescale`` flag. The pipeline min-max normalizes the
    volume before thresholding anything, so an affine rescale would not change
    a segmentation result, and keeping stored values leaves window/level free
    to adjust in downstream tools. Pass ``apply_rescale=True`` for physical
    units.
  - **Voxel spacing.** ``PixelSpacing`` gives the in-plane step; the Z step is
    measured from the gap between consecutive slice positions, which stays
    correct when ``SliceThickness`` disagrees with the real spacing (overlapping
    or gapped reconstructions) or is missing entirely.

Only the pieces the segmentation pipeline needs are recovered -- the full
orientation matrix and physical origin are the business of
``tools/convert_DICOM_NRRD.py``, which writes registered NRRDs. Here the volume
is handed on as a plain array plus a (dz, dy, dx) spacing tuple, matching what
:func:`~data_io.loaders.load_nrrd` and :func:`~data_io.loaders.load_tiff`
return.
"""

import logging
import os
from typing import Literal, overload

import numpy as np

logger = logging.getLogger(__name__)

# Extensions that mark a file as DICOM without reading it. DICOM series are
# frequently written with no extension at all, so an empty suffix counts too
# and the real test is the "DICM" magic (see _is_dicom_file).
DICOM_SUFFIXES = (".dcm", ".dicom", ".ima", "")

# Files that sit alongside a series but are not image slices.
_NON_SLICE_NAMES = {"dicomdir", "version"}


def _is_dicom_file(path: str) -> bool:
    """
    True if `path` looks like a DICOM file, by magic number rather than name.

    Part 10 files carry the ASCII marker "DICM" at byte 128, after the
    preamble. Checking those four bytes is far cheaper than a parse attempt
    across a directory of thousands of slices, and does not depend on a naming
    convention the scanner may not follow.
    """
    if not os.path.isfile(path):
        return False
    if os.path.basename(path).lower() in _NON_SLICE_NAMES:
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(132)[128:132] == b"DICM"
    except OSError:
        return False


def find_dicom_files(directory: str) -> list[str]:
    """
    DICOM slice files directly inside `directory`, unsorted.

    Hidden files are skipped, as are the DICOMDIR index and other non-image
    companions.
    """
    if not os.path.isdir(directory):
        return []
    return [
        path
        for name in sorted(os.listdir(directory))
        if not name.startswith(".")
        and _is_dicom_file(path := os.path.join(directory, name))
    ]


def is_dicom_dir(directory: str) -> bool:
    """True if `directory` holds DICOM slices directly in it."""
    return bool(find_dicom_files(directory))


def _slice_position(ds, normal: np.ndarray | None) -> float:
    """
    Sort key for one slice: its position along the stacking direction.

    Projecting ``ImagePositionPatient`` onto the slice normal orders slices
    correctly even for oblique acquisitions, where no single coordinate of the
    position increases monotonically. ``InstanceNumber`` is the fallback for
    series that omit position tags.
    """
    try:
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        if normal is not None:
            return float(np.dot(position, normal))
        return float(position[2])
    except Exception:
        pass
    try:
        return float(ds.InstanceNumber)
    except Exception:
        return 0.0


def _slice_normal(ds) -> np.ndarray | None:
    """Unit vector along the slice-stacking direction, from the in-plane cosines."""
    try:
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[0:3], iop[3:6])
        norm = np.linalg.norm(normal)
        if norm > 0:
            return normal / norm
    except Exception:
        pass
    return None


def spacing_from_dicom_headers(headers: list) -> tuple[float, float, float] | None:
    """
    Voxel spacing (dz, dy, dx) in mm from a sorted list of slice headers.

    In-plane spacing comes from ``PixelSpacing``, which DICOM stores as
    (row spacing, column spacing) = (dy, dx). The slice step is measured as the
    median distance between consecutive slice positions rather than read from
    ``SliceThickness``: thickness describes the reconstructed slab, which for
    overlapping or gapped reconstructions is not the spacing between slice
    centres. ``SliceThickness`` (then ``SpacingBetweenSlices``) is the fallback
    when positions are missing or degenerate.

    Returns None when the headers carry no usable in-plane spacing, so the
    caller can fall back rather than silently assuming 1 mm.
    """
    if not headers:
        return None

    first = headers[0]

    dy = dx = None
    for keyword in ("PixelSpacing", "ImagerPixelSpacing"):
        try:
            values = getattr(first, keyword)
            dy, dx = (float(v) for v in values)
            if dy > 0 and dx > 0:
                break
            dy = dx = None
        except Exception:
            dy = dx = None

    if dy is None or dx is None:
        logger.warning("DICOM series carries no usable PixelSpacing")
        return None

    dz = None
    if len(headers) >= 2:
        normal = _slice_normal(first)
        try:
            positions = np.asarray(
                [np.asarray(ds.ImagePositionPatient, dtype=float) for ds in headers]
            )
            if normal is not None:
                projected = positions @ normal
            else:
                projected = positions[:, 2]
            steps = np.abs(np.diff(projected))
            steps = steps[steps > 0]
            if steps.size:
                dz = float(np.median(steps))
                # A series with a varying slice step cannot be represented by a
                # single spacing; warn rather than silently averaging it away.
                if steps.size > 1 and np.ptp(steps) > 0.01 * dz:
                    logger.warning(
                        "DICOM slice spacing is not uniform (%.6g-%.6g mm); "
                        "using the median %.6g mm",
                        float(steps.min()), float(steps.max()), dz,
                    )
        except Exception:
            dz = None

    if dz is None or dz <= 0:
        for keyword in ("SpacingBetweenSlices", "SliceThickness"):
            try:
                candidate = float(getattr(first, keyword))
                if candidate > 0:
                    dz = candidate
                    logger.info(
                        "Using %s for the slice spacing (no usable slice positions)",
                        keyword,
                    )
                    break
            except Exception:
                continue

    if dz is None or dz <= 0:
        # In-plane spacing is known and microCT voxels are near-isotropic, so
        # matching the row spacing beats dropping the geometry entirely.
        dz = dy
        logger.warning(
            "DICOM series carries no slice spacing; assuming %.6g mm (= row spacing)",
            dz,
        )

    return (float(dz), float(dy), float(dx))


def _rescale_for(ds) -> tuple[float, float]:
    """
    The (slope, intercept) mapping stored pixel values to physical units.

    ``RescaleSlope``/``RescaleIntercept`` are the standard pair. Some microCT
    and RT exporters instead write ``DoseGridScaling`` as a bare multiplier,
    which is honoured here when no rescale slope is present.
    """
    slope = None
    intercept = 0.0
    try:
        slope = float(ds.RescaleSlope)
    except Exception:
        slope = None
    try:
        intercept = float(ds.RescaleIntercept)
    except Exception:
        intercept = 0.0

    if slope is None:
        try:
            candidate = float(ds.DoseGridScaling)
            if candidate != 0:
                slope = candidate
        except Exception:
            slope = None

    if slope is None or slope == 0:
        slope = 1.0
    return slope, intercept


def _volume_dtype(sample: np.ndarray, slope: float, intercept: float, apply_rescale: bool):
    """
    The dtype to assemble the volume in.

    Rescaling with a non-integer slope or intercept produces non-integer
    values, so those cases need float32. An integer slope and intercept map
    integers to integers, but can overflow the stored width (a slope of 1000 on
    int16 data will not fit), so int32 is used there. When nothing is applied
    the stored dtype is kept, which halves peak memory on a full-resolution
    scan versus promoting everything to float.
    """
    if not apply_rescale or (slope == 1.0 and intercept == 0.0):
        return sample.dtype
    if not np.issubdtype(sample.dtype, np.integer):
        return np.dtype(np.float32)
    if float(slope).is_integer() and float(intercept).is_integer():
        return np.dtype(np.int32)
    return np.dtype(np.float32)


@overload
def load_dicom_series(
    directory: str,
    return_spacing: Literal[False] = False,
    apply_rescale: bool = False,
) -> np.ndarray | None: ...
@overload
def load_dicom_series(
    directory: str,
    return_spacing: Literal[True],
    apply_rescale: bool = False,
) -> tuple[np.ndarray | None, tuple[float, float, float] | None]: ...
def load_dicom_series(
    directory: str,
    return_spacing: bool = False,
    apply_rescale: bool = False,
) -> (
    np.ndarray | None
    | tuple[np.ndarray | None, tuple[float, float, float] | None]
):
    """
    Load a directory of DICOM slices into a 3D volume.

    Slices are sorted by physical position and stacked into (Z, Y, X). Voxel
    values are left as stored: no rescaling or normalization is applied, which
    matches ``tools/convert_DICOM_NRRD.py`` and keeps window/level free to
    adjust downstream. The pipeline min-max normalizes before thresholding, so
    an affine rescale would not move a segmentation boundary anyway.

    Parameters:
        directory: Path to a directory containing DICOM slice files.
        return_spacing: If True, return ``(volume, spacing_zyx_mm)`` instead of
            just the volume. ``spacing`` is None when the headers carry no
            usable geometry.
        apply_rescale: Apply the header's rescale slope/intercept, converting
            stored values to the series' physical units. Default False (raw
            stored values).

    Returns:
        3D numpy array with shape (depth, height, width), or None if the
        directory holds no readable DICOM slices. With ``return_spacing``, a
        ``(volume, spacing)`` tuple (``(None, None)`` on failure).
    """
    import pydicom

    failed = (None, None) if return_spacing else None

    if not os.path.isdir(directory):
        logger.error("DICOM directory %s does not exist", directory)
        return failed

    paths = find_dicom_files(directory)
    if not paths:
        logger.error("No DICOM files found in %s", directory)
        return failed

    logger.info("Reading %d DICOM headers from %s", len(paths), directory)
    entries = []
    for path in paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as e:
            logger.warning("Skipping unreadable DICOM %s: %s", path, e)
            continue
        # Header-only files (e.g. a stray structured report) carry no image.
        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue
        entries.append((path, ds))

    if not entries:
        logger.error("No readable DICOM image slices in %s", directory)
        return failed

    # A directory can hold more than one series; mixing them would interleave
    # unrelated slices into one volume. Keep the largest series and say so.
    series_uids = {getattr(ds, "SeriesInstanceUID", None) for _, ds in entries}
    if len(series_uids) > 1:
        counts: dict = {}
        for _, ds in entries:
            counts.setdefault(getattr(ds, "SeriesInstanceUID", None), 0)
            counts[getattr(ds, "SeriesInstanceUID", None)] += 1
        keep = max(counts, key=counts.__getitem__)
        logger.warning(
            "%s holds %d DICOM series; loading the largest (%d of %d slices)",
            directory, len(series_uids), counts[keep], len(entries),
        )
        entries = [(p, ds) for p, ds in entries if getattr(ds, "SeriesInstanceUID", None) == keep]

    normal = _slice_normal(entries[0][1])
    entries.sort(key=lambda item: _slice_position(item[1], normal))
    headers = [ds for _, ds in entries]
    sorted_paths = [path for path, _ in entries]

    spacing = spacing_from_dicom_headers(headers)

    slope, intercept = _rescale_for(headers[0])
    if slope != 1.0 or intercept != 0.0:
        if apply_rescale:
            logger.info(
                "Applying DICOM rescale: value = stored * %g + %g", slope, intercept
            )
        else:
            logger.info(
                "Keeping raw stored values; the header's rescale "
                "(value = stored * %g + %g) is not applied",
                slope, intercept,
            )

    first = pydicom.dcmread(sorted_paths[0])
    first_array = first.pixel_array
    if first_array.ndim != 2:
        logger.error(
            "Expected single-frame DICOM slices; %s has shape %s",
            sorted_paths[0], first_array.shape,
        )
        return failed

    rows, cols = first_array.shape
    dtype = _volume_dtype(first_array, slope, intercept, apply_rescale)
    logger.info(
        "Assembling DICOM volume: %d x %d x %d, dtype=%s",
        len(sorted_paths), rows, cols, dtype,
    )

    volume = np.empty((len(sorted_paths), rows, cols), dtype=dtype)
    for i, path in enumerate(sorted_paths):
        ds = pydicom.dcmread(path) if i else first
        array = ds.pixel_array
        if array.shape != (rows, cols):
            raise ValueError(
                f"DICOM slice {path} has inconsistent dimensions: "
                f"expected {(rows, cols)}, got {array.shape}"
            )
        if apply_rescale:
            # Each slice may carry its own rescale; honouring the per-slice
            # values keeps a series with a varying intercept on one scale.
            s, b = _rescale_for(ds)
            if s != 1.0 or b != 0.0:
                array = array.astype(np.float64, copy=False) * s + b
        volume[i] = array.astype(dtype, copy=False)

    logger.info(
        "DICOM volume loaded: shape=%s, dtype=%s, min=%s, max=%s",
        volume.shape, volume.dtype, np.min(volume), np.max(volume),
    )

    if return_spacing:
        if spacing is not None:
            logger.info("DICOM voxel spacing (Z, Y, X) mm: %s", spacing)
        else:
            logger.warning("DICOM series %s carries no voxel spacing", directory)
        return volume, spacing

    return volume
