"""
I/O utilities: loading BMP stacks, TIFF volumes, and NRRD files, saving TIFF/NRRD outputs.

Outputs are written by :func:`save_volume` / :func:`save_mask`, which take the
container format (``"tif"`` or ``"nrrd"``) as an argument so the CLI's
``--format`` flag selects it. Both bake the voxel spacing into the file, so a
saved result opens at true physical size instead of as a unitless voxel grid.
"""

import glob
import json
import logging
import os
from typing import Literal, overload

import nrrd
import numpy as np
import tifffile as tiff
from PIL import Image

from .spacing import (
    DEFAULT_SPACING_MM,
    spacing_from_nrrd_header,
    spacing_from_tiff,
)

logger = logging.getLogger(__name__)


def load_bmp_stack(
    directory: str,
    file_pattern: str = "*.bmp",
    exclude_pattern: str = '',
    sort_by: str = "name",
) -> np.ndarray:
    """
    Load a stack of BMP files into a 3D volume.

    Parameters:
        directory: Path to directory containing BMP files.
        file_pattern: Glob pattern for file matching.
        exclude_pattern: Glob pattern for files to exclude.
        sort_by: Sorting method ('name', 'number', 'date').

    Returns:
        3D numpy array with shape (depth, height, width).
    """
    pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(pattern)

    if exclude_pattern != '':
        full_exclude_pattern = os.path.join(directory, exclude_pattern)
        excluded_files = set(
            os.path.join(directory, os.path.basename(p))
            for p in glob.glob(full_exclude_pattern)
        )
        logger.info("Excluding files: %s", excluded_files)
        file_list = [f for f in file_list if f not in excluded_files]

    if not file_list:
        raise FileNotFoundError(f"No BMP files found in {directory}")

    if sort_by == "name":
        file_list.sort()
    elif sort_by == "number":
        file_list.sort(
            key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))))
        )
    elif sort_by == "date":
        file_list.sort(key=lambda x: int(os.path.getmtime(x)))

    logger.info("Loading %d BMP files from %s", len(file_list), directory)

    first_img = Image.open(file_list[0]).convert("L")
    width, height = first_img.size
    volume = np.zeros((len(file_list), height, width), dtype=np.uint8)

    for i, file_path in enumerate(file_list):
        img = Image.open(file_path).convert("L")
        img_array = np.array(img)
        if img_array.shape != (height, width):
            raise ValueError(
                f"Image {file_path} has inconsistent dimensions: "
                f"expected {(height, width)}, got {img_array.shape}"
            )
        volume[i] = img_array

    logger.info("Volume loaded successfully: %s", volume.shape)
    return volume


SpacingMM = tuple[float, float, float]


@overload
def load_nrrd(input_path: str, return_spacing: Literal[False] = False) -> np.ndarray | None: ...
@overload
def load_nrrd(
    input_path: str, return_spacing: Literal[True]
) -> tuple[np.ndarray | None, SpacingMM | None]: ...
def load_nrrd(input_path: str, return_spacing: bool = False) -> np.ndarray | None | tuple[np.ndarray | None, SpacingMM | None]:
    """
    Load an NRRD file into a 3D numpy array.

    Parameters:
        input_path: Path to the .nrrd file.
        return_spacing: If True, return ``(volume, spacing_zyx_mm)`` instead of
            just the volume. ``spacing`` is None when the header carries no
            usable geometry.

    Returns:
        3D numpy array with shape (depth, height, width), or None if the
        file does not exist. With ``return_spacing``, a ``(volume, spacing)``
        tuple (``(None, None)`` on a missing file).
    """
    if not os.path.exists(input_path):
        logger.error("File %s does not exist", input_path)
        return (None, None) if return_spacing else None

    data, header = nrrd.read(input_path)
    logger.info(
        "Loaded NRRD: shape=%s, dtype=%s, space=%s",
        data.shape, data.dtype, header.get("space", "N/A"),
    )

    # NRRD files often store data as (X, Y, Z). Transpose to (Z, Y, X) to
    # match the (depth, height, width) convention used by the rest of the
    # pipeline.
    if data.ndim == 3 and header.get("space", "").startswith("left-posterior-superior"):
        data = np.transpose(data, (2, 1, 0))
        logger.info("Transposed LPS NRRD to (Z, Y, X): %s", data.shape)
    elif data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
        logger.info("Transposed NRRD to (Z, Y, X): %s", data.shape)

    if return_spacing:
        spacing = spacing_from_nrrd_header(header)
        if spacing is not None:
            logger.info("NRRD voxel spacing (Z, Y, X) mm: %s", spacing)
        else:
            logger.warning("NRRD %s carries no voxel spacing in its header", input_path)
        return data, spacing

    return data


@overload
def load_tiff(input_path: str, return_spacing: Literal[False] = False) -> np.ndarray | None: ...
@overload
def load_tiff(
    input_path: str, return_spacing: Literal[True]
) -> tuple[np.ndarray | None, SpacingMM | None]: ...
def load_tiff(input_path: str, return_spacing: bool = False) -> np.ndarray | None | tuple[np.ndarray | None, SpacingMM | None]:
    """
    Load a multi-page TIFF file.

    Parameters:
        input_path: Path to the TIFF file.
        return_spacing: If True, return ``(volume, spacing_zyx_mm)`` instead of
            just the volume. ``spacing`` is None when the file has no
            resolution tags.

    Returns:
        3D numpy array, or None if the file does not exist. With
        ``return_spacing``, a ``(volume, spacing)`` tuple (``(None, None)`` on
        a missing file).
    """
    if not os.path.exists(input_path):
        logger.error("File %s does not exist", input_path)
        return (None, None) if return_spacing else None
    volume = tiff.imread(input_path)
    if return_spacing:
        spacing = spacing_from_tiff(input_path)
        if spacing is not None:
            logger.info("TIFF voxel spacing (Z, Y, X) mm: %s", spacing)
        return volume, spacing
    return volume


OUTPUT_FORMATS = ("tif", "nrrd")

# NRRD "space directions" rows are (L, P, S) vectors per array axis. This is
# the axial, no-orientation-tags convention used by tools/volume_pipeline.py:
# columns (array X) run along L, rows (array Y) along P, slices (array Z)
# along S. Matching it means a segmentation output overlays the converted
# scan it came from in Slicer/ITK-SNAP.
def _directions_from_spacing(spacing_zyx: tuple[float, float, float]) -> list[list[float]]:
    dz, dy, dx = (float(v) for v in spacing_zyx)
    return [
        [0.0, 0.0, dz],   # array axis 0 (Z / slices)  -> S
        [0.0, dy, 0.0],   # array axis 1 (Y / rows)    -> P
        [dx, 0.0, 0.0],   # array axis 2 (X / columns) -> L
    ]


def _resolve_output_dir(output_dir: str, base_dir: str | None) -> str:
    root = base_dir if base_dir is not None else "./segmentation_results"
    out_root = os.path.join(root, output_dir)
    os.makedirs(out_root, exist_ok=True)
    return out_root


def _normalize_format(output_format: str) -> str:
    fmt = str(output_format).lower().lstrip(".")
    if fmt == "tiff":
        fmt = "tif"
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format {output_format!r}; expected one of {OUTPUT_FORMATS}"
        )
    return fmt


def _spacing_or_default(
    spacing: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    """
    The (dz, dy, dx) spacing as a plain 3-tuple of floats, or 1 mm isotropic
    when none is known. Callers may hand in any 3-sequence (a list, a NumPy
    array read back from a cache), so it is unpacked rather than trusted.
    """
    if spacing is None:
        return DEFAULT_SPACING_MM
    dz, dy, dx = spacing
    return (float(dz), float(dy), float(dx))


def _write_tiff(
    path: str,
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    extra_metadata: dict | None = None,
) -> None:
    """
    Write a (Z, Y, X) volume as a multi-page TIFF with ImageJ voxel geometry.

    TIFF has no native 3-D geometry field, so the voxel size goes in the
    ImageJ-style tags Fiji reads back: XResolution/YResolution as
    pixels-per-unit, and the Z step as ``spacing`` in the ImageDescription.
    Units are mm throughout, matching the NRRD writer.
    """
    dz, dy, dx = (float(v) for v in spacing_zyx)

    # imagej=True cannot be combined with BigTIFF (which the volumes here need),
    # and tifffile's metadata= dict would be written as JSON, which ImageJ does
    # not parse. So the ImageJ ImageDescription is built by hand: it is the only
    # place a TIFF can record the Z step, and Fiji reads the voxel depth from
    # exactly these newline-separated key=value lines.
    description_fields = {
        "ImageJ": "1.54f",
        "images": int(volume.shape[0]),
        "slices": int(volume.shape[0]),
        "unit": "mm",
        "spacing": dz,
    }
    if extra_metadata:
        description_fields.update(extra_metadata)
    description = "\n".join(f"{k}={v}" for k, v in description_fields.items()) + "\n"

    tiff.imwrite(
        path,
        volume,
        dtype=volume.dtype,
        compression="zlib",
        bigtiff=True,
        # In-plane size goes in the standard resolution tags as px-per-mm.
        resolution=(1.0 / dx, 1.0 / dy),
        resolutionunit="NONE",
        description=description,
    )


def _write_nrrd(
    path: str,
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    extra_header: dict | None = None,
) -> None:
    """
    Write a (Z, Y, X) volume as NRRD with voxel geometry in the header.

    The array is transposed to (X, Y, Z) on the way out -- the layout every
    other NRRD in this project uses -- and the direction rows are reversed
    alongside it so the physical geometry is unchanged by the storage order.
    """
    data = np.ascontiguousarray(np.transpose(volume, (2, 1, 0)))
    directions = _directions_from_spacing(spacing_zyx)[::-1]

    header = {
        "space": "left-posterior-superior",
        "space directions": directions,
        "space origin": [0.0, 0.0, 0.0],
        "kinds": ["domain", "domain", "domain"],
    }
    if extra_header:
        header.update(extra_header)

    nrrd.write(path, data, header, compression_level=1)


def save_volume(
    volume: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
    output_format: str = "tif",
    spacing: tuple[float, float, float] | None = None,
) -> str:
    """
    Save a 3D CT volume as a multi-page TIFF or an NRRD, with voxel size baked in.

    Parameters:
        volume: 3D array of CT data (e.g. int16 or float32), in (Z, Y, X) order.
        output_dir: Sub-directory name under the save root.
        name: Base filename (without extension).
        base_dir: Root directory for output. Defaults to ./segmentation_results.
        output_format: "tif" or "nrrd".
        spacing: Voxel size (dz, dy, dx) in mm. Falls back to 1 mm isotropic
            when the input carried no spacing.

    Returns:
        The path written.
    """
    fmt = _normalize_format(output_format)
    out_root = _resolve_output_dir(output_dir, base_dir)
    spacing_zyx = _spacing_or_default(spacing)

    logger.info(
        "Saving volume - shape: %s, dtype: %s, min: %s, max: %s, spacing (mm): %s",
        volume.shape,
        volume.dtype,
        np.min(volume),
        np.max(volume),
        spacing_zyx,
    )

    if volume.dtype == np.float64:
        logger.info("Downgrading float64 to float32")
        volume = volume.astype(np.float32)

    if volume.dtype not in [np.int16, np.float32]:
        volume = volume.astype(np.int16)

    output_path = os.path.join(out_root, f"{name}.{fmt}")
    if fmt == "tif":
        _write_tiff(output_path, volume, spacing_zyx)
    else:
        _write_nrrd(output_path, volume, spacing_zyx)

    logger.info("Saved volume to %s", output_path)
    return output_path


def save_mask(
    mask: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
    output_format: str = "tif",
    spacing: tuple[float, float, float] | None = None,
) -> str:
    """
    Save a binary mask as a uint8 (0/1) TIFF or NRRD readable by 3D Slicer.

    The mask shares the voxel geometry of the volume it was derived from, so
    the two overlay exactly when both are loaded into a viewer.

    Parameters:
        mask: 3D boolean or uint8 array (non-zero = foreground), (Z, Y, X).
        output_dir: Sub-directory name under the save root.
        name: Base filename (without extension).
        base_dir: Root directory for output. Defaults to ./segmentation_results.
        output_format: "tif" or "nrrd".
        spacing: Voxel size (dz, dy, dx) in mm.

    Returns:
        The path written.
    """
    fmt = _normalize_format(output_format)
    out_root = _resolve_output_dir(output_dir, base_dir)
    spacing_zyx = _spacing_or_default(spacing)

    mask_uint8 = (mask > 0).astype(np.uint8)
    output_path = os.path.join(out_root, f"{name}.{fmt}")
    if fmt == "tif":
        _write_tiff(output_path, mask_uint8, spacing_zyx)
    else:
        _write_nrrd(output_path, mask_uint8, spacing_zyx)

    logger.info("Saved mask to %s", output_path)
    return output_path


def save_ct_volume_as_tiff(
    volume: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> None:
    """
    Save a 3D CT volume as a multi-page TIFF.

    Thin wrapper kept for callers that always want TIFF; new code should call
    :func:`save_volume` with an explicit ``output_format``.
    """
    save_volume(
        volume, output_dir, name,
        base_dir=base_dir, output_format="tif", spacing=spacing,
    )


def save_mask_as_tiff(
    mask: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> None:
    """
    Save a binary mask as a uint8 TIFF (0/1 values) readable by 3D Slicer.

    Thin wrapper kept for callers that always want TIFF; new code should call
    :func:`save_mask` with an explicit ``output_format``.
    """
    save_mask(
        mask, output_dir, name,
        base_dir=base_dir, output_format="tif", spacing=spacing,
    )


def save_as_nrrd(
    volume: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> None:
    """
    Save a 3D volume as an NRRD file.

    Thin wrapper kept for callers that always want NRRD; new code should call
    :func:`save_volume` with an explicit ``output_format``.
    """
    save_volume(
        volume, output_dir, name,
        base_dir=base_dir, output_format="nrrd", spacing=spacing,
    )


def write_spacing_sidecar(
    output_dir: str,
    spacing: tuple[float, float, float] | None,
    base_dir: str | None = None,
    name: str = "voxel_size",
) -> str:
    """
    Record the voxel size next to a sample's outputs as JSON.

    Both containers already carry the spacing, but a plain-text record keeps
    it readable for downstream analysis scripts that measure in physical units
    without having to reopen a multi-GB volume.
    """
    out_root = _resolve_output_dir(output_dir, base_dir)
    spacing_zyx = _spacing_or_default(spacing)
    payload = {
        "spacing_mm_zyx": [float(v) for v in spacing_zyx],
        "unit": "mm",
        "is_default": spacing is None,
    }
    output_path = os.path.join(out_root, f"{name}.json")
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
