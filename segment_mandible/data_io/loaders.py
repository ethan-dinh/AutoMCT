"""
I/O utilities: loading BMP stacks, TIFF volumes, and NRRD files, saving TIFF/NRRD outputs.
"""

import glob
import logging
import os

import nrrd
import numpy as np
import tifffile as tiff
from PIL import Image

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


def load_nrrd(input_path: str) -> np.ndarray | None:
    """
    Load an NRRD file into a 3D numpy array.

    Parameters:
        input_path: Path to the .nrrd file.

    Returns:
        3D numpy array with shape (depth, height, width), or None if the
        file does not exist.
    """
    if not os.path.exists(input_path):
        logger.error("File %s does not exist", input_path)
        return None

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

    return data


def load_tiff(input_path: str) -> np.ndarray | None:
    """
    Load a multi-page TIFF file.

    Parameters:
        input_path: Path to the TIFF file.

    Returns:
        3D numpy array, or None if the file does not exist.
    """
    if not os.path.exists(input_path):
        logger.error("File %s does not exist", input_path)
        return None
    return tiff.imread(input_path)


def save_ct_volume_as_tiff(
    volume: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
) -> None:
    """
    Save a 3D CT volume as a multi-page TIFF.

    Parameters:
        volume: 3D array of CT data (e.g. int16 or float32).
        output_dir: Sub-directory name under the save root.
        name: Base filename (without extension).
        base_dir: Root directory for output. Defaults to ./segmentation_results.
    """
    root = base_dir if base_dir is not None else "./segmentation_results"
    out_root = os.path.join(root, output_dir)
    os.makedirs(out_root, exist_ok=True)

    logger.info(
        "Saving volume — shape: %s, dtype: %s, min: %s, max: %s",
        volume.shape,
        volume.dtype,
        np.min(volume),
        np.max(volume),
    )

    if volume.dtype == np.float64:
        logger.info("Downgrading float64 to float32")
        volume = volume.astype(np.float32)

    if volume.dtype not in [np.int16, np.float32]:
        volume = volume.astype(np.int16)

    output_path = os.path.join(out_root, f"{name}.tif")
    tiff.imwrite(output_path, volume, dtype=volume.dtype, compression="zlib", bigtiff=True)


def save_mask_as_tiff(
    mask: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
) -> None:
    """
    Save a binary mask as a uint8 TIFF (0/1 values) readable by 3D Slicer.

    Using TIFF guarantees axis alignment with volumes saved by save_ct_volume_as_tiff,
    since both use tifffile with the same (Z, Y, X) memory layout.

    Parameters:
        mask: 3D boolean or uint8 array (non-zero = foreground).
        output_dir: Sub-directory name under the save root.
        name: Base filename (without extension).
        base_dir: Root directory for output. Defaults to ./segmentation_results.
    """
    root = base_dir if base_dir is not None else "./segmentation_results"
    out_root = os.path.join(root, output_dir)
    os.makedirs(out_root, exist_ok=True)

    mask_uint8 = (mask > 0).astype(np.uint8)
    output_path = os.path.join(out_root, f"{name}.tif")
    tiff.imwrite(output_path, mask_uint8, dtype=np.uint8, compression="zlib", bigtiff=True)
    logger.info("Saved mask to %s", output_path)


def save_as_nrrd(
    volume: np.ndarray,
    output_dir: str,
    name: str,
    base_dir: str | None = None,
) -> None:
    """
    Save a 3D volume as an NRRD file.

    The volume is stored in (Z, Y, X) order consistent with the rest of the
    pipeline, then transposed to (X, Y, Z) for NRRD convention.

    Parameters:
        volume: 3D array to save.
        output_dir: Sub-directory name under the save root.
        name: Base filename (without extension).
        base_dir: Root directory for output. Defaults to ./segmentation_results.
    """
    root = base_dir if base_dir is not None else "./segmentation_results"
    out_root = os.path.join(root, output_dir)
    os.makedirs(out_root, exist_ok=True)

    if volume.dtype == np.float64:
        volume = volume.astype(np.float32)

    # Transpose from pipeline (Z, Y, X) back to NRRD (X, Y, Z)
    data = np.transpose(volume, (2, 1, 0))

    output_path = os.path.join(out_root, f"{name}.nrrd")
    nrrd.write(output_path, data, compression_level=9)
    logger.info("Saved NRRD to %s", output_path)
