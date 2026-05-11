"""
Segmentation helper functions: thresholding, labeling, morphological ops.
Replaces the automct.segmentation dependency.
"""

import logging

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    ball,
    closing,
    opening,
    disk,
    remove_small_objects,
)
from tqdm import tqdm
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Remove small islands
# ------------------------------------------------------------------
def remove_small_islands(mask: np.ndarray, min_voxels: int = 2000, connectivity: int = 3) -> np.ndarray:
    """
    Remove connected components smaller than min_voxels from a 3D binary mask.
    connectivity=1 -> 6-connected
    connectivity=2 -> 18-connected
    connectivity=3 -> 26-connected
    """
    structure = ndi.generate_binary_structure(rank=mask.ndim, connectivity=connectivity)
    labels, num_labels = ndi.label(mask, structure=structure) # type: ignore

    if num_labels == 0:
        return mask

    counts = np.bincount(labels.ravel())
    # counts[0] is background
    keep_labels = np.flatnonzero(counts >= min_voxels)
    keep_labels = keep_labels[keep_labels != 0]

    cleaned = np.isin(labels, keep_labels)
    return cleaned


# ------------------------------------------------------------------
# Slice-level segmentation
# ------------------------------------------------------------------

def segment_slice(
    slice_2d: np.ndarray,
    method: str = "otsu",
    min_area: int = 100,
    **kwargs,
) -> np.ndarray:
    """
    Segment a 2D slice and return a labeled image.

    Parameters:
        slice_2d: 2D grayscale array.
        method: 'otsu' (only supported method here).
        min_area: Minimum region area to keep.
        **kwargs: Passed to the method implementation (e.g. nbins).

    Returns:
        Labeled integer array.
    """
    if method == "otsu":
        return _segment_otsu(slice_2d, min_area, **kwargs)
    raise ValueError(f"Unknown segmentation method: {method}")


def _segment_otsu(
    slice_2d: np.ndarray,
    min_area: int = 100,
    **kwargs,
) -> np.ndarray:
    thresh = threshold_otsu(slice_2d)
    binary = slice_2d > thresh

    if kwargs.get("morphology", True):
        binary = opening(binary)
        binary = closing(binary)

    binary = remove_small_objects(binary, min_size=min_area)
    return label(binary) # type: ignore


def label_slice(slice_2d: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Label connected components in a 2D binary mask."""
    return label(slice_2d, connectivity=connectivity) # type: ignore


# ------------------------------------------------------------------
# Volume-level segmentation
# ------------------------------------------------------------------

def segment_volume(
    volume: np.ndarray,
    method: str = "otsu",
    min_area: int = 100,
    **kwargs,
) -> np.ndarray:
    """
    Segment a 3D volume slice-by-slice and return a labeled volume.
    """
    depth = volume.shape[0]
    labeled_volume = np.zeros_like(volume, dtype=np.int32)
    logger.info("Segmenting %d slices using '%s' method", depth, method)
    for i in tqdm(range(depth), desc="Segmenting slices"):
        labeled_volume[i] = segment_slice(volume[i], method, min_area, **kwargs)
    return labeled_volume


def label_3d_volume(binary_volume: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Label connected components in a 3D binary volume."""
    return label(binary_volume, connectivity=connectivity) # type: ignore


def convert_to_binary(volume: np.ndarray, fill_holes: bool = False) -> np.ndarray:
    """
    Convert a volume to a boolean binary mask.

    Parameters:
        fill_holes: If True, apply closing before thresholding to fill gaps.
    """
    if fill_holes:
        volume = closing(volume)
    return volume > 0


# ------------------------------------------------------------------
# Region queries
# ------------------------------------------------------------------

def get_largest_region(labeled_item: np.ndarray) -> int:
    """Return the label of the largest region in a labeled array."""
    regions = regionprops(labeled_item)
    return max(regions, key=lambda r: r.area).label


def get_region_intensity(
    slice_2d: np.ndarray,
    labeled_slice: np.ndarray,
    region_label: int,
) -> float:
    """
    Return the mean intensity of a labeled region.

    Returns 0.0 if the region is not found.
    """
    for region in regionprops(labeled_slice, intensity_image=slice_2d):
        if region.label == region_label:
            return float(region.intensity_mean)
    return 0.0


# ------------------------------------------------------------------
# Morphological operations
# ------------------------------------------------------------------

def erode_mask(
    mask: np.ndarray,
    radius: int = 1,
    if_2d: bool = False,
) -> np.ndarray:
    """Erode a binary mask with a disk (2D) or ball (3D) structuring element."""
    footprint = disk(radius) if if_2d else ball(radius)
    return binary_erosion(mask, footprint)


def dilate_mask(
    mask: np.ndarray,
    radius: int = 1,
    if_2d: bool = False,
) -> np.ndarray:
    """Dilate a binary mask with a disk (2D) or ball (3D) structuring element."""
    footprint = disk(radius) if if_2d else ball(radius)
    return binary_dilation(mask, footprint)


def morphological_closing(volume: np.ndarray, ball_radius: int = 1) -> np.ndarray:
    """Apply 3D binary closing with a ball structuring element."""
    return closing(volume, ball(ball_radius))
