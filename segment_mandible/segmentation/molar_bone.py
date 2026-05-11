import logging

import numpy as np

from segmentation.utils import (
    erode_mask,
    label_3d_volume,
)

logger = logging.getLogger(__name__)

MIN_COMPONENT_SIZE = 3500


def _filter_small_components(
    labeled_volume: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, int]:
    """
    Remove connected components smaller than min_size voxels.
    Returns:
        filtered_labeled_volume, n_labels
    """
    labels, counts = np.unique(labeled_volume, return_counts=True)

    valid_labels = labels[(labels != 0) & (counts > min_size)]

    if valid_labels.size == 0:
        return np.zeros_like(labeled_volume), 0

    filtered = np.zeros_like(labeled_volume)
    for new_label, old_label in enumerate(valid_labels, start=1):
        filtered[labeled_volume == old_label] = new_label

    return filtered, int(valid_labels.size)


def segment_molar_bone(
    preprocessed_volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment molar and bone via component selection using max intensity.

    Strategy:
      - Segment mineralized tissue (foreground).
      - Label connected components.
      - Remove components smaller than MIN_COMPONENT_SIZE.
      - If fewer than 2 components exist, erode foreground and re-label
        until at least 2 components appear.
      - Component with the highest max intensity → molar.
      - All remaining components → bone.

    Returns:
        (bone_mask, molar_mask) — boolean masks.
    """
    foreground = preprocessed_volume > 0

    labeled_volume = label_3d_volume(foreground, connectivity=1)
    labeled_volume, n_labels = _filter_small_components(
        labeled_volume,
        MIN_COMPONENT_SIZE,
    )

    erosion_count = 0
    max_erosions = 20

    logger.info("Initial segmentation found %d component(s) > %d voxels", n_labels, MIN_COMPONENT_SIZE)

    while n_labels < 2 and erosion_count < max_erosions:
        foreground = erode_mask(foreground, radius=1)
        if not np.any(foreground):
            break

        labeled_volume = label_3d_volume(foreground, connectivity=1)
        labeled_volume, n_labels = _filter_small_components(
            labeled_volume,
            MIN_COMPONENT_SIZE,
        )
        erosion_count += 1

    if erosion_count > 0:
        logger.info(
            "Needed %d erosion(s) to reach %d component(s) > %d voxels",
            erosion_count,
            n_labels,
            MIN_COMPONENT_SIZE,
        )

    if n_labels == 0:
        return np.zeros_like(foreground), np.zeros_like(foreground)

    if n_labels == 1:
        logger.warning(
            "Only 1 valid component found after %d erosions; returning as bone",
            max_erosions,
        )
        return labeled_volume > 0, np.zeros_like(foreground)

    molar_label = max(
        range(1, n_labels + 1),
        key=lambda lbl: float(preprocessed_volume[labeled_volume == lbl].max()),
    )

    molar_mask = labeled_volume == molar_label
    bone_mask = (labeled_volume > 0) & ~molar_mask

    return bone_mask, molar_mask