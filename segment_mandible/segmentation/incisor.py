import logging

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from tqdm import tqdm

from segmentation.utils import (
    convert_to_binary,
    dilate_mask,
    get_largest_region,
    label_3d_volume,
    label_slice,
    segment_slice,
)

logger = logging.getLogger(__name__)

def segment_incisor(
    preprocessed_volume: np.ndarray,
    min_tip_intensity: float,
) -> np.ndarray:
    """
    Segment the incisor from the mandible volume.
    """
    from scipy.ndimage import binary_propagation

    def _try_erode_separate(
        mask_2d: np.ndarray, ref_centroid: np.ndarray, max_iters: int = 15
    ):
        """
        Iteratively erode mask_2d until ≥2 components appear, then reconstruct
        the incisor component by watershed-partitioning the original mask using
        the eroded components as seeds. Returns None if separation never occurs.
        """
        from skimage.segmentation import watershed

        struct_cross = ndi.generate_binary_structure(2, 1)
        struct_full = np.ones((3, 3), dtype=bool)
        eroded = mask_2d.copy()
        for i in range(1, max_iters + 1):
            eroded = ndi.binary_erosion(eroded, structure=struct_cross)
            if not np.any(eroded):
                break
            labeled_eroded, n_labels = ndi.label(eroded, structure=struct_full) # type: ignore
            if n_labels >= 2:
                logger.debug(
                    "Erosion separated merged region into %d parts after %d iteration(s)",
                    n_labels,
                    i,
                )
                best_label, best_dist = None, np.inf
                for lbl in range(1, n_labels + 1):
                    ys, xs = np.where(labeled_eroded == lbl)
                    c = np.array([ys.mean(), xs.mean()])
                    dist = float(np.linalg.norm(c - ref_centroid))
                    if dist < best_dist:
                        best_dist, best_label = dist, lbl
                partitioned = watershed(
                    np.zeros(mask_2d.shape, dtype=np.uint8),
                    markers=labeled_eroded,
                    mask=mask_2d,
                )
                return partitioned == best_label
        return None

    volume_shape = preprocessed_volume.shape
    rough_mask = np.zeros(volume_shape, dtype=bool)
    seed_mask_3d = np.zeros(volume_shape, dtype=bool)

    prev_centroid = None
    prev_area = None
    seed_found = False

    area_jump_factor = 1.25

    for z in tqdm(
        reversed(range(volume_shape[0])),
        desc="Segmenting incisor",
        total=volume_shape[0],
        unit="slice",
        leave=False,
    ):
        slice_2d = preprocessed_volume[z]

        if np.sum(slice_2d) == 0:
            continue

        segmented_slice = segment_slice(slice_2d, method="otsu", min_area=100, nbins=2048)
        labeled_slic = label_slice(segmented_slice)
        props = regionprops(labeled_slic, intensity_image=slice_2d)

        if not props:
            continue

        if prev_centroid is None:
            tip_candidates = [p for p in props if p.intensity_mean >= min_tip_intensity]
            if not tip_candidates:
                continue

            incisor_region = max(
                tip_candidates,
                key=lambda p: (p.intensity_mean, -p.area),
            )
            seed_found = True
        else:
            def _score_region(p):
                dist = float(
                    np.linalg.norm(
                        np.array(p.centroid, dtype=float)
                        - np.array(prev_centroid, dtype=float)
                    )
                )
                return (-dist, p.intensity_mean, -p.area)

            incisor_region = max(props, key=_score_region)

        slice_mask_full = labeled_slic == incisor_region.label

        current_area = int(np.count_nonzero(slice_mask_full))
        if (
            prev_area is not None
            and prev_area > 0
            and current_area > area_jump_factor * prev_area
            and prev_centroid is not None
        ):
            logger.debug(
                "Area jump on slice %d (prev=%d, curr=%d); attempting erosion separation",
                z, prev_area, current_area,
            )
            separated = _try_erode_separate(slice_mask_full, prev_centroid)
            if separated is not None:
                slice_mask_full = separated
                current_area = int(np.count_nonzero(slice_mask_full))
                logger.debug("Separation successful on slice %d (new area=%d)", z, current_area)
            else:
                logger.debug("Separation failed on slice %d; keeping merged region", z)

        rough_mask[z] = slice_mask_full
        ys, xs = np.where(slice_mask_full)
        prev_centroid = np.array([ys.mean(), xs.mean()])
        prev_area = current_area

        if seed_found and not np.any(seed_mask_3d):
            seed_mask_3d[z] = slice_mask_full

    if not np.any(seed_mask_3d):
        logger.warning("No reliable incisor seed found; returning empty mask.")
        return np.zeros(volume_shape, dtype=bool)

    logger.info("Creating corridor mask around rough incisor segmentation")
    corridor_mask = dilate_mask(rough_mask, radius=1, if_2d=False)

    base_mask = convert_to_binary(preprocessed_volume, fill_holes=True).astype(bool)
    allowed_mask = base_mask & corridor_mask

    logger.info("Performing seeded 3D region growing for incisor")
    grown_mask = binary_propagation(
        seed_mask_3d,
        mask=allowed_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
    )

    logger.info("Cleaning up incisor mask")
    incisor_volume = np.where(grown_mask, preprocessed_volume, 0)
    incisor_volume = convert_to_binary(incisor_volume, fill_holes=True)
    incisor_volume = label_3d_volume(incisor_volume, connectivity=1)
    incisor_label = get_largest_region(incisor_volume)
    incisor_mask = (incisor_volume == incisor_label)

    # logger.info("Dilating incisor mask to recover thin structures")
    # incisor_mask = dilate_mask(incisor_mask, radius=2, if_2d=False)

    return incisor_mask.astype(bool)
