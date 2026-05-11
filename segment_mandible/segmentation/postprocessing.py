"""
Post-processing for incisor and molar segmentation masks.

Incisor
-------
The segmented incisor may carry bone-attachment bridges — narrow necks where
the tooth body is still fused to surrounding cortical bone.  ``postprocess_incisor``
breaks those necks by iteratively eroding the mask until the main tooth body
separates into its own component, then reconstructs the full incisor body via
binary propagation within the original mask boundary, leaving detached fragments
behind.

Molars
------
The molar segmentation returns a single fused mask that covers all three molars.
``split_molars`` uses 3-D distance-transform watershed to partition the mask into
individual molar instances:

1. Compute Euclidean distance transform (EDT) inside the mask.
2. Find N seeds at the EDT local maxima (one per expected molar).
3. Watershed on the negative EDT, seeded at those maxima.
4. Merge any watershed region that is too small to be a real molar into its
   nearest neighbour.
"""

import logging
from typing import Optional

import numpy as np
from scipy.ndimage import binary_propagation, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import ball
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Incisor
# ---------------------------------------------------------------------------

def postprocess_incisor(
    incisor_mask: np.ndarray,
    max_erosions: int = 50,
    min_body_fraction: float = 0.5,
) -> np.ndarray:
    """
    Remove bone-attachment bridges from the incisor mask.

    Iteratively erodes the mask with a 3-D ball until the main incisor body
    separates from any attached bone fragments (i.e. ≥ 2 components appear and
    the largest holds at least ``min_body_fraction`` of all eroded voxels).  The
    clean incisor body is then reconstructed by flood-filling the largest
    component within the original mask boundary, discarding the detached
    fragments.

    If no separation is achieved within ``max_erosions`` rounds the original
    mask is returned unchanged with a warning.

    Parameters
    ----------
    incisor_mask : np.ndarray
        Boolean mask of the segmented incisor (may include bone attachments).
    max_erosions : int
        Maximum number of ball-1 erosion rounds before giving up.
    min_body_fraction : float
        The largest component must represent at least this fraction of all
        eroded foreground voxels to be accepted as the main body.

    Returns
    -------
    np.ndarray
        Cleaned boolean incisor mask.
    """
    if not incisor_mask.any():
        return incisor_mask

    original_mask = incisor_mask.astype(bool)
    struct = ball(1).astype(bool)
    flood_struct = np.ones((3, 3, 3), dtype=bool)
    eroded = original_mask.copy()

    for i in range(1, max_erosions + 1):
        from scipy.ndimage import binary_erosion
        candidate = binary_erosion(eroded, structure=struct)

        if not candidate.any():
            logger.warning(
                "Incisor post-processing: mask fully eroded after %d round(s) "
                "without separation — returning original mask.", i,
            )
            return original_mask

        labeled, n = label(candidate, connectivity=1, return_num=True)  # type: ignore

        if n >= 2:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0  # exclude background
            largest_label = int(counts.argmax())
            largest_size = int(counts[largest_label])
            total_size = int(counts.sum())

            logger.info(
                "Incisor post-processing: %d component(s) after %d erosion(s); "
                "largest = %d voxels (%.1f%% of eroded foreground).",
                n, i, largest_size, 100.0 * largest_size / max(total_size, 1),
            )

            if largest_size >= min_body_fraction * total_size:
                core = labeled == largest_label
                # Flood-fill the core within the original mask — this recovers
                # the full incisor body without reattaching the detached bone.
                reconstructed = binary_propagation(
                    core, mask=original_mask, structure=flood_struct,
                )
                removed = int(original_mask.sum()) - int(reconstructed.sum())
                logger.info(
                    "Incisor post-processing: removed %d bridge voxel(s) "
                    "(%.2f%% of original mask).",
                    removed, 100.0 * removed / max(int(original_mask.sum()), 1),
                )
                return reconstructed.astype(bool)

        eroded = candidate

    logger.warning(
        "Incisor post-processing: no clean separation after %d erosion(s) — "
        "returning original mask.", max_erosions,
    )
    return original_mask


# ---------------------------------------------------------------------------
# Molars
# ---------------------------------------------------------------------------

def split_molars(
    molar_mask: np.ndarray,
    intensity_volume: Optional[np.ndarray] = None,
    n_molars: int = 3,
    min_molar_voxels: int = 1000,
    dentin_percentile: float = 60.0,
) -> np.ndarray:
    """
    Split a fused molar mask into individual molar instances using
    intensity-guided dentin seeding followed by EDT watershed.

    Why intensity-guided seeding?
    - Adjacent molars are connected through a **broad enamel bridge** that
      geometry-only methods (plain EDT peaks) cannot reliably cut.
    - Bi-rooted molars have an intra-molar EDT valley between their two roots;
      geometry-only watershed splits along that valley instead of between teeth.

    The key anatomical insight is that **dentin is less bright than enamel**.
    Even when enamel bridges adjacent molars, the dentin interior of each
    tooth is still spatially separate.  Each molar's dentin also connects
    both of its roots at the crown, so it forms a single connected component
    per tooth despite multi-rooted anatomy.

    Algorithm
    ---------
    1. Exclude the brightest ``(100 - dentin_percentile)`` % of voxels inside
       the molar mask (enamel) to obtain a dentin-only sub-mask.
    2. Label the dentin sub-mask; keep the ``n_molars`` largest components as
       one seed region per tooth.
    3. Compute the EDT inside the full molar mask.
    4. Run watershed on the negative EDT using those dentin seeds, constrained
       to the molar mask.  The watershed boundary falls at the enamel bridge
       saddle rather than at a root valley.
    5. Merge any watershed region smaller than ``min_molar_voxels`` into its
       nearest neighbor.

    If no ``intensity_volume`` is supplied, or if fewer than 2 dentin seeds are
    found, falls back to EDT-peak seeding (suitable when molars are already
    separated).

    Parameters
    ----------
    molar_mask : np.ndarray
        Boolean mask covering all molars (may be fused at the enamel).
    intensity_volume : np.ndarray, optional
        Intensity array co-registered with ``molar_mask`` (e.g. the
        preprocessed normalised volume).  Required for dentin seeding.
    n_molars : int
        Expected number of molars (default 3).
    min_molar_voxels : int
        Watershed regions below this size are merged into a neighbor.
    dentin_percentile : float
        Voxels inside the molar mask with intensity **below** this percentile
        are treated as dentin and used for seeding (default 60 — excludes the
        brightest 40 % as enamel).

    Returns
    -------
    np.ndarray
        int32 label array (same shape as ``molar_mask``).
        0 = background; 1 … k = individual molars (k ≤ n_molars).
    """
    if not molar_mask.any():
        logger.warning("split_molars: empty molar mask — returning zeros.")
        return np.zeros_like(molar_mask, dtype=np.int32)

    dt: np.ndarray = distance_transform_edt(molar_mask)  # type: ignore[assignment]
    logger.info(
        "split_molars: EDT max radius = %.1f vx, mask voxels = %d",
        float(dt.max()), int(molar_mask.sum()),
    )

    markers = _dentin_seeds(molar_mask, intensity_volume, n_molars, dentin_percentile)

    if markers is None:
        logger.info("split_molars: falling back to EDT-peak seeding")
        markers = _edt_peak_seeds(dt, molar_mask, n_molars)

    if markers is None or int((markers > 0).sum()) < 2:
        logger.warning(
            "split_molars: fewer than 2 seeds — cannot split; "
            "returning mask as a single label."
        )
        result = np.zeros_like(molar_mask, dtype=np.int32)
        result[molar_mask] = 1
        return result

    labels_ws = watershed(-dt, markers=markers, mask=molar_mask)
    labels_ws = _merge_small_regions(labels_ws, molar_mask, min_molar_voxels)

    logger.info("split_molars: output contains %d molar label(s)", int(labels_ws.max()))
    return labels_ws.astype(np.int32)


def _dentin_seeds(
    molar_mask: np.ndarray,
    intensity_volume: Optional[np.ndarray],
    n_molars: int,
    dentin_percentile: float,
) -> Optional[np.ndarray]:
    """
    Build a marker array by finding the N largest connected components of the
    dentin sub-mask (molar voxels below the enamel intensity threshold).

    Returns None if ``intensity_volume`` is not provided or fewer than 2
    components are found.
    """
    if intensity_volume is None:
        return None

    molar_vals = intensity_volume[molar_mask]
    enamel_thresh = float(np.percentile(molar_vals, dentin_percentile))
    logger.info(
        "split_molars: enamel threshold = %.4f (%.0f-th percentile of molar intensities)",
        enamel_thresh, dentin_percentile,
    )

    dentin_mask = molar_mask & (intensity_volume < enamel_thresh)
    if not dentin_mask.any():
        logger.warning("split_molars: dentin sub-mask is empty — check dentin_percentile")
        return None

    labeled_dentin, n = label(dentin_mask, connectivity=1, return_num=True)  # type: ignore
    logger.info("split_molars: %d dentin component(s) found", n)

    if n < 2:
        return None

    # Keep the n_molars largest dentin components as one seed region per tooth.
    counts = np.bincount(labeled_dentin.ravel())
    counts[0] = 0  # background
    top_labels = np.argsort(counts)[-n_molars:][::-1]
    top_labels = top_labels[counts[top_labels] > 0]

    if len(top_labels) < 2:
        return None

    markers = np.zeros(molar_mask.shape, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(top_labels, start=1):
        markers[labeled_dentin == old_lbl] = new_lbl

    logger.info(
        "split_molars: dentin seeds — %d marker region(s), sizes %s",
        len(top_labels),
        [int(counts[l]) for l in top_labels],
    )
    return markers


def _edt_peak_seeds(
    dt: np.ndarray,
    molar_mask: np.ndarray,
    n_molars: int,
    min_distance: int = 50,
) -> Optional[np.ndarray]:
    """Fallback: seed from the N largest EDT local maxima."""
    coords = peak_local_max(dt, num_peaks=n_molars, min_distance=min_distance, labels=molar_mask)
    if len(coords) < 2:
        return None
    markers = np.zeros(molar_mask.shape, dtype=np.int32)
    for idx, coord in enumerate(coords, start=1):
        markers[tuple(coord)] = idx
    return markers


def _merge_small_regions(
    labels: np.ndarray,
    mask: np.ndarray,
    min_voxels: int,
) -> np.ndarray:
    """
    Reassign watershed regions smaller than *min_voxels* to the label of the
    spatially nearest large region, then renumber labels to be contiguous.
    """
    labels = labels.copy()
    unique, counts = np.unique(labels[mask], return_counts=True)

    small_labels = unique[counts < min_voxels]
    if small_labels.size == 0:
        # Renumber to be safe (watershed labels may have gaps)
        return _renumber(labels, mask)

    large_mask = np.isin(labels, unique[counts >= min_voxels]) & mask

    # Distance transform from large-label region: gives each voxel the
    # (z, y, x) index of its nearest large-label voxel via return_indices.
    _, nearest_idx = distance_transform_edt(~large_mask, return_indices=True)

    for sl in small_labels:
        sl_voxels = labels == sl
        if not sl_voxels.any():
            continue

        # Look up the nearest large-label voxel for every small-label voxel.
        nz = nearest_idx[0][sl_voxels]
        ny = nearest_idx[1][sl_voxels]
        nx = nearest_idx[2][sl_voxels]
        target_labels = labels[nz, ny, nx]

        # Assign to the most common target label.
        nonzero = target_labels[target_labels > 0]
        if nonzero.size == 0:
            continue
        target = int(np.bincount(nonzero).argmax())

        logger.info(
            "split_molars: merging small region (label=%d, size=%d) → label %d",
            int(sl), int(sl_voxels.sum()), target,
        )
        labels[sl_voxels] = target

    return _renumber(labels, mask)


def _renumber(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Renumber foreground labels to be contiguous starting at 1."""
    unique_labels = np.unique(labels[mask])
    renumbered = np.zeros_like(labels)
    for new_lbl, old_lbl in enumerate(unique_labels, start=1):
        renumbered[labels == old_lbl] = new_lbl
    return renumbered
