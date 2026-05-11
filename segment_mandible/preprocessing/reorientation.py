"""
Mandible reorientation: crops to the hemimandible bounding box and produces
a canonical orientation so that all specimens are aligned identically.

Canonical orientation
---------------------
- Axis 0 (Z): anterior–posterior long axis; incisor tip at HIGH Z.
- Axis 1 (Y): cross-sectional height; incisor tip at LOW Y (teeth toward
  index 0 — the "top" of the cross-section).
- Axis 2 (X): cross-sectional width; X >= Y (incisor lies flat, not vertical).

Algorithm
---------
1. Binary mask of non-zero voxels (hemimandible after background removal).
2. Compute the tight 3D axis-aligned bounding box of the mask.
3. Crop the volume (and any companions) to that bounding box.
4. Permute axes so the longest bounding-box edge is axis 0.
5. Detect the incisor tip as the small, very high-intensity enamel cluster.
6. Flip axis 0 so the incisor tip is at HIGH Z.
7. Swap axes 1 and 2 if Y > X so that X >= Y (incisor lies flat).
8. Flip axis 1 so the incisor tip is at LOW Y (teeth at the top).

All operations are lossless (crop, transpose, flip) — no interpolation is
introduced and the output contains exactly the foreground voxels.
"""

import logging
from typing import Optional

import numpy as np
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mandible_mask(volume: np.ndarray) -> np.ndarray:
    """Binary mask based on non-zero voxels."""
    return volume > 0


def _bounding_box(mask: np.ndarray) -> tuple[slice, ...]:
    """Tight N-D bounding box returned as a tuple of slices."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return tuple(slice(None) for _ in range(mask.ndim))
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(mask.ndim))


def _incisor_tip_centroid(
    volume: np.ndarray,
    mask: np.ndarray,
    intensity_percentile: float = 97.0,
    max_component_fraction: float = 0.02,
    edge_fraction: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Locate the incisor tip enamel as the most intense, spatially compact cluster
    near either end of axis 0.

    Only the first and last `edge_fraction` of slices are searched.
    Returns a float array [z, y, x] in the current volume's coordinate frame,
    or None if no candidate is found.
    """
    bone_vals = volume[mask]
    if bone_vals.size == 0:
        return None

    zdim = volume.shape[0]
    edge_slices = max(1, int(np.ceil(zdim * edge_fraction)))

    # Search both ends of the mandible along axis 0.
    search_regions = [
        (slice(0, edge_slices), "low-z"),
        (slice(zdim - edge_slices, zdim), "high-z"),
    ]

    thresh = np.percentile(bone_vals, intensity_percentile)
    size_cap = max(1, int(mask.sum() * max_component_fraction))

    best_region = None

    for zslice, region_name in search_regions:
        local_mask = mask[zslice]
        local_volume = volume[zslice]

        bright = (local_volume > thresh) & local_mask
        if not bright.any():
            logger.debug(
                "No voxels above %.1f-th percentile in %s edge window",
                intensity_percentile,
                region_name,
            )
            continue

        labeled = label(bright, connectivity=1)
        regions = regionprops(labeled, intensity_image=local_volume)
        if not regions:
            continue

        small = [r for r in regions if r.area <= size_cap]
        candidates = small if small else regions

        tip = max(candidates, key=lambda r: r.intensity_mean)
        logger.debug(
            "Incisor tip candidate in %s window: label=%d area=%d intensity_mean=%.4f",
            region_name,
            tip.label,
            tip.area,
            tip.intensity_mean,
        )

        if best_region is None or tip.intensity_mean > best_region[0].intensity_mean:
            best_region = (tip, zslice.start)

    if best_region is None:
        return None

    tip, z_offset = best_region
    centroid = np.array(tip.centroid, dtype=float)
    centroid[0] += z_offset  # shift local z centroid back to global coordinates
    return centroid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reorient_mandible(
    volume: np.ndarray,
    companion_volumes: Optional[list[np.ndarray]] = None,
    intensity_percentile: float = 95.0,
    debug: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Crop to the hemimandible bounding box and produce a canonical orientation
    so that all specimens are aligned identically.

    Canonical orientation
    ---------------------
    - Axis 0 (Z): long axis; incisor tip at HIGH Z.
    - Axis 1 (Y): cross-sectional height; incisor tip at LOW Y (teeth at top).
    - Axis 2 (X): cross-sectional width; X >= Y (incisor lies flat).

    The incisor tip centroid is detected once after the axis permutation and
    then tracked through every subsequent flip/swap, so no information is lost
    and all three axes are pinned deterministically.

    All operations are lossless (no interpolation).

    Parameters
    ----------
    volume : np.ndarray
        Background-removed float volume, shape (Z, Y, X).
    companion_volumes : list[np.ndarray] | None
        Additional volumes to transform with the same operations (e.g. the
        original uint8 BMP stack used for final intensity masking).
    intensity_percentile : float
        Percentile used to isolate enamel-level signal for tip detection.

    Returns
    -------
    (reoriented_volume, reoriented_companions)
        ``reoriented_companions`` is an empty list when ``companion_volumes``
        is None.
    """
    companions: list[np.ndarray] = list(companion_volumes) if companion_volumes else []

    logger.info("Reorientation — computing binary mandible mask")
    mask = _mandible_mask(volume)

    if not mask.any():
        logger.warning("Reorientation skipped: empty mask after thresholding")
        return volume, companions

    # ------------------------------------------------------------------
    # Step 1: crop to tight bounding box
    # ------------------------------------------------------------------
    bbox = _bounding_box(mask)
    dims = tuple(s.stop - s.start for s in bbox)
    logger.info(
        "Hemimandible bounding box  Z=%s  Y=%s  X=%s  →  dims=%s",
        *bbox, dims,
    )

    volume = volume[bbox].copy()
    mask = mask[bbox].copy()
    companions = [c[bbox].copy() for c in companions]

    # ------------------------------------------------------------------
    # Step 2: permute axes so the longest edge is axis 0
    # ------------------------------------------------------------------
    longest_axis = int(np.argmax(dims))
    if longest_axis != 0:
        perm = [longest_axis] + [i for i in range(3) if i != longest_axis]
        logger.info(
            "Permuting axes %s so longest dimension (%d voxels, axis %d) becomes axis 0",
            perm, dims[longest_axis], longest_axis,
        )
        volume = np.transpose(volume, perm).copy()
        mask = np.transpose(mask, perm).copy()
        companions = [np.transpose(c, perm).copy() for c in companions]
    else:
        logger.info("Longest dimension already on axis 0 — no permutation needed")

    # ------------------------------------------------------------------
    # Detect incisor tip centroid once, then track it through all
    # subsequent flips and swaps.
    # ------------------------------------------------------------------
    logger.info("Detecting incisor tip centroid")
    tip = _incisor_tip_centroid(
        volume,
        mask,
        intensity_percentile=intensity_percentile,
        edge_fraction=0.25,
    )

    if tip is None:
        logger.warning(
            "Incisor tip not detected — orientation along axes 0, 1, and 2 "
            "is unchanged. Manual inspection recommended."
        )
        return volume, companions

    logger.info(
        "Tip centroid  Z=%.1f  Y=%.1f  X=%.1f  (volume shape %s)",
        tip[0], tip[1], tip[2], volume.shape,
    )

    # ------------------------------------------------------------------
    # Step 3: flip axis 0 so the incisor tip is at HIGH Z
    # ------------------------------------------------------------------
    mid_z = volume.shape[0] / 2.0
    if tip[0] < mid_z:
        logger.info(
            "Flipping axis 0: tip Z=%.1f is below midpoint %.1f → moving to high Z",
            tip[0], mid_z,
        )
        volume = np.flip(volume, axis=0).copy()
        companions = [np.flip(c, axis=0).copy() for c in companions]
        tip[0] = (volume.shape[0] - 1) - tip[0]
    else:
        logger.info("Axis 0: tip already at high Z (Z=%.1f >= midpoint %.1f)", tip[0], mid_z)

    # ------------------------------------------------------------------
    # Step 4: swap axes 1 and 2 if Y > X so that X >= Y (incisor flat)
    # ------------------------------------------------------------------
    if volume.shape[1] > volume.shape[2]:
        logger.info(
            "Swapping axes 1 and 2: Y=%d > X=%d — rotating so incisor lies flat",
            volume.shape[1], volume.shape[2],
        )
        volume = np.swapaxes(volume, 1, 2).copy()
        companions = [np.swapaxes(c, 1, 2).copy() for c in companions]
        tip[1], tip[2] = tip[2], tip[1]
    else:
        logger.info(
            "Axes 1/2: X=%d >= Y=%d — incisor already flat, no swap needed",
            volume.shape[2], volume.shape[1],
        )

    # ------------------------------------------------------------------
    # Step 5: flip axis 1 so the incisor tip is at LOW Y (teeth at top)
    # ------------------------------------------------------------------
    mid_y = volume.shape[1] / 2.0
    if tip[1] > mid_y:
        logger.info(
            "Flipping axis 1: tip Y=%.1f is above midpoint %.1f → moving to low Y",
            tip[1], mid_y,
        )
        volume = np.flip(volume, axis=1).copy()
        companions = [np.flip(c, axis=1).copy() for c in companions]
        tip[1] = (volume.shape[1] - 1) - tip[1]
    else:
        logger.info("Axis 1: tip already at low Y (Y=%.1f <= midpoint %.1f)", tip[1], mid_y)

    logger.info(
        "Reorientation complete — output shape %s  tip at Z=%.1f  Y=%.1f  X=%.1f",
        volume.shape, tip[0], tip[1], tip[2],
    )

    return volume, companions
