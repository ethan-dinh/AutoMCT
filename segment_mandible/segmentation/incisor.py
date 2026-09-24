import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_propagation
from skimage.measure import regionprops
from skimage.segmentation import watershed
from tqdm import tqdm

from logging_setup import progress_disabled
from segmentation.utils import (
    convert_to_binary,
    dilate_mask,
    get_largest_region,
    label_3d_volume,
    label_slice,
    segment_slice,
)

logger = logging.getLogger(__name__)

# Largest per-axis margin, in voxels, still handled by the ball-erosion fast
# path. A ball of radius r is a (2r+1)^3 footprint, so the work per voxel grows
# as r^3: at 3 voxels that is a 343-element footprint on bool, far cheaper than
# a float64 EDT over the same region, while at 10 it is 9261 and the EDT's
# volume-independent cost wins. The margin shrink exists to shave a
# partial-volume rim of one or two voxels, so the fast path covers the intended
# use and the EDT still backs anything wider.
_MAX_BALL_EROSION_RADIUS_VOXELS = 4


def _mask_bounding_box(
    mask: np.ndarray,
    margin_shrink_mm: float,
    spacing: tuple[float, float, float] | None,
) -> tuple[slice, ...] | None:
    """
    Tight bounding box of the mask's foreground, padded by the margin.

    The shrink only ever removes voxels, so every voxel outside the mask's
    bounding box is already background and stays background. Transforming the
    whole array therefore computes -- and then discards at the threshold -- a
    distance for every one of them. On a mouse incisor that is the overwhelming
    majority of the scan: the tooth occupies about 1% of the reoriented volume,
    and even its (diagonal, so untight) bounding box is under a third.

    The box is padded by the margin's worth of voxels per axis so that a
    foreground voxel near the box face still sees the background beyond it.
    Without the pad the transform would measure to the crop face and report a
    too-large distance there, keeping a rim that the full-volume transform
    removes. Padding is clipped at the array edge, which is correct: the EDT
    treats outside-the-array as foreground, and so does the cropped one.

    Returns None when the mask is empty (no box to take) or when the padded box
    covers essentially the whole array, in which case cropping only adds a copy.
    """
    if not mask.any():
        return None

    steps = (1.0, 1.0, 1.0) if spacing is None else spacing
    box = []
    for axis, step in enumerate(steps):
        present = np.nonzero(mask.any(axis=tuple(i for i in range(mask.ndim) if i != axis)))[0]
        pad = int(np.ceil(margin_shrink_mm / step)) + 1
        lo = max(0, int(present[0]) - pad)
        hi = min(mask.shape[axis], int(present[-1]) + 1 + pad)
        box.append(slice(lo, hi))

    box = tuple(box)
    cropped = int(np.prod([s.stop - s.start for s in box]))
    if cropped > 0.9 * mask.size:
        return None
    return box


def _ball_footprint(
    margin_shrink_mm: float,
    spacing: tuple[float, float, float] | None,
) -> np.ndarray | None:
    """
    Footprint of every voxel within ``margin_shrink_mm`` of the centre.

    Eroding by this footprint keeps exactly the voxels whose whole
    margin-radius neighbourhood is foreground -- which is the same set the EDT
    threshold keeps, because a voxel's distance to background exceeds the
    margin iff no background voxel lies within it. The two agree voxel for
    voxel; this one just never materialises the distances it would immediately
    discard.

    The offsets are measured in mm via ``spacing``, so the footprint is a true
    constant-distance ball on anisotropic scans, not a box. That distinction is
    the point: a box footprint trims sqrt(3) times deeper along the diagonals
    than along the axes, and this mask's surface is what downstream
    intensity measurements are taken inside of.

    Returns None when the margin is wide enough that the footprint would be
    large enough for the EDT to be the cheaper route.
    """
    steps = (1.0, 1.0, 1.0) if spacing is None else spacing
    radii = [int(np.ceil(margin_shrink_mm / step)) for step in steps]
    if max(radii) > _MAX_BALL_EROSION_RADIUS_VOXELS:
        return None

    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii)]
    squared: np.ndarray = sum((g * step) ** 2 for g, step in zip(grids, steps))  # type: ignore[assignment]
    # Inclusive at the margin, which is what makes this agree with the EDT
    # path's exclusive `distance > margin`: requiring every neighbour within
    # (and at) the margin to be foreground is exactly the condition that no
    # background lies at or below the margin, i.e. that the distance to the
    # nearest background voxel is strictly greater than it.
    return squared <= margin_shrink_mm ** 2


def _shrink_by_distance(
    mask: np.ndarray,
    margin_shrink_mm: float,
    spacing: tuple[float, float, float] | None,
    workers: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Erode a mask by a physical distance, exactly.

    Two routes, both producing the identical result and differing only in cost:

    [1] A ball erosion, for a margin of a few voxels. Keeping the voxels whose
        entire margin-radius neighbourhood is foreground is the same set as
        thresholding a distance transform, and it runs on bool rather than
        materialising a float64 distance per voxel. This is the intended case
        -- the shrink exists to shave a one-to-two-voxel partial-volume rim.

    [2] A Euclidean distance transform for anything wider, where the footprint
        would grow as the cube of the radius (a half-millimetre margin on an
        8 um scan means a 125^3 element, and enough memory traffic to get the
        process OOM-killed) while the EDT's cost stays flat in the margin.

    Both are restricted to the mask's padded bounding box, since a voxel
    outside it is already background and cannot survive a step that only
    removes voxels. The result is written back into a full-size array, so the
    caller always gets a mask in its own coordinate space.

    ``sampling`` carries the voxel size into the transform, so distances come
    out in mm directly. That keeps the margin exact on anisotropic scans rather
    than rounding it to a whole number of voxels per axis, and it means the
    trimmed surface is a true constant-distance offset instead of an ellipsoid
    approximation of one. The ball footprint is built in mm for the same reason.

    Without spacing (e.g. a BMP stack) the margin is read as a voxel count,
    which is the same computation with unit sampling.

    Parameters:
        workers: Threads to split the transform across. None (default) uses
            every available core; 1 forces the serial path. Unused by the ball
            erosion, which is already cheap enough not to need splitting.

    Returns:
        (shrunk mask, greatest surface distance in mm). The depth is exact on
        the serial path; on the chunked path it is a lower bound, since a
        distance larger than the slab halo is truncated by the slab edge. The
        ball path reports 0.0, since it never computes distances -- it is only
        used to explain an over-large margin, and the caller recomputes it
        exactly in that case.
    """
    if workers is None:
        workers = os.cpu_count() or 1

    box = _mask_bounding_box(mask, margin_shrink_mm, spacing)
    if box is not None:
        shrunk_box, depth = _shrink_by_distance(
            mask[box], margin_shrink_mm, spacing, workers,
        )
        # Back into the caller's coordinate space: everything outside the box
        # was background before the shrink and remains background after it.
        out = np.zeros(mask.shape, dtype=bool)
        out[box] = shrunk_box
        return out, depth

    footprint = _ball_footprint(margin_shrink_mm, spacing)
    if footprint is not None:
        # border_value=1 treats outside-the-array as foreground, matching the
        # EDT, which measures only to background voxels actually present.
        return ndi.binary_erosion(mask, footprint, border_value=1), 0.0

    if workers > 1:
        slabs = _plan_slabs(mask.shape[0], margin_shrink_mm, spacing, workers)
        if len(slabs) > 1:
            return _shrink_by_distance_parallel(mask, margin_shrink_mm, spacing, slabs)

    distance: np.ndarray = ndi.distance_transform_edt(mask, sampling=spacing)  # type: ignore[assignment]
    return distance > margin_shrink_mm, float(distance.max(initial=0.0))


def _plan_slabs(
    depth: int,
    margin_shrink_mm: float,
    spacing: tuple[float, float, float] | None,
    workers: int,
) -> list[tuple[int, int, int, int]]:
    """
    Split the Z axis into overlapping slabs for a chunked distance transform.

    A voxel's nearest background voxel may sit outside its own slab, so slabs
    cannot simply be cut apart -- the transform would measure to the cut face
    and report a too-small distance near every seam. Each slab is therefore
    padded by the margin's worth of voxels on each side: any background within
    the margin is visible inside the padded slab, and voxels deeper than the
    margin are kept regardless of their exact distance, so the thresholded
    result is identical to the serial one.

    Returns (core_start, core_stop, padded_start, padded_stop) per slab, or a
    single whole-volume entry when padding would make chunking pointless.
    """
    step_z = 1.0 if spacing is None else spacing[0]
    # Enough voxels to see every background voxel that could set a distance
    # at or below the margin; +1 for the rounding.
    halo = int(np.ceil(margin_shrink_mm / step_z)) + 1

    # With a halo this large relative to the volume, every slab would re-do
    # most of the work; the serial transform is cheaper.
    max_slabs = max(1, depth // max(1, 4 * halo))
    n_slabs = min(workers, max_slabs)
    if n_slabs <= 1:
        return [(0, depth, 0, depth)]

    bounds = np.linspace(0, depth, n_slabs + 1).astype(int)
    slabs = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= start:
            continue
        slabs.append(
            (int(start), int(stop), int(max(0, start - halo)), int(min(depth, stop + halo)))
        )
    return slabs


def _shrink_by_distance_parallel(
    mask: np.ndarray,
    margin_shrink_mm: float,
    spacing: tuple[float, float, float] | None,
    slabs: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, float]:
    """
    Run the distance transform on overlapping Z slabs across threads.

    SciPy's EDT releases the GIL, so threads give real parallelism without
    copying the volume into worker processes -- which for a multi-GB mask
    would cost more in pickling and memory than the transform itself saves.
    Each slab writes only its own core rows into the shared output, so the
    writes never overlap and no locking is needed.
    """
    out = np.zeros(mask.shape, dtype=bool)

    def _run(slab):
        core_start, core_stop, pad_start, pad_stop = slab
        distance: np.ndarray = ndi.distance_transform_edt(  # type: ignore[assignment]
            mask[pad_start:pad_stop], sampling=spacing
        )
        core = slice(core_start - pad_start, core_stop - pad_start)
        out[core_start:core_stop] = distance[core] > margin_shrink_mm
        # Only core rows are valid: a padded row's distance may be truncated by
        # the slab edge, and the halo caps distances at the margin anyway.
        return float(distance[core].max(initial=0.0))

    with ThreadPoolExecutor(max_workers=len(slabs)) as pool:
        depths = list(pool.map(_run, slabs))

    return out, max(depths, default=0.0)


def _smallest_voxel_step(spacing: tuple[float, float, float] | None) -> float:
    """Finest voxel dimension in mm -- the smallest margin that can remove anything."""
    return 1.0 if spacing is None else min(spacing)


def segment_incisor(
    preprocessed_volume: np.ndarray,
    min_tip_intensity: float,
    seed_out: dict | None = None,
) -> np.ndarray:
    """
    Segment the incisor from the mandible volume.

    Parameters:
        seed_out: If given, filled in with the seed the 3D region grow started
            from -- ``slice_index``, ``area`` and ``intensity_mean`` -- so a
            caller can check the seed independently of the mask it produced. A
            wrong seed yields a mask that looks entirely plausible, so it is
            worth validating on its own.
    """
    def _try_erode_separate(
        mask_2d: np.ndarray, ref_centroid: np.ndarray, max_iters: int = 15
    ):
        """
        Iteratively erode mask_2d until ≥2 components appear, then reconstruct
        the incisor component by watershed-partitioning the original mask using
        the eroded components as seeds. Returns None if separation never occurs.
        """
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

    def _record_seed(z: int, area: int, intensity_mean: float) -> None:
        """Log the chosen seed and hand it to the caller for validation."""
        logger.info(
            "Incisor seed taken from slice %d: %d voxels, mean intensity %.4g "
            "(floor %.4g)",
            z, area, float(intensity_mean), min_tip_intensity,
        )
        if seed_out is not None:
            seed_out.update(
                slice_index=int(z),
                area=int(area),
                intensity_mean=float(intensity_mean),
            )

    volume_shape = preprocessed_volume.shape
    rough_mask = np.zeros(volume_shape, dtype=bool)
    seed_mask_3d = np.zeros(volume_shape, dtype=bool)

    if seed_out is not None:
        seed_out.update(slice_index=None, area=0, intensity_mean=None)

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
        disable=progress_disabled(),
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
            _record_seed(z, current_area, incisor_region.intensity_mean)

    tracked_slices = int(np.count_nonzero(rough_mask.reshape(volume_shape[0], -1).sum(axis=1)))
    logger.info(
        "Slice tracking followed the incisor across %d of %d slices",
        tracked_slices, volume_shape[0],
    )

    if not np.any(seed_mask_3d):
        logger.warning(
            "No reliable incisor seed found — no slice held a region with mean "
            "intensity above the %.4g floor; returning an empty mask.",
            min_tip_intensity,
        )
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
    incisor_mask = incisor_volume == incisor_label

    # Keeping only the largest component silently discards whatever else the
    # grow reached. How much it discarded says whether the grow was clean or
    # whether it leaked and the cleanup is papering over it.
    grown_voxels = int(np.count_nonzero(grown_mask))
    kept = int(np.count_nonzero(incisor_mask))
    logger.info(
        "Region growing reached %d voxels; kept the largest component (%d voxels, %.1f%%)",
        grown_voxels, kept, 100.0 * kept / grown_voxels if grown_voxels else 0.0,
    )

    return incisor_mask.astype(bool)


def shrink_incisor_margin(
    incisor_mask: np.ndarray,
    margin_shrink_mm: float = 0.0,
    spacing: tuple[float, float, float] | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """
    Trim a margin off the incisor surface.

    The segmentation boundary falls in the partial-volume band where the
    incisor meets the surrounding bone, so those edge voxels are a blend of
    both tissues. Eroding by a small margin pulls the mask back to
    unambiguously-incisor voxels, which matters for intensity-based measures
    (mineral density, enamel thickness) that a rim of bone-contaminated voxels
    would bias.

    This is incisor-specific: bone and molar masks are left alone.

    Parameters:
        incisor_mask: Boolean incisor mask to trim.
        margin_shrink_mm: Distance to trim, in mm. 0 (default) returns the
            mask unchanged. Read as a voxel count when spacing is unavailable.
        spacing: Voxel size (dz, dy, dx) in mm.
        workers: Threads to split the distance transform across. None
            (default) uses every available core; 1 forces the serial path.

    Returns:
        The trimmed mask, or the original when trimming is disabled or the
        margin is finer than a voxel.

    Raises:
        ValueError: if the margin is wide enough to erase the mask entirely.
    """
    if margin_shrink_mm <= 0:
        return incisor_mask

    if spacing is not None:
        for axis, step in enumerate(spacing):
            if step <= 0:
                raise ValueError(f"Voxel spacing must be positive (axis {axis} = {step})")

    finest_step = _smallest_voxel_step(spacing)
    if margin_shrink_mm < finest_step:
        logger.warning(
            "Incisor margin shrink of %.4g mm is below the finest voxel dimension "
            "(%.4g mm); leaving the mask unshrunk.",
            margin_shrink_mm, finest_step,
        )
        return incisor_mask

    logger.info(
        "Shrinking incisor margin by %.4g mm (voxel size %s mm)",
        margin_shrink_mm, spacing,
    )
    shrunk, _ = _shrink_by_distance(
        incisor_mask, margin_shrink_mm, spacing, workers,
    )

    # A margin wider than the incisor's own half-thickness erases everything.
    # Only then is the exact deepest-point distance worth a second full pass:
    # it tells the caller what margin range this specimen can actually take.
    if not shrunk.any():
        distance: np.ndarray = ndi.distance_transform_edt(incisor_mask, sampling=spacing)  # type: ignore[assignment]
        max_depth = float(distance.max(initial=0.0))
        raise ValueError(
            f"Incisor margin shrink of {margin_shrink_mm:.4g} mm would remove the "
            f"entire mask: the incisor's deepest point is only {max_depth:.4g} mm "
            f"from its surface. Use a margin well below that -- a shrink is meant "
            f"to shave the partial-volume rim (roughly one to two voxels, "
            f"{finest_step:.4g}-{2 * finest_step:.4g} mm here), not to hollow the tooth."
        )

    before = int(np.count_nonzero(incisor_mask))
    after = int(np.count_nonzero(shrunk))
    logger.info(
        "Incisor mask shrunk from %d to %d voxels (%.1f%% retained)",
        before, after, 100.0 * after / before if before else 0.0,
    )
    return shrunk
