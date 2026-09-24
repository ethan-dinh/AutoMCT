"""
Incisor segmentation by fusion-aware threshold descent.

Why this exists
---------------
The slice-tracking approach in ``incisor.py`` finds the tooth (recall ~0.90 on
specimen 6181-3AVG) but leaks badly into surrounding bone (precision ~0.30,
3x too much volume): its final ``binary_propagation`` floods through any
voxel-level bridge between incisor and bone.

The incisor is 4-6x brighter than adjacent tissue through most of its length,
but a *global* intensity threshold does not separate it either -- molar enamel
and dense cortical bone occupy the same intensity range (best global threshold
scores Dice 0.47). What does separate it is intensity *plus* connectivity: at a
high enough threshold the incisor is its own 3D connected component; lower the
threshold far enough and it fuses to bone through a bridge.

Between those two regimes the seeded component grows slowly and smoothly
(1-2% per threshold step, as the tooth surface is recovered). At fusion it
jumps discontinuously -- on 6181-3AVG, 3.9x in a single step. That
discontinuity is the signal, and detecting it needs no ground truth: descend
the threshold until the component explodes, then take the last threshold
before it.

This yields Dice 0.979 on 6181-3AVG versus 0.445 for the tracker.
"""

import logging

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)


def _tissue_floor(volume: np.ndarray) -> float:
    """
    Intensity separating background (air) from tissue.

    A small fraction of the volume's maximum. Deliberately permissive: this
    only needs to exclude air before percentiles are taken, and setting it too
    high is actively harmful. Otsu's threshold, for instance, lands at 6000 on
    6181-3AVG, which pushes the sweep's lower bound above the true fusion
    point (~9400) and stops the descent early.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0
    return 0.05 * float(np.nanmax(finite))


def find_incisor_seed(
    volume: np.ndarray,
    percentile: float = 99.0,
    tissue_floor: float | None = None,
    min_component_voxels: int = 200,
) -> np.ndarray:
    """
    Locate a seed inside the incisor among the brightest voxels, choosing the
    component with the greatest extent along the volume's long axis.

    Selecting the *largest* bright component does not work: molar cusps are
    denser than the incisor and can form a larger cluster, so the seed jumps
    between the incisor and the molars depending on the exact percentile
    (measured on 6181-3AVG: 100% inside the incisor at the 99.5th percentile,
    0% at the 99.0th). Extent along axis 0 is the stable discriminator -- the
    incisor is a long tube spanning most of the volume while molars are
    compact clusters. Seeding by extent lands wholly inside the incisor across
    the entire 98.0-99.9 percentile range.

    Parameters
    ----------
    volume : np.ndarray
        Grayscale volume (Z, Y, X), long axis first.
    percentile : float
        Intensity percentile (over tissue voxels) defining "brightest".
    tissue_floor : float | None
        Voxels at or below this are background and excluded from the
        percentile. Defaults to Otsu's background/tissue threshold.
    min_component_voxels : int
        Components smaller than this are ignored, so a thin noise streak
        cannot win on extent alone.

    Returns
    -------
    np.ndarray
        Boolean seed mask. Empty if no candidate was found.
    """
    if tissue_floor is None:
        tissue_floor = _tissue_floor(volume)

    tissue = volume[volume > tissue_floor]
    if tissue.size == 0:
        logger.warning("Incisor seed: no voxels above tissue floor %.4g", tissue_floor)
        return np.zeros(volume.shape, dtype=bool)

    thresh = float(np.percentile(tissue, percentile))
    bright = volume > thresh
    if not bright.any():
        logger.warning("Incisor seed: no voxels above %.1f percentile", percentile)
        return np.zeros(volume.shape, dtype=bool)

    labels, n = ndi.label(bright)  # type: ignore
    if n == 0:
        return np.zeros(volume.shape, dtype=bool)

    boxes = ndi.find_objects(labels)
    best_label, best_span, best_size = None, -1, 0
    for i, box in enumerate(boxes, start=1):
        if box is None:
            continue
        size = int((labels[box] == i).sum())
        if size < min_component_voxels:
            continue
        span = box[0].stop - box[0].start
        if span > best_span:
            best_label, best_span, best_size = i, span, size

    if best_label is None:
        logger.warning("Incisor seed: no component larger than %d voxels",
                       min_component_voxels)
        return np.zeros(volume.shape, dtype=bool)

    seed = labels == best_label
    logger.info(
        "Incisor seed: %d voxels, z-extent %d, above intensity %.4g "
        "(%.1fth percentile of tissue)",
        best_size, best_span, thresh, percentile,
    )
    return seed


def segment_incisor_by_threshold_descent(
    volume: np.ndarray,
    seed: np.ndarray | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    n_steps: int = 60,
    fusion_ratio: float = 2.5,
    refine: bool = True,
    warmup_steps: int = 5,
    max_total_growth: float = 4.0,
    refine_tolerance: float = 1.5,
) -> tuple[np.ndarray, float]:
    """
    Segment the incisor by lowering an intensity threshold until the seeded
    connected component fuses with surrounding bone, then taking the last
    threshold before fusion.

    Parameters
    ----------
    volume : np.ndarray
        Grayscale volume (Z, Y, X).
    seed : np.ndarray | None
        Boolean seed inside the incisor. Computed via ``find_incisor_seed``
        when omitted.
    t_start, t_stop : float | None
        Threshold sweep bounds. ``t_start`` defaults to the 99.5th percentile
        of tissue intensity; ``t_stop`` to a fraction of the seed's own median
        intensity. The lower bound is deliberately generous: the fusion
        detector ends the descent well before it, so extra headroom costs only
        a few wasted steps but protects specimens whose incisor stays isolated
        to a lower threshold. (On 6181-3AVG fusion occurs near the 65th
        percentile of tissue, so an 85th-percentile floor stopped the sweep
        early and cost ~0.10 Dice.)
    n_steps : int
        Number of thresholds tested between ``t_start`` and ``t_stop``. More
        steps locate the fusion point more precisely; 60 is ample.
    fusion_ratio : float
        A step whose component grows by more than this factor is treated as
        fusion. Measured on 6181-3AVG the true fusion step grows 3.99x while
        the largest legitimate step grows 1.21x, so anything in roughly
        1.5-3.0 works; 2.5 sits in the middle of that gap.
    warmup_steps : int
        Number of initial sweep steps exempt from the fusion test. Near the
        top of the sweep the component is still forming and grows fast in
        relative terms (1.53x on 6181-3AVG) purely because it is small;
        without this guard that early growth is misread as fusion and the
        sweep aborts at 3% recall. A step count is used rather than a size
        threshold because the component's absolute size varies with specimen
        and seed percentile, whereas the forming phase is always the first
        handful of steps.
    refine : bool
        After finding the fusion step, re-sweep at 10x finer resolution
        between the last safe threshold and the fusion threshold.
    max_total_growth : float
        Stop the descent once the component has grown by this factor over its
        size at the top of the sweep, even if no single step tripped
        ``fusion_ratio``.

        This is the guard for fusion that happens *gradually*. The per-step
        ratio assumes the tooth and bone are separated by a sharp intensity
        boundary, so that admitting the bridge admits the whole of bone in one
        step. Denoising softens that boundary: measured on P82-Odam-CTR-2, the
        same scan fuses in a single 3.36x step when raw, but in a 1.99x step
        once NLM and TV denoising have been applied -- under the 2.5 ratio, so
        the descent ran to the bottom of its sweep and returned 1.33M voxels
        of fused tooth-and-bone instead of the 264k the raw scan gives.

        A cumulative bound catches that case without lowering ``fusion_ratio``
        into the range where ordinary surface recovery (up to ~1.2x per step)
        would trip it. The tooth roughly triples between the top of the sweep
        and its true extent, so 4x leaves headroom while still being far below
        the 20x that full fusion reaches.
    refine_tolerance : float
        How much larger than the coarse pass's mask a refinement may be and
        still be accepted. The refine sweep covers one coarse step, which is
        too narrow for either fusion test to fire, so without this bound it
        simply returns the largest component in the window -- the fused one.
        Legitimate refinement only recovers the surface the coarse step's
        granularity missed, which is a small addition.

    Returns
    -------
    (mask, threshold)
        Boolean incisor mask and the threshold that produced it.
    """
    if seed is None:
        seed = find_incisor_seed(volume)
    if not seed.any():
        logger.error("Incisor threshold descent: empty seed, returning empty mask")
        return np.zeros(volume.shape, dtype=bool), float("nan")

    floor = _tissue_floor(volume)
    tissue = volume[volume > floor]

    if t_start is None:
        t_start = float(np.percentile(tissue, 99.5))
    if t_stop is None:
        # Anchor the floor to the seed's own intensity rather than to a
        # percentile of the tissue population. A percentile floor is only
        # meaningful when background has been excluded cleanly; anchoring to
        # the seed keeps the sweep inside the structure's own intensity scale
        # regardless of how much air surrounds it.
        seed_level = float(np.median(volume[seed]))
        t_stop = max(
            0.35 * seed_level,
            float(np.percentile(tissue, 50.0)) * 0.5,
        )

    def _seeded_component(t: float):
        """Component containing the most seed voxels at threshold t."""
        labels, n = ndi.label(volume > t)  # type: ignore
        if n == 0:
            return None, 0
        ids, counts = np.unique(labels[seed], return_counts=True)
        keep = ids > 0
        ids, counts = ids[keep], counts[keep]
        if ids.size == 0:
            return None, 0
        best = int(ids[np.argmax(counts)])
        comp = labels == best
        return comp, int(comp.sum())

    def _descend(hi: float, lo: float, steps: int, warmup: int = 0):
        """Return (last_safe_threshold, last_safe_mask, fusion_threshold|None)."""
        prev_size = None
        prev_mask = None
        prev_t = hi
        base_size = None
        for i, t in enumerate(np.linspace(hi, lo, steps)):
            comp, size = _seeded_component(float(t))
            if comp is None:
                continue
            # Measured from the end of the warmup, where the component has
            # finished forming: from before it, the ratio would mostly reflect
            # that early growth rather than any fusion.
            if base_size is None and i >= warmup:
                base_size = size
            # Ignore the forming phase; see warmup_steps.
            if (
                prev_size is not None
                and size > fusion_ratio * prev_size
                and i >= warmup
            ):
                logger.info(
                    "Fusion detected at threshold %.4g (component %d -> %d, %.2fx)",
                    t, prev_size, size, size / max(prev_size, 1),
                )
                return prev_t, prev_mask, float(t)
            if (
                base_size
                and prev_mask is not None
                and size > max_total_growth * base_size
            ):
                logger.info(
                    "Gradual fusion detected at threshold %.4g: component has "
                    "grown %.1fx since the start of the sweep (%d -> %d), past "
                    "the %.1fx bound, without any single step exceeding %.1fx",
                    t, size / base_size, base_size, size, max_total_growth,
                    fusion_ratio,
                )
                return prev_t, prev_mask, float(t)
            prev_size, prev_mask, prev_t = size, comp, float(t)
        logger.info(
            "No fusion detected down to threshold %.4g; using it", lo,
        )
        return prev_t, prev_mask, None

    safe_t, safe_mask, fusion_t = _descend(t_start, t_stop, n_steps,
                                           warmup=warmup_steps)

    if refine and fusion_t is not None:
        logger.info("Refining between %.4g and %.4g", safe_t, fusion_t)
        r_t, r_mask, _ = _descend(safe_t, fusion_t, 12)
        # Only accept a refinement that stays close to what the coarse pass
        # judged safe. The refine sweep spans a single coarse step, so it is
        # too short for either fusion test to have room to fire: the per-step
        # ratio sees only small increments, and the cumulative bound measures
        # from a base that is already just shy of fusion. Left unchecked it
        # therefore returns the largest component in the window, which is the
        # fused one it was supposed to stop before. Measured on P82-Odam-CTR-2
        # that turned a 265k mask into 635k.
        if r_mask is not None:
            safe_size = int(safe_mask.sum()) if safe_mask is not None else 0
            if safe_size == 0 or int(r_mask.sum()) <= refine_tolerance * safe_size:
                safe_t, safe_mask = r_t, r_mask
            else:
                logger.info(
                    "Discarding refinement at %.4g: %d voxels against the "
                    "coarse pass's %d is past the %.2fx tolerance, so the "
                    "refinement has stepped into the fusion it was narrowing.",
                    r_t, int(r_mask.sum()), safe_size, refine_tolerance,
                )

    if safe_mask is None:
        logger.error("Incisor threshold descent: no component found")
        return np.zeros(volume.shape, dtype=bool), float("nan")

    logger.info(
        "Incisor segmented at threshold %.4g: %d voxels", safe_t, int(safe_mask.sum()),
    )
    return safe_mask.astype(bool), safe_t


def recover_cervical_loop(
    volume: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    relative_threshold: float = 0.60,
    corridor_dilation: int = 2,
    max_iterations: int = 20,
    apical_fraction: float = 0.25,
    max_growth: float = 0.5,
) -> np.ndarray:
    """
    Recover the cervical loop at the incisor's apical end.

    Why this is separate from ``recover_low_density_margin``
    -------------------------------------------------------
    The cervical loop is the tooth's growth region: continuously forming, and
    so the least mineralised part of it. A rodent incisor grows throughout
    life, which makes this region the point of the scan for many studies --
    but it is exactly what a single global threshold cannot reach. Measured on
    P82 scans, the descent's mask tapers to 50-80 voxels per slice there while
    the surrounding corridor holds 2-3x that many voxels of genuine tissue.

    Two simpler things were tried first and are worth not repeating:

    - Relaxing ``recover_low_density_margin``'s threshold (0.90 -> 0.50).
      Apical coverage plateaus around 130 voxels per slice, because that
      function adds only components already touching the confident mask in one
      pass; the loop is reached through a chain of progressively dimmer voxels,
      not in a single step.
    - Iterating that relaxation over the whole mask. This does reach the loop
      (220-260 voxels per slice) but inflates the total mask to 440k-507k
      voxels -- the same leak into bone that the threshold descent exists to
      avoid, arrived at from the other direction.

    What works is to iterate *and* confine the relaxation to the apical end.
    The loop's location is not incidental: it is always at one end of the
    tooth, so restricting growth to the apical fraction of the mask's own Z
    span lets the threshold drop far enough to follow the mineral gradient
    into the loop while bone alongside the tooth's mineralised body -- which
    is where a uniform relaxation leaks -- stays out of reach. On P82-Odam-CTR-1
    this lifts apical coverage 80 -> 307 voxels per slice for a total of 317k,
    against 507k for the unrestricted version at comparable coverage.

    Parameters
    ----------
    volume : np.ndarray
        The grayscale volume the mask was segmented from.
    mask : np.ndarray
        Confident incisor mask from the threshold descent.
    threshold : float
        The threshold that produced ``mask``.
    relative_threshold : float
        Fraction of ``threshold`` applied inside the apical corridor. The loop
        is genuinely faint, so this is far lower than the margin recovery's
        0.90; it is safe only because of the apical restriction.
    corridor_dilation : int
        Corridor half-width, in voxels, re-grown around the mask each pass.
    max_iterations : int
        Growth stops early once a pass adds nothing, so this only bounds the
        worst case. Coverage still climbs between 10 and 20 passes; beyond
        that it flattens.
    apical_fraction : float
        Portion of the mask's Z span, measured from the apical end, in which
        growth is allowed. Raising it to 0.35 added ~55k voxels for no gain in
        apical coverage -- growth beyond the loop is into bone.
    max_growth : float
        Stop once the mask has grown by this fraction of its original size.
        The apical zone limits *where* growth can happen but not how much, and
        if the loop happens to sit close to the bone's intensity the corridor
        can still find a path into it. A cervical loop is a minority of the
        tooth, so a mask that has grown by half has stopped tracking the loop
        and started filling something else. This bounds the damage instead of
        relying on the intensity gap alone.

    Returns
    -------
    np.ndarray
        Boolean mask, always a superset of ``mask``.
    """
    if not mask.any() or not np.isfinite(threshold):
        return mask.astype(bool)

    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        return mask.astype(bool)

    # The apical end is the low-Z end of the span: reorientation puts the
    # incisal tip last, so the tooth's forming end comes first.
    low, high = int(occupied[0]), int(occupied[-1])
    cutoff = low + int(apical_fraction * (high - low))

    # Work on a crop rather than the whole volume. Growth is confined to the
    # apical zone anyway, but dilating and labelling all of a 1772-slice scan
    # once per iteration made this the single slowest step in a full-resolution
    # run (141 s of a 462 s pipeline). The crop keeps a margin above the
    # cutoff so the corridor around the mask's own boundary is intact, and the
    # result is written back into the full-size mask.
    margin = corridor_dilation + 1
    top = min(mask.shape[0], cutoff + 1 + margin)
    sub_volume = volume[:top]
    out_sub = mask[:top].astype(bool)

    zone = np.zeros(top, dtype=bool)
    zone[: cutoff + 1] = True
    zone = zone[:, None, None]

    level = threshold * relative_threshold
    start_size = int(mask.sum())
    budget = start_size * (1.0 + max_growth)
    # Everything outside the crop is untouched, so measure the budget against
    # the whole mask while comparing sizes within the crop.
    outside = start_size - int(out_sub.sum())
    for _ in range(max_iterations):
        corridor = ndi.binary_dilation(out_sub, iterations=corridor_dilation) & zone
        candidate = (sub_volume > level) & corridor
        labels, n = ndi.label(candidate)  # type: ignore
        if n == 0:
            break
        # Only components already touching the mask: the corridor alone would
        # admit a detached fleck of bone that happens to sit inside it.
        touching = np.unique(labels[out_sub])
        touching = touching[touching > 0]
        grown = np.isin(labels, touching) | out_sub
        if int(grown.sum()) == int(out_sub.sum()):
            break
        if outside + int(grown.sum()) > budget:
            # Take nothing from the pass that broke the budget: it is the one
            # that ran into whatever the growth should have stopped at, so
            # keeping part of it would keep part of that.
            logger.warning(
                "Cervical loop recovery stopped at %d voxels: the next pass "
                "would reach %d, past the %.0f%% growth budget on %d. The "
                "loop is probably close in intensity to nearby bone here.",
                outside + int(out_sub.sum()), outside + int(grown.sum()),
                100 * max_growth, start_size,
            )
            break
        out_sub = grown

    out = mask.astype(bool).copy()
    out[:top] = out_sub

    logger.info(
        "Cervical loop recovery: %d -> %d voxels (apical %.0f%% of Z span, "
        "threshold %.4g)",
        int(mask.sum()), int(out.sum()), 100 * apical_fraction, level,
    )
    return out.astype(bool)


def _tooth_axis(mask: np.ndarray, fit_slices: int = 80, degree: int = 2):
    """
    Fit the tooth's centre line near its apical end, as a function of Z.

    Only the apical-most ``fit_slices`` are fitted rather than the whole tooth:
    the incisor is a long arc, and a curve fitted over its entire length is
    dominated by the far end and extrapolates badly into the near one. A
    quadratic over the nearby stretch tracks the local curvature, which is what
    the extrapolation needs.

    Returns (fy, fx), each mapping a Z index to a predicted centroid coordinate,
    or None when the mask has too few slices to fit.
    """
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    occupied = np.flatnonzero(counts)
    if occupied.size < degree + 1:
        return None

    window = occupied[:fit_slices]
    if window.size < degree + 1:
        window = occupied

    centroids = np.array([ndi.center_of_mass(mask[z]) for z in window])
    fit_y = np.polyfit(window, centroids[:, 0], degree)
    fit_x = np.polyfit(window, centroids[:, 1], degree)
    return fit_y, fit_x


def _axis_slice_area(volume, mask, z, threshold, relative_threshold, radius):
    """
    Voxels the tracker would accept in slice ``z``, used to size the taper.

    Measuring what tracking actually yields next to the join gives a
    specimen-specific reference for "how wide the loop is here", which a fixed
    voxel count cannot.
    """
    if not (0 <= z < volume.shape[0]):
        return 0
    axis = _tooth_axis(mask)
    if axis is None:
        return 0
    fit_y, fit_x = axis
    centre_y = float(np.polyval(fit_y, z))
    centre_x = float(np.polyval(fit_x, z))
    grid_y = np.arange(volume.shape[1])[:, None]
    grid_x = np.arange(volume.shape[2])[None, :]
    disk = ((grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2) <= radius ** 2
    candidate = (volume[z] > threshold * relative_threshold) & disk
    if not candidate.any():
        return 0
    labels, n = ndi.label(candidate)  # type: ignore
    if n == 0:
        return 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return int(sizes.max())



def track_cervical_loop(
    volume: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    relative_threshold: float = 0.5,
    radius: float = 12.0,
    max_slices: int = 120,
    min_slice_voxels: int = 8,
    max_centroid_drift: float = 10.0,
    start_min_voxels: int = 10,
    taper_fraction: float = 0.25,
    regrowth_ratio: float = 1.4,
) -> np.ndarray:
    """
    Extend the incisor mask apically by following the tooth's own axis.

    Why a different approach is needed
    ----------------------------------
    The cervical loop is the tooth's forming end, and it is *less dense than
    the bone beside it*. That single fact defeats every method built on a
    global intensity threshold plus connectivity, including
    ``recover_cervical_loop`` above. Measured on P82-OdamcKo-CTR-2: relaxing
    the descent's threshold to 0.9x leaves a sane 241k-voxel component that
    stops at z=139, while 0.7x jumps to 1.07M voxels spanning the whole volume
    -- the tooth has fused to the mandible. There is no level in between that
    reaches the loop without also taking the jaw.

    What separates the loop from the bone around it is therefore not how bright
    it is but *where* it is: it continues the tooth's trajectory. So this
    tracks rather than thresholds. The tooth's centre line is fitted near its
    apical end and extrapolated one slice at a time; in each new slice only
    tissue within ``radius`` of the predicted centre is considered, and the
    largest blob there is accepted only if it is near the prediction. The
    geometric restriction is what makes a threshold as low as 0.5x safe --
    bone at that intensity is abundant, but it is not on the tooth's axis.

    On P82-OdamcKo-CTR-2 the axis-local tissue apical to the mask holds
    100-150 voxels per slice at mean intensity 0.30-0.35, against a descent
    threshold of 0.44, and stays within 2-8 voxels of the predicted axis for
    40 slices past where the mask ends.

    Parameters
    ----------
    volume : np.ndarray
        The grayscale volume the mask was segmented from.
    mask : np.ndarray
        Incisor mask to extend.
    threshold : float
        The threshold that produced ``mask``.
    relative_threshold : float
        Fraction of ``threshold`` applied inside the tracking disk. Safe this
        low only because of the geometric restriction.
    radius : float
        Radius, in voxels, of the disk around the predicted centre searched in
        each slice. Wide enough to hold the loop, narrow enough to exclude the
        bone beside it.
    max_slices : int
        Hard bound on how far apically to track. The taper test below is what
        normally ends a track; this only stops a pathological one.
    min_slice_voxels : int
        A slice yielding fewer voxels than this ends the track: the loop has
        either ended or been lost.
    max_centroid_drift : float
        How far, in voxels, the accepted blob's centroid may sit from the
        predicted axis. This is the check that stops the track stepping sideways
        onto bone that happens to lie within the disk.
    start_min_voxels : int
        Begin tracking at the apical-most slice holding at least this many
        voxels, rather than at the mask's very first slice, and re-track the
        thin tail below it rather than keeping it.

        Kept low on purpose. Raising it does smooth the join -- the descent's
        tail is what pinches the profile there -- but it also moves the start
        further up the tooth, past where the loop already begins, so the taper
        test ends the track almost at once. Measured across four P82 scans,
        raising it from 10 to 40 cut two specimens' extensions from +35 and
        +59 slices to -4 and +3. The residual step where the tracked run meets
        the descent's taper is left alone: that taper is the tooth genuinely
        thinning, not an artefact, and widening it was tried and found to
        inflate the mask rather than smooth it.
    taper_fraction : float
        End the track once the cross-section falls below this fraction of the
        largest one seen, which is how the loop itself ends.

        A fixed slice budget cannot do this job, because how far the loop
        extends varies by specimen. Measured across four P82 scans the tracked
        cross-section always rises to a peak around 20-35 slices in and then
        declines; three of the four then ended on their own after another 9-21
        slices, while the fourth carried on for 95 more at a roughly constant
        36-50 voxels -- no longer tapering, so no longer the loop. Ending on
        the taper stops that case without truncating the three that were still
        genuinely descending.

    regrowth_ratio : float
        End the track if the cross-section grows back by this factor after
        having tapered, which means it is no longer following the loop.

        The loop narrows once past its widest point, so a sustained
        *re*-widening is the signature of the track having stepped onto
        something else lying along the same axis. On P82-OdamcKo-CTR-2 the
        cross-section falls to 0.54 of its peak by step 48 -- a real taper --
        then climbs back to 0.84 by step 96 as the track follows a separate
        structure out to the edge of the volume. Its absolute size never drops
        far enough for ``taper_fraction`` to fire, so the turnaround is what
        identifies it.

    Returns
    -------
    np.ndarray
        Boolean mask, always a superset of ``mask``.
    """
    if not mask.any() or not np.isfinite(threshold):
        return mask.astype(bool)

    axis = _tooth_axis(mask)
    if axis is None:
        logger.warning("Cervical loop tracking: mask too short to fit an axis")
        return mask.astype(bool)
    fit_y, fit_x = axis

    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    occupied = np.flatnonzero(counts)

    # Start from the first slice with real substance, not from the mask's
    # apical-most voxel. The descent tapers out over its last several slices
    # (1, 11, 27, 48 ... voxels on P82-OdamcKo-CTR-3), and starting below that
    # tail leaves it stranded between the tracked run and the body of the
    # mask -- a 180 -> 244 -> 11 -> 27 profile that reads downstream as the
    # mask jumping onto another structure. Beginning at the substantial part
    # and re-tracking over the tail replaces those slices instead.
    substantial = occupied[counts[occupied] >= start_min_voxels]
    start = int(substantial[0]) if substantial.size else int(occupied[0])

    # Advance past the descent's taper so those slices are re-tracked too.
    #
    # The taper is under-segmented rather than genuinely thin: the descent's
    # threshold is the one that keeps the tooth clear of bone, and it loses the
    # loop's faint edges, so its last slices thin out (10, 17, 29, 44 ...) while
    # the tracker, working at half that threshold, recovers 115-180 voxels per
    # slice just below them. Leaving the taper in place puts that valley
    # between two correctly-segmented runs, which reads downstream as the mask
    # jumping onto another structure. Re-tracking it fills the valley with the
    # same criterion used either side of it.
    #
    # The threshold is the tracker's own first-slice yield, so this adapts to
    # the specimen instead of assuming a width.
    probe = _axis_slice_area(volume, mask, start - 1, threshold,
                             relative_threshold, radius)
    if probe:
        while (
            start + 1 < mask.shape[0]
            and 0 < counts[start] < taper_fraction * probe
        ):
            start += 1


    grid_y = np.arange(volume.shape[1])[:, None]
    grid_x = np.arange(volume.shape[2])[None, :]
    level = threshold * relative_threshold

    out = mask.astype(bool).copy()
    # Drop only the thin tail immediately below the start slice: it is about
    # to be re-tracked, and leaving it would keep the very slices the new start
    # exists to replace. Slices further down are left alone -- clearing the
    # whole range below `start` would discard anything the mask legitimately
    # holds there.
    for z in range(start - 1, -1, -1):
        if counts[z] == 0:
            break
        out[z] = False
    added_slices = 0
    peak_area = 0
    trough_area = np.inf
    for step in range(1, max_slices + 1):
        z = start - step
        if z < 0:
            break

        centre_y = float(np.polyval(fit_y, z))
        centre_x = float(np.polyval(fit_x, z))
        disk = ((grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2) <= radius ** 2

        candidate = (volume[z] > level) & disk
        if candidate.sum() < min_slice_voxels:
            logger.debug("Cervical loop tracking stopped at z=%d: too little tissue", z)
            break

        labels, n = ndi.label(candidate)  # type: ignore
        if n == 0:
            break
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        blob = labels == int(sizes.argmax())

        centroid: tuple[float, float] = ndi.center_of_mass(blob)  # type: ignore[assignment]
        blob_y, blob_x = centroid
        drift = float(np.hypot(blob_y - centre_y, blob_x - centre_x))
        if drift > max_centroid_drift:
            logger.debug(
                "Cervical loop tracking stopped at z=%d: blob %.1f voxels off axis",
                z, drift,
            )
            break

        area = int(blob.sum())

        # The narrowest cross-section seen since the peak, so re-growth is
        # measured from the bottom of the taper rather than from the peak.
        if area < peak_area:
            trough_area = min(trough_area, area)
        peak_area = max(peak_area, area)

        if trough_area < peak_area and area > regrowth_ratio * trough_area:
            logger.debug(
                "Cervical loop tracking stopped at z=%d: cross-section grew "
                "back from %d to %d after tapering from a peak of %d — the "
                "track has left the loop",
                z, int(trough_area), area, peak_area,
            )
            break

        if peak_area and area < taper_fraction * peak_area:
            logger.debug(
                "Cervical loop tracking stopped at z=%d: cross-section %d is "
                "below %.0f%% of its peak %d — the loop has tapered out",
                z, area, 100 * taper_fraction, peak_area,
            )
            break

        out[z] |= blob
        added_slices += 1

    logger.info(
        "Cervical loop tracking: %d -> %d voxels, extended %d slices apically "
        "from z=%d (threshold %.4g, %.2fx the descent's)",
        int(mask.sum()), int(out.sum()), added_slices, start, level,
        relative_threshold,
    )
    return out


def polish_incisor_mask(
    mask: np.ndarray,
    closing_radius: int = 0,
    fill_3d: bool = True,
) -> np.ndarray:
    """
    Light cleanup of a threshold-descent incisor mask.

    Deliberately conservative. In particular hole filling is done in **3D
    only, never per-slice**: the hand-drawn ground truth excludes the pulp
    cavity, and because that cavity is open at the tooth's base it stays
    connected to the exterior in 3D (a 3D fill is a near no-op). Filling
    slice-by-slice would close each cross-sectional ring and add ~37% spurious
    volume, capping Dice at 0.84 no matter how good the segmentation is.

    Parameters
    ----------
    mask : np.ndarray
        Boolean incisor mask.
    closing_radius : int
        Radius of an optional 3D binary closing. 0 disables it.
    fill_3d : bool
        Apply 3D hole filling (safe -- see above).
    """
    out: np.ndarray = mask.astype(bool)

    if closing_radius > 0:
        from skimage.morphology import ball
        out = ndi.binary_closing(out, structure=ball(closing_radius))

    if fill_3d:
        out = ndi.binary_fill_holes(out)  # type: ignore[assignment]

    labels, n = ndi.label(out)  # type: ignore
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        out = labels == int(np.argmax(sizes))

    return out.astype(bool)


def recover_low_density_margin(
    volume: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    relative_threshold: float = 0.90,
    corridor_dilation: int = 2,
) -> np.ndarray:
    """
    Recover under-segmented low-density tissue immediately around ``mask``.

    A single global threshold cannot capture the whole tooth: at the apical
    (forming) end roughly a quarter of the true voxels are *less dense* than
    the threshold the fusion point permits, so they are dropped, while
    lowering the global threshold to reach them fuses the tooth to bone.

    This recovers them by relaxing the threshold **only inside a thin corridor
    around the confident mask**, and only for components that touch it. The
    corridor is what makes the lower threshold safe: bone that would fuse at
    this intensity is mostly outside it, and any candidate component not
    already touching the mask is discarded.

    On 6181-3AVG this lifts recall 0.968 -> 0.998 and Dice 0.980 -> 0.992,
    with precision easing only 0.992 -> 0.986.

    Parameters
    ----------
    volume : np.ndarray
        The same grayscale volume the mask was segmented from.
    mask : np.ndarray
        Confident incisor mask from the threshold descent.
    threshold : float
        The threshold that produced ``mask``.
    relative_threshold : float
        Fraction of ``threshold`` used inside the corridor. Lower recovers
        more but leaks more: measured 0.90 -> Dice 0.992, 0.85 -> 0.983,
        0.80 -> 0.975.
    corridor_dilation : int
        Corridor half-width in voxels. Wider does not help (2 and 3 score
        within 0.001) since the recovered tissue hugs the existing surface.

    Returns
    -------
    np.ndarray
        Boolean mask, always a superset of ``mask``.
    """
    if not mask.any() or not np.isfinite(threshold):
        return mask.astype(bool)

    corridor = ndi.binary_dilation(mask, iterations=corridor_dilation)
    candidate = (volume > threshold * relative_threshold) & corridor

    labels, n = ndi.label(candidate)  # type: ignore
    if n == 0:
        return mask.astype(bool)

    touching = np.unique(labels[mask])
    touching = touching[touching > 0]
    grown = np.isin(labels, touching) | mask

    logger.info(
        "Low-density margin recovery: %d -> %d voxels (threshold %.4g -> %.4g)",
        int(mask.sum()), int(grown.sum()), threshold,
        threshold * relative_threshold,
    )
    return grown.astype(bool)
