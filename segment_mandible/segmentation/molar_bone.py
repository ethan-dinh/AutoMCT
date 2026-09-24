import logging
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from skimage.segmentation import watershed

from segmentation.utils import (
    erode_mask,
    label_3d_volume,
)
from visualization import create_3d_visualization

logger = logging.getLogger(__name__)

MIN_COMPONENT_SIZE = 3500

# Minimum voxels for an enamel cluster to count as a molar anchor.
_MIN_ENAMEL_SEED_SIZE = 50

# Histogram resolution used to locate the bone and enamel peaks.
_ENAMEL_THRESH_NBINS = 128
_ENAMEL_THRESH_SMOOTH_SIGMA = 2.25

# Minimum prominence, in natural-log units of voxel count, for a peak in the
# log histogram to count as the enamel peak. 0.3 is a ~26% dip on either side.
# On WT-M-1 (f0004019) the enamel peak has a log prominence of 0.71 while the
# dentin shoulder between bone and enamel stays under 0.3.
_ENAMEL_MIN_LOG_PROMINENCE = 0.3

# An enamel peak must hold at least this fraction of the bone peak's height,
# so a handful of saturated voxels in the far tail cannot pose as enamel.
_ENAMEL_MIN_RELATIVE_HEIGHT = 1e-3

# Fallback threshold as a fraction of the foreground max, used only if the
# histogram has no discernible peak (e.g. near-empty foreground).
_ENAMEL_THRESH_FALLBACK_FRACTION = 0.9

# Fraction of the enamel peak's local half-width to shift the threshold left
# (toward the bone knee) once the enamel peak has been located. Knee fallback only.
_ENAMEL_PEAK_LEFT_BIAS = 0.5

# The bone anchor is the foreground outside the enamel's bounding box grown by
# this fraction of the volume's long axis on every side. 4% is ~80 voxels
# (0.5 mm) on a 2000-slice, 6 um scan. That is NOT far enough to clear the
# molar roots: on WT-M-1 they run ~136 voxels (0.8 mm) past the crown's enamel
# box, so the anchor always overlaps the tooth itself. Growing the margin to
# clear them would eat into the bone the anchor must still hold, so instead
# the attachment test asks how much of the anchor a molar holds, not whether
# it touches it at all (see _ANCHOR_ATTACHED_SHARE).
_ANCHOR_MARGIN_FRACTION = 0.04

# An enamel-bearing component counts as attached to bone when it holds at
# least this fraction of the above-threshold anchor voxels. Measured on WT-M-1:
# while the molar is fused to bone its component holds 99.7% of the anchor
# (t=0.18); one step later (t=0.19) it has split off and holds 0.44% -- only
# its own root tips. 0.1 sits well clear of both. Testing for *any* overlap
# instead read those root tips as bone, and the search climbed to 0.4333,
# where the roots are finally thresholded away, discarding the dentin and
# roots with them.
_ANCHOR_ATTACHED_SHARE = 0.1

# Coarse step of the separation search, as a fraction of the range between
# the foreground floor and the enamel threshold, and the bisection tolerance.
_SEPARATION_COARSE_FRACTION = 0.1
_SEPARATION_FINE_FRACTION = 0.005

# Thickness, in voxels, of the shell the molar mask may regrow into after
# separation. See _regrow_molar_rim.
_MOLAR_RIM_VOXELS = 2


def _plot_enamel_threshold_histogram(
    bin_centres: np.ndarray,
    counts: np.ndarray,
    smoothed: np.ndarray,
    peak_idx: Optional[int],
    threshold: float,
    enamel_idx: Optional[int] = None,
) -> None:
    """Show the foreground intensity histogram (log scale) with the bone peak, enamel peak and threshold."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bin_centres, counts, width=bin_centres[1] - bin_centres[0], color="lightgray", label="raw histogram")
    ax.plot(bin_centres, smoothed, color="tab:blue", label="smoothed")
    ax.set_yscale("log")

    if peak_idx is not None:
        ax.axvline(bin_centres[peak_idx], color="tab:green", linestyle="--", label="bone peak")
    if enamel_idx is not None:
        ax.axvline(bin_centres[enamel_idx], color="tab:orange", linestyle="--", label="enamel peak")
    ax.axvline(threshold, color="tab:red", linestyle="--", label="enamel threshold")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Voxel count (log)")
    ax.set_title("Foreground intensity histogram — enamel threshold")
    ax.legend()
    plt.show()
    plt.close(fig)


def _find_enamel_threshold(
    fg_vals: np.ndarray,
    nbins: int = _ENAMEL_THRESH_NBINS,
    smooth_sigma: float = _ENAMEL_THRESH_SMOOTH_SIGMA,
    min_log_prominence: float = _ENAMEL_MIN_LOG_PROMINENCE,
    debug: bool = False,
) -> float:
    """
    Locate the enamel intensity threshold at the histogram valley below the
    enamel peak.

    Enamel is the brightest tissue but a tiny fraction of the foreground: on
    WT-M-1 its peak is ~55k voxels per bin against ~890k for bone. On a
    linear histogram its prominence (~29k) falls under any peak detector
    scaled to the bone peak, which is why the enamel went unseen. On a *log*
    histogram it is an unmistakable second mode (log prominence 0.71), so the
    peaks are found there.

    The bone peak is the most prominent peak of the linear histogram; the
    enamel peak is the brightest sufficiently prominent peak of the log
    histogram to its right. The threshold is the lowest point of the valley
    between the enamel peak and the peak before it -- the natural enamel /
    dentin boundary, which keeps the whole enamel cap rather than only the
    brighter half of its peak.

    Falls back to the older knee-and-tail method when no log peak exists.
    """
    fallback = float(fg_vals.max() * _ENAMEL_THRESH_FALLBACK_FRACTION) if fg_vals.size else 0.0
    if fg_vals.size == 0:
        return fallback

    counts, edges = np.histogram(fg_vals, bins=nbins)
    bin_centres = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma)

    # Bone: the dominant peak by prominence. Not argmax -- the foreground cut
    # piles partial-volume voxels into the first bin, which can outnumber the
    # bone peak but is an edge, not a peak.
    lin_peaks, lin_props = find_peaks(smoothed, prominence=max(float(smoothed.max()) * 0.05, 1e-8))
    bone_idx = (
        int(lin_peaks[np.argmax(lin_props["prominences"])])
        if len(lin_peaks) else int(np.argmax(smoothed))
    )

    log_smoothed = np.log1p(smoothed)
    log_peaks, _ = find_peaks(log_smoothed, prominence=min_log_prominence)
    min_height = _ENAMEL_MIN_RELATIVE_HEIGHT * float(smoothed[bone_idx])
    candidates = [
        int(p) for p in log_peaks
        if p > bone_idx and smoothed[p] >= min_height
    ]

    if not candidates:
        logger.warning(
            "Enamel threshold: no enamel peak in the log histogram right of "
            "the bone peak (%.4f) — falling back to the knee method",
            float(bin_centres[bone_idx]),
        )
        return _find_enamel_threshold_by_knee(fg_vals, nbins, smooth_sigma, debug=debug)

    # Brightest candidate: enamel is the densest tissue, and a dentin shoulder
    # that happens to clear the prominence bar sits to its left.
    enamel_idx = max(candidates)
    left_bound = max([p for p in log_peaks if p < enamel_idx] + [bone_idx])
    valley_idx = left_bound + int(np.argmin(smoothed[left_bound:enamel_idx + 1]))
    threshold = float(bin_centres[valley_idx])

    logger.info(
        "Enamel threshold: bone peak at %.4f, enamel peak at %.4f, valley "
        "threshold at %.4f (%d voxels at or above it)",
        float(bin_centres[bone_idx]), float(bin_centres[enamel_idx]),
        threshold, int((fg_vals >= threshold).sum()),
    )

    if debug:
        _plot_enamel_threshold_histogram(
            bin_centres, counts, smoothed,
            peak_idx=bone_idx, threshold=threshold, enamel_idx=enamel_idx,
        )

    return threshold


def _find_enamel_threshold_by_knee(
    fg_vals: np.ndarray,
    nbins: int = _ENAMEL_THRESH_NBINS,
    smooth_sigma: float = _ENAMEL_THRESH_SMOOTH_SIGMA,
    fallback_fraction: float = _ENAMEL_THRESH_FALLBACK_FRACTION,
    left_bias: float = _ENAMEL_PEAK_LEFT_BIAS,
    debug: bool = False,
) -> float:
    """
    Fallback enamel threshold: walk the bone peak's descending slope to its
    knee, refit a histogram on the tail beyond it, and place the threshold a
    little left of the tail's peak.

    Used only when the log histogram shows no separate enamel mode.
    """
    fallback = float(fg_vals.max() * fallback_fraction)
    if fg_vals.size == 0:
        return fallback

    counts, edges = np.histogram(fg_vals, bins=nbins)
    bin_centres = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma)

    prom = max(float(smoothed.max()) * 0.05, 1e-8)
    peaks, props = find_peaks(smoothed, prominence=prom)

    if len(peaks) == 0:
        logger.warning("Enamel threshold: no histogram peak found — using fallback")
        return fallback

    peak_a = int(peaks[np.argmax(props["prominences"])])

    x = bin_centres[peak_a:]
    y = smoothed[peak_a:]

    if len(y) < 4:
        logger.warning("Enamel threshold: insufficient tail points — using fallback")
        return fallback

    x0, y0 = float(x[0]), float(y[0])
    x1, y1 = float(x[-1]), float(y[-1])
    dx, dy = x1 - x0, y1 - y0
    denom = float(np.hypot(dx, dy))

    if denom == 0.0:
        logger.warning("Enamel threshold: degenerate tail chord — using fallback")
        return fallback

    dist = np.abs(dy * (x - x0) - dx * (y - y0)) / denom
    slope = np.gradient(y, x)
    valid = np.isfinite(dist) & np.isfinite(slope) & (slope <= 0)

    if not np.any(valid):
        logger.warning("Enamel threshold: no descending tail points — using fallback")
        return fallback

    masked_dist = np.where(valid, dist, -np.inf)
    knee_idx = int(np.argmax(masked_dist))
    knee_threshold = float(x[knee_idx])

    logger.info(
        "Enamel threshold (knee): bone peak at %.4f, knee at %.4f",
        float(bin_centres[peak_a]), knee_threshold,
    )

    tail_vals = fg_vals[fg_vals > knee_threshold]
    if tail_vals.size == 0:
        logger.warning("Enamel threshold: no voxels beyond knee — using knee as fallback")
        return knee_threshold

    tail_counts, tail_edges = np.histogram(tail_vals, bins=nbins)
    tail_bin_centres = (tail_edges[:-1] + tail_edges[1:]) / 2.0
    tail_smoothed = gaussian_filter1d(tail_counts.astype(np.float64), sigma=smooth_sigma)

    tail_prom = max(float(tail_smoothed.max()) * 0.05, 1e-8)
    tail_peaks, tail_props = find_peaks(tail_smoothed, prominence=tail_prom)

    if len(tail_peaks) == 0:
        logger.warning(
            "Enamel threshold: no peak found in tail beyond knee — using knee as fallback"
        )
        threshold = knee_threshold
        if debug:
            _plot_enamel_threshold_histogram(
                bin_centres, counts, smoothed,
                peak_idx=peak_a, threshold=threshold,
            )
        return threshold

    enamel_peak_idx = int(tail_peaks[np.argmax(tail_props["prominences"])])
    enamel_peak_x = float(tail_bin_centres[enamel_peak_idx])

    widths, *_ = peak_widths(tail_smoothed, [enamel_peak_idx], rel_height=0.5)
    half_width_bins = float(widths[0]) / 2.0
    bin_step = float(tail_bin_centres[1] - tail_bin_centres[0]) if len(tail_bin_centres) > 1 else 0.0
    half_width_x = half_width_bins * bin_step

    threshold = enamel_peak_x - left_bias * half_width_x
    threshold = max(threshold, knee_threshold)

    logger.info(
        "Enamel threshold (knee): enamel peak at %.4f (half-width=%.4f), threshold at %.4f",
        enamel_peak_x, half_width_x, threshold,
    )

    if debug:
        _plot_enamel_threshold_histogram(
            bin_centres, counts, smoothed,
            peak_idx=peak_a, threshold=threshold,
        )

    return threshold


def _filter_small_components(
    labeled_volume: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, int]:
    """Remove connected components smaller than min_size. Returns (filtered, n_labels)."""
    labels, counts = np.unique(labeled_volume, return_counts=True)
    valid_labels = labels[(labels != 0) & (counts > min_size)]

    if valid_labels.size == 0:
        return np.zeros_like(labeled_volume), 0

    filtered = np.zeros_like(labeled_volume)
    for new_label, old_label in enumerate(valid_labels, start=1):
        filtered[labeled_volume == old_label] = new_label

    return filtered, int(valid_labels.size)


def _bone_anchor(
    foreground: np.ndarray,
    enamel_mask: np.ndarray,
    margin_fraction: float = _ANCHOR_MARGIN_FRACTION,
) -> np.ndarray:
    """
    Foreground voxels that are certainly bone: everything outside the enamel's
    bounding box grown by a margin.

    The old test for "enamel still attached to bone" identified bone as the
    largest connected component. That is not stable as the threshold rises:
    bone thins faster than the molars, so past some level the molar *is* the
    largest component and enamel reads as "connected to bone" again. Measured
    on WT-M-1: separated at 0.20-0.35, "reconnected" from 0.40 up -- so a
    sweep that stepped over the window reported that the molars never
    separated. A fixed geometric anchor makes connectivity monotone in the
    threshold (raising it only removes voxels), which is what lets the search
    bracket and bisect.

    The anchor is not guaranteed to be free of tooth: molar roots reach past
    any margin small enough to leave the anchor holding bone. Callers must
    therefore treat a small overlap with the anchor as a root tip, not as a
    bone connection -- see _ANCHOR_ATTACHED_SHARE.
    """
    box = ndi.find_objects(enamel_mask.astype(np.uint8))[0]
    margin = max(1, int(round(margin_fraction * max(foreground.shape))))
    grown = tuple(
        slice(max(s.start - margin, 0), min(s.stop + margin, n))
        for s, n in zip(box, foreground.shape)
    )
    anchor = foreground.copy()
    anchor[grown] = False
    return anchor


def _find_molar_separation_threshold(
    volume: np.ndarray,
    enamel_mask: np.ndarray,
    anchor: np.ndarray,
    floor: float,
    ceiling: float,
    coarse_fraction: float = _SEPARATION_COARSE_FRACTION,
    fine_fraction: float = _SEPARATION_FINE_FRACTION,
    attached_share: float = _ANCHOR_ATTACHED_SHARE,
) -> tuple[Optional[np.ndarray], float, int]:
    """
    Find the lowest intensity threshold at which no enamel-bearing component
    is still part of the bone -- i.e. holds ``attached_share`` or more of the
    above-threshold anchor -- by a coarse ascending sweep followed by
    bisection.

    Dentin and bone overlap in intensity, so there is no single level that
    cuts cleanly everywhere; but raising the threshold only ever removes
    voxels, so once enamel detaches from the anchor it stays detached. The
    coarse pass climbs from the foreground floor in large steps until the
    first detached level, bracketing the transition; bisection then narrows
    that bracket. The lowest detached level is wanted because every step up
    shaves real tissue off the molar surface.

    This replaces a fixed 0.01 sweep from zero, which re-labelled the whole
    volume up to 200 times (3.3 s each on WT-M-1) -- the first ~14 of them
    below the foreground floor, where nothing can change. The search here
    needed 11 labellings.

    Returns (labels at the separation threshold, threshold, n_labellings), or
    (None, ceiling, n) if enamel never detaches below ``ceiling``.
    """
    structure = ndi.generate_binary_structure(3, 1)
    n_labellings = 0

    def _connected(t: float) -> tuple[bool, np.ndarray]:
        nonlocal n_labellings
        n_labellings += 1
        above = volume > t
        labels, n = ndi.label(above, structure=structure)  # type: ignore
        anchor_counts = np.bincount(labels[anchor & above], minlength=n + 1)
        anchor_counts[0] = 0
        total = int(anchor_counts.sum())
        enamel_counts = np.bincount(labels[enamel_mask], minlength=n + 1)
        enamel_counts[0] = 0
        enamel_ids = np.flatnonzero(enamel_counts)
        share = (
            float(anchor_counts[enamel_ids].max()) / total
            if total and enamel_ids.size else 0.0
        )
        attached = share >= attached_share
        logger.debug(
            "Molar separation probe: threshold=%.4f -> enamel component holds "
            "%.2f%% of the bone anchor (attached=%s)", t, 100.0 * share, attached,
        )
        return attached, labels

    span = ceiling - floor
    coarse_step = max(span * coarse_fraction, 1e-6)
    tolerance = max(span * fine_fraction, 1e-6)

    lo = floor
    hi_labels = None
    hi = lo
    while hi < ceiling:
        hi = min(lo + coarse_step, ceiling)
        attached, labels = _connected(hi)
        if not attached:
            hi_labels = labels
            break
        lo = hi

    if hi_labels is None:
        return None, ceiling, n_labellings

    logger.info(
        "Molar separation bracketed in (%.4f, %.4f] after %d coarse step(s)",
        lo, hi, n_labellings,
    )

    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        attached, labels = _connected(mid)
        if attached:
            lo = mid
        else:
            hi, hi_labels = mid, labels

    return hi_labels, hi, n_labellings


def _regrow_molar_rim(
    volume: np.ndarray,
    foreground: np.ndarray,
    molar_core: np.ndarray,
    threshold: float,
    rim_voxels: int = _MOLAR_RIM_VOXELS,
) -> np.ndarray:
    """
    Give back the molar surface that the separation threshold shaved off.

    Every voxel at or below the separation threshold is dropped from the
    core, including the partial-volume rim of the tooth itself. Letting the
    molar regrow freely into the foreground recovers that rim but also runs
    down thin, bone-intensity stems of alveolar bone that touch the tooth
    (seen on WT-M-1). So growth is confined to a shell a couple of voxels
    thick, and a marker watershed on inverted intensity decides each shell
    voxel between molar and the bone above threshold, so a contested voxel
    goes to whichever side it is joined to through brighter tissue, and the
    split lands in the dark periodontal gap.
    """
    if rim_voxels <= 0 or not molar_core.any():
        return molar_core

    box = ndi.find_objects(molar_core.astype(np.uint8))[0]
    pad = rim_voxels + 1
    box = tuple(
        slice(max(s.start - pad, 0), min(s.stop + pad, n))
        for s, n in zip(box, volume.shape)
    )
    core = molar_core[box]
    vol = volume[box]
    fg = foreground[box]

    shell = ndi.binary_dilation(core, ndi.generate_binary_structure(3, 1), iterations=rim_voxels)
    markers = np.zeros(core.shape, dtype=np.int32)
    markers[fg & (vol > threshold) & ~core] = 2
    markers[core] = 1
    grown = watershed(-vol, markers=markers, mask=fg & (shell | (markers == 2))) == 1

    out = molar_core.copy()
    out[box] |= grown
    return out


def segment_molar_bone(
    preprocessed_volume: np.ndarray,
    min_enamel_seed_size: int = _MIN_ENAMEL_SEED_SIZE,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment molars and bone: find the lowest intensity threshold at which
    every enamel cap is detached from bone, then take the enamel-bearing
    components at that threshold as the molars.

    Dentin (the bulk of a molar) and bone sit at overlapping intensities, so
    a single global threshold either fuses molars to bone or cuts into real
    tissue. Enamel is strictly brighter than both, so it can always be
    isolated cleanly, and it only exists in the molars once the incisor is
    removed. The steps:

    1. Enamel threshold from the valley below the enamel peak of the log
       foreground histogram (see ``_find_enamel_threshold``).
    2. Bone anchor: foreground well away from any enamel.
    3. Coarse-then-bisect search for the lowest threshold at which no
       enamel component still holds a real share of the anchor (a root
       tip reaching into it does not count).
    4. Molars = components holding enamel at that threshold, regrown by a
       thin rim to recover the surface the threshold shaved off.
    5. Bone = the rest of the foreground.

    Falls back to the erosion-based max-intensity method if no enamel is
    found or it never detaches from bone.

    Parameters:
        preprocessed_volume: Intensity volume with incisor removed and
            background zeroed.
        min_enamel_seed_size: Minimum voxels for an enamel cluster to count,
            filtering out noise speckles.
        debug: If True, show the enamel histogram and open napari viewers of
            the enamel seeds and the final bone/molar split.

    Returns:
        (bone_mask, molar_mask) — boolean masks.
    """
    foreground_full = preprocessed_volume > 0

    if not foreground_full.any():
        empty = np.zeros_like(foreground_full)
        return empty, empty

    # Everything below runs on the foreground's bounding box. Labelling is the
    # dominant cost and scales with the array, not with the tissue in it.
    crop = ndi.find_objects(foreground_full.astype(np.uint8))[0]
    volume = preprocessed_volume[crop]
    foreground = foreground_full[crop]

    fg_vals = volume[foreground]
    enamel_threshold = _find_enamel_threshold(fg_vals, debug=debug)
    floor = float(fg_vals.min())
    del fg_vals

    enamel_labels, _ = ndi.label(foreground & (volume >= enamel_threshold))  # type: ignore
    enamel_counts = np.bincount(enamel_labels.ravel())
    enamel_counts[0] = 0
    kept = np.flatnonzero(enamel_counts >= min_enamel_seed_size)
    enamel_mask = np.isin(enamel_labels, kept)
    del enamel_labels
    logger.info(
        "Enamel: %d voxels in %d cluster(s) of >= %d voxels (largest %s)",
        int(enamel_mask.sum()), kept.size, min_enamel_seed_size,
        sorted((int(c) for c in enamel_counts[kept]), reverse=True)[:5],
    )

    if debug:
        create_3d_visualization(
            volume,
            additional_volumes={
                "Foreground": (foreground, "gray"),
                "Enamel": (enamel_mask, "yellow"),
            },
            title="[debug] Enamel vs Foreground Mask",
        )

    if enamel_mask.any():
        anchor = _bone_anchor(foreground, enamel_mask)
        if not anchor.any():
            logger.warning("Molar separation: enamel spans the whole volume, no bone anchor left")
        else:
            labels, threshold, n_labellings = _find_molar_separation_threshold(
                volume, enamel_mask, anchor,
                floor=floor, ceiling=enamel_threshold,
            )
            if labels is None:
                logger.warning(
                    "Enamel never detached from bone below the enamel threshold "
                    "%.4f (%d labellings)", enamel_threshold, n_labellings,
                )
            else:
                molar_ids = np.unique(labels[enamel_mask])
                molar_ids = molar_ids[molar_ids > 0]
                sizes = np.bincount(labels.ravel())[molar_ids]
                molar_ids = molar_ids[sizes >= MIN_COMPONENT_SIZE]
                core = np.isin(labels, molar_ids)
                del labels

                molar = _regrow_molar_rim(volume, foreground, core, threshold)
                logger.info(
                    "Molars separated at threshold %.4f (%d labellings): %d "
                    "component(s), %d core voxels -> %d after rim regrowth",
                    threshold, n_labellings, molar_ids.size,
                    int(core.sum()), int(molar.sum()),
                )

                molar_mask = np.zeros_like(foreground_full)
                molar_mask[crop] = molar
                bone_mask = foreground_full & ~molar_mask

                if debug:
                    create_3d_visualization(
                        preprocessed_volume,
                        additional_volumes={
                            "Bone": (bone_mask, "blue"),
                            "Molar": (molar_mask, "red"),
                        },
                        title=f"[debug] Bone/Molar Split (threshold={threshold:.4f})",
                    )
                return bone_mask, molar_mask

    logger.warning(
        "No usable enamel separation — falling back to erosion-based component selection"
    )
    return _segment_molar_bone_by_erosion(preprocessed_volume, foreground_full)


def _segment_molar_bone_by_erosion(
    preprocessed_volume: np.ndarray,
    foreground: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Original fallback: erode until components separate, pick highest-intensity
    component as molar.
    """
    labeled_volume = label_3d_volume(foreground, connectivity=1)
    labeled_volume, n_labels = _filter_small_components(labeled_volume, MIN_COMPONENT_SIZE)

    erosion_count = 0
    max_erosions = 20

    logger.info("Fallback: initial segmentation found %d component(s) > %d voxels", n_labels, MIN_COMPONENT_SIZE)

    while n_labels < 2 and erosion_count < max_erosions:
        foreground = erode_mask(foreground, radius=1)
        if not np.any(foreground):
            break
        labeled_volume = label_3d_volume(foreground, connectivity=1)
        labeled_volume, n_labels = _filter_small_components(labeled_volume, MIN_COMPONENT_SIZE)
        erosion_count += 1

    if erosion_count > 0:
        logger.info(
            "Fallback: needed %d erosion(s) to reach %d component(s)",
            erosion_count, n_labels,
        )

    if n_labels == 0:
        empty = np.zeros_like(foreground)
        return empty, empty

    if n_labels == 1:
        logger.warning("Fallback: only 1 valid component found; returning as bone")
        return labeled_volume > 0, np.zeros_like(foreground)

    molar_label = max(
        range(1, n_labels + 1),
        key=lambda lbl: float(preprocessed_volume[labeled_volume == lbl].max()),
    )
    molar_mask = labeled_volume == molar_label
    bone_mask = (labeled_volume > 0) & ~molar_mask
    return bone_mask, molar_mask
