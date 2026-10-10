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

# ...and the molars count as still fused to bone only while the components
# that hold the anchor carry at least this share of the enamel. Less than that
# is not a tooth but a dense patch of bone that reached the enamel threshold:
# it is part of the bone at every level, so waiting for it to "detach" climbs
# until only the enamel caps are left. On P42_WT_WT_1 (no enamel peak, knee
# threshold 0.58) a 4,586-voxel patch of cortical bone, 0.8% of the enamel,
# held the search up to 0.551; without it the molars part from bone at 0.307
# and keep their dentin and roots (0.75M -> 3.0M core voxels). The smallest
# real tooth, M3, carries 11.5-16.4% of the enamel on the P22 scans.
_FUSED_ENAMEL_MIN_SHARE = 0.03

# Coarse step of the separation search, as a fraction of the range between
# the foreground floor and the enamel threshold, and the bisection tolerance.
_SEPARATION_COARSE_FRACTION = 0.1
_SEPARATION_FINE_FRACTION = 0.005

# Thickness, in voxels, of the shell the molar mask may regrow into after
# separation. See _regrow_molar_rim.
_MOLAR_RIM_VOXELS = 2

# Background trim (see _trim_attached_background). Molar voxels dimmer than
# the tissue floor are kept only within this many voxels of brighter tooth,
# which is the blurred surface of the tooth itself. On WT-M-1 (denoised) the
# surface ramps from background to dentin over ~3-4 voxels; 2 was picked by
# eye over 3 and 4 as the cut that leaves the crowns cleanest.
_BACKGROUND_RIM_VOXELS = 2

# A piece of dim material is background only if it comes within this many
# voxels of enamel, i.e. it rests on the crown. On WT-M-1, soft tissue coating
# the crowns comes within ~4.5 voxels of enamel, while the dim lining of the
# pulp chamber never comes closer than 12 (dentin lies in between) and root
# tips sit 16-150 voxels away.
_BACKGROUND_ENAMEL_CONTACT_VOXELS = 5

# ...and within a piece that rests on the crown, only voxels this close to
# enamel are removed. A dim sheet can run from the crown down a root surface
# to the dim, tapering root tip, joining both into one piece; the cap stops
# the trim partway. It trades completeness for cleanliness: at 15 every root
# tip on WT-M-1 survived, but the crowns kept visible background; at 30 the
# crowns come out clean and the dimmest end of a root arm lying within 30
# voxels of enamel is trimmed with it.
_BACKGROUND_ENAMEL_REACH_VOXELS = 30

# Everywhere on the tooth -- crown, roots, pulp walls -- molar voxels below
# this fraction of the way from the separation threshold up to the tissue
# floor are background, with no position test. On WT-M-1 that is 0.230: the
# faintest ~1-voxel skin (0.15-0.23), which was judged by eye to be background
# all round the tooth. It also undoes _regrow_molar_rim, whose voxels all sit
# below the separation threshold; the regrowth still decides which of them
# are kept out of the bone mask.
_BACKGROUND_SURFACE_CUT_FRACTION = 0.5

# Bone cleanup (see clean_bone_mask). The foreground the bone mask is cut from
# starts at the background/bone knee, which lets in the soft tissue round the
# teeth -- the incisor's pulp, the periodontal space, the gingiva on the
# crowns -- at under half the density of bone. On the 13 P22 TGM2 scans it was
# 30-43% of the bone mask: a soft-tissue mode at 4,500-5,500 (scanner units)
# beside bone's at 10,200-11,900, parted by a valley at 6,900-7,900.
#
# A mode must stand this far out of the smoothed histogram, as a fraction of
# its tallest bin, to count.
_BONE_PEAK_MIN_PROMINENCE = 0.05

# The valley must dip below this fraction of the soft-tissue side (its peak,
# or the histogram's low edge where soft tissue shows only as a falling edge)
# to be cut at. The two modes overlap through the partial-volume surface of
# the bone, so the dip is shallow: to 0.50-0.86 of the soft-tissue peak on
# the P22 scans. A mask that is already clean rises from its low edge straight
# to the bone peak, with no dip at all, so it is left as it is: cleaning the
# 13 cleaned masks again removed nothing.
_BONE_FLOOR_MAX_DIP = 0.95

# ...and the soft-tissue mode must sit in the lower part of the range below
# bone: at most this fraction of the way from the histogram's low end up to
# the bone peak (0.39-0.54 on the P22 scans). A mode just under the bone peak
# is more bone -- trabecular beside cortical, say -- not soft tissue, and is
# not cut away.
_BONE_SOFT_MAX_POSITION = 0.6


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
    info: dict | None = None,
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
    ``info``, when given, is told which way the threshold was found
    (``info["method"]``): "log-valley", "knee", or "max-fraction" (a share of
    the brightest value, when the histogram has no usable peak at all).
    """
    fallback = float(fg_vals.max() * _ENAMEL_THRESH_FALLBACK_FRACTION) if fg_vals.size else 0.0
    if info is not None:
        info["method"] = "max-fraction"
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
        threshold = _find_enamel_threshold_by_knee(fg_vals, nbins, smooth_sigma, debug=debug)
        if info is not None:
            # The knee method's own last resort is the same share of the brightest value.
            info["method"] = "max-fraction" if threshold == fallback else "knee"
        return threshold

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

    if info is not None:
        info["method"] = "log-valley"
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


def _fused_with_bone(
    labels: np.ndarray,
    ids: np.ndarray,
    anchor: np.ndarray,
    attached_share: float = _ANCHOR_ATTACHED_SHARE,
) -> np.ndarray:
    """
    The components among ``ids`` that are still part of the bone: each holds
    at least ``attached_share`` of the anchor voxels that are labelled at all
    (those above the threshold ``labels`` was taken at).
    """
    counts = np.bincount(labels[anchor], minlength=int(labels.max()) + 1)
    counts[0] = 0
    return ids[counts[ids] >= attached_share * max(int(counts.sum()), 1)]


def _find_molar_separation_threshold(
    volume: np.ndarray,
    enamel_mask: np.ndarray,
    anchor: np.ndarray,
    floor: float,
    ceiling: float,
    coarse_fraction: float = _SEPARATION_COARSE_FRACTION,
    fine_fraction: float = _SEPARATION_FINE_FRACTION,
    attached_share: float = _ANCHOR_ATTACHED_SHARE,
    enamel_share: float = _FUSED_ENAMEL_MIN_SHARE,
) -> tuple[Optional[np.ndarray], float, int]:
    """
    Find the lowest intensity threshold at which no enamel-bearing component
    is still part of the bone -- i.e. holds ``attached_share`` or more of the
    above-threshold anchor -- by a coarse ascending sweep followed by
    bisection. Components that do hold the anchor but carry under
    ``enamel_share`` of the enamel are dense bone, not teeth, and do not keep
    the search climbing (see _FUSED_ENAMEL_MIN_SHARE).

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
        fused = enamel_ids[anchor_counts[enamel_ids] >= attached_share * total] \
            if total else enamel_ids[:0]
        held = float(enamel_counts[fused].sum()) / max(int(enamel_counts.sum()), 1)
        attached = share >= attached_share and held >= enamel_share
        logger.debug(
            "Molar separation probe: threshold=%.4f -> enamel component holds "
            "%.2f%% of the bone anchor, fused components carry %.2f%% of the "
            "enamel (attached=%s)", t, 100.0 * share, 100.0 * held, attached,
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


def _find_tissue_floor(
    molar_vals: np.ndarray,
    low: float,
    high: float,
    nbins: int = _ENAMEL_THRESH_NBINS,
    smooth_sigma: float = _ENAMEL_THRESH_SMOOTH_SIGMA,
) -> Optional[float]:
    """
    The intensity below which a molar voxel is not solid tooth: the valley
    between the dim mode of the molar mask's histogram and its dentin peak.

    The dim mode is everything the separation threshold let in that is not
    dentin or enamel -- soft tissue resting on the crown and the blurred
    surface of the tooth. On WT-M-1 the valley sits at 0.275 against a dentin
    peak at 0.51; the pipeline's bone floor on the same scan is 0.273.

    The histogram spans ``low`` (the separation threshold) to ``high`` (the
    enamel threshold), so the enamel peak cannot outrank dentin. On WT-M-1
    the dim mode is not a peak but a falling edge from ``low``; when it is a
    peak of its own, the valley is searched right of it, since the stretch
    between ``low`` and that peak can be emptier than the valley itself.
    Returns None when there is no dentin peak to anchor the valley.
    """
    counts, edges = np.histogram(molar_vals, bins=nbins, range=(low, high))
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma)
    peaks, props = find_peaks(smoothed, prominence=max(float(smoothed.max()) * 0.05, 1e-8))
    if not len(peaks):
        return None
    dentin_idx = int(peaks[np.argmax(props["prominences"])])
    left_bound = max([int(p) for p in peaks if p < dentin_idx] + [0])
    valley_idx = left_bound + int(np.argmin(smoothed[left_bound:dentin_idx + 1]))
    return float((edges[valley_idx] + edges[valley_idx + 1]) / 2.0)


def _trim_attached_background(
    volume: np.ndarray,
    molar: np.ndarray,
    enamel_mask: np.ndarray,
    low: float,
    high: float,
    tissue_floor: Optional[float] = None,
    rim_voxels: int = _BACKGROUND_RIM_VOXELS,
    contact_voxels: int = _BACKGROUND_ENAMEL_CONTACT_VOXELS,
    reach_voxels: int = _BACKGROUND_ENAMEL_REACH_VOXELS,
    surface_cut_fraction: float = _BACKGROUND_SURFACE_CUT_FRACTION,
) -> np.ndarray:
    """
    Remove the dim background that comes away with the molars.

    The separation threshold is low -- 0.18 on WT-M-1, under the bone floor
    of 0.27 -- because it is the lowest level at which the molars let go of
    bone, and so the molar components still hold whatever dim material
    touches them: soft tissue and debris on the crowns, which shows as a
    lumpy coating, and a faint skin all round the tooth.

    The faint skin goes wherever it is: every voxel below
    ``surface_cut_fraction`` of the way from ``low`` up to the tissue floor
    (``tissue_floor``, or ``_find_tissue_floor`` when not given).

    Above that cut, dim material (below the tissue floor) is still a mix. The
    tooth's own blurred surface, the tapering root tips and the dim lining of
    the pulp chamber must stay; the coating on the crowns must go. They are
    told apart by where they sit:

    - The blurred surface lies within ``rim_voxels`` of solid tooth. Only dim
      voxels farther out than that are candidates.
    - Background rests on the crown, which is where the enamel is. A
      connected piece of candidates is background only if it comes within
      ``contact_voxels`` of enamel. The pulp lining is walled off from enamel
      by dentin, and root tips are far from it.
    - Within such a piece, only voxels within ``reach_voxels`` of enamel are
      removed, so a dim sheet running down a root surface does not carry the
      trim to the root tip.

    Anything left disconnected from an enamel-bearing molar afterwards was
    joined to it only through background and goes too. Measured on WT-M-1
    with the defaults: 1.04M of 6.38M voxels removed (16.3%). Nearly all of
    it is under the surface cut -- a voxel-deep skin over the whole tooth,
    plus most of the crown coating -- and the position test adds the ~5k
    voxels of coating brighter than the cut.
    """
    if not molar.any() or not enamel_mask.any():
        return molar

    box = ndi.find_objects(molar.astype(np.uint8))[0]
    box = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, n)) for s, n in zip(box, molar.shape))
    mol = molar[box]
    vol = volume[box]
    enamel = enamel_mask[box] & mol

    floor = tissue_floor if tissue_floor is not None else _find_tissue_floor(vol[mol], low, high)
    if floor is None or not enamel.any():
        logger.warning("Background trim: no dentin peak in the molar histogram — skipping")
        return molar

    surface_cut = low + surface_cut_fraction * (floor - low)
    faint = mol & (vol < surface_cut)

    solid = mol & (vol > floor)
    dim = mol & (
        np.asarray(ndi.distance_transform_edt(~solid), dtype=np.float64) > rim_voxels
    )
    del solid

    background = np.zeros_like(mol)
    n_resting = n_pieces = 0
    if dim.any():
        enamel_dist: np.ndarray = np.asarray(
            ndi.distance_transform_edt(~enamel), dtype=np.float64
        )
        pieces, n_pieces = ndi.label(dim, structure=ndi.generate_binary_structure(3, 3))  # type: ignore
        resting = np.unique(pieces[dim & (enamel_dist <= contact_voxels)])
        resting = resting[resting > 0]
        n_resting = resting.size
        background = np.isin(pieces, resting) & (enamel_dist <= reach_voxels)
        del pieces, enamel_dist
    del dim

    trimmed = mol & ~background & ~faint
    labels, n = ndi.label(trimmed, structure=ndi.generate_binary_structure(3, 1))  # type: ignore
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    enamel_counts = np.bincount(labels[enamel], minlength=n + 1)
    keep = np.flatnonzero((enamel_counts > 0) & (sizes >= MIN_COMPONENT_SIZE))
    keep = keep[keep > 0]
    trimmed = np.isin(labels, keep)
    del labels

    n_faint = int(faint.sum())
    n_coating = int((background & ~faint).sum())
    n_detached = int(mol.sum()) - n_faint - n_coating - int(trimmed.sum())
    logger.info(
        "Background trim: tissue floor %.4f, surface cut %.4f -> %d faint-skin "
        "voxels removed; %d of %d dim piece(s) rest on the crowns -> %d more; "
        "%d left detached (%.2f%% of the molar mask in all)",
        floor, surface_cut, n_faint, n_resting, n_pieces, n_coating, n_detached,
        100.0 * (n_faint + n_coating + n_detached) / max(int(mol.sum()), 1),
    )

    out = molar.copy()
    out[box] = trimmed
    return out


def find_bone_floor(
    values: np.ndarray,
    low: float,
    high: float,
    nbins: int = _ENAMEL_THRESH_NBINS,
    smooth_sigma: float = _ENAMEL_THRESH_SMOOTH_SIGMA,
) -> Optional[float]:
    """
    The intensity below which a bone-mask voxel is soft tissue: the valley
    under the bone peak of the mask's histogram, or None when there is none.

    Bone is the brightest mode between ``low`` and ``high``. Soft tissue is the
    tallest mode below it, or, where it shows only as a falling edge from
    ``low``, that edge. The valley between them is where the density is least
    common, which is also where the partial-volume surface of bone is cut
    halfway, as a threshold between two materials should cut it. None when
    the histogram does not dip below ``_BONE_FLOOR_MAX_DIP`` of the soft
    tissue side -- no soft tissue, or a mask already cleaned -- or when the
    lower mode sits too close under bone to be soft tissue
    (``_BONE_SOFT_MAX_POSITION``).
    """
    values = np.asarray(values)
    if values.size == 0 or not high > low:
        return None
    counts, edges = np.histogram(values, bins=nbins, range=(low, high))
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma)
    peaks, _ = find_peaks(smoothed, prominence=max(float(smoothed.max()) * _BONE_PEAK_MIN_PROMINENCE, 1e-8))
    if not len(peaks):
        return None
    bone = int(peaks[-1])
    below = [int(p) for p in peaks if p < bone]
    soft = max(below, key=lambda p: smoothed[p]) if below else 0
    valley = soft + int(np.argmin(smoothed[soft:bone + 1]))
    if valley in (soft, bone) or smoothed[valley] > _BONE_FLOOR_MAX_DIP * smoothed[soft] \
            or soft > _BONE_SOFT_MAX_POSITION * bone:
        return None
    return float((edges[valley] + edges[valley + 1]) / 2.0)


def clean_bone_mask(
    volume: np.ndarray,
    candidate: np.ndarray,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """
    Keep only the bone: the voxels above the soft-tissue floor, in the one
    large piece the jaw is.

    ``candidate`` is everything thresholded in that is neither incisor nor
    molar. Two kinds of it are not bone. Soft tissue round the teeth passes
    the background threshold (see ``_BONE_PEAK_MIN_PROMINENCE``) and is cut at
    ``find_bone_floor``. Islands -- specks of noise, and slivers left
    stranded when the teeth were cut out -- are dropped by keeping only the
    largest 26-connected piece. ``low`` and ``high`` bound the histogram the
    floor is read from (by default the candidate's own range).

    Returns (bone mask, {"floor", "soft_tissue", "islands", "kept"}), voxel
    counts for the log.
    """
    values = volume[candidate]
    info = {"floor": None, "soft_tissue": 0, "islands": 0, "kept": 0}
    if values.size == 0:
        return candidate.copy(), info
    lo = float(values.min()) if low is None else float(low)
    hi = float(values.max()) if high is None else float(high)
    floor = find_bone_floor(values, lo, hi)
    del values
    bone = candidate & (volume > floor) if floor is not None else candidate.copy()
    info["floor"] = floor
    info["soft_tissue"] = int(candidate.sum()) - int(bone.sum())

    labels, n = ndi.label(bone, structure=ndi.generate_binary_structure(3, 3))  # type: ignore
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        info["islands"] = int(sizes.sum()) - int(sizes[largest])
        bone = labels == largest
    del labels
    info["kept"] = int(bone.sum())
    return bone, info


def _log_bone_cleanup(info: dict, n_candidate: int) -> None:
    logger.info(
        "Bone cleanup: soft-tissue floor %s -> %d soft-tissue voxels and %d in islands removed; "
        "%d of %d kept (%.1f%%)",
        "none found" if info["floor"] is None else f"{info['floor']:.4g}",
        info["soft_tissue"], info["islands"], info["kept"], n_candidate,
        100.0 * info["kept"] / max(n_candidate, 1),
    )


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
    5. Bone = the rest of the foreground, less the soft tissue round the
       teeth and the islands (see ``clean_bone_mask``).
    6. Dim background resting on the crowns is trimmed off the molars and
       assigned to neither mask (see ``_trim_attached_background``).

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
                # A component that bears "enamel" but is still part of the
                # bone at the separation threshold is the bone itself, with a
                # dense patch in it (see _FUSED_ENAMEL_MIN_SHARE): it stays
                # bone, and its patch stops counting as enamel.
                fused = _fused_with_bone(labels, molar_ids, anchor)
                if fused.size:
                    patch = enamel_mask & np.isin(labels, fused)
                    logger.warning(
                        "Molar separation: %d enamel voxel(s) (%.1f%% of the enamel) "
                        "stay part of the bone at threshold %.4f; kept as dense "
                        "bone, not tooth", int(patch.sum()),
                        100.0 * float(patch.sum()) / max(int(enamel_mask.sum()), 1),
                        threshold,
                    )
                    enamel_mask = enamel_mask & ~patch
                    molar_ids = np.setdiff1d(molar_ids, fused)
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

                # Bone is cut against the untrimmed molars: the background
                # trimmed off a crown is not bone, so it belongs to neither.
                # Nor are the soft tissue round the teeth and the islands.
                candidate = foreground & ~molar
                bone, bone_info = clean_bone_mask(volume, candidate, low=floor, high=enamel_threshold)
                _log_bone_cleanup(bone_info, int(candidate.sum()))
                del candidate
                bone_mask = np.zeros_like(foreground_full)
                bone_mask[crop] = bone
                del bone
                molar = _trim_attached_background(
                    volume, molar, enamel_mask, low=threshold, high=enamel_threshold,
                )
                molar_mask = np.zeros_like(foreground_full)
                molar_mask[crop] = molar

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
    bone_mask, molar_mask = _segment_molar_bone_by_erosion(preprocessed_volume, foreground_full)
    if bone_mask.any():
        candidate = bone_mask
        bone_mask, bone_info = clean_bone_mask(preprocessed_volume, candidate)
        _log_bone_cleanup(bone_info, int(candidate.sum()))
        del candidate
    return bone_mask, molar_mask


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
