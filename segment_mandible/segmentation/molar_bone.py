import logging
from typing import Optional

import numpy as np
from scipy.ndimage import binary_propagation, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from skimage.measure import label

from preprocessing import cut_bridges_slice
from segmentation.utils import (
    dilate_mask,
    erode_mask,
    label_3d_volume,
)
from visualization import create_3d_visualization

logger = logging.getLogger(__name__)

MIN_COMPONENT_SIZE = 3500

# Minimum voxels for an enamel cluster to count as a molar anchor.
_MIN_ENAMEL_SEED_SIZE = 50

# Histogram resolution used to locate the bone-peak's tail knee.
_ENAMEL_THRESH_NBINS = 128
_ENAMEL_THRESH_SMOOTH_SIGMA = 2.25

# Fallback threshold as a fraction of the foreground max, used only if the
# histogram has no discernible peak (e.g. near-empty foreground).
_ENAMEL_THRESH_FALLBACK_FRACTION = 0.9

# Fraction of the enamel peak's local half-width to shift the threshold left
# (toward the bone knee) once the enamel peak has been located.
_ENAMEL_PEAK_LEFT_BIAS = 0.5


def _plot_enamel_threshold_histogram(
    bin_centres: np.ndarray,
    counts: np.ndarray,
    smoothed: np.ndarray,
    peak_idx: Optional[int],
    threshold: float,
) -> None:
    """Show the foreground intensity histogram with the detected bone peak and knee."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bin_centres, counts, width=bin_centres[1] - bin_centres[0], color="lightgray", label="raw histogram")
    ax.plot(bin_centres, smoothed, color="tab:blue", label="smoothed")

    if peak_idx is not None:
        ax.axvline(bin_centres[peak_idx], color="tab:green", linestyle="--", label="bone peak")
    ax.axvline(threshold, color="tab:red", linestyle="--", label="enamel threshold (knee)")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Voxel count")
    ax.set_title("Foreground intensity histogram — enamel threshold")
    ax.legend()
    plt.show()


def _find_enamel_threshold(
    fg_vals: np.ndarray,
    nbins: int = _ENAMEL_THRESH_NBINS,
    smooth_sigma: float = _ENAMEL_THRESH_SMOOTH_SIGMA,
    fallback_fraction: float = _ENAMEL_THRESH_FALLBACK_FRACTION,
    left_bias: float = _ENAMEL_PEAK_LEFT_BIAS,
    debug: bool = False,
) -> float:
    """
    Adaptively locate the enamel intensity threshold.

    Bone dominates the foreground by voxel count and forms the histogram's
    main peak; enamel is a sparse, much brighter tail beyond it. A fixed
    percentile can't tell those apart — a sample with unusually bright or
    voluminous bone would push bone voxels past a fixed cutoff. Instead we
    find the bone peak, then walk its right (descending) slope to the knee
    where the curve flattens into that sparse bright tail. The knee alone is
    too conservative as an enamel cutoff — it just marks where bone ends, not
    where enamel is centered. So we discard everything at or below the knee,
    refit a histogram on the remaining tail voxels, and locate the enamel
    peak within that tail. The threshold sits at that peak, shifted slightly
    left (toward the knee) by a fraction of the peak's local half-width so we
    don't clip the peak's left shoulder.
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

    # Bone is the dominant (most prominent) peak in the foreground histogram.
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
        "Enamel threshold: bone peak at %.4f, knee at %.4f",
        float(bin_centres[peak_a]), knee_threshold,
    )

    # Refit on the raw tail voxels beyond the knee to locate the enamel peak.
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

    # Shift left from the enamel peak by a fraction of its local half-width.
    widths, *_ = peak_widths(tail_smoothed, [enamel_peak_idx], rel_height=0.5)
    half_width_bins = float(widths[0]) / 2.0
    bin_step = float(tail_bin_centres[1] - tail_bin_centres[0]) if len(tail_bin_centres) > 1 else 0.0
    half_width_x = half_width_bins * bin_step

    threshold = enamel_peak_x - left_bias * half_width_x
    threshold = max(threshold, knee_threshold)

    logger.info(
        "Enamel threshold: enamel peak at %.4f (half-width=%.4f), threshold at %.4f",
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


def _enamel_seeds(
    volume: np.ndarray,
    foreground: np.ndarray,
    min_seed_size: int,
    max_molars: int,
    debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Identify molar anchors from enamel clusters.

    After incisor removal, enamel only exists in molars. We adaptively
    threshold the foreground at the knee beyond the dominant bone peak to
    isolate candidate enamel voxels, label the resulting clusters, discard
    any smaller than min_seed_size (noise speckles), then keep the top N
    by mean intensity as seeds — one per molar. Ranking by brightness
    rather than size matters because a large, merely-bright patch of bone
    that crossed the threshold can otherwise outrank a smaller but truly
    enamel-bright cluster.

    Returns a label array (same shape as volume, dtype int32) with each enamel
    seed cluster assigned a unique label 1..N, or None if fewer than 1 seed is
    found.
    """
    fg_vals = volume[foreground]
    if fg_vals.size == 0:
        return None

    enamel_thresh = _find_enamel_threshold(fg_vals, debug=debug)

    enamel_mask = foreground & (volume >= enamel_thresh)
    if not enamel_mask.any():
        logger.warning("No enamel voxels found above threshold — will fall back")
        return None

    labeled_enamel, n = label(enamel_mask, connectivity=1, return_num=True)  # type: ignore
    logger.info("Enamel clusters found: %d", n)

    if n == 0:
        return None

    counts = np.bincount(labeled_enamel.ravel())
    counts[0] = 0

    candidate_labels = [lbl for lbl in range(1, n + 1) if counts[lbl] >= min_seed_size]

    if len(candidate_labels) == 0:
        logger.warning("No enamel clusters met the minimum size — will fall back")
        return None

    # Keep up to max_molars brightest clusters (by mean intensity) among
    # those that passed the minimum-size filter.
    mean_intensities = {
        lbl: float(volume[labeled_enamel == lbl].mean()) for lbl in candidate_labels
    }
    top_labels = sorted(
        candidate_labels, key=lambda lbl: mean_intensities[lbl], reverse=True
    )[:max_molars]

    logger.info(
        "Enamel seeds selected: %d cluster(s), sizes %s, mean intensities %s",
        len(top_labels),
        [int(counts[l]) for l in top_labels],
        [round(mean_intensities[l], 4) for l in top_labels],
    )

    seeds = np.zeros(volume.shape, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(top_labels, start=1):
        seeds[labeled_enamel == old_lbl] = new_lbl

    return seeds


def _cut_foreground_bridges(foreground: np.ndarray) -> np.ndarray:
    """
    Sever thin bone bridges connecting molars to each other/the jaw, slice by
    slice, so seeded propagation can't leak through a thin neck from a molar
    into bone (or a neighboring molar).
    """
    cut = np.zeros_like(foreground)
    for i in range(foreground.shape[0]):
        sl = foreground[i]
        if not sl.any():
            continue
        cut_slice = cut_bridges_slice(sl, strategy="reconstruct")
        cut[i] = cut_slice if cut_slice is not None else sl
    return cut


def _grow_from_seeds(
    seeds: np.ndarray,
    foreground: np.ndarray,
) -> np.ndarray:
    """
    Grow each seed label into the full foreground via binary propagation.

    Each molar seed is grown independently so they compete for foreground
    voxels. Foreground voxels not reached by any seed → bone.

    Propagation is confined to a bridge-cut version of the foreground mask,
    which severs the thin bone necks that would otherwise let a molar seed
    flood into surrounding bone or a neighboring molar. Voxels trimmed away
    by the bridge cut are re-added to whichever label reaches them via
    dilation, so the final masks still cover the full original foreground.

    Returns a label array with the same labels as seeds (0 = ungrown).
    """
    n_seeds = int(seeds.max())
    grown = np.zeros_like(seeds)

    bridge_cut_foreground = _cut_foreground_bridges(foreground)

    for lbl in range(1, n_seeds + 1):
        seed_mask = seeds == lbl
        # Dilate seed slightly before propagating to bridge thin enamel gaps.
        seed_mask = dilate_mask(seed_mask, radius=2, if_2d=False)
        grown_lbl = binary_propagation(
            seed_mask,
            mask=bridge_cut_foreground,
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        # Only claim voxels not already assigned to a prior seed.
        unassigned = grown == 0
        grown[grown_lbl & unassigned] = lbl

    # Reclaim voxels removed by the bridge cut (thin necks) by growing each
    # label a couple voxels back into the full foreground, still respecting
    # earlier claims so cut bridges stay split between whichever side is closer.
    for _ in range(2):
        for lbl in range(1, n_seeds + 1):
            claimed = grown == lbl
            if not claimed.any():
                continue
            grown_back = dilate_mask(claimed, radius=1, if_2d=False) & foreground
            unassigned = grown == 0
            grown[grown_back & unassigned] = lbl

    return grown


def segment_molar_bone(
    preprocessed_volume: np.ndarray,
    max_molars: int = 4,
    min_enamel_seed_size: int = _MIN_ENAMEL_SEED_SIZE,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment molars and bone using enamel-anchored region growing.

    After incisor removal, enamel (the brightest tissue in microCT) is only
    present in molars. We locate enamel clusters as seeds for each molar, then
    grow each seed into the full foreground. Foreground not reached by any seed
    is classified as bone.

    This is robust to disconnected molar components because each tooth is
    anchored independently by its own enamel signal rather than relying on
    connected-component selection.

    Falls back to the erosion-based max-intensity method if no enamel seeds
    are found (e.g. very low-resolution scans where enamel is indistinct).

    Parameters:
        preprocessed_volume: Intensity volume with incisor removed.
        max_molars: Upper bound on the number of molar seeds to find.
        min_enamel_seed_size: Minimum voxels for an enamel cluster to be used
            as a seed. Filters out noise speckles.
        debug: If True, open a napari viewer showing the enamel seeds
            relative to the foreground mask before growing.

    Returns:
        (bone_mask, molar_mask) — boolean masks.
    """
    foreground = preprocessed_volume > 0

    if not foreground.any():
        empty = np.zeros_like(foreground)
        return empty, empty

    # --- Primary path: enamel-seeded region growing -----------------------
    seeds = _enamel_seeds(
        preprocessed_volume, foreground,
        min_enamel_seed_size, max_molars,
        debug=debug,
    )

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            additional_volumes={
                "Foreground": (foreground, "gray"),
                "Enamel Seeds": (seeds, "red") if seeds is not None else (np.zeros_like(foreground), "red"),
            },
            title="[debug] Enamel Seeds vs Foreground Mask",
        )

    if seeds is not None:
        logger.info("Segmenting molars via enamel-anchored region growing")
        grown = _grow_from_seeds(seeds, foreground)

        molar_mask = grown > 0
        bone_mask = foreground & ~molar_mask

        logger.info(
            "Enamel-seeded segmentation: molar voxels=%d, bone voxels=%d",
            int(molar_mask.sum()), int(bone_mask.sum()),
        )
        return bone_mask, molar_mask

    # --- Fallback: erosion-based connected-component selection ------------
    logger.warning(
        "Enamel seeding failed — falling back to erosion-based component selection"
    )
    return _segment_molar_bone_by_erosion(preprocessed_volume, foreground)


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