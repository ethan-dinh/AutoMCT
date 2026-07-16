import logging
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from skimage.measure import label

from segmentation.utils import (
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
    plt.close(fig)


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


def _bone_label(counts: np.ndarray) -> int:
    """
    Identify the bone component: the largest connected component by voxel
    count. Bone is the easiest region to detect because it contains the
    most voxels; molars are smaller connected components.
    """
    return int(np.argmax(counts))


def _enamel_connected_to_bone(
    enamel_mask: np.ndarray,
    labeled_foreground: np.ndarray,
    bone_label: int,
) -> bool:
    """True if any enamel voxel lies inside the bone component."""
    return bool(np.any(labeled_foreground[enamel_mask] == bone_label))


def _find_molar_separation_threshold(
    volume: np.ndarray,
    enamel_mask: np.ndarray,
    *,
    min_component_size: int,
    threshold_step: float,
    max_iters: int,
) -> tuple[np.ndarray, float, int]:
    """
    Escalate the foreground intensity threshold from 0 (the full foreground)
    until every enamel-bearing component is disconnected from the bone
    component (the largest component by voxel count).

    Dentin and bone sit at overlapping intensities, so a single global
    threshold either fuses molars to bone or cuts into real tissue. Enamel
    is strictly brighter than both, so it can always be isolated cleanly --
    this uses that asymmetry to find a working separation threshold instead
    of building the molar mask directly off enamel.

    Returns (labeled_foreground, threshold, n_iters) at the point of
    disconnection -- bone is the largest component; every other large
    component that contains enamel is a molar candidate.
    """
    threshold = 0.0

    for it in range(max_iters):
        candidate = (volume > threshold) & (volume > 0)
        labeled = label_3d_volume(candidate, connectivity=1)
        counts = np.bincount(labeled.ravel())
        counts[0] = 0

        if not counts.any():
            logger.warning("Molar separation: threshold=%.4f -> empty foreground; stopping", threshold)
            break

        bone_lbl = _bone_label(counts)
        still_connected = _enamel_connected_to_bone(enamel_mask, labeled, bone_lbl)

        logger.info(
            "Molar separation iter %3d: threshold=%.4f -> bone=%d voxels, "
            "enamel connected to bone=%s",
            it, threshold, int(counts[bone_lbl]), still_connected,
        )

        if not still_connected:
            logger.info(
                "Enamel disconnected from bone after %d iteration(s) at threshold=%.4f",
                it, threshold,
            )
            return labeled, threshold, it

        threshold += threshold_step

    logger.warning(
        "Enamel never disconnected from bone after %d iterations; using last threshold %.4f",
        max_iters, threshold,
    )
    candidate = (volume > threshold) & (volume > 0)
    labeled = label_3d_volume(candidate, connectivity=1)
    return labeled, threshold, max_iters


def _extract_molar_mask(
    labeled_foreground: np.ndarray,
    enamel_mask: np.ndarray,
    *,
    min_component_size: int,
) -> np.ndarray:
    """
    Given the disconnected labeling from _find_molar_separation_threshold,
    return a boolean mask of every component that contains enamel and isn't
    the bone component -- i.e. all molar candidates.
    """
    counts = np.bincount(labeled_foreground.ravel())
    counts[0] = 0
    if not counts.any():
        return np.zeros_like(labeled_foreground, dtype=bool)

    bone_lbl = _bone_label(counts)
    large_labels = np.flatnonzero(counts >= min_component_size)

    molar_mask = np.zeros_like(labeled_foreground, dtype=bool)
    for lbl in large_labels:
        if lbl == bone_lbl:
            continue
        component = labeled_foreground == lbl
        if np.any(enamel_mask & component):
            molar_mask |= component

    return molar_mask


def segment_molar_bone(
    preprocessed_volume: np.ndarray,
    min_enamel_seed_size: int = _MIN_ENAMEL_SEED_SIZE,
    threshold_step: float = 0.01,
    max_iters: int = 200,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment molars and bone by escalating an intensity threshold until
    enamel disconnects from bone.

    Dentin (the bulk of a molar) and bone sit at overlapping intensities, so
    a single global threshold either fuses molars to bone or cuts into real
    tissue. Enamel is strictly brighter than both, so it can always be
    isolated cleanly. We use that asymmetry to find a working separation
    threshold: starting from the full foreground, raise the threshold until
    every enamel-bearing connected component is disconnected from the bone
    component (identified as the largest component by voxel count -- bone
    dominates by volume, so this is the simplest reliable way to identify
    it). Every other large component that contains enamel is then a molar.

    This is robust to disconnected molar components because each tooth is
    identified by its own enamel signal rather than relying on a single
    global threshold to directly build the molar mask.

    Falls back to the erosion-based max-intensity method if no enamel is
    found at all (e.g. very low-resolution scans where enamel is
    indistinct).

    Parameters:
        preprocessed_volume: Intensity volume with incisor removed.
        min_enamel_seed_size: Minimum voxels for an enamel cluster to count,
            filtering out noise speckles.
        threshold_step: Amount to raise the foreground intensity threshold
            each escalation iteration.
        max_iters: Safety cap on escalation steps.
        debug: If True, open a napari viewer showing enamel vs. foreground
            before separation, then the final bone/molar split.

    Returns:
        (bone_mask, molar_mask) — boolean masks.
    """
    foreground = preprocessed_volume > 0

    if not foreground.any():
        empty = np.zeros_like(foreground)
        return empty, empty

    # --- Primary path: enamel-disconnection threshold escalation -----------
    fg_vals = preprocessed_volume[foreground]
    enamel_threshold = _find_enamel_threshold(fg_vals, debug=debug)
    enamel_mask = foreground & (preprocessed_volume >= enamel_threshold)

    labeled_enamel, _ = label(enamel_mask, connectivity=1, return_num=True)  # type: ignore
    enamel_counts = np.bincount(labeled_enamel.ravel())
    enamel_counts[0] = 0
    enamel_mask = enamel_mask & np.isin(
        labeled_enamel, np.flatnonzero(enamel_counts >= min_enamel_seed_size)
    )

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            additional_volumes={
                "Foreground": (foreground, "gray"),
                "Enamel": (enamel_mask, "yellow"),
            },
            title="[debug] Enamel vs Foreground Mask",
        )

    if enamel_mask.any():
        logger.info("Segmenting molars via enamel-disconnection threshold escalation")
        labeled_foreground, threshold, n_iters = _find_molar_separation_threshold(
            preprocessed_volume, enamel_mask,
            min_component_size=MIN_COMPONENT_SIZE,
            threshold_step=threshold_step,
            max_iters=max_iters,
        )

        molar_mask = _extract_molar_mask(
            labeled_foreground, enamel_mask, min_component_size=MIN_COMPONENT_SIZE,
        )
        bone_mask = (labeled_foreground > 0) & ~molar_mask

        logger.info(
            "Enamel-disconnection segmentation: threshold=%.4f (%d iterations), "
            "molar voxels=%d, bone voxels=%d",
            threshold, n_iters, int(molar_mask.sum()), int(bone_mask.sum()),
        )

        if debug:
            create_3d_visualization(
                preprocessed_volume,
                additional_volumes={
                    "Bone": (bone_mask, "blue"),
                    "Molar": (molar_mask, "red"),
                },
                title=f"[debug] Bone/Molar Split (threshold={threshold:.4f}, iters={n_iters})",
            )

        return bone_mask, molar_mask

    # --- Fallback: erosion-based connected-component selection ------------
    logger.warning(
        "No enamel found — falling back to erosion-based component selection"
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