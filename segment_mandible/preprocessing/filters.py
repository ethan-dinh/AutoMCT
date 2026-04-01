"""
Preprocessing utilities: normalization, Gaussian filtering, intensity clipping,
bridge cutting, and bone intensity estimation.
"""

import logging
from typing import Literal, Optional

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from skimage.morphology import (
    disk,
    reconstruction as recon2d,
    remove_small_objects,
    skeletonize,
)
from skimage.segmentation import clear_border
from skimage.morphology import erosion as bin_erode2d

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

def normalize_volume(volume: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize volume values.

    Parameters:
        method: 'minmax' (default) or 'zscore'.
    """
    if method == "minmax":
        vmin, vmax = volume.min(), volume.max()
        return (volume - vmin) / (vmax - vmin)
    elif method == "zscore":
        mean, std = volume.mean(), volume.std()
        return (volume - mean) / std
    raise ValueError(f"Unknown normalization method: {method}")


def gaussian_filter_volume(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply a Gaussian filter to a 3D volume."""
    logger.info("Applying Gaussian filter with sigma = %f", sigma)
    return gaussian_filter(volume, sigma=sigma)


# ------------------------------------------------------------------
# Intensity clipping / rescaling
# ------------------------------------------------------------------

def rescale_to_unit(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linear rescale to [0, 1] with clipping. Safe for NaNs."""
    if np.isclose(vmax, vmin):
        return np.zeros_like(img, dtype=np.float32)
    scaled = (img - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------
# Bridge cutting
# ------------------------------------------------------------------

def _suggest_params_from_slice(
    sl: np.ndarray, thin_percentile: float = 20.0
) -> tuple[int, float]:
    """
    Estimate r_xy (for reconstruct) and t_xy (for EDT) from a 2D binary slice.
    """
    sl = sl.astype(bool)
    if not sl.any():
        return 1, 1.0
    skel = skeletonize(sl)
    edt = distance_transform_edt(sl)

    if edt is None:
        return 1, 1.0

    vals = edt[skel] if skel.any() else edt[sl]
    thin_half = np.percentile(vals, thin_percentile) if vals.size else 1.0
    t_xy = float(max(1.0, thin_half))
    r_xy = int(max(1, np.floor(t_xy / 1.5)))
    return r_xy, t_xy


def cut_bridges_slice(
    slice_mask: np.ndarray,
    *,
    strategy: Literal["reconstruct", "edt"] = "reconstruct",
    r_xy: Optional[int] = None,
    t_xy: Optional[float] = None,
    min_component_pixels: int = 0,
    clear_touching_border: bool = False,
    auto_param_percentile: float = 20.0,
) -> np.ndarray | None:
    """
    Cut thin bridges in a single 2D binary mask.

    Parameters:
        slice_mask: 2D bool/0-1 array; foreground is True or 1.
        strategy: 'reconstruct' (erode + reconstruct) or 'edt' (EDT threshold).
        r_xy: Disk radius for 'reconstruct'. Auto-estimated if None.
        t_xy: EDT threshold for 'edt'. Auto-estimated if None.
        min_component_pixels: Remove components smaller than this.
        clear_touching_border: Drop components touching the image border.
        auto_param_percentile: Percentile used for auto parameter estimation.

    Returns:
        2D boolean mask with bridges severed.
    """
    sl = slice_mask.astype(bool)
    if not sl.any():
        return sl

    r_est, t_est = _suggest_params_from_slice(sl, thin_percentile=auto_param_percentile)

    if strategy == "reconstruct":
        r_use = r_est if r_xy is None else r_xy
        se = disk(r_use)
        eroded = bin_erode2d(sl, footprint=se).astype(np.uint8)
        out = recon2d(seed=eroded, mask=sl.astype(np.uint8), method="dilation") > 0

    elif strategy == "edt":
        t_use = t_est if t_xy is None else t_xy
        edt = distance_transform_edt(sl)
        pruned = sl & (edt >= t_use) # type: ignore
        out = binary_fill_holes(pruned)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if clear_touching_border:
        out = clear_border(out)

    if min_component_pixels and min_component_pixels > 0:
        out = remove_small_objects(out, min_size=min_component_pixels)

    return out


# ------------------------------------------------------------------
# Valley threshold (background / bone separation)
# ------------------------------------------------------------------
def find_threshold(
    volume: np.ndarray,
    middle_fraction: float = 0.20,
    nbins: int = 128,
    smooth_sigma: float = 5.0,
    fallback_fraction: float = 0.45,
    debug: bool = False,
    second_peak_min_sep: float = 0.08,
    second_peak_prom_frac: float = 0.01,
    bias: float = -0.45, # Shift threshold toward background (negative) or bone (positive). Default -0.3 is a moderate shift toward background to help preserve faint bone voxels.
    strategy: Literal["valley", "knee"] = "valley",
    conservative: bool = False,
) -> float:
    """
    Find a threshold separating background from bone.

    Peak finding is done on the smoothed histogram only.
    The raw histogram is used only for debug plotting.

    Parameters
    ----------
    second_peak_min_sep:
        Minimum intensity separation, in histogram x-units, to begin searching
        for the second peak after the first one.
    second_peak_prom_frac:
        Relative prominence threshold for the second peak search, expressed as
        a fraction of the global smoothed histogram maximum.
    bias:
        Fractional shift of the threshold toward the larger (bone) peak.
        0.0 (default) places the threshold at the valley minimum. 1.0 moves
        it all the way to the bone peak. Negative values shift toward the
        background peak. Must be in [-1, 1].
    strategy:
        Overall thresholding strategy.
        ``"valley"`` (default) finds two histogram peaks and places the
        threshold at the valley between them.
        ``"knee"`` finds the dominant (low-intensity) peak and places the
        threshold where its right descending slope flattens out, without
        searching for a secondary peak.
    conservative:
        Only used when ``strategy="knee"``. If ``True``, selects the
        ``argmax`` of grad2 in the post-crossing region (earlier, more
        conservative threshold). If ``False`` (default), selects the
        ``argmin`` (further right, more permissive threshold).
    """
    depth = volume.shape[0]
    half = max(1, int(depth * middle_fraction / 2))
    mid = depth // 2
    start = max(0, mid - half)
    end = min(depth, mid + half)
    roi = volume[start:end]

    logger.info(
        "Valley threshold ROI: slices %d-%d (%d slices, %.0f%% of volume)",
        start, end, roi.shape[0], middle_fraction * 100,
    )

    voxels = roi[roi > 0]
    if voxels.size == 0:
        logger.warning("Valley threshold: no nonzero voxels in ROI - returning 0")
        return 0.0

    counts, edges = np.histogram(voxels, bins=nbins)
    bin_centres = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma)

    def _plot(
        title: str,
        threshold: float,
        peak_a: int | None = None,
        peak_b: int | None = None,
        valley_idx: int | None = None,
        peaks_arr: np.ndarray | None = None,
        curve: np.ndarray | None = None,
    ) -> None:
        if debug:
            _plot_valley_histogram(
                bin_centres,
                counts,
                curve if curve is not None else smoothed,
                peaks_arr if peaks_arr is not None else np.array([], dtype=int),
                valley_idx=valley_idx,
                threshold=threshold,
                peak_a=peak_a,
                peak_b=peak_b,
                title=title,
            )

    def _find_peaks(curve: np.ndarray, prom_frac: float) -> tuple[np.ndarray, dict]:
        prom = max(float(curve.max()) * prom_frac, 1e-8)
        return find_peaks(curve, prominence=prom)

    fallback = float(voxels.max() * fallback_fraction)

    # 1) Find the dominant low-intensity peak.
    peaks_all, props_all = _find_peaks(smoothed, prom_frac=0.05)

    if len(peaks_all) == 0:
        logger.warning(
            "Valley threshold: no peaks found on first pass - entering retry loop "
            "(up to 5 attempts, trimming bottom 2%% of intensity range each time)"
        )
        _active_voxels = voxels
        _active_counts, _active_edges = counts, edges
        _active_centres, _active_smoothed = bin_centres, smoothed

        for _attempt in range(1, 6):
            _clip_pct = _attempt * 2
            _low = float(np.percentile(_active_voxels, 2))
            _active_voxels = _active_voxels[_active_voxels > _low]

            if _active_voxels.size == 0:
                logger.warning(
                    "Valley threshold retry %d/%d: no voxels remain after trimming "
                    "bottom %d%% - stopping early",
                    _attempt, 5, _clip_pct,
                )
                break

            _active_counts, _active_edges = np.histogram(_active_voxels, bins=nbins)
            _active_centres = (_active_edges[:-1] + _active_edges[1:]) / 2.0
            _active_smoothed = gaussian_filter1d(
                _active_counts.astype(np.float64), sigma=smooth_sigma
            )
            peaks_all, props_all = _find_peaks(_active_smoothed, prom_frac=0.05)

            if len(peaks_all) > 0:
                logger.info(
                    "Valley threshold retry %d/%d: found %d peak(s) after trimming "
                    "bottom %d%% of intensity range (intensity floor raised to %.4f)",
                    _attempt, 5, len(peaks_all), _clip_pct, _low,
                )
                bin_centres = _active_centres
                smoothed = _active_smoothed
                counts = _active_counts
                break

            logger.warning(
                "Valley threshold retry %d/%d: still no peaks after trimming "
                "bottom %d%% of intensity range (intensity floor raised to %.4f)",
                _attempt, 5, _clip_pct, _low,
            )

        if len(peaks_all) == 0:
            logger.warning(
                "Valley threshold: no peaks found after 5 retries - "
                "falling back to %.4f (%.0f%% of ROI max)",
                fallback, fallback_fraction * 100,
            )
            _plot(
                title="Valley Threshold (fallback: no peaks after retries)",
                threshold=fallback,
                peaks_arr=np.array([], dtype=int),
                curve=smoothed,
            )
            return fallback

    # 1b) If more than two peaks remain, iteratively trim the bottom 2% of the
    #     intensity range until only two peaks are left (up to 5 attempts).
    if len(peaks_all) > 2:
        logger.warning(
            "Valley threshold: %d peaks found - trimming bottom 2%% iteratively "
            "to reduce to two peaks (up to 5 attempts)",
            len(peaks_all),
        )
        _active_voxels = voxels
        for _attempt in range(1, 6):
            _clip_pct = _attempt * 2
            _low = float(np.percentile(_active_voxels, 2))
            _active_voxels = _active_voxels[_active_voxels > _low]

            if _active_voxels.size == 0:
                logger.warning(
                    "Valley threshold peak-reduction retry %d/%d: no voxels remain "
                    "after trimming bottom %d%% - stopping early",
                    _attempt, 5, _clip_pct,
                )
                break

            _active_counts, _active_edges = np.histogram(_active_voxels, bins=nbins)
            _active_centres = (_active_edges[:-1] + _active_edges[1:]) / 2.0
            _active_smoothed = gaussian_filter1d(
                _active_counts.astype(np.float64), sigma=smooth_sigma
            )
            peaks_all, props_all = _find_peaks(_active_smoothed, prom_frac=0.05)

            logger.info(
                "Valley threshold peak-reduction retry %d/%d: %d peak(s) remain "
                "after trimming bottom %d%% (intensity floor raised to %.4f)",
                _attempt, 5, len(peaks_all), _clip_pct, _low,
            )

            if len(peaks_all) <= 2:
                bin_centres = _active_centres
                smoothed = _active_smoothed
                counts = _active_counts
                break

        if len(peaks_all) > 2:
            logger.warning(
                "Valley threshold: still %d peaks after 5 reduction attempts - "
                "proceeding with the two most prominent",
                len(peaks_all),
            )
            top2 = np.argsort(props_all["prominences"])[-2:]
            peaks_all = peaks_all[np.sort(top2)]
            props_all = {k: v[np.sort(top2)] for k, v in props_all.items()}

    main_idx = int(np.argmax(props_all["prominences"]))
    peak_a = int(peaks_all[main_idx])

    # 2a) Knee strategy: find where the right slope of peak_a flattens and return.
    if strategy == "knee":
        logger.info(
            "Valley threshold: strategy='knee' - finding right-slope knee of peak_a @ %.4f",
            float(bin_centres[peak_a]),
        )
        _right_curve = smoothed[peak_a:]
        _right_centres = bin_centres[peak_a:]
        _threshold_from_slope: float | None = None

        if len(_right_curve) >= 4:
            _grad1 = np.gradient(_right_curve)
            _grad2 = np.gradient(_grad1)
            _sign_diff = np.diff(np.sign(_grad2[1:]))
            _crossings = np.where(_sign_diff > 0)[0] + 2
            if len(_crossings) > 0:
                _search_start = int(_crossings[0])
                _g1_tail = _grad1[_search_start:]
                _g2_tail = _grad2[_search_start:]
                _flat_local = _search_start + int(
                    np.argmax(_g2_tail) if conservative else np.argmin(_g2_tail)
                )
                _flat_local = min(_flat_local, len(_right_centres) - 1)
                _threshold_from_slope = float(_right_centres[_flat_local])
                logger.info(
                    "Valley threshold (knee): right-slope knee at intensity %.4f "
                    "(grad1=%.4g, grad2=%.4g, bin index %d from peak_a)",
                    _threshold_from_slope,
                    float(_g1_tail[_flat_local - _search_start]),
                    float(_g2_tail[_flat_local - _search_start]),
                    _flat_local,
                )
            else:
                logger.warning("Valley threshold (knee): no inflection found on right slope")

        if _threshold_from_slope is None or _threshold_from_slope <= 0.0:
            logger.warning(
                "Valley threshold (knee): slope-flattening failed - "
                "falling back to %.4f (%.0f%% of ROI max)",
                fallback, fallback_fraction * 100,
            )
            _plot(
                title="Valley Threshold knee (fallback: slope failed)",
                threshold=fallback,
                peak_a=peak_a,
                peaks_arr=peaks_all,
                curve=smoothed,
            )
            return fallback

        _plot(
            title="Valley Threshold (knee)",
            threshold=_threshold_from_slope,
            peak_a=peak_a,
            peaks_arr=peaks_all,
            curve=smoothed,
        )
        return _threshold_from_slope

    # 2b) Valley strategy: search for the second peak only on the right side.
    x_a = float(bin_centres[peak_a])
    right_mask = bin_centres >= (x_a + second_peak_min_sep)

    if not np.any(right_mask):
        logger.warning(
            "Valley threshold: no right-side search region - falling back to %.4f",
            fallback,
        )
        _plot(
            title="Valley Threshold (fallback: no right-side region)",
            threshold=fallback,
            peak_a=peak_a,
            peaks_arr=peaks_all,
            curve=smoothed,
        )
        return fallback

    right_curve = smoothed[right_mask]
    right_x = bin_centres[right_mask]

    peaks_right, props_right = _find_peaks(right_curve, prom_frac=second_peak_prom_frac)

    if len(peaks_right) == 0:
        logger.warning(
            "Valley threshold (valley): no secondary peak found - "
            "falling back to %.4f (%.0f%% of ROI max)",
            fallback, fallback_fraction * 100,
        )
        _plot(
            title="Valley Threshold (fallback: no secondary peak)",
            threshold=fallback,
            peak_a=peak_a,
            peaks_arr=peaks_all,
            curve=smoothed,
        )
        return fallback

    # Pick the tallest peak in the right-side region.
    second_local_idx = int(peaks_right[np.argmax(props_right["prominences"])])
    peak_b = int(np.flatnonzero(right_mask)[second_local_idx])

    # 3) Valley is the minimum between the two peak centers.
    search_left = min(peak_a, peak_b)
    search_right = max(peak_a, peak_b)

    if search_right <= search_left + 1:
        logger.warning(
            "Valley threshold: degenerate peak spacing - falling back to %.4f",
            fallback,
        )
        _plot(
            title="Valley Threshold (fallback: degenerate spacing)",
            threshold=fallback,
            peak_a=peak_a,
            peak_b=peak_b,
            peaks_arr=peaks_all,
            curve=smoothed,
        )
        return fallback

    valley_rel = int(np.argmin(smoothed[search_left:search_right + 1]))
    valley_idx = search_left + valley_rel
    valley_x = float(bin_centres[valley_idx])
    peak_b_x = float(bin_centres[peak_b])
    threshold = valley_x + bias * (peak_b_x - valley_x)

    logger.info(
        "Valley threshold: peak A @ %.4f, peak B @ %.4f, valley @ %.4f",
        bin_centres[peak_a], bin_centres[peak_b], threshold,
    )

    _plot(
        title="Valley Threshold",
        threshold=threshold,
        peak_a=peak_a,
        peak_b=peak_b,
        valley_idx=valley_idx,
        peaks_arr=np.array([peak_a, peak_b], dtype=int),
        curve=smoothed,
    )

    if threshold <= 0.0 or not np.isfinite(threshold):
        logger.warning(
            "Valley threshold invalid (%.4f) - falling back to %.4f",
            threshold, fallback,
        )
        return fallback

    return threshold


def _plot_valley_histogram(
    bin_centres: np.ndarray,
    counts: np.ndarray,
    smoothed: np.ndarray,
    peaks: np.ndarray,
    *,
    valley_idx: Optional[int],
    threshold: float,
    peak_a: Optional[int] = None,
    peak_b: Optional[int] = None,
    title: str = "Valley Threshold",
) -> None:
    """Render the histogram debug plot and block until the window is closed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(title)

    ax.bar(bin_centres, counts, width=(bin_centres[1] - bin_centres[0]),
           color="steelblue", alpha=0.4, label="Raw histogram")
    ax.plot(bin_centres, smoothed, color="steelblue", linewidth=1.5, label="Smoothed")

    # All detected peaks (grey)
    ax.plot(bin_centres[peaks], smoothed[peaks],
            "o", color="grey", markersize=5, label="All peaks")

    # The two chosen peaks (orange)
    if peak_a is not None and peak_b is not None:
        for p, lbl in [(peak_a, "Background peak"), (peak_b, "Bone peak")]:
            ax.plot(bin_centres[p], smoothed[p],
                    "^", color="darkorange", markersize=9, label=lbl)

    # Valley / threshold (red)
    if valley_idx is not None:
        ax.plot(bin_centres[valley_idx], smoothed[valley_idx],
                "v", color="red", markersize=9, label=f"Valley @ {threshold:.4f}")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Voxel count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Bone intensity estimation
# ------------------------------------------------------------------

def find_min_intensity_of_bone(preprocessed_volume: np.ndarray) -> float:
    """
    Estimate the minimum bone intensity from middle slices of the volume.

    Uses Otsu segmentation on Gaussian-filtered middle slices and returns
    the minimum mean intensity across the largest region in each slice.
    """
    # Lazy import to avoid the circular dependency at module load time.
    from segmentation.utils import get_largest_region, get_region_intensity, segment_volume  # noqa: PLC0415

    num_slices = preprocessed_volume.shape[0]
    mid = num_slices // 2
    middle_slices = preprocessed_volume[mid - 15 : mid + 15]

    middle_slices = gaussian_filter_volume(middle_slices, sigma=10)
    segmented_middle = segment_volume(middle_slices, method="otsu", min_area=200)

    intensities = []
    for i in range(segmented_middle.shape[0]):
        labeled_slice = segmented_middle[i]
        slice_2d = middle_slices[i]
        largest_label = get_largest_region(labeled_slice)
        intensity = get_region_intensity(slice_2d, labeled_slice, largest_label)
        intensities.append(intensity)

    return float(np.min(intensities))
