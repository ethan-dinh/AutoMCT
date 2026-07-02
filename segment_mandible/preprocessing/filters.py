"""
Preprocessing utilities: normalization, Gaussian filtering, intensity clipping,
bridge cutting, and bone intensity estimation.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from skimage.morphology import (
    disk,
    erosion as bin_erode2d,
    reconstruction as recon2d,
    remove_small_objects,
    skeletonize,
)
from skimage.segmentation import clear_border
from tqdm import tqdm

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


def non_local_means_filter(
    volume: np.ndarray,
    *,
    h: Optional[float] = None,
    patch_size: int = 5,
    patch_distance: int = 6,
    fast_mode: bool = True,
    per_slice: bool = True,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """
    Apply non-local means denoising to a 3D volume.

    Reduces noise while preserving edges and fine structural details — important
    for accurate thresholding and segmentation of microCT data.

    Parameters:
        h: Filter strength (cut-off distance for patch-weight decay).
           Estimated from the local noise level per slice when None.
        patch_size: Side length (in pixels) of the square patches compared.
        patch_distance: Maximum search radius (in pixels) for similar patches.
        fast_mode: Use the approximated fast NLM variant (slightly lower
                   quality but orders of magnitude faster).
        per_slice: When True (default), run 2D NLM on each axial slice
                   independently.  This is far faster and uses far less memory
                   than full 3D NLM, which is impractical for large volumes.
        n_jobs: Number of parallel threads for per-slice mode.
                None (default) uses all available CPUs; 1 disables parallelism.
                skimage's NLM is Cython-backed and releases the GIL, so threads
                provide true parallelism without pickling overhead.
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma

    volume = volume.astype(np.float32)
    workers = n_jobs if n_jobs is not None else os.cpu_count()
    logger.info(
        "Applying non-local means filter (per_slice=%s, patch_size=%d, "
        "patch_distance=%d, fast_mode=%s, n_jobs=%d)",
        per_slice, patch_size, patch_distance, fast_mode, workers,
    )

    if per_slice:
        result = np.empty_like(volume)

        def _denoise_slice(i: int) -> tuple[int, np.ndarray]:
            sl = volume[i]
            h_use = h if h is not None else float(0.8 * estimate_sigma(sl))
            return i, denoise_nl_means(
                sl,
                h=h_use,
                patch_size=patch_size,
                patch_distance=patch_distance,
                fast_mode=fast_mode,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_denoise_slice, i): i for i in range(volume.shape[0])}
            with tqdm(total=volume.shape[0], desc="Applying non-local means filter") as pbar:
                for future in as_completed(futures):
                    i, denoised = future.result()
                    result[i] = denoised
                    pbar.update(1)

        return result

    h_use = h if h is not None else float(0.8 * estimate_sigma(volume))  # type: ignore
    return denoise_nl_means(
        volume,
        h=h_use,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=fast_mode,
    ).astype(np.float32)


def tv_denoise_volume(
    volume: np.ndarray,
    *,
    weight: float = 0.075,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """
    Apply Total Variation (TV) denoising to a 3D volume, slice by slice.

    TV denoising enforces piecewise-constant regions, which sharpens the
    boundary between bone and background and produces cleaner histogram peaks
    for thresholding. It is best applied after NLM denoising, which removes
    stochastic noise first.

    Parameters:
        weight: Denoising strength. Higher values produce smoother, more
                cartoon-like output; lower values stay closer to the input.
                Typical range for normalized microCT data: 0.05 – 0.3.
        n_jobs: Number of parallel threads. None uses all available CPUs.
    """
    from skimage.restoration import denoise_tv_chambolle

    volume = volume.astype(np.float32)
    workers = n_jobs if n_jobs is not None else os.cpu_count()
    logger.info(
        "Applying TV denoising (weight=%.3f, n_jobs=%d)", weight, workers,
    )

    result = np.empty_like(volume)

    def _denoise_slice(i: int) -> tuple[int, np.ndarray]:
        return i, denoise_tv_chambolle(volume[i], weight=weight).astype(np.float32)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_denoise_slice, i): i for i in range(volume.shape[0])}
        with tqdm(total=volume.shape[0], desc="Applying TV denoising") as pbar:
            for future in as_completed(futures):
                i, denoised = future.result()
                result[i] = denoised
                pbar.update(1)

    return result


# ------------------------------------------------------------------
# CLAHE contrast enhancement
# ------------------------------------------------------------------

def clahe_volume(
    volume: np.ndarray,
    *,
    clip_limit: float = 0.01,
    nbins: int = 256,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) slice-by-slice.

    Enhances local contrast without amplifying noise globally. Useful before
    incisor segmentation to sharpen the intensity difference between the dense
    incisor tip and surrounding bone — giving Otsu a cleaner bimodal histogram.

    The input is expected to be float32 in [0, 1]. Output is float32 in [0, 1].

    Parameters:
        clip_limit: Contrast limiting threshold (fraction of the histogram peak).
                    Lower values = less contrast boost. Typical range: 0.005–0.03.
        nbins: Number of histogram bins used internally by CLAHE.
        n_jobs: Parallel threads. None uses all available CPUs.
    """
    from skimage.exposure import equalize_adapthist

    volume = volume.astype(np.float32)
    workers = n_jobs if n_jobs is not None else os.cpu_count()
    logger.info(
        "Applying CLAHE (clip_limit=%.4f, nbins=%d, n_jobs=%d)",
        clip_limit, nbins, workers,
    )

    result = np.empty_like(volume)

    def _clahe_slice(i: int) -> tuple[int, np.ndarray]:
        sl = volume[i]
        enhanced = equalize_adapthist(sl, nbins=nbins, clip_limit=clip_limit)
        return i, enhanced.astype(np.float32)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_clahe_slice, i): i for i in range(volume.shape[0])}
        with tqdm(total=volume.shape[0], desc="Applying CLAHE") as pbar:
            for future in as_completed(futures):
                i, enhanced = future.result()
                result[i] = enhanced
                pbar.update(1)

    return result


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
def _find_histogram_peaks(curve: np.ndarray, prom_frac: float) -> tuple[np.ndarray, dict]:
    prom = max(float(curve.max()) * prom_frac, 1e-8)
    return find_peaks(curve, prominence=prom)


def _threshold_knee(
    bin_centres: np.ndarray,
    smoothed: np.ndarray,
    counts: np.ndarray,
    peak_a: int,
    peaks_all: np.ndarray,
    fallback: float,
    fallback_fraction: float,
    debug: bool,
) -> float:
    """
    Return a normalized threshold in [0, 1] at the knee/elbow on the
    right descending side of peak_a.

    This uses the maximum perpendicular distance from the descending
    smooth curve to the chord connecting the peak to the tail.

    Why this instead of curvature?
    - Curvature can get pulled into the flat tail or noisy regions.
    - For histogram-like decay, the "bottom of the hill" is better
      approximated by the elbow / max distance to chord.

    Returns:
        threshold in [0, 1]
    """
    n = len(smoothed)
    if n < 4:
        logger.warning(
            "Knee threshold: insufficient points - falling back to %.4f (%.0f%% of ROI max)",
            fallback,
            fallback_fraction * 100,
        )
        if debug:
            _plot_valley_histogram(
                bin_centres,
                counts,
                smoothed,
                peaks_all,
                valley_idx=None,
                threshold=fallback,
                peak_a=peak_a,
                title="Knee Threshold (fallback: insufficient points)",
            )
        return float(np.clip(fallback, 0.0, 1.0))

    # Work in a normalized coordinate system on [0, 1]
    x = np.linspace(0.0, 1.0, n)
    y = np.asarray(smoothed, dtype=float)

    peak_a = int(np.clip(peak_a, 0, n - 1))
    x_r = x[peak_a:]
    y_r = y[peak_a:]

    logger.info(
        "Knee threshold: finding elbow on right slope of peak_a @ normalized x=%.4f",
        float(x[peak_a]),
    )

    threshold: float | None = None

    if len(y_r) >= 4:
        # Chord from start of descending branch to the tail
        x0, y0 = float(x_r[0]), float(y_r[0])
        x1, y1 = float(x_r[-1]), float(y_r[-1])

        dx = x1 - x0
        dy = y1 - y0
        denom = float(np.hypot(dx, dy))

        if denom > 0.0:
            # Perpendicular distance from each point to the chord
            dist = np.abs(dy * (x_r - x0) - dx * (y_r - y0)) / denom

            # Prefer the descending part of the curve to avoid odd bumps
            slope = np.gradient(y_r, x_r)
            valid = np.isfinite(dist) & np.isfinite(slope) & (slope <= 0)

            if np.any(valid):
                masked_dist = np.where(valid, dist, -np.inf)
                knee_local = int(np.argmax(masked_dist))
                threshold = float(x_r[knee_local])

                logger.info(
                    "Knee threshold: max distance at normalized x=%.4f "
                    "(dist=%.4g, y=%.4g, local index %d from peak_a)",
                    threshold,
                    float(dist[knee_local]),
                    float(y_r[knee_local]),
                    knee_local,
                )
            else:
                logger.warning(
                    "Knee threshold: no valid descending-slope points found; using fallback"
                )
        else:
            logger.warning(
                "Knee threshold: chord length is zero; using fallback"
            )

    if threshold is None or not np.isfinite(threshold):
        logger.warning(
            "Knee threshold: falling back to %.4f (%.0f%% of ROI max)",
            fallback,
            fallback_fraction * 100,
        )
        threshold = float(np.clip(fallback, 0.0, 1.0))

    threshold = float(np.clip(threshold, 0.0, 1.0))

    if debug:
        _plot_valley_histogram(
            bin_centres,
            counts,
            smoothed,
            peaks_all,
            valley_idx=None,
            threshold=threshold,
            peak_a=peak_a,
            title="Knee Threshold",
        )

    return threshold

def _threshold_valley(
    bin_centres: np.ndarray,
    smoothed: np.ndarray,
    counts: np.ndarray,
    peak_a: int,
    peaks_all: np.ndarray,
    fallback: float,
    fallback_fraction: float,
    debug: bool,
    second_peak_min_sep: float,
    second_peak_prom_frac: float,
    bias: float,
) -> float:
    """
    Return the intensity at the valley between ``peak_a`` and the next
    prominent peak to its right, optionally shifted by ``bias``.
    """
    x_a = float(bin_centres[peak_a])
    right_mask = bin_centres >= (x_a + second_peak_min_sep)

    if not np.any(right_mask):
        logger.warning(
            "Valley threshold: no right-side search region - falling back to %.4f",
            fallback,
        )
        if debug:
            _plot_valley_histogram(
                bin_centres, counts, smoothed, peaks_all,
                valley_idx=None, threshold=fallback,
                peak_a=peak_a, title="Valley Threshold (fallback: no right-side region)",
            )
        return fallback

    peaks_right, props_right = _find_histogram_peaks(
        smoothed[right_mask], prom_frac=second_peak_prom_frac
    )

    if len(peaks_right) == 0:
        logger.warning(
            "Valley threshold: no secondary peak found - falling back to %.4f (%.0f%% of ROI max)",
            fallback, fallback_fraction * 100,
        )
        if debug:
            _plot_valley_histogram(
                bin_centres, counts, smoothed, peaks_all,
                valley_idx=None, threshold=fallback,
                peak_a=peak_a, title="Valley Threshold (fallback: no secondary peak)",
            )
        return fallback

    second_local_idx = int(peaks_right[np.argmax(props_right["prominences"])])
    peak_b = int(np.flatnonzero(right_mask)[second_local_idx])

    search_left = min(peak_a, peak_b)
    search_right = max(peak_a, peak_b)

    if search_right <= search_left + 1:
        logger.warning(
            "Valley threshold: degenerate peak spacing - falling back to %.4f",
            fallback,
        )
        if debug:
            _plot_valley_histogram(
                bin_centres, counts, smoothed, peaks_all,
                valley_idx=None, threshold=fallback,
                peak_a=peak_a, peak_b=peak_b,
                title="Valley Threshold (fallback: degenerate spacing)",
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

    if threshold <= 0.0 or not np.isfinite(threshold):
        logger.warning(
            "Valley threshold invalid (%.4f) - falling back to %.4f",
            threshold, fallback,
        )
        return fallback

    if debug:
        _plot_valley_histogram(
            bin_centres, counts, smoothed,
            np.array([peak_a, peak_b], dtype=int),
            valley_idx=valley_idx, threshold=threshold,
            peak_a=peak_a, peak_b=peak_b, title="Valley Threshold",
        )
    return threshold


def find_threshold(
    volume: np.ndarray,
    middle_fraction: float = 0.25,
    nbins: int = 128,
    smooth_sigma: float = 2.25,
    fallback_fraction: float = 0.38,
    debug: bool = False,
    second_peak_min_sep: float = 0.08,
    second_peak_prom_frac: float = 0.01,
    bias: float = 0.4,
    strategy: Literal["valley", "knee"] = "valley",
) -> float:
    """
    Find a threshold separating background from bone.

    Extracts a central ROI from the volume, builds a smoothed histogram,
    resolves the dominant peak (with retries if needed), then delegates to
    ``_threshold_knee`` or ``_threshold_valley`` depending on ``strategy``.

    Parameters
    ----------
    second_peak_min_sep:
        Minimum intensity separation (in histogram x-units) before searching
        for the second peak.  Valley strategy only.
    second_peak_prom_frac:
        Relative prominence threshold for the secondary-peak search, as a
        fraction of the global smoothed-histogram maximum.  Valley strategy only.
    bias:
        Fractional shift of the threshold toward the bone peak.
        0 = valley minimum; 1 = bone peak; negative = toward background.
        Valley strategy only.
    strategy:
        ``"valley"`` places the threshold at the valley between the two
        dominant histogram peaks.
        ``"knee"`` uses the point of maximum curvature on the right
        descending slope of the dominant peak.
    """
    depth = volume.shape[0]
    half = max(1, int(depth * middle_fraction / 2))
    mid = depth // 2
    start = max(0, mid - half)
    end = min(depth, mid + half)
    roi = volume[start:end]

    logger.info(
        "find_threshold ROI: slices %d-%d (%d slices, %.0f%% of volume)",
        start, end, roi.shape[0], middle_fraction * 100,
    )

    voxels = roi[roi > 0]
    if voxels.size == 0:
        logger.warning("find_threshold: no nonzero voxels in ROI - returning 0")
        return 0.0

    counts, edges = np.histogram(voxels, bins=nbins)
    bin_centres = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma)
    fallback = float(voxels.max() * fallback_fraction)

    # --- Peak finding with zero-peak retry loop ---
    peaks_all, props_all = _find_histogram_peaks(smoothed, prom_frac=0.05)

    if len(peaks_all) == 0:
        logger.warning(
            "find_threshold: no peaks on first pass"
        )
        for _attempt in range(1, 21):
            _pct = _attempt * 5  # 5%, 10%, ..., 100% of original distribution
            _low = float(np.percentile(voxels, _pct))
            _active_voxels = voxels[voxels > _low]
            if _active_voxels.size == 0:
                logger.warning(
                    "retry %d/20: no voxels remain - stopping early",
                    _attempt,
                )
                break
            _c, _e = np.histogram(_active_voxels, bins=nbins)
            _bc = (_e[:-1] + _e[1:]) / 2.0
            _sm = gaussian_filter1d(_c.astype(np.float64), sigma=smooth_sigma)
            peaks_all, props_all = _find_histogram_peaks(_sm, prom_frac=0.05)
            if len(peaks_all) > 0:
                logger.info(
                    "retry %d/20: found %d peak(s) "
                    "(intensity floor raised to %.4f, %d%% percentile of original)",
                    _attempt, len(peaks_all), _low, _pct,
                )
                bin_centres, smoothed, counts = _bc, _sm, _c
                break
            logger.warning(
                "retry %d/20: still no peaks "
                "(floor raised to %.4f, %d%% percentile of original)",
                _attempt, _low, _pct,
            )

    if len(peaks_all) == 0:
        logger.warning(
            "find_threshold: no peaks after retries - "
            "falling back to %.4f (%.0f%% of ROI max)",
            fallback, fallback_fraction * 100,
        )
        if debug:
            _plot_valley_histogram(
                bin_centres, counts, smoothed, np.array([], dtype=int),
                valley_idx=None, threshold=fallback,
                title="find_threshold (fallback: no peaks after retries)",
            )
        return fallback

    # --- >2-peak reduction loop ---
    if len(peaks_all) > 2:
        logger.warning(
            "find_threshold: %d peaks found - stepping through 5%% percentile "
            "intervals to reduce to two (up to 20 attempts)",
            len(peaks_all),
        )
        for _attempt in range(1, 21):
            _pct = _attempt * 5  # 5%, 10%, ..., 100% of original distribution
            _low = float(np.percentile(voxels, _pct))
            _active_voxels = voxels[voxels > _low]
            if _active_voxels.size == 0:
                logger.warning(
                    "find_threshold peak-reduction retry %d/20: no voxels remain",
                    _attempt,
                )
                break
            _c, _e = np.histogram(_active_voxels, bins=nbins)
            _bc = (_e[:-1] + _e[1:]) / 2.0
            _sm = gaussian_filter1d(_c.astype(np.float64), sigma=smooth_sigma)
            peaks_all, props_all = _find_histogram_peaks(_sm, prom_frac=0.05)
            logger.info(
                "find_threshold peak-reduction retry %d/20: %d peak(s) remain "
                "(intensity floor raised to %.4f, %d%% percentile of original)",
                _attempt, len(peaks_all), _low, _pct,
            )
            if len(peaks_all) <= 2:
                bin_centres, smoothed, counts = _bc, _sm, _c
                break

        if len(peaks_all) > 2:
            logger.warning(
                "find_threshold: still %d peaks after 20 attempts - "
                "keeping the two most prominent",
                len(peaks_all),
            )
            top2 = np.argsort(props_all["prominences"])[-2:]
            peaks_all = peaks_all[np.sort(top2)]
            props_all = {k: v[np.sort(top2)] for k, v in props_all.items()}

    peak_a = int(peaks_all[np.argmax(props_all["prominences"])])

    if strategy == "knee":
        return _threshold_knee(
            bin_centres, smoothed, counts,
            peak_a, peaks_all, fallback, fallback_fraction, debug,
        )
    return _threshold_valley(
        bin_centres, smoothed, counts,
        peak_a, peaks_all, fallback, fallback_fraction, debug,
        second_peak_min_sep, second_peak_prom_frac, bias,
    )


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
