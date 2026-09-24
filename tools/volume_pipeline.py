"""
Shared crop → bin → reorient → filter → write pipeline for the microCT volume
converters.

``convert_DICOM_NRRD.py`` and ``convert_ISQ_NRRD.py`` differ only in how they
get a voxel array off disk: one assembles a sorted DICOM series, the other
parses a Scanco ISQ header and reads the raw 16-bit block that follows.
Everything after that — interactive cropping, binning to a target voxel size,
long-axis reorientation, the filter chain, CLAHE, NRRD geometry, metadata — is
identical, and lives here so the two tools cannot drift apart.

Each converter supplies a ``read_source`` callback returning
``(volume, direction_matrix, origin, source_meta)`` and a ``PipelineOptions``
built from the shared CLI arguments; ``convert_volume`` does the rest. Given
the same option values, both tools therefore run the same code on the same
array and produce byte-identical outputs.

A run writes at most two volumes: ``<stem>_prefiltered.nrrd`` (with
--save-raw), holding the volume after every geometric stage — crop, binning,
reorientation — but before any filter, and ``<stem>.nrrd``, the same volume
filtered. Both are written at the target voxel size and in the same frame, so
they overlay exactly and differ only by the filter chain.

Geometry convention throughout: arrays are (Z, Y, X); ``direction_matrix``
row *i* is array axis *i*'s (L, P, S) direction vector scaled by that axis's
physical spacing in mm — exactly what NRRD's "space directions" field wants.
``origin`` is the (L, P, S) position of voxel [0, 0, 0].

Reorientation (--reorient) runs between binning and filtering: the specimen is
segmented, its minimum-volume oriented bounding box is fitted, and the volume
is rotated so array axis Z runs along the tooth's long axis (tip → base) rather
than whatever direction the scanner happened to stack slices in.

Filters applied in order:
    0. Noise-floor threshold    — flatten the air band to a constant, so no
                                   later stage spends itself on background
                                   noise (optional, --noise-floor)
    1. Gaussian pre-smoothing   — suppress isolated hot pixels before NLM
    2. Non-local means (NLM)    — edge-preserving noise reduction
    3. Median filter            — remove remaining salt-and-pepper noise
                                   (true 3-D, numba-JIT + multi-threaded)
    4. Anisotropic diffusion    — edge-preserving smoothing: flattens noise
                                   within tissues but not across boundaries
    5. Richardson-Lucy deconv   — recover detail lost to the imaging PSF
    6. TV (Chambolle) denoise   — piecewise-constant smoothing for cleaner
                                   thresholding boundaries
    7. Unsharp mask             — high-pass boost that steepens boundary ramps

Stage 0 and stages 3-7 are off by default; stages 1 and 2 are disabled by passing
``--gauss-sigma 0`` / ``--nlm-h 0``, which makes either tool a plain
read-and-write converter.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import nrrd
from rich.logging import RichHandler
from numba import njit, prange
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.ndimage import label as ndi_label
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

log = logging.getLogger(__name__)


# A reader returns (volume [Z,Y,X], direction_matrix [3,3], origin [3], meta).
SourceReader = Callable[[], "tuple[np.ndarray, np.ndarray, np.ndarray, dict]"]


@dataclass
class PipelineOptions:
    """Every knob shared by the two converters, with the shared defaults.

    These defaults *are* the parity contract: both CLIs build this object from
    the same argument definitions, so an unspecified flag means the same thing
    in either tool.
    """

    workers: int | None = None
    # Recorded in metadata only — each reader applies its own format's
    # stored-value → physical-units conversion while reading.
    apply_rescale: bool = False
    save_raw: bool = False
    crop: bool = False
    autocrop: bool = False
    autocrop_margin_mm: float = 1.0
    autocrop_sample_dim: int = 192
    # Binning / downsampling
    target_voxel_um: float | None = None
    bin_method: str = "mean"
    # Long-axis reorientation
    reorient: bool = False
    reorient_order: int = 1
    reorient_snap_deg: float = 5.0
    reorient_tight: bool = False
    reorient_margin_mm: float = 1.0
    reorient_tip_at: str = "start"
    reorient_voxel_mm: float | None = None
    reorient_sample_dim: int = 192
    reorient_mask_smooth_mm: float = 0.06
    # Output encoding
    compression_level: int = 1
    save_dtype: str = "float32"
    # Noise-floor threshold (runs before the filter chain)
    noise_floor: float | None = None
    noise_floor_method: str = "absolute"
    noise_floor_mode: str = "floor"
    noise_floor_fill: float | None = None
    # Gaussian pre-smoother
    gauss_sigma: float = 0.0
    # NLM
    nlm_patch: int = 5
    nlm_search: int = 11
    nlm_h: float = 1.0
    nlm_3d: bool = False
    # Median
    median_radius: int = 0
    # Anisotropic diffusion (Perona-Malik)
    aniso_iterations: int = 0
    aniso_kappa: float = 50.0
    aniso_gamma: float = 0.1
    # Unsharp mask
    unsharp_amount: float = 0.0
    unsharp_sigma: float = 1.0
    # Richardson-Lucy deconvolution
    rl_iterations: int = 0
    rl_psf_sigma: float = 1.0
    rl_psf_path: Path | None = None
    # TV (Chambolle) denoise
    tv_weight: float = 0.0
    # CLAHE contrast enhancement (viewing copy only)
    enhance_contrast: bool = False
    clahe_clip_limit: float = 0.01
    clahe_kernel_size: int | None = None
    # Final floor applied after the filter chain, just before the write. The
    # ISQ reader's --clamp-negative only clamps on read, and the unsharp mask
    # (and, less often, deconvolution) can push a voxel sitting at the floor
    # back below it where it borders dense tissue, so the output needs its own
    # clamp to actually contain no values under the floor.
    output_floor: float | None = None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def spacing_from_directions(direction_matrix: np.ndarray) -> np.ndarray:
    """Per-axis voxel spacing (mm) as the length of each direction-matrix row.

    Taking the row norm rather than the diagonal keeps this correct for
    oblique or gantry-tilted acquisitions, where an array axis is not aligned
    with any single L/P/S axis.
    """
    return np.linalg.norm(np.asarray(direction_matrix, dtype=float), axis=1)


def directions_from_spacing(dz: float, dy: float, dx: float) -> np.ndarray:
    """Axis-aligned "space directions" matrix for a scan with no orientation tags.

    Rows are array axes (Z, Y, X); columns are (L, P, S). The mapping matches
    what an axial DICOM series with the identity ``ImageOrientationPatient``
    ([1,0,0, 0,1,0]) produces in ``_read_direction_matrix``: columns (array X)
    run along L, rows (array Y) along P, and slices (array Z) along S.

    Getting this the same way round in both readers is what lets an ISQ scan
    and a DICOM series of the same specimen overlay in Slicer/ITK-SNAP. Mapping
    Z→L and X→S instead — the natural-looking diagonal, since the array is
    (Z, Y, X) — transposes the volume in physical space relative to every
    DICOM-derived file, even though the voxel array is identical.

    Used by the ISQ reader, whose header carries no patient orientation, and by
    the DICOM reader as its fallback when ``ImageOrientationPatient`` is
    missing, so both land on the same geometry for the same spacing.
    """
    return np.array([
        [0.0, 0.0, dz],   # array axis 0 (Z / slices)  → S
        [0.0, dy, 0.0],   # array axis 1 (Y / rows)    → P
        [dx, 0.0, 0.0],   # array axis 2 (X / columns) → L
    ], dtype=float)


# ---------------------------------------------------------------------------
# Interactive crop selection
# ---------------------------------------------------------------------------

def select_crop_bbox(volume: np.ndarray) -> tuple[slice, slice, slice]:
    """
    Open a napari viewer for the user to select a 3-D crop region.

    Draw one rectangle on a Shapes layer to set the X/Y bounding box, and use
    the slice slider (or the Shapes layer's own z-extent, if drawn across
    slices) to pick the Z range. The window must be closed to confirm the
    selection.

    Returns:
        (z_slice, y_slice, x_slice) to apply to a (Z, Y, X) volume.
    """
    import napari

    depth, height, width = volume.shape
    mid = depth // 2

    viewer = napari.Viewer(title="Select crop region — draw a rectangle, then close the window")
    viewer.add_image(volume, name="Volume", colormap="gray")
    shapes = viewer.add_shapes(
        name="Crop box",
        shape_type="rectangle",
        edge_color="red",
        face_color="transparent",
        edge_width=3,
        ndim=volume.ndim,
    )
    viewer.dims.set_current_step(0, mid)

    napari.run()

    if len(shapes.data) == 0:
        log.warning("No crop rectangle drawn — using the full volume extent.")
        return slice(0, depth), slice(0, height), slice(0, width)

    # A rectangle drawn over a 3-D image comes back as an (N, ndim) array of
    # corner coordinates in *volume axis order* — (Z, Y, X) here, because the
    # Shapes layer was created with ndim=volume.ndim above. The Z column is
    # constant (the slice it was drawn on) and is ignored: Z is picked
    # separately via the slice-range prompt below.
    #
    # Older napari versions, and a Shapes layer that ended up 2-D anyway,
    # return only the two displayed axes, so take the *last two* columns
    # rather than columns 0 and 1 — with a 3-column array the latter reads Z
    # as Y and Y as X, producing a crop far tighter than what was drawn.
    rect = np.asarray(shapes.data[0], dtype=float)
    if rect.ndim != 2 or rect.shape[1] < 2:
        log.warning("Unexpected shape data %s — using the full volume extent.", rect.shape)
        return slice(0, depth), slice(0, height), slice(0, width)

    if len(shapes.data) > 1:
        log.warning("%d shapes drawn — using the first one.", len(shapes.data))

    y0, y1 = float(np.min(rect[:, -2])), float(np.max(rect[:, -2]))
    x0, x1 = float(np.min(rect[:, -1])), float(np.max(rect[:, -1]))

    # Corners can fall outside the image (napari lets you drag past the edge)
    # and are floats; floor/ceil so the drawn region is fully contained rather
    # than rounded inward, then clamp to the volume.
    y_slice = slice(
        int(np.clip(np.floor(y0), 0, height - 1)),
        int(np.clip(np.ceil(y1), 1, height)),
    )
    x_slice = slice(
        int(np.clip(np.floor(x0), 0, width - 1)),
        int(np.clip(np.ceil(x1), 1, width)),
    )

    z_start, z_end = _prompt_z_range(depth)
    z_slice = slice(z_start, z_end)

    log.info(
        "Crop selected — Z: %d:%d, Y: %d:%d, X: %d:%d  (from %d×%d×%d)",
        z_slice.start, z_slice.stop, y_slice.start, y_slice.stop,
        x_slice.start, x_slice.stop, depth, height, width,
    )
    return z_slice, y_slice, x_slice


def autocrop_bbox(
    volume: np.ndarray,
    *,
    margin_mm: float = 1.0,
    spacing: np.ndarray | None = None,
    max_dim: int = 192,
) -> tuple[slice, slice, slice]:
    """Axis-aligned bounding box of the specimen, found without user input.

    Same segmentation as the reorientation path uses: the volume is strided
    down to at most *max_dim* voxels per axis, Otsu-thresholded to separate
    specimen from air, and the largest connected component kept so the sample
    holder, mounting putty and stray debris don't drag the box outward. The
    component's extent is then padded by *margin_mm* and mapped back to
    full-resolution indices.

    This is the non-interactive counterpart to ``select_crop_bbox``. It differs
    from ``--reorient-tight`` in *when* and *in what frame* it crops: this runs
    before binning and cuts an axis-aligned box out of the original array,
    which is what makes it a memory and speed win on a large scan;
    ``--reorient-tight`` crops in the rotated frame, after binning, and so
    tracks the specimen's own axes more tightly.

    *spacing* is per-axis voxel size in mm (Z, Y, X), used only to convert
    *margin_mm* into voxels; without it the margin is treated as voxels.

    Returns ``(z_slice, y_slice, x_slice)`` for a (Z, Y, X) volume.
    """
    shape = np.array(volume.shape)
    steps = tuple(max(1, int(np.ceil(n / max_dim))) for n in volume.shape)
    sub = np.asarray(volume[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32)

    threshold = float(threshold_otsu(sub))
    mask = sub > threshold
    if not mask.any():
        log.warning("Autocrop: Otsu selected no voxels — keeping the full volume.")
        return tuple(slice(0, int(n)) for n in shape)

    labels, n_components = ndi_label(mask)
    if n_components > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = int(counts.argmax())
        log.info(
            "Autocrop: %d connected components — keeping the largest "
            "(%d of %d masked voxels, %.1f%%).",
            n_components, int(counts[largest]), int(mask.sum()),
            100.0 * counts[largest] / float(mask.sum()),
        )
        mask = labels == largest

    idx = np.argwhere(mask)
    lo_sub, hi_sub = idx.min(axis=0), idx.max(axis=0)

    # Back to full-resolution indices. The stride means a masked sample at
    # sub-index i covers full indices [i*step, (i+1)*step), so the high edge
    # is expanded by one full step to cover the whole sampled block.
    steps_arr = np.asarray(steps)
    lo = lo_sub * steps_arr
    hi = (hi_sub + 1) * steps_arr

    if margin_mm > 0:
        if spacing is None:
            pad = np.full(3, margin_mm)
        else:
            spacing = np.asarray(spacing, dtype=float)
            pad = np.where(spacing > 0, margin_mm / np.maximum(spacing, 1e-12), 0.0)
        lo = lo - pad
        hi = hi + pad

    lo = np.clip(np.floor(lo).astype(int), 0, shape - 1)
    hi = np.clip(np.ceil(hi).astype(int), lo + 1, shape)

    kept = float(np.prod(hi - lo)) / float(np.prod(shape))
    log.info(
        "Autocrop (Otsu threshold %.4g, margin %.2f mm) — Z: %d:%d, Y: %d:%d, "
        "X: %d:%d  (from %d×%d×%d, keeping %.1f%% of voxels)",
        threshold, margin_mm, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
        *shape, 100.0 * kept,
    )
    return slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2]))


def _prompt_z_range(depth: int) -> tuple[int, int]:
    """Prompt the user for a Z (slice) start/end range on the terminal."""
    log.info("Volume has %d slices (0-%d).", depth, depth - 1)
    while True:
        raw = input(f"Enter Z range as 'start end' (blank = full range 0 {depth}): ").strip()
        if not raw:
            return 0, depth
        try:
            start_str, end_str = raw.split()
            start, end = int(start_str), int(end_str)
        except ValueError:
            print("Please enter two integers separated by a space, e.g. '20 480'.")
            continue
        start = max(0, start)
        end = min(depth, end)
        if start >= end:
            print(f"Start ({start}) must be less than end ({end}).")
            continue
        return start, end


# ---------------------------------------------------------------------------
# Binning / downsampling
# ---------------------------------------------------------------------------

def _bin_volume(
    volume: np.ndarray,
    factors: tuple[int, int, int],
    direction_matrix: np.ndarray,
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Bin *volume* by integer factors per axis, returning (volume, directions).

    Factors normally come from ``_factors_for_target_um``, which derives them
    from the requested physical voxel size.

    Averaging each (fz, fy, fx) block into one voxel cuts file size and
    downstream compute by the product of the factors (2× isotropic = 8× less
    data) and, because averaging N independent samples reduces noise by √N,
    it doubles SNR at 2× — so binning is itself a denoising step, not just a
    size reduction.

    The physical voxel size grows by the same factors, so each row of the
    "space directions" matrix is scaled to match; the volume stays
    geometrically correct in Slicer/ITK-SNAP at its new, coarser resolution.

    Trailing voxels that don't fill a complete block are trimmed (a partial
    block would otherwise be averaged over fewer samples, giving that edge
    voxel a different noise level and a half-size physical extent).

    ``method="mean"`` preserves intensities and suppresses noise, and is
    right for grayscale attenuation data. ``method="max"`` keeps peak values
    — useful when thin high-density structures (e.g. enamel) would otherwise
    be diluted by averaging with neighbouring soft tissue.

    Output is float32 regardless of input dtype: block means are not integers,
    and rounding them here rather than at write time would quantize the input
    to every later filter. Integer output on disk is a separate choice, made
    once by ``--save-dtype``.
    """
    fz, fy, fx = factors
    if fz == fy == fx == 1:
        return volume, direction_matrix

    depth, height, width = volume.shape
    tz, ty, tx = (depth // fz) * fz, (height // fy) * fy, (width // fx) * fx
    if (tz, ty, tx) != (depth, height, width):
        log.info(
            "Trimming %d×%d×%d → %d×%d×%d so dimensions divide evenly by the bin factors.",
            depth, height, width, tz, ty, tx,
        )
    trimmed = volume[:tz, :ty, :tx]

    log.info("Binning by %d×%d×%d (Z×Y×X, method=%s)…", fz, fy, fx, method)
    reshaped = trimmed.reshape(tz // fz, fz, ty // fy, fy, tx // fx, fx)
    if method == "max":
        binned = reshaped.max(axis=(1, 3, 5))
    else:
        # Average in float32 regardless of input dtype so integer inputs
        # don't truncate or overflow while summing each block.
        binned = reshaped.astype(np.float32).mean(axis=(1, 3, 5))

    binned = np.ascontiguousarray(binned, dtype=np.float32)

    # Row i of the direction matrix is axis i's direction vector scaled by
    # its spacing — binning multiplies that spacing by the bin factor.
    scaled_directions = direction_matrix * np.array([[fz], [fy], [fx]], dtype=float)

    new_spacing = spacing_from_directions(scaled_directions)
    log.info("Binned volume shape: %s", binned.shape)
    log.info(
        "New voxel spacing (Z×Y×X): %.4f × %.4f × %.4f mm",
        new_spacing[0], new_spacing[1], new_spacing[2],
    )
    return binned, scaled_directions


def _factors_for_target_um(
    direction_matrix: np.ndarray,
    target_um: float,
) -> tuple[int, int, int]:
    """Choose per-axis integer bin factors to reach *target_um* voxels.

    Native spacing per axis is the length of that axis's row in the direction
    matrix, so this stays correct for oblique/gantry-tilted acquisitions where
    the axes aren't aligned with L/P/S.

    Binning can only combine whole voxels, so the target is reachable exactly
    only when it is an integer multiple of the native spacing. Each axis uses
    ``round(target / native)`` — the factor landing closest to the requested
    size in either direction — clamped to >= 1 so an already-coarse axis is
    left untouched rather than being (impossibly) upsampled.

    Factors are computed per axis from that axis's own spacing, so anisotropic
    acquisitions (finer in-plane than through-plane, typical of stacked-slice
    microCT) converge toward isotropic voxels instead of preserving the
    original anisotropy.
    """
    native_um = spacing_from_directions(direction_matrix) * 1000.0
    factors = []
    for spacing_um in native_um:
        if spacing_um <= 0:
            factors.append(1)
            continue
        factors.append(max(1, int(round(target_um / spacing_um))))
    return factors[0], factors[1], factors[2]


def _log_binning_plan(
    direction_matrix: np.ndarray,
    factors: tuple[int, int, int],
    target_um: float,
) -> None:
    """Report the achievable voxel size versus the requested target."""
    native_um = spacing_from_directions(direction_matrix) * 1000.0
    achieved = tuple(float(n) * f for n, f in zip(native_um, factors))
    log.info(
        "Target voxel size %.1f µm — native %.2f × %.2f × %.2f µm (Z×Y×X) → "
        "bin %d×%d×%d → %.2f × %.2f × %.2f µm",
        target_um, *native_um, *factors, *achieved,
    )
    if any(abs(a - target_um) > 0.01 * target_um for a in achieved):
        log.warning(
            "Exact target not reachable — binning combines whole voxels, so "
            "only integer multiples of the native spacing are achievable. "
            "Using %.2f × %.2f × %.2f µm (closest available).",
            *achieved,
        )
    if all(f == 1 for f in factors):
        log.warning(
            "Target %.1f µm is at or below the native voxel size — no binning "
            "applied (this tool can only make voxels larger, not smaller).",
            target_um,
        )


# ---------------------------------------------------------------------------
# Reorientation to the specimen's long axis
# ---------------------------------------------------------------------------

def _sample_object_points(
    volume: np.ndarray,
    direction_matrix: np.ndarray,
    origin: np.ndarray,
    *,
    max_dim: int = 192,
    mask_smooth_mm: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Physical (L, P, S) coordinates of a strided sample of specimen voxels.

    The bounding box only needs the specimen's *shape*, not its full sampling,
    so the volume is strided down to at most ``max_dim`` voxels per axis before
    thresholding. That keeps the convex hull below and the Otsu threshold cheap
    (milliseconds) on a full-resolution microCT scan, and it does not move the
    answer: the long axis of a tooth is a centimetre-scale feature.

    Otsu separates specimen from air, and the largest connected component
    discards the sample holder, mounting putty and stray debris — any of which
    would otherwise drag the bounding box off the tooth's own axis.

    *mask_smooth_mm* Gaussian-blurs the subsample before thresholding, and is
    what keeps a tapered specimen in one piece. A rodent incisor's tip is its
    *densest* region (mature enamel), but it is also thin, so at full sampling
    the mask there is a lacework of hundreds of disconnected blobs rather than
    one solid mass — and "keep the largest component" then throws the tip away
    as if it were debris. Blurring bridges those gaps: on a 3168² test scan it
    collapsed 1218 components to 46 and took the largest component's share of
    the mask from 0.768 to 0.998, recovering 1.4 mm of tip that was otherwise
    cropped off by --reorient-tight. The recovered extent is flat from 1.0 mm
    to well past 4 mm of smoothing, so the exact value is not critical.

    The blur is applied to a *copy* used only to build the mask; the returned
    coordinates and *background* come from the unsmoothed data, so this defines
    a bounding box without touching the intensities anything downstream sees.

    Note the ordering this depends on: smoothing is safe here only because any
    crop has already run. On a full uncropped scan ~99% of voxels are air, and
    blurring smears that air population across the boundary hard enough to
    collapse the Otsu split into the noise band — which merges the holder into
    the specimen and balloons the box. Crop first, then smooth, then threshold.

    Returns ``(points [N, 3] in mm, background_level)``, where the background
    level is the median intensity below the threshold — the value to pad with
    when the volume is rotated into a larger grid.
    """
    steps = tuple(max(1, int(np.ceil(n / max_dim))) for n in volume.shape)
    sub = np.asarray(volume[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32)

    # Threshold a smoothed copy, but measure the unsmoothed one. Sigma is given
    # in mm and converted per axis, because the stride varies with volume size
    # (a cropped scan subsamples more finely than the full one): a sigma in
    # subsample voxels would otherwise mean a different physical distance on
    # every scan.
    mask_source = sub
    if mask_smooth_mm > 0:
        spacing = spacing_from_directions(direction_matrix)
        sub_spacing = spacing * np.asarray(steps, dtype=float)
        sigma = np.where(sub_spacing > 0, mask_smooth_mm / np.maximum(sub_spacing, 1e-12), 0.0)
        if np.any(sigma > 0):
            log.info(
                "Smoothing the specimen mask with sigma %.2f mm (%.2f × %.2f × "
                "%.2f subsample voxels) so a thin tip stays connected.",
                mask_smooth_mm, *sigma,
            )
            mask_source = gaussian_filter(sub, sigma=sigma)

    threshold = float(threshold_otsu(mask_source))
    mask = mask_source > threshold
    if not mask.any():
        raise ValueError("Otsu threshold selected no voxels — cannot find a long axis.")
    background = float(np.median(sub[~mask])) if np.any(~mask) else float(sub.min())

    labels, n_components = ndi_label(mask)
    if n_components > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = int(counts.argmax())
        kept = int(counts[largest])
        log.info(
            "Specimen mask: %d connected components — keeping the largest "
            "(%d of %d masked voxels, %.1f%%).",
            n_components, kept, int(mask.sum()), 100.0 * kept / float(mask.sum()),
        )
        mask = labels == largest

    idx = np.argwhere(mask).astype(float) * np.asarray(steps, dtype=float)
    points = np.asarray(origin, dtype=float) + idx @ np.asarray(direction_matrix, dtype=float)
    log.info(
        "Long-axis estimate uses %d sampled specimen voxels (stride %d×%d×%d, "
        "Otsu threshold %.4g).",
        len(points), *steps, threshold,
    )
    return points, background


def _min_area_rect(points_2d: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimum-area enclosing rectangle of a 2-D point set (rotating calipers).

    A minimum-area rectangle always has one side flush with an edge of the
    convex hull, so only the hull's own edge directions have to be tested.

    Returns ``(area, u, v)`` with ``u``, ``v`` the rectangle's orthonormal axes.
    """
    hull = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]

    edges = np.roll(hull_pts, -1, axis=0) - hull_pts
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-12
    if not keep.any():
        return 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    dirs = edges[keep] / lengths[keep, None]
    perp = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)

    # (E, M): every hull point projected onto every candidate edge direction.
    along = dirs @ hull_pts.T
    across = perp @ hull_pts.T
    areas = (along.max(axis=1) - along.min(axis=1)) * (across.max(axis=1) - across.min(axis=1))
    best = int(np.argmin(areas))
    return float(areas[best]), dirs[best], perp[best]


def _pca_axes(points: np.ndarray) -> np.ndarray:
    """Principal axes of *points* as orthonormal rows, longest variance first.

    Fallback for degenerate point sets where the convex hull can't be built
    (a perfectly flat or collinear specimen mask).
    """
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt


def min_volume_obb(points: np.ndarray, *, max_faces: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-volume oriented bounding box of a 3-D point set.

    Uses the standard hull-face result: a minimum-volume enclosing box has at
    least one face flush with a face of the convex hull. So for each hull face
    the points are projected onto that face's plane, the minimum-area rectangle
    of the projection is found by rotating calipers, and the box volume is that
    area times the extent along the face normal. The smallest wins.

    This is what distinguishes the box from a PCA frame: PCA minimizes squared
    distance to a fitted line, which a heavy root or a bulging cervical margin
    pulls off-axis, whereas the minimum-volume box is fixed by the specimen's
    outline alone — the answer wanted here, "which direction is the tooth long
    in".

    Duplicate face normals (co-planar facets of the same flat surface) are
    collapsed, and the candidate list is capped at *max_faces* by facet area so
    runtime stays bounded on a finely tessellated hull; the discarded facets are
    the slivers, which cannot define the box's flush face.

    Returns ``(axes [3, 3] orthonormal rows, extents [3] in the same units)``,
    both sorted by extent, longest first — so ``axes[0]`` is the long axis.
    """
    points = np.asarray(points, dtype=float)
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        log.warning("Convex hull failed — falling back to PCA axes for the long axis.")
        axes = _pca_axes(points)
        projected = points @ axes.T
        extents = projected.max(axis=0) - projected.min(axis=0)
        order = np.argsort(extents)[::-1]
        return axes[order], extents[order]

    verts = points[hull.vertices]
    normals = hull.equations[:, :3]

    # Collapse co-planar facets, then keep the largest-area candidates.
    _, unique_idx = np.unique(np.round(normals, 4), axis=0, return_index=True)
    if len(unique_idx) > max_faces:
        simplices = points[hull.simplices[unique_idx]]
        areas = 0.5 * np.linalg.norm(
            np.cross(simplices[:, 1] - simplices[:, 0], simplices[:, 2] - simplices[:, 0]),
            axis=1,
        )
        unique_idx = unique_idx[np.argsort(areas)[::-1][:max_faces]]
    normals = normals[unique_idx]

    best_volume = np.inf
    best_axes = None
    for normal in normals:
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        w = normal / norm
        # Any unit vector not parallel to w gives a valid in-plane basis.
        seed = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(w, seed)
        u /= np.linalg.norm(u)
        v = np.cross(w, u)

        planar = verts @ np.stack([u, v], axis=1)
        try:
            area, ru, rv = _min_area_rect(planar)
        except (QhullError, ValueError):
            continue
        along_normal = verts @ w
        volume = area * float(along_normal.max() - along_normal.min())
        if volume < best_volume:
            best_volume = volume
            best_axes = np.stack([w, ru[0] * u + ru[1] * v, rv[0] * u + rv[1] * v])

    if best_axes is None:
        log.warning("No valid hull face produced a box — falling back to PCA axes.")
        best_axes = _pca_axes(points)

    projected = verts @ best_axes.T
    extents = projected.max(axis=0) - projected.min(axis=0)
    order = np.argsort(extents)[::-1]
    return best_axes[order], extents[order]


def _tip_is_at_low_end(projection: np.ndarray, *, window: float = 0.2) -> bool:
    """True if the specimen's *thin* end sits at the low end of *projection*.

    An incisor tapers: the incisal tip has far less cross-section than the
    apical/base end. Comparing how many masked voxels fall in the outer 20% of
    the long axis at each end therefore identifies which end is the tip without
    needing any anatomical landmark.
    """
    low, high = float(projection.min()), float(projection.max())
    span = high - low
    if span <= 0:
        return True
    n_low = int(np.count_nonzero(projection < low + window * span))
    n_high = int(np.count_nonzero(projection > high - window * span))
    log.info(
        "Taper check along the long axis: %d voxels in the first %.0f%%, %d in "
        "the last %.0f%% → tip is at the %s end.",
        n_low, 100 * window, n_high, 100 * window, "low" if n_low <= n_high else "high",
    )
    return n_low <= n_high


def _snap_to_axis_permutation(
    frame: np.ndarray,
    direction_matrix: np.ndarray,
    snap_deg: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Return (perm, signs) if *frame* is a signed permutation of the array axes.

    When the specimen's long axis already lies within *snap_deg* of an existing
    array axis, a transpose-and-flip reproduces the requested orientation
    exactly. Taking that path instead of resampling keeps every voxel value
    bit-identical and costs nothing — worth checking, because the common case
    here (a scan whose Z runs buccal→lingual) is exactly a 90° axis swap.

    ``perm[a]`` is the old array axis that becomes new axis *a*; ``signs[a]``
    is -1 when that axis must also be reversed.
    """
    if snap_deg <= 0:
        return None
    spacing = spacing_from_directions(direction_matrix)
    if np.any(spacing <= 0):
        return None
    unit_axes = np.asarray(direction_matrix, dtype=float) / spacing[:, None]

    # cos[a, b] = angle between new axis a and old array axis b.
    cos = frame @ unit_axes.T
    tolerance = np.cos(np.deg2rad(snap_deg))
    best = np.argmax(np.abs(cos), axis=1)
    if np.any(np.abs(cos[np.arange(3), best]) < tolerance):
        return None
    if len(set(best.tolist())) != 3:
        return None
    signs = tuple(int(np.sign(cos[a, best[a]])) or 1 for a in range(3))
    return tuple(int(b) for b in best), signs


def _frame_spacing(
    frame: np.ndarray,
    direction_matrix: np.ndarray,
    voxel_mm: float | None = None,
) -> np.ndarray:
    """Voxel size (mm) to give each axis of the reoriented grid.

    Reorientation is a rotation, not a rescaling: it should not silently change
    how big a voxel is. Each new axis therefore inherits the spacing of the
    original array axis it is most nearly aligned with, matched one-to-one by
    an optimal assignment (so a 90° swap of an anisotropic scan carries the
    spacings around with the axes rather than mixing them up).

    This is exactly what the transpose/flip path does by construction, so both
    reorientation paths report the same voxel size for the same scan — a
    dataset that lands just either side of ``--reorient-snap-deg`` does not
    change resolution depending on which branch it took.

    For the isotropic scans these converters normally see (ISQ always, microCT
    DICOM nearly always) all three spacings are equal and this is a no-op.
    Pass *voxel_mm* to override with an explicit isotropic size instead.
    """
    spacing = spacing_from_directions(direction_matrix)
    if voxel_mm is not None:
        return np.full(3, float(voxel_mm), dtype=float)

    unit_axes = np.asarray(direction_matrix, dtype=float) / spacing[:, None]
    # Maximize total |cos| between new and old axes, constrained to a
    # one-to-one matching: a greedy argmax could assign two new axes the same
    # old one on a strongly oblique rotation.
    new_axis, old_axis = linear_sum_assignment(-np.abs(frame @ unit_axes.T))
    assigned = np.empty(3, dtype=float)
    assigned[new_axis] = spacing[old_axis]

    if float(spacing.max() - spacing.min()) > 1e-6 * float(spacing.max()):
        log.warning(
            "Anisotropic voxels (%.4f × %.4f × %.4f mm) rotated onto an oblique "
            "frame: each new axis keeps the spacing of its nearest original axis "
            "(%.4f × %.4f × %.4f mm), which preserves voxel size but samples a "
            "mixed-resolution direction at the coarser rate. Pass "
            "--reorient-voxel-mm to resample to an explicit isotropic size instead.",
            *spacing, *assigned,
        )
    return assigned


def reorient_to_long_axis(
    volume: np.ndarray,
    direction_matrix: np.ndarray,
    origin: np.ndarray,
    *,
    order: int = 1,
    snap_deg: float = 5.0,
    tight: bool = False,
    margin_mm: float = 1.0,
    tip_at: str = "start",
    voxel_mm: float | None = None,
    sample_max_dim: int = 192,
    mask_smooth_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Rotate *volume* so array axis Z runs along the specimen's long axis.

    Scans of a mounted incisor routinely come off the scanner with Z cutting
    buccal→lingual instead of tip→base, which makes every slice-wise step
    downstream (slice montages, per-slice enamel area, CMPR) cut the tooth the
    wrong way. This finds the tooth's own frame from the minimum-volume
    oriented bounding box of its segmented voxels and resamples the volume into
    it, so that afterwards:

        Z — long axis of the tooth (tip → base by default, see *tip_at*)
        Y — second-longest bounding box axis
        X — shortest bounding box axis

    Geometry is preserved, not faked: every voxel keeps its physical (L, P, S)
    position, and the NRRD "space directions"/"space origin" written afterwards
    describe the rotated frame, so the reoriented volume still overlays the
    original in Slicer/ITK-SNAP.

    When the required rotation is within *snap_deg* of a pure axis swap the
    volume is transposed and flipped instead of interpolated, which is exact —
    no resampling blur at all. Otherwise ``scipy.ndimage.affine_transform``
    resamples at spline *order* (1 = trilinear).

    Voxel size is carried through unchanged by both paths: the transpose path
    permutes the spacings along with the axes, and the resample path gives each
    new axis the spacing of its nearest original axis (see ``_frame_spacing``),
    so the two agree for the same scan. Pass *voxel_mm* to resample to an
    explicit isotropic voxel size instead.

    With *tight*, the output is cropped to the bounding box plus *margin_mm* of
    padding; otherwise the grid covers the whole input volume, so no data is
    lost to the rotation.

    Returns ``(volume, direction_matrix, origin, info)``.
    """
    points, background = _sample_object_points(
        volume, direction_matrix, origin, max_dim=sample_max_dim,
        mask_smooth_mm=mask_smooth_mm,
    )
    axes, extents = min_volume_obb(points)
    log.info(
        "Minimum-volume bounding box extents (long → short): "
        "%.2f × %.2f × %.2f mm", *extents,
    )

    long_axis, mid_axis = axes[0].copy(), axes[1].copy()

    # Orient the long axis tip → base (or base → tip), decided by which end
    # carries less cross-section.
    tip_low = _tip_is_at_low_end(points @ long_axis)
    want_tip_low = tip_at != "end"
    if tip_low != want_tip_low:
        long_axis = -long_axis

    # Keep the in-plane axes as close as possible to the original Y/X so the
    # reoriented volume isn't gratuitously mirrored relative to the scan.
    spacing = spacing_from_directions(direction_matrix)
    unit_axes = np.asarray(direction_matrix, dtype=float) / spacing[:, None]
    if mid_axis @ unit_axes[1] < 0:
        mid_axis = -mid_axis
    mid_axis = mid_axis - (mid_axis @ long_axis) * long_axis
    mid_axis /= np.linalg.norm(mid_axis)
    short_axis = np.cross(long_axis, mid_axis)  # right-handed by construction
    frame = np.stack([long_axis, mid_axis, short_axis])

    log.info(
        "Long axis (L, P, S) = [%.4f, %.4f, %.4f]; rotation vs. current Z axis "
        "= %.1f°",
        *long_axis,
        np.degrees(np.arccos(np.clip(abs(float(long_axis @ unit_axes[0])), -1.0, 1.0))),
    )

    # Physical bounds of the new grid, expressed along the new axes.
    if tight:
        bounds_source = points @ frame.T
        lo = bounds_source.min(axis=0) - margin_mm
        hi = bounds_source.max(axis=0) + margin_mm
    else:
        shape = np.array(volume.shape, dtype=float) - 1.0
        corner_idx = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T
        corners = np.asarray(origin, dtype=float) + (corner_idx * shape) @ np.asarray(
            direction_matrix, dtype=float
        )
        projected = corners @ frame.T
        lo = projected.min(axis=0)
        hi = projected.max(axis=0)

    info = {
        "spacing_mm_before_zyx": [float(v) for v in spacing],
        "long_axis_lps": [float(v) for v in long_axis],
        "frame_rows_lps": [[float(v) for v in row] for row in frame],
        "obb_extents_mm": [float(v) for v in extents],
        "tip_at": "start" if want_tip_low else "end",
        "tight": bool(tight),
        "margin_mm": float(margin_mm) if tight else None,
        "mask_smooth_mm": float(mask_smooth_mm),
    }

    snapped = _snap_to_axis_permutation(frame, direction_matrix, snap_deg)
    if snapped is not None:
        perm, signs = snapped
        log.info(
            "Long axis is within %.1f° of an existing array axis — reorienting "
            "by transpose %s with flips %s (exact, no interpolation).",
            snap_deg, perm, signs,
        )
        out = np.transpose(volume, perm)
        new_directions = np.asarray(direction_matrix, dtype=float)[list(perm)] * np.array(
            signs, dtype=float
        )[:, None]
        new_origin = np.asarray(origin, dtype=float).copy()
        flipped = []
        for new_axis, sign in enumerate(signs):
            if sign < 0:
                old_axis = perm[new_axis]
                new_origin = new_origin + (volume.shape[old_axis] - 1) * np.asarray(
                    direction_matrix, dtype=float
                )[old_axis]
                flipped.append(new_axis)
        if flipped:
            out = np.flip(out, axis=tuple(flipped))
        out = np.ascontiguousarray(out)

        # With --reorient-tight, trim to the bounding box. Bounds are recomputed
        # in the *snapped* frame (the array axes themselves) rather than the
        # box's own frame, so the crop lands on exact voxel boundaries.
        if tight:
            new_spacing = spacing_from_directions(new_directions)
            snap_frame = new_directions / new_spacing[:, None]
            proj = points @ snap_frame.T
            lo_snap = proj.min(axis=0) - margin_mm
            hi_snap = proj.max(axis=0) + margin_mm
            start_proj = np.asarray(new_origin, dtype=float) @ snap_frame.T
            shape_arr = np.array(out.shape)
            starts = np.clip(
                np.floor((lo_snap - start_proj) / new_spacing).astype(int), 0, shape_arr - 1
            )
            stops = np.clip(
                np.ceil((hi_snap - start_proj) / new_spacing).astype(int) + 1,
                starts + 1, shape_arr,
            )
            out = np.ascontiguousarray(
                out[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]]
            )
            new_origin = new_origin + starts.astype(float) @ new_directions

        info["method"] = "transpose"
        info["permutation_zyx"] = list(perm)
        info["flips_zyx"] = list(signs)
        info["spacing_mm_after_zyx"] = [
            float(v) for v in spacing_from_directions(new_directions)
        ]
        log.info("Reoriented volume shape (Z, Y, X): %s", out.shape)
        return out, new_directions, new_origin, info

    # General case: resample onto a grid aligned with the box, keeping each
    # axis's voxel size (see _frame_spacing) unless one was requested.
    new_spacing = _frame_spacing(frame, direction_matrix, voxel_mm)
    new_directions = frame * new_spacing[:, None]
    new_origin = lo @ frame
    out_shape = tuple(
        int(np.ceil(span / step)) + 1 for span, step in zip(hi - lo, new_spacing)
    )

    est_mb = float(np.prod(out_shape)) * 4 / 1e6
    log.info(
        "Resampling to the long-axis frame: %s voxels at %.4f × %.4f × %.4f mm "
        "(Z×Y×X; %.0f MB, spline order %d)…",
        out_shape, *new_spacing, est_mb, order,
    )

    # affine_transform maps output index n → input index M @ n + offset, and
    # index → physical is p = origin + n @ D for each grid, so
    # M = (D_new @ inv(D))ᵀ and offset = ((origin_new - origin_old) @ inv(D))ᵀ.
    inv_directions = np.linalg.inv(np.asarray(direction_matrix, dtype=float))
    matrix = (new_directions @ inv_directions).T
    offset = (np.asarray(new_origin, dtype=float) - np.asarray(origin, dtype=float)) @ inv_directions

    out = affine_transform(
        np.asarray(volume, dtype=np.float32),
        matrix,
        offset=offset,
        output_shape=out_shape,
        order=order,
        mode="constant",
        cval=background,
        prefilter=order > 1,
    )

    info["method"] = "resample"
    info["spacing_mm_after_zyx"] = [float(v) for v in new_spacing]
    info["voxel_mm_requested"] = float(voxel_mm) if voxel_mm is not None else None
    info["spline_order"] = int(order)
    log.info("Reoriented volume shape (Z, Y, X): %s", out.shape)
    return out, new_directions, np.asarray(new_origin, dtype=float), info


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _resolve_threshold(
    volume: np.ndarray,
    value: float | None,
    method: str,
    *,
    sample_max_dim: int = 192,
) -> float | None:
    """Turn a --noise-floor* option pair into one absolute intensity level.

    *method* selects how *value* is read:

    - ``"absolute"`` — *value* is already an intensity in the volume's own
      units (raw scanner counts, or cm^-1 after --apply-scaling). Returned
      unchanged.
    - ``"percentile"`` — *value* is a percentile of the intensity histogram,
      so ``--noise-floor 40 --noise-floor-method percentile`` clears the
      dimmest 40% of voxels. Useful when scans differ in overall gain and a
      single absolute count would not transfer between them.
    - ``"otsu"`` — *value* is ignored and the specimen/air split is found
      automatically. This is the most aggressive of the three: Otsu sits
      partway up the boundary ramp rather than at the noise floor, so it
      removes the low-density soft tissue and pulp along with the air.

    The percentile and Otsu levels are computed on a strided subsample (at
    most *sample_max_dim* voxels per axis), the same trick the autocrop and
    reorientation stages use: both are whole-histogram statistics, and a
    microCT volume is enormously oversampled relative to what either needs.

    Returns ``None`` when no thresholding was requested.
    """
    if method == "otsu":
        steps = tuple(max(1, int(np.ceil(n / sample_max_dim))) for n in volume.shape)
        sub = np.asarray(volume[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32)
        return float(threshold_otsu(sub))

    if value is None:
        return None

    if method == "percentile":
        if not 0.0 <= value <= 100.0:
            raise ValueError(
                f"--noise-floor must be a percentile in [0, 100] when "
                f"--noise-floor-method is 'percentile', got {value}."
            )
        steps = tuple(max(1, int(np.ceil(n / sample_max_dim))) for n in volume.shape)
        sub = np.asarray(volume[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32)
        return float(np.percentile(sub, value))

    return float(value)


def _threshold_filter(
    volume: np.ndarray,
    level: float | None,
    *,
    fill: float | None = None,
    mode: str = "floor",
) -> tuple[np.ndarray, dict | None]:
    """Clear everything below *level* to a constant, leaving the rest exact.

    This is a noise *floor*, not a segmentation. Air in a microCT scan is not
    empty — it is a band of low-amplitude counts around zero, and every later
    stage pays for it: NLM and the median filter spend their kernel budget
    averaging that band, anisotropic diffusion sees spurious gradients in it,
    and gzip cannot compress it because no two air voxels agree. Flattening it
    to one value first makes each of those cheaper and, for the filters,
    better behaved at the specimen boundary.

    Two modes, differing only in what happens to the voxels that survive:

    - ``"floor"`` (default) — sub-threshold voxels become *fill*; everything
      at or above *level* keeps its exact original value. Non-destructive for
      the specimen, which is why it is the default: densitometry, enamel
      thresholding and every downstream measurement still see true counts.
    - ``"shift"`` — the same, then *level* is subtracted from the survivors so
      the specimen starts at zero. Rescales the values, so it invalidates any
      absolute-density calibration; use it only for viewing or for tools that
      want a zero-based range.

    *fill* defaults to *level* in floor mode (a flat plateau at the cut, which
    keeps the boundary ramp monotonic) and to 0 in shift mode.

    Returns ``(volume, info)`` where *info* records the level and the fraction
    of voxels cleared, for the metadata sidecar. *info* is ``None`` when no
    threshold was applied.
    """
    if level is None:
        return volume, None

    if mode not in ("floor", "shift"):
        raise ValueError(f"Unknown threshold mode {mode!r}; expected 'floor' or 'shift'.")

    if fill is None:
        fill = 0.0 if mode == "shift" else float(level)

    volume = volume.astype(np.float32)
    below = volume < level
    cleared = float(below.mean())

    log.info(
        "Noise floor at %.4g (%s): clearing %.1f%% of voxels to %.4g…",
        level, mode, cleared * 100.0, fill,
    )
    if cleared > 0.98:
        log.warning(
            "That threshold clears %.1f%% of the volume — it is almost "
            "certainly above the specimen, not just the air. Check the level "
            "against the scan's histogram.", cleared * 100.0,
        )

    if mode == "shift":
        volume = volume - float(level)
        volume[below] = fill
    else:
        volume[below] = fill

    return volume.astype(np.float32), {
        "level": float(level),
        "mode": mode,
        "fill": float(fill),
        "fraction_cleared": cleared,
    }


def _gaussian_presmoother(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Light isotropic Gaussian to suppress isolated hot pixels before NLM."""
    if sigma <= 0:
        return volume
    log.info("Gaussian pre-smoothing (σ=%.2f)…", sigma)
    return gaussian_filter(volume.astype(np.float32), sigma=sigma)


def _nlm_worker(args: tuple[int, np.ndarray, int, int, float]) -> tuple[int, np.ndarray]:
    """Picklable per-slice NLM worker for ProcessPoolExecutor."""
    idx, slc, patch_size, patch_distance, h = args
    result = denoise_nl_means(
        slc,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=True,
        preserve_range=True,
    )
    return idx, result.astype(np.float32)


def _nlm_filter(
    volume: np.ndarray,
    patch_size: int,
    patch_distance: int,
    h_factor: float,
    mode_3d: bool,
    workers: int | None = None,
) -> np.ndarray:
    """Non-local means denoising.

    2-D slice mode (default): slices are denoised in parallel across workers.
    3-D mode (--nlm-3d): denoises the whole volume jointly on a single process
    — better results on isotropic acquisitions but uses significantly more RAM.

    ``h_factor <= 0`` skips the stage entirely. h=0 would weight only exactly
    matching patches, i.e. return the input, after paying the full (dominant)
    cost of the search — so skipping is both the same answer and the fast one,
    and it is what makes a filter-free raw conversion possible.
    """
    if h_factor <= 0:
        log.info("NLM filter disabled (--nlm-h 0).")
        return volume

    volume = volume.astype(np.float32)

    if mode_3d:
        log.info("NLM filter — 3-D mode (patch=%d, search=%d, h×σ=%.1f)…",
                 patch_size, patch_distance, h_factor)
        sigma_est = float(np.mean(estimate_sigma(volume)))
        log.debug("  Estimated noise σ = %.4f", sigma_est)
        return denoise_nl_means(
            volume,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h_factor * sigma_est,
            fast_mode=True,
            preserve_range=True,
        ).astype(np.float32)

    log.info("NLM filter — 2-D parallel mode (patch=%d, search=%d, h×σ=%.1f)…",
             patch_size, patch_distance, h_factor)
    # Estimate noise from a representative subset of slices for speed
    sample_idx = np.linspace(0, volume.shape[0] - 1, min(10, volume.shape[0]), dtype=int)
    sigma_est = float(np.mean([
        np.mean(estimate_sigma(volume[i])) for i in sample_idx
    ]))
    log.debug("  Estimated noise σ = %.4f (from %d sample slices)", sigma_est, len(sample_idx))
    h = h_factor * sigma_est

    out = np.empty_like(volume)
    nlm_args = [(i, volume[i], patch_size, patch_distance, h) for i in range(volume.shape[0])]
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(_nlm_worker, a) for a in nlm_args]
        with tqdm(total=volume.shape[0], desc="NLM", unit="slice") as pbar:
            for fut in as_completed(futures):
                idx, slc = fut.result()
                out[idx] = slc
                pbar.update()
    return out


@njit(parallel=True, cache=True, nogil=True)
def _median_filter_3d_kernel(volume: np.ndarray, radius: int) -> np.ndarray:
    """
    True 3-D sliding-window median with reflect boundary handling, parallelized
    over Z with numba. Equivalent to scipy.ndimage.median_filter(volume,
    size=2*radius+1, mode="reflect") but multi-threaded across CPU cores
    instead of scipy's single-threaded C loop.
    """
    depth, height, width = volume.shape
    out = np.empty_like(volume)
    window_len = (2 * radius + 1) ** 3

    for z in prange(depth):
        window = np.empty(window_len, dtype=volume.dtype)
        for y in range(height):
            for x in range(width):
                n = 0
                for dz in range(-radius, radius + 1):
                    zz = z + dz
                    # reflect boundary (scipy default: "reflect" mirrors the
                    # edge value, e.g. ... 2 1 | 0 1 2 ... for a 1-D axis)
                    if zz < 0:
                        zz = -zz - 1
                    elif zz >= depth:
                        zz = 2 * depth - zz - 1
                    for dy in range(-radius, radius + 1):
                        yy = y + dy
                        if yy < 0:
                            yy = -yy - 1
                        elif yy >= height:
                            yy = 2 * height - yy - 1
                        for dx in range(-radius, radius + 1):
                            xx = x + dx
                            if xx < 0:
                                xx = -xx - 1
                            elif xx >= width:
                                xx = 2 * width - xx - 1
                            window[n] = volume[zz, yy, xx]
                            n += 1
                out[z, y, x] = np.median(window)

    return out


def _median_filter(volume: np.ndarray, radius: int) -> np.ndarray:
    """3-D median filter to remove residual salt-and-pepper noise.

    ``radius <= 0`` skips the stage: a 1×1×1 window's median is the voxel
    itself, so the pass would rewrite the volume unchanged at full cost.
    """
    if radius <= 0:
        return volume
    size = 2 * radius + 1
    log.info("Median filter (size=%d, numba parallel 3-D)…", size)
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    return _median_filter_3d_kernel(volume, radius)


def _gaussian_psf(sigma: float, radius: int | None = None) -> np.ndarray:
    """Build a normalized isotropic 3-D Gaussian point-spread function."""
    r = radius if radius is not None else max(1, int(round(3 * sigma)))
    ax = np.arange(-r, r + 1, dtype=np.float32)
    zz, yy, xx = np.meshgrid(ax, ax, ax, indexing="ij")
    psf = np.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    return (psf / psf.sum()).astype(np.float32)


def _richardson_lucy_filter(
    volume: np.ndarray,
    psf_sigma: float,
    iterations: int,
    psf_path: Path | None = None,
) -> np.ndarray:
    """Richardson-Lucy deconvolution to recover detail lost to the imaging PSF.

    Run after denoising (Gaussian/NLM/median) since RL amplifies whatever
    noise remains in the input with each iteration — deconvolving first would
    sharpen noise as if it were signal. Uses a synthetic isotropic Gaussian
    PSF by default (--psf-sigma); pass --psf-file to load a measured PSF
    (e.g. from a calibration bead scan) instead.
    """
    if iterations <= 0:
        return volume

    from skimage.restoration import richardson_lucy

    if psf_path is not None:
        log.info("Richardson-Lucy deconvolution — loading PSF from %s…", psf_path)
        psf, _ = nrrd.read(str(psf_path))
        psf = (psf / psf.sum()).astype(np.float32)
    else:
        log.info("Richardson-Lucy deconvolution — synthetic Gaussian PSF (σ=%.2f)…", psf_sigma)
        psf = _gaussian_psf(psf_sigma)

    log.info("Richardson-Lucy deconvolution (%d iterations)…", iterations)
    volume = volume.astype(np.float32)
    vmin = float(volume.min())
    # richardson_lucy requires non-negative input; shift up if needed and
    # restore the original offset afterward so intensities stay comparable.
    shifted = volume - vmin if vmin < 0 else volume
    deconvolved = richardson_lucy(shifted, psf, num_iter=iterations, clip=False)
    if vmin < 0:
        deconvolved = deconvolved + vmin
    return deconvolved.astype(np.float32)


@njit(parallel=True, cache=True, nogil=True)
def _anisotropic_diffusion_kernel(
    volume: np.ndarray,
    iterations: int,
    kappa: float,
    gamma: float,
) -> np.ndarray:
    """
    3-D Perona-Malik anisotropic diffusion, parallelized over Z with numba.

    Each iteration diffuses intensity between a voxel and its 6 face
    neighbours, but weights each flux by g(|grad I|) = exp(-(grad/kappa)^2) —
    so smoothing runs freely *along* a homogeneous region and is throttled
    *across* a step edge. That is the opposite behaviour to the Gaussian /
    median stages above, which blur an enamel/dentin boundary just as hard as
    they blur noise inside dentin.

    The exponential conduction function (Perona-Malik eq. 2) is used rather
    than the 1/(1+(grad/kappa)^2) form; it favours high-contrast edges more
    strongly, which suits the large enamel-to-dentin attenuation step.
    """
    depth, height, width = volume.shape
    current = volume.copy()
    out = np.empty_like(volume)
    inv_kappa_sq = np.float32(1.0 / (kappa * kappa))
    gamma32 = np.float32(gamma)

    # The 6-neighbourhood as explicit integer offsets. Kept as an array rather
    # than branching on an axis index so numba infers a single integer type
    # for the neighbour coordinates below.
    offsets = np.array(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=np.int64,
    )

    for _ in range(iterations):
        for z in prange(depth):
            for y in range(height):
                for x in range(width):
                    centre = current[z, y, x]
                    total = np.float32(0.0)
                    # A voxel on the volume face simply has no flux in that
                    # direction (zero-flux Neumann boundary), which keeps the
                    # outer surface from drifting inward.
                    for k in range(6):
                        zz = z + offsets[k, 0]
                        if zz < 0 or zz >= depth:
                            continue
                        yy = y + offsets[k, 1]
                        if yy < 0 or yy >= height:
                            continue
                        xx = x + offsets[k, 2]
                        if xx < 0 or xx >= width:
                            continue
                        grad = current[zz, yy, xx] - centre
                        total += np.exp(-grad * grad * inv_kappa_sq) * grad
                    out[z, y, x] = centre + gamma32 * total
        current = out.copy()

    return current


def _anisotropic_diffusion_filter(
    volume: np.ndarray,
    iterations: int,
    kappa: float,
    gamma: float,
) -> np.ndarray:
    """Edge-preserving Perona-Malik diffusion (see kernel for the math).

    Placed before RL deconvolution: diffusion flattens the noise that RL
    would otherwise amplify, while leaving the boundary step that RL is meant
    to sharpen intact.

    ``kappa`` is the conduction threshold in *intensity units of this volume*
    — gradients well above it are treated as edges and preserved, gradients
    below it are smoothed away. It therefore has to be set relative to the
    actual enamel/dentin intensity step, not copied from another dataset.
    """
    if iterations <= 0:
        return volume

    log.info(
        "Anisotropic diffusion (Perona-Malik, %d iterations, κ=%.1f, γ=%.2f)…",
        iterations, kappa, gamma,
    )
    volume = np.ascontiguousarray(volume, dtype=np.float32)
    return _anisotropic_diffusion_kernel(volume, iterations, kappa, gamma)


def _unsharp_mask_filter(
    volume: np.ndarray,
    amount: float,
    sigma: float,
) -> np.ndarray:
    """Unsharp masking: boost what a Gaussian blur would have removed.

    ``out = I + amount * (I - blur(I))``. The difference term is the high
    frequency content — dominated by the boundary ramp — so adding it back
    steepens every edge, which is exactly what a downstream Sobel/gradient
    edge detector keys off.

    Runs last, after all denoising: unsharp masking is a linear high-pass
    boost with no notion of what is signal, so any noise still present is
    amplified by the same factor as the edges. Keep ``amount`` modest (0.5-1.5)
    — heavier settings produce haloes at the enamel surface that read as a
    false second edge.
    """
    if amount <= 0:
        return volume

    log.info("Unsharp mask (amount=%.2f, σ=%.2f)…", amount, sigma)
    volume = volume.astype(np.float32)
    blurred = gaussian_filter(volume, sigma=sigma)
    return (volume + amount * (volume - blurred)).astype(np.float32)


def _tv_worker(args: tuple[int, np.ndarray, float]) -> tuple[int, np.ndarray]:
    """Picklable per-slice TV-denoise worker for ProcessPoolExecutor."""
    idx, slc, weight = args
    from skimage.restoration import denoise_tv_chambolle
    return idx, denoise_tv_chambolle(slc, weight=weight).astype(np.float32)


def _tv_filter(volume: np.ndarray, weight: float, workers: int | None = None) -> np.ndarray:
    """Total variation (Chambolle) denoising, run slice-by-slice in parallel.

    TV denoising enforces piecewise-constant regions, sharpening the
    bone/background boundary and producing cleaner histogram peaks for
    thresholding. Run last in the filter pipeline — after Gaussian/NLM/median
    have removed stochastic and salt-and-pepper noise — since TV works best on
    data that's already had that noise stripped out; applied too early it can
    smear noise into false plateaus instead of true edges.
    """
    if weight <= 0:
        return volume

    log.info("TV (Chambolle) denoising (weight=%.3f)…", weight)
    volume = volume.astype(np.float32)
    out = np.empty_like(volume)
    tv_args = [(i, volume[i], weight) for i in range(volume.shape[0])]
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(_tv_worker, a) for a in tv_args]
        with tqdm(total=volume.shape[0], desc="TV denoise", unit="slice") as pbar:
            for fut in as_completed(futures):
                idx, slc = fut.result()
                out[idx] = slc
                pbar.update()
    return out


def _clahe_enhance(volume: np.ndarray, clip_limit: float, kernel_size: int | None) -> np.ndarray:
    """Slice-wise CLAHE for a viewing copy with enamel made visually distinct.

    Unlike percentile contrast stretching, CLAHE boosts *local* contrast so the
    enamel/dentin/bone boundary becomes visible without crushing the rest of the
    dynamic range. Applied per-slice (axial) since skimage's CLAHE expects
    2-D images and per-slice equalization keeps RAM low. Output is rescaled
    back to the input's original intensity range so it stays comparable to
    the raw volume, just with enhanced local contrast.
    """
    from skimage.exposure import equalize_adapthist

    log.info("CLAHE contrast enhancement (clip_limit=%.3f)…", clip_limit)
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax == vmin:
        return volume.astype(np.float32)

    # equalize_adapthist expects input scaled to [0, 1] float
    normed = ((volume - vmin) / (vmax - vmin)).astype(np.float32)
    out = np.empty_like(normed)
    for i in tqdm(range(normed.shape[0]), desc="CLAHE", unit="slice"):
        out[i] = equalize_adapthist(
            normed[i],
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )
    # Rescale back to original intensity range so the file stays compatible
    # with the same window/level habits used on the raw volume.
    return (out * (vmax - vmin) + vmin).astype(np.float32)


def apply_filter_chain(volume: np.ndarray, opts: PipelineOptions) -> np.ndarray:
    """Run every filter stage in the fixed order documented at module top."""
    volume = _gaussian_presmoother(volume, opts.gauss_sigma)
    volume = _nlm_filter(volume, opts.nlm_patch, opts.nlm_search, opts.nlm_h,
                         opts.nlm_3d, opts.workers)
    volume = _median_filter(volume, opts.median_radius)
    volume = _anisotropic_diffusion_filter(volume, opts.aniso_iterations,
                                           opts.aniso_kappa, opts.aniso_gamma)
    volume = _richardson_lucy_filter(volume, opts.rl_psf_sigma, opts.rl_iterations,
                                     opts.rl_psf_path)
    volume = _tv_filter(volume, opts.tv_weight, opts.workers)
    volume = _unsharp_mask_filter(volume, opts.unsharp_amount, opts.unsharp_sigma)
    return volume


# ---------------------------------------------------------------------------
# NRRD writer
# ---------------------------------------------------------------------------

def write_nrrd(
    path: Path,
    volume: np.ndarray,
    direction_matrix: np.ndarray,
    *,
    origin: np.ndarray | None = None,
    compression_level: int = 1,
    dtype: str = "float32",
) -> None:
    """Write *volume* to NRRD with physical voxel geometry encoded in the header.

    *volume* is the pipeline's internal (Z, Y, X) array and *direction_matrix*
    has one row per axis in that same order. Both are transposed to (X, Y, Z)
    on the way out, because that is the layout every other NRRD in this project
    uses — Slicer/ITK-SNAP exports, and ``segment_mandible``'s own
    ``save_as_nrrd``, which likewise flips (Z, Y, X) → (X, Y, Z) before
    writing. Matching it means ``load_nrrd``'s transpose back to (Z, Y, X) is
    correct for converter output and hand-exported files alike, instead of
    being right for one and silently putting the buccal-lingual axis on Z for
    the other.

    NRRD convention: "space directions" row i is the (L, P, S) direction
    vector for array axis i, scaled by that axis's physical spacing. Reversing
    the row order alongside the array keeps the two consistent, so the volume
    still lands in the same physical place in a viewer — only the storage
    order changes, never the geometry.

    Write speed notes (this dominates wall time on large volumes):
      - ``compression_level`` maps to zlib's level. pynrrd defaults to 9,
        which is single-threaded and by far the slowest setting for only a
        few percent size gain on noisy microCT data. Level 1 is typically
        10-20x faster; level 0 writes uncompressed raw (fastest, largest).
      - ``index_order`` is deliberately left at pynrrd's default of "F", which
        writes "sizes" in the same order as the array it is handed — here
        (X, Y, Z), matching the transposed "space directions" rows below.
      - ``dtype="int16"`` halves the byte count versus float32. Safe for raw
        stored microCT values (integers by construction), but filtered
        volumes are rounded, so it is not the default.

    ``origin`` is the (L, P, S) position of voxel [0, 0, 0]. It must be
    carried through a crop — a cropped volume starts at a different physical
    point, and leaving the origin at zero silently shifts it relative to the
    uncropped scan in Slicer/ITK-SNAP. Defaults to the origin. The origin is a
    physical point, so it is unaffected by the axis reversal.
    """
    # (Z, Y, X) → (X, Y, Z), rows of the direction matrix reversed to match.
    volume = np.transpose(volume, (2, 1, 0))
    direction_matrix = np.asarray(direction_matrix, dtype=float)[::-1]

    volume = np.ascontiguousarray(volume)

    target = np.dtype(dtype)
    if volume.dtype == target:
        # Already the requested dtype — nothing to round or clip. Worth
        # short-circuiting because np.rint promotes an integer array to float
        # just to round values that are integers already, doubling peak memory
        # on a volume that needs no conversion at all.
        pass
    elif dtype in ("int16", "uint16"):
        info = np.iinfo(target)
        vmin, vmax = float(volume.min()), float(volume.max())
        if vmin < info.min or vmax > info.max:
            log.warning(
                "Volume range [%.1f, %.1f] exceeds %s [%d, %d] — values will "
                "be clipped. Use --save-dtype float32 to preserve them.",
                vmin, vmax, dtype, info.min, info.max,
            )
            volume = np.clip(volume, info.min, info.max)
        volume = np.rint(volume).astype(target)
    else:
        volume = volume.astype(np.float32, copy=False)

    origin_lps = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    header = {
        "space": "left-posterior-superior",
        "space directions": np.asarray(direction_matrix, dtype=float).tolist(),
        "space origin": origin_lps.tolist(),
        "kinds": ["domain", "domain", "domain"],
    }
    if compression_level <= 0:
        header["encoding"] = "raw"

    est_mb = volume.nbytes / 1e6
    log.info(
        "Writing NRRD → %s (%s, %s, %.0f MB uncompressed)",
        path, volume.dtype, "raw" if compression_level <= 0 else f"gzip-{compression_level}", est_mb,
    )
    nrrd.write(
        str(path),
        volume,
        header,
        compression_level=max(1, compression_level),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Metadata keys this module derives on every run. Stripped from a cached
# <stem>_meta.json before it is reused so stale values from the run that made
# <stem>_prefiltered.nrrd can't survive into a run with different settings.
# (The _binning/_reorientation values are read off the cache *before* this
# strip, to know which geometric stages that file already went through.)
_DERIVED_META_KEYS = (
    "_spacing_mm", "_native_spacing_mm", "_binning", "_reorientation",
    "_noise_floor", "_filter_params",
)


def convert_volume(
    *,
    output_dir: Path,
    stem: str,
    read_source: SourceReader,
    opts: PipelineOptions,
) -> dict[str, Path]:
    """Crop → bin → reorient → filter → write, shared by both converters.

    Two volumes are written per run:

    - ``<stem>_prefiltered.nrrd`` (when ``--save-raw`` is set) — the volume
      after every *geometric* stage that was requested (crop, binning,
      reorientation) but before any filter runs. This is the "compressed"
      copy: it is already at the target voxel size and orientation, so it is
      as small as the filtered output and directly comparable to it.
    - ``<stem>.nrrd`` — the same volume after the filter chain.

    ``read_source`` is called only when there is no reusable
    ``<stem>_prefiltered.nrrd``; it returns the source volume (Z, Y, X), its
    direction matrix, the (L, P, S) origin of voxel [0, 0, 0], and a dict of
    format-specific header metadata to seed ``<stem>_meta.json``.

    That prefiltered file doubles as a resume cache: a later run reloads it and
    skips the source import, the crop prompt, and whichever of binning and
    reorientation were already baked into it (recorded in ``<stem>_meta.json``).
    Re-binning an already-binned volume would compound the factors — 20 µm
    twice is 40 µm — so those stages are skipped rather than repeated.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    source_meta: dict | None = None

    prefiltered_path = output_dir / f"{stem}_prefiltered.nrrd"
    cached_meta_path = output_dir / f"{stem}_meta.json"

    # Geometric stages already applied to a reloaded prefiltered volume; those
    # are skipped below instead of being applied a second time.
    done_binning = False
    done_reorient = False
    cached_binning: dict | None = None
    cached_reorient: dict | None = None

    if prefiltered_path.exists():
        # A previous --save-raw run already produced a cropped/binned/reoriented
        # but unfiltered volume. Reuse it so re-running with different filter
        # parameters doesn't require re-reading the source, re-selecting the
        # crop region, or repeating the binning and rotation.
        log.info("Found existing %s — loading it and skipping source import.",
                 prefiltered_path)
        volume, nrrd_header = nrrd.read(str(prefiltered_path))
        volume = volume.astype(np.float32)
        direction_matrix = np.asarray(nrrd_header["space directions"], dtype=float)
        origin = np.asarray(nrrd_header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
        # write_nrrd stores (X, Y, Z); undo that to get back to the pipeline's
        # internal (Z, Y, X), reversing the direction rows to match.
        volume = np.ascontiguousarray(np.transpose(volume, (2, 1, 0)))
        direction_matrix = direction_matrix[::-1]
        log.info("Loaded direction matrix (Z,Y,X rows, LPS columns):\n%s", direction_matrix)
        log.info("Loaded volume shape (Z, Y, X): %s", volume.shape)
        outputs["prefiltered"] = prefiltered_path

        # Which geometric stages does that file already carry? The metadata
        # written alongside it is the record; without it, assume none and warn,
        # since re-binning would silently change the voxel size.
        if cached_meta_path.exists():
            with open(cached_meta_path) as fh:
                prior = json.load(fh)
            cached_binning = prior.get("_binning") or None
            if cached_binning and tuple(cached_binning.get("factors_zyx") or ()) not in ((), (1, 1, 1)):
                done_binning = True
            cached_reorient = prior.get("_reorientation") or None
            if cached_reorient and cached_reorient.get("enabled") is not False:
                done_reorient = True
        else:
            log.warning(
                "%s has no companion %s, so the stages already applied to it "
                "are unknown. Assuming none — check the result if this file "
                "came from a run with --target-voxel-um or --reorient.",
                prefiltered_path.name, cached_meta_path.name,
            )

        if done_binning and opts.target_voxel_um is not None:
            log.info(
                "Skipping binning: %s is already binned by %s (%.1f µm target). "
                "Delete it to re-bin the source to a different voxel size.",
                prefiltered_path.name,
                tuple(cached_binning.get("factors_zyx")),
                cached_binning.get("target_voxel_um") or float("nan"),
            )
        if done_reorient and opts.reorient:
            log.info(
                "Skipping reorientation: %s is already on the specimen's long axis.",
                prefiltered_path.name,
            )

    else:
        volume, direction_matrix, origin, source_meta = read_source()
        log.info("Volume shape (Z, Y, X): %s", volume.shape)
        log.info(
            "Voxel spacing (Z×Y×X): %.4f × %.4f × %.4f mm",
            *spacing_from_directions(direction_matrix),
        )

        # Optionally crop before anything else, so every downstream step runs
        # on fewer voxels. --crop asks the user; --autocrop finds the specimen
        # itself. An explicit --crop wins if somehow both are given.
        crop_slices = None
        if opts.crop:
            log.info("Opening napari for interactive crop selection…")
            crop_slices = select_crop_bbox(volume)
            if opts.autocrop:
                log.info("--crop given, so --autocrop is ignored for this run.")
        elif opts.autocrop:
            log.info("Detecting the specimen for automatic cropping…")
            crop_slices = autocrop_bbox(
                volume,
                margin_mm=opts.autocrop_margin_mm,
                spacing=spacing_from_directions(direction_matrix),
                max_dim=opts.autocrop_sample_dim,
            )

        if crop_slices is not None:
            z_sl, y_sl, x_sl = crop_slices
            volume = np.ascontiguousarray(volume[z_sl, y_sl, x_sl])
            log.info("Cropped volume shape: %s", volume.shape)

            # Voxel [0, 0, 0] of the cropped volume is voxel
            # [z_start, y_start, x_start] of the original, so shift the origin
            # along each axis's own direction vector by that many voxels.
            starts = np.array([z_sl.start, y_sl.start, x_sl.start], dtype=float)
            origin = origin + starts @ direction_matrix

    # Bin (downsample) to the target voxel size, so the filters, CLAHE and every
    # write below operate on the reduced volume.
    native_directions = direction_matrix
    bin_factors = (1, 1, 1)
    if opts.target_voxel_um is not None and not done_binning:
        bin_factors = _factors_for_target_um(direction_matrix, opts.target_voxel_um)
        _log_binning_plan(direction_matrix, bin_factors, opts.target_voxel_um)
        volume, direction_matrix = _bin_volume(
            volume, bin_factors, direction_matrix, opts.bin_method
        )
    elif done_binning and cached_binning:
        # Carry the earlier run's factors into this run's metadata, and treat
        # the reloaded spacing as post-binning so _native_spacing_mm still
        # reports the scanner's own resolution rather than the binned one.
        bin_factors = tuple(cached_binning.get("factors_zyx") or (1, 1, 1))
        native_directions = direction_matrix / np.array(
            [[max(1, f)] for f in bin_factors], dtype=float
        )

    # Rotate the volume so Z runs along the specimen's long axis. Placed after
    # binning so the bounding-box search and the resample both run on the
    # reduced volume, and before the prefiltered write so both outputs share
    # one orientation.
    reorient_info: dict | None = cached_reorient if done_reorient else None
    if opts.reorient and not done_reorient:
        volume, direction_matrix, origin, reorient_info = reorient_to_long_axis(
            volume, direction_matrix, origin,
            order=opts.reorient_order,
            snap_deg=opts.reorient_snap_deg,
            tight=opts.reorient_tight,
            margin_mm=opts.reorient_margin_mm,
            tip_at=opts.reorient_tip_at,
            voxel_mm=opts.reorient_voxel_mm,
            sample_max_dim=opts.reorient_sample_dim,
            mask_smooth_mm=opts.reorient_mask_smooth_mm,
        )

    # Save the unfiltered volume — cropped, binned and reoriented as requested,
    # so it is the same size and in the same frame as the filtered output below
    # and the two can be compared voxel-for-voxel. Skipped when it was itself
    # the source of this run, since nothing about it has changed.
    if opts.save_raw and "prefiltered" not in outputs:
        write_nrrd(prefiltered_path, volume, direction_matrix, origin=origin,
                   compression_level=opts.compression_level, dtype=opts.save_dtype)
        outputs["prefiltered"] = prefiltered_path

    # Flatten the air band before the filters see it, so NLM, the median
    # filter and anisotropic diffusion aren't spending their kernels on
    # background noise or dragging it across the specimen boundary. Placed
    # after the prefiltered write, so that cached copy stays a faithful
    # unfiltered record and a later run can re-threshold it differently.
    threshold_level = _resolve_threshold(
        volume, opts.noise_floor, opts.noise_floor_method,
    )
    volume, threshold_info = _threshold_filter(
        volume, threshold_level,
        fill=opts.noise_floor_fill,
        mode=opts.noise_floor_mode,
    )

    volume = apply_filter_chain(volume, opts)

    if opts.output_floor is not None:
        below = int(np.count_nonzero(volume < opts.output_floor))
        if below:
            log.info(
                "Filter chain pushed %d voxel(s) below the floor %g — clamping "
                "them back before writing.", below, opts.output_floor,
            )
            volume = np.maximum(volume, np.asarray(opts.output_floor, dtype=volume.dtype))

    nrrd_path = output_dir / f"{stem}.nrrd"
    write_nrrd(nrrd_path, volume, direction_matrix, origin=origin,
               compression_level=opts.compression_level, dtype=opts.save_dtype)
    outputs["filtered"] = nrrd_path

    # Optional CLAHE-enhanced viewing copy (enamel visibility), separate file
    if opts.enhance_contrast:
        enhanced = _clahe_enhance(volume, opts.clahe_clip_limit, opts.clahe_kernel_size)
        enhanced_path = output_dir / f"{stem}_enhanced.nrrd"
        write_nrrd(enhanced_path, enhanced, direction_matrix, origin=origin,
                   compression_level=opts.compression_level, dtype=opts.save_dtype)
        outputs["enhanced"] = enhanced_path
        del enhanced

    meta_path = output_dir / f"{stem}_meta.json"
    if source_meta is not None:
        meta = dict(source_meta)
    elif cached_meta_path.exists():
        # The source wasn't re-read this run (loaded from
        # <stem>_prefiltered.nrrd), so carry forward the header fields captured
        # by the original run.
        with open(cached_meta_path) as fh:
            meta = json.load(fh)
        for key in _DERIVED_META_KEYS:
            meta.pop(key, None)
    else:
        meta = {}

    axis_spacing = spacing_from_directions(direction_matrix)
    meta["_spacing_mm"] = {
        "z": float(axis_spacing[0]), "y": float(axis_spacing[1]), "x": float(axis_spacing[2]),
    }
    native_spacing = spacing_from_directions(native_directions)
    meta["_native_spacing_mm"] = {
        "z": float(native_spacing[0]), "y": float(native_spacing[1]), "x": float(native_spacing[2]),
    }
    binned_this_run = opts.target_voxel_um is not None and not done_binning
    meta["_binning"] = {
        "target_voxel_um": (
            opts.target_voxel_um if binned_this_run
            else (cached_binning or {}).get("target_voxel_um") if done_binning
            else None
        ),
        "factors_zyx": list(bin_factors),
        "method": (
            opts.bin_method if binned_this_run
            else (cached_binning or {}).get("method") if done_binning
            else None
        ),
    }
    meta["_reorientation"] = reorient_info if reorient_info is not None else {"enabled": False}
    meta["_noise_floor"] = threshold_info if threshold_info is not None else {"enabled": False}
    meta["_filter_params"] = {
        "noise_floor": opts.noise_floor,
        "noise_floor_method": (
            opts.noise_floor_method if threshold_info is not None else None
        ),
        "noise_floor_mode": (
            opts.noise_floor_mode if threshold_info is not None else None
        ),
        "cropped": opts.crop or opts.autocrop or "prefiltered" in outputs,
        "autocrop": opts.autocrop and not opts.crop,
        "autocrop_margin_mm": (
            opts.autocrop_margin_mm if opts.autocrop and not opts.crop else None
        ),
        "apply_rescale": opts.apply_rescale,
        "compression_level": opts.compression_level,
        "save_dtype": opts.save_dtype,
        "gauss_sigma": opts.gauss_sigma,
        "nlm_patch": opts.nlm_patch,
        "nlm_search": opts.nlm_search,
        "nlm_h_factor": opts.nlm_h,
        "nlm_3d": opts.nlm_3d,
        "median_radius": opts.median_radius,
        "aniso_iterations": opts.aniso_iterations,
        "aniso_kappa": opts.aniso_kappa if opts.aniso_iterations > 0 else None,
        "aniso_gamma": opts.aniso_gamma if opts.aniso_iterations > 0 else None,
        "unsharp_amount": opts.unsharp_amount,
        "unsharp_sigma": opts.unsharp_sigma if opts.unsharp_amount > 0 else None,
        "rl_iterations": opts.rl_iterations,
        "rl_psf_sigma": opts.rl_psf_sigma if opts.rl_psf_path is None else None,
        "rl_psf_path": str(opts.rl_psf_path) if opts.rl_psf_path is not None else None,
        "tv_weight": opts.tv_weight,
        "enhance_contrast": opts.enhance_contrast,
        "clahe_clip_limit": opts.clahe_clip_limit if opts.enhance_contrast else None,
        "clahe_kernel_size": opts.clahe_kernel_size if opts.enhance_contrast else None,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info("Metadata → %s", meta_path)
    outputs["meta"] = meta_path

    return outputs


# ---------------------------------------------------------------------------
# Shared CLI
# ---------------------------------------------------------------------------

def add_pipeline_arguments(p: argparse.ArgumentParser) -> None:
    """Add every option in ``PipelineOptions`` to *p*.

    Both converters call this, so a flag's name, default and help text are
    defined once — the guarantee behind "same parameters, same output". Each
    tool adds only its own input-format arguments on top.
    """
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes for parallel slice decoding and filtering")
    p.add_argument("--save-raw", action="store_true",
                   help="Also write <stem>_prefiltered.nrrd: the volume after "
                        "the crop, binning and reorientation requested below "
                        "but before any filter runs. Same size and frame as "
                        "<stem>.nrrd, so the two compare voxel-for-voxel. It is "
                        "also reused as a resume cache on a later run, skipping "
                        "the source import, the crop prompt and the geometric "
                        "stages already baked into it")
    p.add_argument("--crop", action="store_true",
                   help="Interactively select a 3-D crop region (napari) before "
                        "filtering; all outputs reflect the crop.")
    p.add_argument("--autocrop", action="store_true",
                   help="Crop to the specimen automatically, with no napari "
                        "prompt: the volume is Otsu-thresholded, the largest "
                        "connected component kept (discarding holder and "
                        "debris), and its axis-aligned bounding box plus "
                        "--autocrop-margin-mm taken. Runs before binning, so "
                        "every later stage works on the reduced volume. "
                        "Ignored when --crop is given")
    p.add_argument("--autocrop-margin-mm", type=float, default=1.0, metavar="MM",
                   help="Padding kept around the detected specimen with --autocrop")
    p.add_argument("--autocrop-sample-dim", type=int, default=192, metavar="N",
                   help="Max voxels per axis in the strided copy used to detect "
                        "the specimen for --autocrop. Higher is slower and "
                        "rarely moves a millimetre-scale bounding box")

    g = p.add_argument_group(
        "Binning (downsampling)",
        "Averages blocks of voxels into one to reach a coarser target voxel "
        "size, applied right after the crop so every filter and every output "
        "file uses the reduced volume. Cuts file size and filter runtime by the "
        "product of the resulting bin factors (e.g. 10 um -> 20 um is 8x "
        "smaller) and raises SNR by sqrt(N), at the cost of resolution. Voxel "
        "spacing in the NRRD header is scaled to match.",
    )
    g.add_argument("--target-voxel-um", type=float, default=None, metavar="UM",
                   help="Bin the volume to approximately this voxel size in "
                        "micrometres (e.g. 20). Per-axis integer bin factors are "
                        "derived from the scan's native spacing; since only whole "
                        "voxels can be combined, the closest achievable size is "
                        "used and reported. Targets at or below the native size "
                        "leave the volume unchanged")
    g.add_argument("--bin-method", choices=["mean", "max"], default="mean",
                   help="'mean' averages each block (denoises, preserves intensities); "
                        "'max' keeps the block's peak value, preserving thin "
                        "high-density structures like enamel that averaging would dilute")

    g = p.add_argument_group(
        "Reorientation (long axis)",
        "Rotates the volume so array axis Z runs along the specimen's long "
        "axis instead of whatever direction the scanner stacked slices in. The "
        "frame comes from the minimum-volume oriented bounding box of the "
        "Otsu-segmented specimen: Z is the box's longest axis, Y the next, X "
        "the shortest. Applied after binning, so every output file "
        "(raw/filtered/enhanced) shares the new orientation, and the NRRD "
        "space directions/origin are updated so the result still overlays the "
        "original scan in Slicer/ITK-SNAP. Voxel size is preserved - the "
        "spacings travel with the axes - unless --reorient-voxel-mm asks "
        "otherwise.",
    )
    g.add_argument("--reorient", action="store_true",
                   help="Reorient the volume onto the specimen's long axis "
                        "(tip -> base along +Z for an incisor)")
    g.add_argument("--reorient-tip-at", choices=["start", "end"], default="start",
                   help="Which end of the new Z axis the tapered (incisal tip) "
                        "end goes to. The thin end is identified by comparing "
                        "specimen cross-section at the two ends of the long axis")
    g.add_argument("--reorient-order", type=int, default=1, choices=range(0, 6),
                   metavar="0-5",
                   help="Spline interpolation order for the rotation resample. "
                        "0 = nearest (no new values, blocky), 1 = trilinear "
                        "(default), 3 = cubic (sharper, slower). Unused when the "
                        "rotation snaps to a pure axis swap")
    g.add_argument("--reorient-snap-deg", type=float, default=5.0, metavar="DEG",
                   help="If the long axis is within this many degrees of an "
                        "existing array axis, reorient by transpose/flip instead "
                        "of resampling - exact, no interpolation blur. 0 forces "
                        "a resample always")
    g.add_argument("--reorient-tight", action="store_true",
                   help="Crop the reoriented volume to the specimen's bounding "
                        "box plus --reorient-margin-mm, instead of keeping the "
                        "full field of view")
    g.add_argument("--reorient-margin-mm", type=float, default=1.0, metavar="MM",
                   help="Padding kept around the bounding box with --reorient-tight")
    g.add_argument("--reorient-voxel-mm", type=float, default=None, metavar="MM",
                   help="Resample the reoriented volume to this isotropic voxel "
                        "size. By default voxel size is preserved: each new axis "
                        "keeps the spacing of the original axis it is most nearly "
                        "aligned with, so reorienting never silently changes "
                        "resolution. Ignored when the rotation snaps to a pure "
                        "axis swap, which is always exact")
    g.add_argument("--reorient-sample-dim", type=int, default=192, metavar="N",
                   help="Max voxels per axis in the strided copy used to segment "
                        "the specimen and fit the bounding box. Higher is slower "
                        "and rarely changes a centimetre-scale axis")
    g.add_argument("--reorient-mask-smooth-mm", type=float, default=0.06, metavar="MM",
                   help="Gaussian sigma used to smooth the specimen mask before "
                        "it is thresholded, in mm. A tapered tip is thin enough "
                        "that thresholding breaks it into hundreds of specks, "
                        "which the largest-connected-component step then discards "
                        "as debris - cropping the tip off with --reorient-tight. "
                        "Smoothing bridges those gaps. Affects only the mask used "
                        "to find the bounding box, never the saved voxel data. "
                        "Set 0 to disable")

    g = p.add_argument_group(
        "Output encoding",
        "Controls how NRRD files are written. Writing is often the slowest "
        "step: pynrrd's default gzip level 9 is single-threaded and gives only "
        "a few percent size gain over level 1 on noisy microCT data.",
    )
    g.add_argument("--compression-level", type=int, default=1, choices=range(0, 10),
                   metavar="0-9",
                   help="gzip level for NRRD output. 0 = uncompressed raw "
                        "(fastest write, largest file); 1 = fast compression "
                        "(default); 9 = smallest but 10-20x slower to write")
    g.add_argument("--save-dtype", choices=["float32", "int16", "uint16"], default="float32",
                   help="Voxel dtype on disk. int16/uint16 halve file size versus "
                        "float32; values are rounded, and out-of-range values are "
                        "clipped with a warning")

    g = p.add_argument_group(
        "Noise-floor threshold",
        "Flatten the low-intensity air band to a constant before any filter "
        "runs. Voxels at or above the level keep their exact original values, "
        "so this removes background noise without touching the specimen.",
    )
    g.add_argument("--noise-floor", type=float, default=None, metavar="LEVEL",
                   help="Clear voxels below LEVEL. Interpreted per "
                        "--noise-floor-method: an absolute intensity by "
                        "default, or a percentile of the histogram. Omitted "
                        "(the default), no thresholding is done and the raw "
                        "stored values pass through untouched")
    g.add_argument("--noise-floor-method", choices=["absolute", "percentile", "otsu"],
                   default="absolute",
                   help="How --noise-floor is read: 'absolute' = an intensity "
                        "in the volume's own units; 'percentile' = a "
                        "percentile of the intensity histogram (0-100); "
                        "'otsu' = find the specimen/air split automatically "
                        "and ignore --noise-floor (aggressive — it also "
                        "removes low-density soft tissue and pulp)")
    g.add_argument("--noise-floor-mode", choices=["floor", "shift"], default="floor",
                   help="'floor' keeps surviving voxels at their true values "
                        "(safe for densitometry); 'shift' also subtracts the "
                        "level so the specimen starts at zero, which rescales "
                        "the data and breaks absolute-density calibration")
    g.add_argument("--noise-floor-fill", type=float, default=None, metavar="VALUE",
                   help="Value written into the cleared voxels "
                        "(default: the threshold level in 'floor' mode, 0 in "
                        "'shift' mode)")

    g = p.add_argument_group("Gaussian pre-smoother")
    g.add_argument("--gauss-sigma", type=float, default=0,
                   help="Gaussian σ in voxels; 0 to disable")

    g = p.add_argument_group("Non-local means filter")
    g.add_argument("--nlm-patch", type=int, default=5,
                   help="Half-size of NLM comparison patch (pixels)")
    g.add_argument("--nlm-search", type=int, default=11,
                   help="Half-size of NLM search window (pixels)")
    g.add_argument("--nlm-h", type=float, default=1.0,
                   help="NLM filter strength as multiple of estimated noise σ; "
                        "higher → smoother (more blurring). 0 disables NLM, "
                        "turning the run into a plain conversion")
    g.add_argument("--nlm-3d", action="store_true",
                   help="Run NLM in true 3-D mode (higher quality, much more RAM)")

    g = p.add_argument_group("Median filter")
    g.add_argument("--median-radius", type=int, default=0,
                   help="Median filter half-kernel size (voxels); 0 to disable")

    g = p.add_argument_group(
        "Anisotropic diffusion (Perona-Malik)",
        "Runs after Gaussian/NLM/median, before RL deconvolution. Unlike those "
        "isotropic smoothers, diffusion is throttled across strong gradients, "
        "so it keeps flattening noise inside dentin/enamel while leaving the "
        "boundary between them sharp.",
    )
    g.add_argument("--aniso-iterations", type=int, default=0,
                   help="Number of diffusion iterations; 0 disables. Typical "
                        "range: 5-20 — more iterations flatten regions further "
                        "and make boundaries stand out more, at the cost of "
                        "fine texture")
    g.add_argument("--aniso-kappa", type=float, default=50.0,
                   help="Conduction threshold in the volume's own intensity "
                        "units: gradients above it are preserved as edges, "
                        "below it are smoothed. Must be set relative to the "
                        "actual enamel/dentin intensity step in your data")
    g.add_argument("--aniso-gamma", type=float, default=0.1,
                   help="Diffusion rate per iteration. Keep at or below 0.143 "
                        "(1/7) for a 6-neighbour 3-D stencil; higher values "
                        "make the update unstable")

    g = p.add_argument_group(
        "Unsharp mask",
        "Runs last, after every denoising stage. Adds back the high-frequency "
        "content a Gaussian blur would remove, steepening the intensity ramp "
        "across each boundary — which is what a downstream gradient/Sobel edge "
        "detector responds to. Amplifies any remaining noise by the same "
        "factor, so denoise first.",
    )
    g.add_argument("--unsharp-amount", type=float, default=0.0,
                   help="Sharpening strength; 0 disables. Typical range: "
                        "0.5-1.5 — higher values produce haloes at the enamel "
                        "surface that can read as a false second edge")
    g.add_argument("--unsharp-sigma", type=float, default=1.0,
                   help="Gaussian σ (voxels) defining which detail counts as "
                        "'high frequency'. Roughly match the width of the "
                        "boundary ramp you want to sharpen")

    g = p.add_argument_group(
        "Richardson-Lucy deconvolution",
        "Runs after Gaussian/NLM/median, before TV denoise. Recovers detail "
        "lost to the imaging PSF; sensitive to residual noise, so it must "
        "follow denoising rather than precede it.",
    )
    g.add_argument("--rl-iterations", type=int, default=0,
                   help="Number of Richardson-Lucy iterations; 0 disables. "
                        "Typical range: 5-20 — more iterations sharpen further "
                        "but amplify noise/ringing")
    g.add_argument("--psf-sigma", type=float, default=1.0, dest="rl_psf_sigma",
                   help="Sigma (voxels) of the synthetic Gaussian PSF used when "
                        "--psf-file is not given")
    g.add_argument("--psf-file", type=Path, default=None, dest="rl_psf_path",
                   help="Path to a measured PSF volume (.nrrd) to use instead "
                        "of the synthetic Gaussian PSF")

    g = p.add_argument_group(
        "TV (Chambolle) denoise",
        "Runs last in the filter pipeline, after Gaussian/NLM/median/RL-deconv. "
        "Enforces piecewise-constant regions, sharpening the bone/background "
        "boundary for cleaner histogram peaks during thresholding.",
    )
    g.add_argument("--tv-weight", type=float, default=0.0,
                   help="TV denoising strength; 0 disables. Typical range for "
                        "normalized microCT data: 0.05-0.3")

    g = p.add_argument_group(
        "CLAHE contrast enhancement",
        "Writes a SEPARATE <stem>_enhanced.nrrd for viewing only; the main "
        "<stem>.nrrd always stays at raw intensities. Boosts local contrast "
        "(e.g. enamel/dentin boundary) without crushing the rest of the volume "
        "the way percentile stretching would.",
    )
    g.add_argument("--enhance-contrast", action="store_true",
                   help="Also write a CLAHE-enhanced <stem>_enhanced.nrrd")
    g.add_argument("--clahe-clip-limit", type=float, default=0.01,
                   help="CLAHE clip limit; higher → stronger local contrast boost")
    g.add_argument("--clahe-kernel-size", type=int, default=None,
                   help="CLAHE tile size in pixels (default: skimage auto, ~1/8 of slice)")

    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")


def options_from_args(
    args: argparse.Namespace, *, apply_rescale: bool | None = None
) -> PipelineOptions:
    """Build a ``PipelineOptions`` from a parser that used ``add_pipeline_arguments``.

    Each format spells its stored-value → physical-units conversion differently
    (DICOM ``--apply-rescale``, ISQ ``--apply-scaling``), but both CLIs accept
    either spelling and store it as ``args.apply_rescale``, so it is read off
    *args* like every other option. Pass *apply_rescale* explicitly only to
    override that. It is recorded in the metadata either way, so both tools
    document the choice identically.
    """
    if apply_rescale is None:
        apply_rescale = bool(getattr(args, "apply_rescale", False))

    return PipelineOptions(
        workers=args.workers,
        apply_rescale=apply_rescale,
        save_raw=args.save_raw,
        crop=args.crop,
        autocrop=args.autocrop,
        autocrop_margin_mm=args.autocrop_margin_mm,
        autocrop_sample_dim=args.autocrop_sample_dim,
        target_voxel_um=args.target_voxel_um,
        bin_method=args.bin_method,
        reorient=args.reorient,
        reorient_order=args.reorient_order,
        reorient_snap_deg=args.reorient_snap_deg,
        reorient_tight=args.reorient_tight,
        reorient_margin_mm=args.reorient_margin_mm,
        reorient_tip_at=args.reorient_tip_at,
        reorient_voxel_mm=args.reorient_voxel_mm,
        reorient_sample_dim=args.reorient_sample_dim,
        reorient_mask_smooth_mm=args.reorient_mask_smooth_mm,
        compression_level=args.compression_level,
        save_dtype=args.save_dtype,
        noise_floor=args.noise_floor,
        noise_floor_method=args.noise_floor_method,
        noise_floor_mode=args.noise_floor_mode,
        noise_floor_fill=args.noise_floor_fill,
        gauss_sigma=args.gauss_sigma,
        nlm_patch=args.nlm_patch,
        nlm_search=args.nlm_search,
        nlm_h=args.nlm_h,
        nlm_3d=args.nlm_3d,
        median_radius=args.median_radius,
        aniso_iterations=args.aniso_iterations,
        aniso_kappa=args.aniso_kappa,
        aniso_gamma=args.aniso_gamma,
        unsharp_amount=args.unsharp_amount,
        unsharp_sigma=args.unsharp_sigma,
        rl_iterations=args.rl_iterations,
        rl_psf_sigma=args.rl_psf_sigma,
        rl_psf_path=args.rl_psf_path,
        tv_weight=args.tv_weight,
        enhance_contrast=args.enhance_contrast,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_kernel_size=args.clahe_kernel_size,
    )


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def report_outputs(outputs: dict[str, Path]) -> None:
    log.info("[bold green]Done.[/bold green]  Output files:")
    for label, path in outputs.items():
        log.info("  [cyan]%s[/cyan]: %s", label, path)
