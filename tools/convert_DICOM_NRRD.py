"""
Convert a UCSF microCT DICOM series to a filtered .nrrd volume.

Usage:
    python convert_DICOM_NRRD.py <dicom_dir> [options]

Filters applied in order:
    1. Gaussian pre-smoothing   — suppress isolated hot pixels before NLM
    2. Non-local means (NLM)    — edge-preserving noise reduction
    3. Median filter            — remove remaining salt-and-pepper noise

Raw stored pixel values are preserved — no rescaling or normalization is
applied, so window/level can be adjusted freely in downstream tools.

Output:
    <stem>.nrrd          — filtered volume with voxel spacing embedded
    <stem>_raw.nrrd      — unfiltered volume (optional, --save-raw)
    <stem>_enhanced.nrrd — CLAHE-boosted viewing copy (optional, --enhance-contrast)
    <stem>_meta.json     — DICOM metadata from the first slice

Notes:
    - Voxel spacing (PixelSpacing + SliceThickness) is written into the NRRD
      header so downstream tools (3D Slicer, ITK-SNAP) load it correctly.
    - NLM is run patch-by-patch in 2-D slice mode to keep RAM manageable on
      large volumes; use --nlm-3d for true 3-D NLM at higher memory cost.
    - All filter parameters are tunable via CLI flags.

If the output looks too blurry:
    - Lower --nlm-h (try 0.5–0.7 instead of the default 1.0) — this is the
      strongest smoothing knob.
    - Set --gauss-sigma 0 to skip the pre-NLM Gaussian smoothing entirely.
    - Set --median-radius 0 to skip the median filter.
    - These can be combined, e.g.:
        python convert_DICOM_NRRD.py <dicom_dir> --gauss-sigma 0 --nlm-h 0.6 --median-radius 0

To make enamel more visible without altering the analysis-ready <stem>.nrrd:
    - Pass --enhance-contrast to additionally write <stem>_enhanced.nrrd with
      slice-wise CLAHE applied. This boosts local contrast at the
      enamel/dentin/bone boundaries without crushing the rest of the dynamic
      range the way percentile stretching does. Use --clahe-clip-limit to
      tune strength (higher = more aggressive, default 0.01).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import nrrd
import pydicom
from pydicom.multival import MultiValue as DicomMultiValue
from rich.logging import RichHandler
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DICOM I/O
# ---------------------------------------------------------------------------

def _collect_dicom_files(dicom_dir: Path) -> list[Path]:
    files: list[Path] = []
    candidates = [p for p in dicom_dir.iterdir() if p.is_file()]
    for p in tqdm(candidates, desc="Scanning", unit="file", leave=False):
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True)
            files.append(p)
        except Exception:
            pass
    return files


def _sort_key(ds: pydicom.Dataset) -> float:
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        pass
    try:
        return int(ds.InstanceNumber)
    except Exception:
        return 0.0


def _decode_slice(args: tuple[int, str, bool]) -> tuple[int, np.ndarray]:
    """Picklable worker for ProcessPoolExecutor."""
    idx, path_str, apply_rescale = args
    ds = pydicom.dcmread(path_str)
    if apply_rescale:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
    else:
        arr = ds.pixel_array.astype(np.float32)
    return idx, arr


_SKIP_TAGS = {"PixelData", "FloatPixelData", "DoubleFloatPixelData"}


def _extract_metadata(ds: pydicom.Dataset) -> dict:
    meta: dict = {}
    for elem in ds:
        if elem.keyword in _SKIP_TAGS or elem.tag.group == 0x7FE0:
            continue
        try:
            val = elem.value
            if isinstance(val, pydicom.Sequence):
                val = f"<Sequence length={len(val)}>"
            elif isinstance(val, bytes):
                val = val.hex()
            elif isinstance(val, DicomMultiValue):
                val = list(val)
            elif not isinstance(val, (int, float, str, bool)):
                val = str(val)
            meta[elem.keyword or str(elem.tag)] = val
        except Exception:
            pass
    return meta


def _read_spacing(headers: list[pydicom.Dataset]) -> tuple[float, float, float]:
    """Return (dz, dy, dx) in mm.  Falls back to 1.0 on missing tags."""
    ds = headers[0]
    try:
        dy, dx = (float(v) for v in ds.PixelSpacing)
    except Exception:
        dx = dy = 1.0

    # SliceThickness is the nominal value; prefer computed spacing from
    # ImagePositionPatient when available (more accurate for microCT).
    dz: float = 1.0
    try:
        if len(headers) >= 2:
            z0 = float(headers[0].ImagePositionPatient[2])
            z1 = float(headers[1].ImagePositionPatient[2])
            dz = abs(z1 - z0)
        else:
            dz = float(ds.SliceThickness)
    except Exception:
        try:
            dz = float(ds.SliceThickness)
        except Exception:
            dz = 1.0

    return dz, dy, dx


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

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
    """
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


def _median_filter(volume: np.ndarray, radius: int) -> np.ndarray:
    """3-D median filter to remove residual salt-and-pepper noise."""
    size = 2 * radius + 1
    log.info("Median filter (size=%d)…", size)
    return median_filter(volume, size=size).astype(np.float32)


def _clahe_enhance(volume: np.ndarray, clip_limit: float, kernel_size: int | None) -> np.ndarray:
    """Slice-wise CLAHE for a viewing copy with enamel made visually distinct.

    Unlike percentile contrast stretching, CLAHE boosts *local* contrast so the
    enamel/dentin/bone boundary becomes visible without crushing the rest of
    the dynamic range. Applied per-slice (axial) since skimage's CLAHE expects
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
    normed = (volume - vmin) / (vmax - vmin)
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


# ---------------------------------------------------------------------------
# NRRD writer
# ---------------------------------------------------------------------------

def _write_nrrd(path: Path, volume: np.ndarray, spacing_zyx: tuple[float, float, float]) -> None:
    """Write *volume* to NRRD with physical voxel spacing encoded in the header.

    NRRD convention: space directions encodes the column-step vectors
    (right-hand, LPS or RAS).  We write a diagonal matrix — pure scaling,
    no shear or rotation — so tools that read NRRD spacing work correctly.
    """
    dz, dy, dx = spacing_zyx
    header = {
        "space": "left-posterior-superior",
        "space directions": [
            [dz, 0.0, 0.0],
            [0.0, dy, 0.0],
            [0.0, 0.0, dx],
        ],
        "space origin": [0.0, 0.0, 0.0],
        "kinds": ["domain", "domain", "domain"],
    }
    log.info("Writing NRRD → %s", path)
    nrrd.write(str(path), volume, header)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def convert(
    dicom_dir: Path,
    output_dir: Path,
    *,
    workers: int | None = None,
    apply_rescale: bool = False,
    save_raw: bool = False,
    # Gaussian pre-smoother
    gauss_sigma: float = 0.5,
    # NLM
    nlm_patch: int = 5,
    nlm_search: int = 11,
    nlm_h: float = 1.0,
    nlm_3d: bool = False,
    # Median
    median_radius: int = 1,
    # CLAHE contrast enhancement (viewing copy only)
    enhance_contrast: bool = False,
    clahe_clip_limit: float = 0.01,
    clahe_kernel_size: int | None = None,
) -> dict[str, Path]:
    dicom_dir = dicom_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dicom_dir.name
    outputs: dict[str, Path] = {}

    # 1. Discover DICOM files
    log.info("Scanning %s for DICOM files…", dicom_dir)
    files = _collect_dicom_files(dicom_dir)
    if not files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    log.info("Found %d slices.", len(files))

    # 2. Read and sort headers
    log.info("Reading and sorting headers…")
    headers: list[pydicom.Dataset] = []
    path_map: dict[int, Path] = {}
    for p in tqdm(files, desc="Reading headers", unit="file"):
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        path_map[id(ds)] = p
        headers.append(ds)
    headers.sort(key=_sort_key)
    sorted_paths = [path_map[id(ds)] for ds in headers]

    spacing_zyx = _read_spacing(headers)
    log.info("Voxel spacing (Z×Y×X): %.4f × %.4f × %.4f mm", *spacing_zyx)

    # 3. Decode pixels in parallel
    first_ds = pydicom.dcmread(str(sorted_paths[0]))
    rows, cols = first_ds.pixel_array.shape
    n_slices = len(sorted_paths)
    log.info("Assembling volume (%d × %d × %d)…", n_slices, rows, cols)

    volume = np.empty((n_slices, rows, cols), dtype=np.float32)
    decode_args = [(i, str(p), apply_rescale) for i, p in enumerate(sorted_paths)]
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(_decode_slice, a) for a in decode_args]
        with tqdm(total=n_slices, desc="Decoding slices", unit="slice") as pbar:
            for fut in as_completed(futures):
                idx, arr = fut.result()
                volume[idx] = arr
                pbar.update()

    # 4. Optionally save raw volume
    if save_raw:
        raw_path = output_dir / f"{stem}_raw.nrrd"
        _write_nrrd(raw_path, volume, spacing_zyx)
        outputs["raw"] = raw_path

    # 5. Apply filter pipeline
    volume = _gaussian_presmoother(volume, gauss_sigma)
    volume = _nlm_filter(volume, nlm_patch, nlm_search, nlm_h, nlm_3d, workers)
    volume = _median_filter(volume, median_radius)

    # 6. Write filtered NRRD
    nrrd_path = output_dir / f"{stem}.nrrd"
    _write_nrrd(nrrd_path, volume, spacing_zyx)
    outputs["filtered"] = nrrd_path

    # 6b. Optional CLAHE-enhanced viewing copy (enamel visibility), separate file
    if enhance_contrast:
        enhanced = _clahe_enhance(volume, clahe_clip_limit, clahe_kernel_size)
        enhanced_path = output_dir / f"{stem}_enhanced.nrrd"
        _write_nrrd(enhanced_path, enhanced, spacing_zyx)
        outputs["enhanced"] = enhanced_path
        del enhanced

    # 7. Save metadata
    meta_path = output_dir / f"{stem}_meta.json"
    meta = _extract_metadata(headers[0])
    meta["_spacing_mm"] = {"z": spacing_zyx[0], "y": spacing_zyx[1], "x": spacing_zyx[2]}
    meta["_filter_params"] = {
        "gauss_sigma": gauss_sigma,
        "nlm_patch": nlm_patch,
        "nlm_search": nlm_search,
        "nlm_h_factor": nlm_h,
        "nlm_3d": nlm_3d,
        "median_radius": median_radius,
        "enhance_contrast": enhance_contrast,
        "clahe_clip_limit": clahe_clip_limit if enhance_contrast else None,
        "clahe_kernel_size": clahe_kernel_size if enhance_contrast else None,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info("Metadata → %s", meta_path)
    outputs["meta"] = meta_path

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a UCSF microCT DICOM series to a filtered .nrrd volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dicom_dir", type=Path, help="Directory containing DICOM files")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: same parent as dicom_dir)")
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes for parallel slice decoding")
    p.add_argument("--apply-rescale", action="store_true",
                   help="Apply DICOM RescaleSlope/Intercept before filtering")
    p.add_argument("--save-raw", action="store_true",
                   help="Also write an unfiltered <stem>_raw.nrrd")

    g = p.add_argument_group("Gaussian pre-smoother")
    g.add_argument("--gauss-sigma", type=float, default=0.5,
                   help="Gaussian σ in voxels; 0 to disable")

    g = p.add_argument_group("Non-local means filter")
    g.add_argument("--nlm-patch", type=int, default=5,
                   help="Half-size of NLM comparison patch (pixels)")
    g.add_argument("--nlm-search", type=int, default=11,
                   help="Half-size of NLM search window (pixels)")
    g.add_argument("--nlm-h", type=float, default=1.0,
                   help="NLM filter strength as multiple of estimated noise σ; "
                        "higher → smoother (more blurring)")
    g.add_argument("--nlm-3d", action="store_true",
                   help="Run NLM in true 3-D mode (higher quality, much more RAM)")

    g = p.add_argument_group("Median filter")
    g.add_argument("--median-radius", type=int, default=1,
                   help="Median filter half-kernel size (voxels); 0 to disable")

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
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )

    dicom_dir: Path = args.dicom_dir
    if not dicom_dir.is_dir():
        log.error("'%s' is not a directory.", dicom_dir)
        sys.exit(1)

    output_dir = args.out if args.out else dicom_dir.parent

    outputs = convert(
        dicom_dir,
        output_dir,
        workers=args.workers,
        apply_rescale=args.apply_rescale,
        save_raw=args.save_raw,
        gauss_sigma=args.gauss_sigma,
        nlm_patch=args.nlm_patch,
        nlm_search=args.nlm_search,
        nlm_h=args.nlm_h,
        nlm_3d=args.nlm_3d,
        median_radius=args.median_radius,
        enhance_contrast=args.enhance_contrast,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_kernel_size=args.clahe_kernel_size,
    )

    log.info("[bold green]Done.[/bold green]  Output files:")
    for label, path in outputs.items():
        log.info("  [cyan]%s[/cyan]: %s", label, path)


if __name__ == "__main__":
    main()
