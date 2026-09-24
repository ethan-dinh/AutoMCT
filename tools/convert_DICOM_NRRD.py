"""
Convert a UCSF microCT DICOM series to a filtered .nrrd volume.

Usage:
    python convert_DICOM_NRRD.py <dicom_dir> [options]

This tool is the DICOM front end for the shared conversion pipeline in
``volume_pipeline.py``; ``convert_ISQ_NRRD.py`` is the Scanco ISQ front end.
Everything after the volume is in memory — cropping, binning, long-axis
reorientation, the filter chain, CLAHE, NRRD geometry and metadata — is that
shared module, so the two tools accept the same options and produce identical
output for identical option values.

Filters applied in order:
    0. Noise-floor threshold    — flatten the air band to a constant, so no
                                   later stage spends itself on background
                                   noise (optional, --noise-floor)
    1. Gaussian pre-smoothing   — suppress isolated hot pixels before NLM
    2. Non-local means (NLM)    — edge-preserving noise reduction
    3. Median filter            — remove remaining salt-and-pepper noise
                                   (true 3-D, numba-JIT + multi-threaded — see Notes)
    4. Anisotropic diffusion    — edge-preserving smoothing: flattens noise
                                   within tissues but not across boundaries
                                   (optional, --aniso-iterations)
    5. Richardson-Lucy deconv   — recover detail lost to the imaging PSF
                                   (optional, --rl-iterations)
    6. TV (Chambolle) denoise   — piecewise-constant smoothing for cleaner
                                   thresholding boundaries (optional, --tv-weight)
    7. Unsharp mask             — high-pass boost that steepens boundary
                                   ramps (optional, --unsharp-amount)

Raw stored pixel values are preserved — no rescaling or normalization is
applied, so window/level can be adjusted freely in downstream tools. Pass
--apply-rescale to convert to physical units with the series' own
RescaleSlope/Intercept, matching what convert_ISQ_NRRD.py's --apply-scaling
does with the ISQ header's mu_scaling. (--apply-scaling is accepted here as
an alias, matching the ISQ converter's flag name.)

Output:
    <stem>_prefiltered.nrrd — cropped/binned/reoriented but unfiltered
                              (optional, --save-raw)
    <stem>.nrrd          — the same volume, filtered, with geometry embedded
    <stem>_enhanced.nrrd — CLAHE-boosted viewing copy (optional, --enhance-contrast)
    <stem>_meta.json     — DICOM metadata from the first slice

Notes:
    - Voxel spacing (PixelSpacing + SliceThickness) is written into the NRRD
      header so downstream tools (3D Slicer, ITK-SNAP) load it correctly.
    - NLM is run patch-by-patch in 2-D slice mode to keep RAM manageable on
      large volumes; use --nlm-3d for true 3-D NLM at higher memory cost.
      Pass --nlm-h 0 to skip NLM entirely and get a plain conversion.
    - The median filter is a custom numba @njit(parallel=True) 3-D sliding
      window (reflect boundary, same semantics as scipy.ndimage.median_filter)
      that scales across all CPU cores — several times faster than scipy's
      single-threaded implementation, with identical output.
    - All filter parameters are tunable via CLI flags.
    - Pass --crop to interactively select a 3-D bounding box (napari) right
      after DICOM assembly, before any filtering runs. This speeds up every
      later step (Gaussian, NLM, median, CLAHE all run on fewer voxels) and
      both output volumes (prefiltered and filtered) reflect the crop.
      The NRRD "space origin" is shifted by the crop offset, so a cropped
      volume still lands in the same physical position as the full scan when
      both are loaded into Slicer/ITK-SNAP.
    - Pass --target-voxel-um to bin the volume up to a coarser voxel size
      (e.g. --target-voxel-um 20). Per-axis integer bin factors are derived
      from the DICOM's native spacing; because binning combines whole
      voxels, the closest achievable size is used and reported. Applied
      right after the crop, so the filters run on the reduced volume and
      every output file shrinks by the same factor. Voxel spacing in the
      NRRD header is scaled to match, so geometry stays correct at the
      coarser resolution. Block averaging also raises SNR by sqrt(N).
    - Pass --reorient to rotate the volume onto the tooth's own long axis. The
      specimen is Otsu-segmented, its minimum-volume oriented bounding box is
      fitted, and the volume is resampled so that Z runs tip -> base along the
      long axis, Y along the second-longest box axis and X along the shortest.
      This fixes scans whose Z happens to cut buccal-lingual, which makes every
      slice-wise step downstream (montages, per-slice enamel area, CMPR) cut the
      tooth the wrong way. Runs after the crop and binning, so all outputs share
      the new orientation. When the long axis is already within
      --reorient-snap-deg of an array axis the rotation is done as a
      transpose/flip, so no interpolation blur is introduced at all. Voxel
      values keep their physical positions: "space directions"/"space origin"
      describe the rotated frame, so the result still overlays the original scan
      in Slicer/ITK-SNAP. Add --reorient-tight to also crop to the bounding box
      (plus --reorient-margin-mm), and --reorient-tip-at end to put the incisal
      tip at the last slice instead of the first.

If saving is slow:
    - NRRD writing defaults here to gzip level 1 (--compression-level).
      pynrrd's own default is level 9, which is single-threaded and 10-20x
      slower to write for only a few percent size gain on noisy microCT
      data. Use --compression-level 0 for uncompressed raw (fastest write,
      largest file), or 9 if archival size matters more than time.
    - --save-dtype int16 halves file size versus float32 (values are
      rounded; out-of-range values are clipped with a warning). As in the
      ISQ converter, this is lossless for an unfiltered, unbinned read of a
      16-bit-stored series without --apply-rescale, since the volume is
      assembled and kept at its native integer width in that case.
    - --target-voxel-um cuts both write time and file size by the product
      of the resulting bin factors (2x per axis = 8x smaller).

If the output looks too blurry:
    - Lower --nlm-h (try 0.5-0.7 instead of the default 1.0) — this is the
      strongest smoothing knob.
    - Set --gauss-sigma 0 to skip the pre-NLM Gaussian smoothing entirely.
    - Set --median-radius 0 to skip the median filter.
    - These can be combined, e.g.:
        python convert_DICOM_NRRD.py <dicom_dir> --gauss-sigma 0 --nlm-h 0.6 --median-radius 0

If the borders between tissues look soft or smeared:
    - The Gaussian, NLM and median stages are all isotropic — they blur an
      enamel/dentin boundary as hard as they blur noise. Reach for the
      edge-aware stages instead:
        --aniso-iterations 10 --aniso-kappa <edge step>   (needs tuning, below)
        --tv-weight 0.1
    - --aniso-kappa is in the volume's own intensity units, so read the actual
      enamel-to-dentin step off a line profile in Slicer and set kappa near
      it. Too low and real boundaries get smoothed; too high and it behaves
      like plain Gaussian blur.
    - Add --unsharp-amount 1.0 last to steepen what remains. Watch for haloes
      at the enamel surface — a bright rim just outside the true border is the
      sign it is set too high, and a gradient-based segmenter will latch onto
      that rim as a false edge.
    - --rl-iterations 10 sharpens by undoing the imaging PSF rather than by
      boosting high frequencies, so it is the more physically grounded option
      of the two, but it is slower and amplifies residual noise each iteration.
    - Boundary sharpness also depends on not throwing resolution away earlier:
      --target-voxel-um bins whole voxels together, so a boundary that fell
      inside one bin is permanently softened. Use --bin-method max to keep thin
      enamel from being diluted if you must bin.

To make enamel more visible without altering the analysis-ready <stem>.nrrd:
    - Pass --enhance-contrast to additionally write <stem>_enhanced.nrrd with
      slice-wise CLAHE applied. This boosts local contrast at the
      enamel/dentin/bone boundaries without crushing the rest of the dynamic
      range the way percentile stretching does. Use --clahe-clip-limit to
      tune strength (higher = more aggressive, default 0.01).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from pydicom.multival import MultiValue as DicomMultiValue
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from volume_pipeline import (  # noqa: E402  (needs the path insert above)
    PipelineOptions,
    add_pipeline_arguments,
    convert_volume,
    directions_from_spacing,
    options_from_args,
    report_outputs,
    setup_logging,
)

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


def _decode_slice(args: tuple[int, str, bool, str]) -> tuple[int, np.ndarray]:
    """Picklable worker for ProcessPoolExecutor.

    *dtype_name* is the dtype the caller pre-allocated the whole volume with;
    each slice is converted to it here so the assembling process never holds a
    second, wider copy.
    """
    idx, path_str, apply_rescale, dtype_name = args
    ds = pydicom.dcmread(path_str)
    if apply_rescale:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
    else:
        arr = ds.pixel_array
    return idx, arr.astype(np.dtype(dtype_name), copy=False)


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
    """Return (dz, dy, dx) in mm.  Falls back to 1.0 on missing tags.

    Used only as the fallback for ``_read_direction_matrix``, which does not
    assume slices are stacked purely along a fixed axis (see that function).
    """
    ds = headers[0]
    try:
        dy, dx = (float(v) for v in ds.PixelSpacing)
    except Exception:
        dx = dy = 1.0

    dz: float = 1.0
    try:
        if len(headers) >= 2:
            p0 = np.asarray(headers[0].ImagePositionPatient, dtype=float)
            p1 = np.asarray(headers[1].ImagePositionPatient, dtype=float)
            dz = float(np.linalg.norm(p1 - p0))
        else:
            dz = float(ds.SliceThickness)
    except Exception:
        try:
            dz = float(ds.SliceThickness)
        except Exception:
            dz = 1.0

    return dz, dy, dx


def _read_direction_matrix(headers: list[pydicom.Dataset]) -> np.ndarray:
    """
    Build the 3x3 NRRD "space directions" matrix (LPS) from actual DICOM
    orientation tags, instead of assuming a fixed axial acquisition.

    Row i of the returned matrix is the (L, P, S) direction vector for array
    axis i, scaled by that axis's physical spacing — i.e. exactly what
    ``nrrd.write`` expects for "space directions" when array axis order is
    (Z, Y, X).

    - Axis 2 (X / columns) and axis 1 (Y / rows) come from
      ``ImageOrientationPatient``'s row/column direction cosines, scaled by
      ``PixelSpacing`` (row spacing, column spacing).
    - Axis 0 (Z / slices) comes from the actual displacement between
      consecutive slices' ``ImagePositionPatient`` — this is the true slice
      direction and is correct even for gantry-tilted or non-axial series,
      unlike assuming a pure +S step.

    Falls back to identity axial (L, P, S) directions if orientation tags
    are missing, with a warning. That fallback is the same geometry the ISQ
    reader always uses, so an unoriented DICOM series and an ISQ scan of the
    same specimen land in the same frame.
    """
    ds = headers[0]
    try:
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        row_cosine = iop[0:3]   # direction of increasing column index (dx)
        col_cosine = iop[3:6]   # direction of increasing row index (dy)
        dy, dx = (float(v) for v in ds.PixelSpacing)

        x_vec = row_cosine * dx   # array axis 2 (X)
        y_vec = col_cosine * dy   # array axis 1 (Y)

        if len(headers) >= 2:
            p0 = np.asarray(headers[0].ImagePositionPatient, dtype=float)
            p1 = np.asarray(headers[1].ImagePositionPatient, dtype=float)
            z_vec = p1 - p0   # array axis 0 (Z) — true inter-slice step
        else:
            slice_cosine = np.cross(row_cosine, col_cosine)
            z_vec = slice_cosine * float(getattr(ds, "SliceThickness", 1.0))

        return np.array([z_vec, y_vec, x_vec], dtype=float)

    except Exception:
        log.warning(
            "Missing/invalid ImageOrientationPatient — assuming axial "
            "acquisition (rows=L, columns=P, slices=S). Non-axial or "
            "oblique scans will have incorrect NRRD orientation."
        )
        return directions_from_spacing(*_read_spacing(headers))


def read_dicom_series(
    dicom_dir: Path,
    *,
    apply_rescale: bool = False,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Assemble a sorted DICOM series into the pipeline's source tuple.

    Returns ``(volume [Z, Y, X], direction_matrix, origin, meta)``.

    As in the ISQ reader, the series' own stored dtype is kept when
    ``apply_rescale`` is off: every downstream stage promotes to float32 on
    demand, so an integer-stored series is assembled at its native width and
    peak memory is halved on a full-resolution scan without changing a single
    output value. With ``apply_rescale`` on, slope/intercept produce
    non-integer values, so the volume is float32.
    """
    log.info("Scanning %s for DICOM files…", dicom_dir)
    files = _collect_dicom_files(dicom_dir)
    if not files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    log.info("Found %d slices.", len(files))

    log.info("Reading and sorting headers…")
    headers: list[pydicom.Dataset] = []
    path_map: dict[int, Path] = {}
    for p in tqdm(files, desc="Reading headers", unit="file"):
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        path_map[id(ds)] = p
        headers.append(ds)
    headers.sort(key=_sort_key)
    sorted_paths = [path_map[id(ds)] for ds in headers]

    direction_matrix = _read_direction_matrix(headers)
    # Physical (LPS) position of voxel [0, 0, 0] — the first slice's
    # ImagePositionPatient. Carried through any crop so cropped outputs stay
    # registered to the original scan.
    try:
        origin = np.asarray(headers[0].ImagePositionPatient, dtype=float)
    except Exception:
        log.warning("Missing ImagePositionPatient — using origin [0, 0, 0].")
        origin = np.zeros(3)

    first_ds = pydicom.dcmread(str(sorted_paths[0]))
    first_arr = first_ds.pixel_array
    rows, cols = first_arr.shape
    n_slices = len(sorted_paths)

    # Rescaling makes values non-integer, so that path is float32. Otherwise
    # keep the stored integer width (see the docstring); a float stored type
    # is already float32-or-wider and needs no widening either.
    if apply_rescale or not np.issubdtype(first_arr.dtype, np.integer):
        volume_dtype = np.dtype(np.float32)
    else:
        volume_dtype = first_arr.dtype
    log.info(
        "Assembling volume (%d × %d × %d, %s)…", n_slices, rows, cols, volume_dtype,
    )

    volume = np.empty((n_slices, rows, cols), dtype=volume_dtype)
    decode_args = [
        (i, str(p), apply_rescale, volume_dtype.name) for i, p in enumerate(sorted_paths)
    ]
    mp_ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(_decode_slice, a) for a in decode_args]
        with tqdm(total=n_slices, desc="Decoding slices", unit="slice") as pbar:
            for fut in as_completed(futures):
                idx, arr = fut.result()
                volume[idx] = arr
                pbar.update()

    return volume, direction_matrix, origin, _extract_metadata(headers[0])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def convert(
    dicom_dir: Path,
    output_dir: Path,
    opts: PipelineOptions | None = None,
) -> dict[str, Path]:
    """Convert *dicom_dir* into *output_dir* using the shared pipeline."""
    opts = opts or PipelineOptions()
    dicom_dir = dicom_dir.resolve()

    return convert_volume(
        output_dir=output_dir,
        stem=dicom_dir.name,
        read_source=lambda: read_dicom_series(
            dicom_dir, apply_rescale=opts.apply_rescale, workers=opts.workers,
        ),
        opts=opts,
    )


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
    p.add_argument("--apply-rescale", "--apply-scaling", action="store_true",
                   dest="apply_rescale",
                   help="Apply DICOM RescaleSlope/Intercept before filtering, "
                        "converting stored values to physical units (the DICOM "
                        "counterpart of convert_ISQ_NRRD.py's --apply-scaling, "
                        "which is accepted here as an alias)")
    add_pipeline_arguments(p)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    setup_logging(args.verbose)

    dicom_dir: Path = args.dicom_dir
    if not dicom_dir.is_dir():
        log.error("'%s' is not a directory.", dicom_dir)
        sys.exit(1)

    output_dir = args.out if args.out else dicom_dir.parent

    outputs = convert(
        dicom_dir,
        output_dir,
        options_from_args(args, apply_rescale=args.apply_rescale),
    )
    report_outputs(outputs)


if __name__ == "__main__":
    main()
