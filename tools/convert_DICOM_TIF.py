"""
Convert a DICOM series directory to a BigTIFF volume.

Usage:
    python convert_DICOM_TIF.py <dicom_dir> [--out <output_dir>] [--workers N] [-v]

Output:
    <dicom_dir_name>.tif       — 3-D BigTIFF volume (slices × rows × cols)
    <dicom_dir_name>_meta.json — metadata from the first DICOM slice

Notes:
    - Pixel values are preserved in their stored DICOM form by default (raw
      stored integers, no RescaleSlope/RescaleIntercept applied, no VOI LUT).
      Use --apply-rescale to convert to physical units and store float32.
    - BigTIFF supports files > 4 GB; pixel spacing is embedded as TIFF
      resolution tags (CENTIMETER units).
    - Slice decoding is parallelized; use --workers to cap the process count.
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
import tifffile
from tifffile import RESUNIT
from pydicom.multival import MultiValue as DicomMultiValue
from rich.logging import RichHandler
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _collect_dicom_files(dicom_dir: Path) -> list[Path]:
    files = []
    candidates = [p for p in dicom_dir.iterdir() if p.is_file()]
    for p in tqdm(candidates, desc="Scanning", unit="file", leave=False):
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True)
            files.append(p)
        except Exception:
            pass
    return files


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Volume assembly
# ---------------------------------------------------------------------------

def _sort_key(ds: pydicom.Dataset):
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        pass
    try:
        return int(ds.InstanceNumber)
    except Exception:
        return 0


def _read_raw_pixels(ds: pydicom.Dataset) -> np.ndarray:
    """Return stored pixel integers with no rescale or VOI transforms applied.

    pydicom 3.x ``pixel_array(raw=True)`` skips RescaleSlope/Intercept, VOI
    LUTs, and palette-colour expansion, returning exactly the integers stored
    in the DICOM file.  This is the correct path for preserving raw intensity
    values.
    """
    from pydicom.pixels.utils import pixel_array as _pa
    return _pa(ds, raw=True)


def _decode_slice(args: tuple[int, str, bool]) -> tuple[int, np.ndarray]:
    # Module-level so it is picklable for ProcessPoolExecutor
    idx, path_str, apply_rescale = args
    ds = pydicom.dcmread(path_str)
    if apply_rescale:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
    else:
        arr = _read_raw_pixels(ds)
    return idx, arr


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    dicom_dir: Path,
    output_dir: Path,
    workers: int | None = None,
    apply_rescale: bool = False,
) -> tuple[Path, Path]:
    """Convert *dicom_dir* → BigTIFF + JSON metadata in *output_dir*."""
    dicom_dir = dicom_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = dicom_dir.name
    tif_path = output_dir / f"{stem}.tif"
    meta_path = output_dir / f"{stem}_meta.json"

    # 1. Discover DICOM files
    log.info("Scanning %s for DICOM files…", dicom_dir)
    files = _collect_dicom_files(dicom_dir)
    if not files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    log.info("Found %d slices.", len(files))

    # 2. Read headers only for sorting (no pixel data)
    log.info("Reading headers from %d files…", len(files))
    headers: list[pydicom.Dataset] = []
    path_map: dict[int, Path] = {}
    with tqdm(total=len(files), desc="Reading headers", unit="file") as pbar:
        for p in files:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            path_map[id(ds)] = p
            headers.append(ds)
            pbar.update()
    headers.sort(key=_sort_key)
    sorted_paths: list[Path] = [path_map[id(ds)] for ds in headers]

    # 3. Read dimensions from first slice
    first_ds = pydicom.dcmread(str(sorted_paths[0]))
    if apply_rescale:
        sample = first_ds.pixel_array.astype(np.float32)
        slope = float(getattr(first_ds, "RescaleSlope", 1.0))
        intercept = float(getattr(first_ds, "RescaleIntercept", 0.0))
        sample = sample * slope + intercept
    else:
        sample = _read_raw_pixels(first_ds)
    rows, cols = sample.shape
    n_slices = len(sorted_paths)
    bytes_per_voxel = np.dtype(sample.dtype).itemsize
    size_mb = rows * cols * n_slices * bytes_per_voxel / 1e6
    log.info("Assembling volume (%d x %d x %d = %.0f MB)…", n_slices, rows, cols, size_mb)

    # Use a memmap temp file — avoids holding the full volume in RAM
    tmp_path = tif_path.with_suffix(".tmp.dat")
    try:
        # Shape: (slices, rows, cols) — standard ZYX order for tifffile
        volume = np.memmap(tmp_path, dtype=sample.dtype, mode="w+", shape=(n_slices, rows, cols))
        volume[0] = sample
        decode_args = [(i, str(p), apply_rescale) for i, p in enumerate(sorted_paths[1:], start=1)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_decode_slice, a) for a in decode_args]
            with tqdm(total=n_slices, desc="Assembling volume", unit="slice", initial=1) as pbar:
                for future in as_completed(futures):
                    idx, arr = future.result()
                    volume[idx] = arr
                    pbar.update()
        volume.flush()

        # Extract pixel spacing for TIFF resolution metadata
        try:
            dy = float(headers[0].PixelSpacing[0])
            dx = float(headers[0].PixelSpacing[1])
        except Exception:
            dx = dy = 1.0

        log.info("Writing BigTIFF → %s", tif_path)
        tifffile.imwrite(
            str(tif_path),
            volume,
            bigtiff=True,
            resolution=(1 / dx, 1 / dy),
            resolutionunit=RESUNIT.MILLIMETER,
            metadata={"axes": "ZYX"},
        )
    finally:
        del volume
        if tmp_path.exists():
            tmp_path.unlink()

    meta = _extract_metadata(headers[0])
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info("Writing metadata → %s", meta_path)

    return tif_path, meta_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a DICOM series to BigTIFF + JSON metadata."
    )
    parser.add_argument("dicom_dir", type=Path, help="Directory containing DICOM files")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: same parent as dicom_dir)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process pool size for parallel decoding (default: os.cpu_count())",
    )
    parser.add_argument(
        "--apply-rescale",
        action="store_true",
        help="Apply DICOM RescaleSlope/RescaleIntercept and save float32 output",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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
    convert(dicom_dir, output_dir, workers=args.workers, apply_rescale=args.apply_rescale)
    log.info("Done.")


if __name__ == "__main__":
    main()
