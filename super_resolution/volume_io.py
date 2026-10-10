"""
Scans in and volumes out, with their physical geometry.

Loading goes through ``segment_mandible``'s own loaders, so every input the
pipeline accepts (.nrrd, .tif, a DICOM series, a BMP stack) is accepted here
too, in the same (Z, Y, X) orientation.

The upscaled volume is written as an NRRD whose header is derived from the
original's, so it sits exactly on top of the original scan in 3D Slicer: the
voxel size shrinks by the scale factor and the origin shifts by the sub-voxel
offset between the two grids. The file is written tile by tile through a
memory map, never held in RAM whole -- at 2x it is eight times the input.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass

import nrrd
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from .classical import add_segment_mandible_to_path

add_segment_mandible_to_path()

# pylint: disable=wrong-import-position,wrong-import-order
from data_io import (  # noqa: E402  (segment_mandible, importable once the path is set)
    is_dicom_dir,
    load_bmp_stack,
    load_dicom_series,
    load_nrrd,
    load_tiff,
)

logger = logging.getLogger(__name__)

LPS = "left-posterior-superior"

Spacing = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Geometry:
    """
    Where a (Z, Y, X) array sits in physical space, in NRRD terms (mm).

    ``directions[i]`` is the physical displacement of one step along array
    axis ``i`` -- rows are in the array's (Z, Y, X) order, *not* the on-disk
    (X, Y, Z) order NRRD headers list them in. ``origin`` is the physical
    position of the centre of voxel (0, 0, 0).
    """

    directions: np.ndarray
    origin: np.ndarray
    space: str = LPS

    @classmethod
    def from_spacing(cls, spacing_zyx: Spacing) -> "Geometry":
        """
        Axis-aligned geometry for an input that carries only a voxel size.

        Same convention as segment_mandible's writers (axis 0 -> S, axis 1 ->
        P, axis 2 -> L), so outputs overlay the pipeline's own outputs.
        """
        dz, dy, dx = (float(v) for v in spacing_zyx)
        directions = np.array([[0.0, 0.0, dz], [0.0, dy, 0.0], [dx, 0.0, 0.0]])
        return cls(directions, np.zeros(3))

    @classmethod
    def from_nrrd_header(cls, header: dict) -> "Geometry | None":
        """Geometry from an NRRD header, or None when it carries no spatial information."""
        rows = []
        directions = header.get("space directions")
        if directions is not None:
            for row in np.asarray(directions, dtype=object):
                # A "none" row marks a non-spatial axis (e.g. colour).
                if row is None or (isinstance(row, float) and np.isnan(row)):
                    continue
                row = np.asarray(row, dtype=float)
                if row.size == 3 and np.all(np.isfinite(row)):
                    rows.append(row)
        if len(rows) == 3 and all(np.linalg.norm(r) > 0 for r in rows):
            origin = np.asarray(header.get("space origin", np.zeros(3)), dtype=float)
            if origin.size != 3 or not np.all(np.isfinite(origin)):
                origin = np.zeros(3)
            space = str(header.get("space", LPS))
            # On-disk rows are (X, Y, Z); the arrays here are (Z, Y, X).
            return cls(np.asarray(rows[::-1], dtype=float), origin, space)

        spacings = header.get("spacings")
        if spacings is not None:
            values = np.asarray(spacings, dtype=float)
            if values.size == 3 and np.all(np.isfinite(values)) and np.all(values > 0):
                return cls.from_spacing((values[2], values[1], values[0]))
        return None

    @property
    def spacing(self) -> Spacing:
        """Voxel size (dz, dy, dx) in mm: the length of each axis step."""
        norms = np.linalg.norm(self.directions, axis=1)
        return (float(norms[0]), float(norms[1]), float(norms[2]))

    def with_spacing(self, spacing_zyx: Spacing) -> "Geometry":
        """Same orientation and origin, new voxel size (for --voxel-size overrides)."""
        unit = self.directions / np.linalg.norm(self.directions, axis=1, keepdims=True)
        return Geometry(unit * np.asarray(spacing_zyx, float)[:, None], self.origin.copy(), self.space)

    def upscaled(self, scale: tuple[int, int, int]) -> "Geometry":
        """
        Geometry of this grid subdivided ``scale`` times per axis.

        Each step shrinks by the factor. The origin moves because it marks a
        voxel *centre*: the first fine voxel's centre lies (s - 1) / 2s of a
        coarse step before the first coarse centre (at 2x, a quarter-voxel), so
        the two grids share the same outer boundary and overlay exactly.
        """
        s = np.asarray(scale, dtype=float)
        shift = (-(s - 1.0) / (2.0 * s)) @ self.directions
        return Geometry(self.directions / s[:, None], self.origin + shift, self.space)

    def shifted(self, start_zyx) -> "Geometry":
        """Geometry of a sub-volume whose first voxel is ``start_zyx`` of this one."""
        return Geometry(
            self.directions.copy(),
            self.origin + np.asarray(start_zyx, dtype=float) @ self.directions,
            self.space,
        )

    def nrrd_header(self) -> dict:
        """The spatial fields of an NRRD header for an array written in (X, Y, Z) disk order."""
        return {
            "space": self.space,
            "space directions": self.directions[::-1].tolist(),
            "space origin": self.origin.tolist(),
            "kinds": ["domain", "domain", "domain"],
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Scan:
    """A loaded scan: its voxels in (Z, Y, X) order and where they sit in space."""

    name: str
    path: str
    source: str
    data: np.ndarray
    geometry: Geometry
    spacing_from_file: bool


def sample_name(path: str) -> str:
    """Output name for a sample, derived exactly as segment_mandible/main.py does."""
    return pathlib.Path(path.rstrip("/\\")).stem


def resolve_scan_source(path: str) -> tuple[str, str]:
    """
    Decide how to read ``path``: returns (kind, file_or_dir).

    Same precedence as the pipeline for a directory -- .nrrd, then .tif, then
    a DICOM series, then a BMP stack -- so a folder resolves to the same data
    here as it does there.
    """
    if os.path.isfile(path):
        lower = path.lower()
        if lower.endswith(".nrrd"):
            return "nrrd", path
        if lower.endswith((".tif", ".tiff")):
            return "tif", path
        raise ValueError(f"Unsupported input file (expected .nrrd, .tif or .tiff): {path}")
    if os.path.isdir(path):
        nrrd_files = sorted(glob.glob(os.path.join(path, "*.nrrd")))
        if nrrd_files:
            return "nrrd", nrrd_files[0]
        tif_files = sorted(
            glob.glob(os.path.join(path, "*.tif")) + glob.glob(os.path.join(path, "*.tiff"))
        )
        if tif_files:
            return "tif", tif_files[0]
        if is_dicom_dir(path):
            return "dicom", path
        if glob.glob(os.path.join(path, "*.bmp")):
            return "bmp", path
        raise ValueError(f"No .nrrd, .tif, DICOM or .bmp data found in {path}")
    raise FileNotFoundError(path)


def load_scan(
    path: str,
    *,
    voxel_size: Spacing | None = None,
    apply_dicom_rescale: bool = False,
) -> Scan:
    """
    Load a scan with segment_mandible's loaders, plus its full geometry.

    ``voxel_size`` (dz, dy, dx in mm, input frame) overrides the file's
    spacing but keeps its orientation, matching the pipeline's --voxel-size.
    """
    kind, source = resolve_scan_source(path)
    geometry: Geometry | None = None
    spacing: Spacing | None = None

    if kind == "nrrd":
        data, spacing = load_nrrd(source, return_spacing=True)
        geometry = Geometry.from_nrrd_header(nrrd.read_header(source))
    elif kind == "tif":
        data, spacing = load_tiff(source, return_spacing=True)
    elif kind == "dicom":
        data, spacing = load_dicom_series(
            source, return_spacing=True, apply_rescale=apply_dicom_rescale,
        )
    else:
        data = load_bmp_stack(source, file_pattern="*.bmp", exclude_pattern="*spr.bmp")
    if data is None:
        raise IOError(f"Could not load {source}")
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {data.shape} from {source}")

    spacing_from_file = geometry is not None or spacing is not None
    if geometry is None:
        geometry = Geometry.from_spacing(spacing if spacing is not None else (1.0, 1.0, 1.0))
    if voxel_size is not None:
        geometry = geometry.with_spacing(voxel_size)
        spacing_from_file = True
    if not spacing_from_file:
        logger.warning(
            "%s carries no voxel size; assuming 1 mm isotropic. Pass --voxel-size "
            "to record the real one.", source,
        )

    return Scan(
        name=sample_name(path), path=path, source=source, data=data,
        geometry=geometry, spacing_from_file=spacing_from_file,
    )


def discover_samples(root: str) -> list[tuple[str, str]]:
    """
    (name, path) for each sample under ``root``, following main.py's rules.

    A file, or a folder that directly holds a BMP stack or DICOM series, is one
    sample. Anything else is a folder of samples: each sub-folder (except
    "Compressed") and each loose .nrrd/.tif/.tiff file is one.
    """
    if os.path.isfile(root) or (
        os.path.isdir(root) and (glob.glob(os.path.join(root, "*.bmp")) or is_dicom_dir(root))
    ):
        return [(sample_name(root), root)]
    samples = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if entry.startswith("."):
            continue
        if os.path.isdir(full) and entry != "Compressed":
            samples.append((entry, full))
        elif os.path.isfile(full) and entry.lower().endswith((".nrrd", ".tif", ".tiff")):
            samples.append((pathlib.Path(entry).stem, full))
    return samples


# ---------------------------------------------------------------------------
# Intensity handling
# ---------------------------------------------------------------------------

def strided_sample(volume: np.ndarray, max_samples: int = 4_000_000) -> np.ndarray:
    """A regular subsample of ``volume`` (flattened copy) of at most ~max_samples voxels."""
    stride = max(1, int(np.ceil((volume.size / max_samples) ** (1.0 / volume.ndim))))
    return np.ascontiguousarray(volume[::stride, ::stride, ::stride]).ravel()


@dataclass(frozen=True)
class IntensityScale:
    """
    Affine map from a scan's stored values to the network's working range.

    Percentiles rather than min/max, so one saturated voxel or a bright
    reconstruction artefact does not squash the tissue into a sliver of the
    range. Values outside the percentiles simply land outside [0, 1]; nothing
    is clipped on the way in.
    """

    low: float
    high: float

    @classmethod
    def fit(
        cls, volume: np.ndarray, low_percentile: float = 0.5, high_percentile: float = 99.9,
    ) -> "IntensityScale":
        sample = strided_sample(volume).astype(np.float64)
        low, high = np.percentile(sample, [low_percentile, high_percentile])
        if not high > low:
            low, high = float(sample.min()), float(sample.max())
        if not high > low:
            high = low + 1.0
        return cls(float(low), float(high))

    def to_unit(self, array: np.ndarray) -> np.ndarray:
        out = np.asarray(array, dtype=np.float32) - np.float32(self.low)
        out /= np.float32(self.high - self.low)
        return out

    def from_unit(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=np.float32) * np.float32(self.high - self.low) + np.float32(self.low)


def tissue_threshold(volume: np.ndarray) -> float:
    """Otsu threshold (stored units) between air and tissue, from a subsample."""
    sample = strided_sample(volume, 2_000_000).astype(np.float32)
    if sample.min() == sample.max():
        return float(sample.min())
    return float(threshold_otsu(sample))


def air_and_tissue_levels(volume: np.ndarray) -> tuple[float, float, float]:
    """(Otsu split, median below it, median above it), in stored units, from a subsample."""
    sample = strided_sample(volume, 2_000_000).astype(np.float32)
    if sample.min() == sample.max():
        level = float(sample.min())
        return level, level, level
    split = float(threshold_otsu(sample))
    air = float(np.median(sample[sample <= split]))
    above = sample[sample > split]
    tissue = float(np.median(above)) if above.size else split
    return split, air, tissue


def specimen_bounding_box(
    volume: np.ndarray, threshold: float, margin: int,
) -> tuple[slice, slice, slice]:
    """
    Bounding box of everything above ``threshold``, grown by ``margin`` voxels.

    Components smaller than 0.1% of the largest are dropped first, so a few
    noisy air voxels above threshold cannot stretch the box to the whole scan.
    Everything else is kept -- including a holder, if the scan has one --
    because this box must never cut into the specimen.
    """
    stride = max(1, int(np.ceil((volume.size / 8_000_000) ** (1 / 3))))
    coarse = volume[::stride, ::stride, ::stride] > threshold
    if not coarse.any():
        return (slice(None), slice(None), slice(None))
    labels, _ = ndimage.label(coarse)
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= max(1, 0.001 * sizes.max())) + 1
    coords = np.argwhere(np.isin(labels, keep))
    lo = coords.min(axis=0) * stride - margin - stride
    hi = (coords.max(axis=0) + 1) * stride + margin + stride
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, volume.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

_NRRD_TYPES = {
    np.dtype(np.uint8): "uint8",
    np.dtype(np.int8): "int8",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.int16): "int16",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.int32): "int32",
    np.dtype(np.uint64): "uint64",
    np.dtype(np.int64): "int64",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}


def _vector(values) -> str:
    return "(" + ",".join(f"{float(v):.17g}" for v in values) + ")"


def create_nrrd_memmap(
    path: str, shape_zyx: tuple[int, int, int], dtype, geometry: Geometry | None,
) -> np.memmap:
    """
    Create an uncompressed NRRD on disk and return a writable (Z, Y, X) view of its data.

    NRRD stores its fastest axis first; a C-ordered (Z, Y, X) array has X
    fastest, so its bytes *are* an NRRD with sizes (X, Y, Z) and can be filled
    in place, tile by tile, without the whole volume ever being in memory.
    Raw encoding is what makes that possible (a gzip stream cannot be written
    out of order); the file is a temporary input to segmentation anyway.

    ``geometry=None`` writes no spatial fields at all, so a reader finds no
    voxel size -- exactly what it would find for the original scan.
    """
    native = np.dtype(dtype).newbyteorder("=")
    if native not in _NRRD_TYPES:
        raise ValueError(f"No NRRD type for dtype {native}")
    # Always little-endian on disk, whatever the machine, and said so in the header.
    dtype = native.newbyteorder("<") if native.itemsize > 1 else native
    z, y, x = (int(v) for v in shape_zyx)
    lines = [
        "NRRD0004",
        "# Super-resolved volume written by MicroCT-Analysis/super_resolution",
        f"type: {_NRRD_TYPES[native]}",
        "dimension: 3",
    ]
    if geometry is not None:
        lines.append(f"space: {geometry.space}")
    lines.append(f"sizes: {x} {y} {z}")
    if geometry is not None:
        lines.append("space directions: " + " ".join(_vector(row) for row in geometry.directions[::-1]))
    lines.append("kinds: domain domain domain")
    if dtype.itemsize > 1:
        lines.append("endian: little")
    lines.append("encoding: raw")
    if geometry is not None:
        lines.append(f"space origin: {_vector(geometry.origin)}")
    header = ("\n".join(lines) + "\n\n").encode("ascii")

    with open(path, "wb") as handle:
        handle.write(header)
        handle.truncate(len(header) + z * y * x * dtype.itemsize)
    return np.memmap(path, dtype=dtype, mode="r+", offset=len(header), shape=(z, y, x))


def close_memmap(array: np.memmap) -> None:
    """
    Flush and release a memory map so its file can be read, moved or deleted.

    Windows refuses to delete a file that is still mapped, so dropping the last
    reference is not enough on its own -- the mapping is closed explicitly.
    """
    array.flush()
    mm = getattr(array, "_mmap", None)
    if mm is not None:
        mm.close()


def write_nrrd(
    path: str, volume_zyx: np.ndarray, geometry: Geometry | None, *, compress: bool = True,
) -> None:
    """Write a (Z, Y, X) array as NRRD with the given geometry (none if None; gzip by default)."""
    header = geometry.nrrd_header() if geometry is not None else {"kinds": ["domain"] * 3}
    if not compress:
        header["encoding"] = "raw"
    data = np.ascontiguousarray(np.transpose(volume_zyx, (2, 1, 0)))
    nrrd.write(path, data, header, compression_level=1)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def free_disk_bytes(directory: str) -> int:
    """Free space on the filesystem holding ``directory``."""
    return shutil.disk_usage(directory).free


def memory_bytes() -> tuple[int | None, int | None]:
    """
    (total, available) physical memory in bytes; either may be None if unknown.

    psutil if it is installed, else the platform's own interface.
    """
    try:
        import psutil  # pylint: disable=import-outside-toplevel

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        pass
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):  # pylint: disable=too-few-public-methods
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return None, None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None, None
    available = None
    try:
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    return int(total), (int(available) if available is not None else None)
