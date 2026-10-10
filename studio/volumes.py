"""
Previews: the downsampled copies of each volume that the viewer streams.

A segmentation output is the whole scan's frame -- often 2000 x 1000 x 700
voxels and several GB -- and nothing in a browser wants that. A preview is the
volume cropped to its tissue, block-averaged down to a voxel budget, quantized
to 16 bits and gzipped. A view then opens in about a second, and every slice,
projection and window change after that happens in the browser with no round
trip to the server.

What stays exact is the intensity scale. Quantization maps the volume's true
[min, max] onto the 16-bit range and records both ends, so the viewer reads
back scanner units (to 1/65535 of the range) and a window set in the viewer
means the same numbers as a threshold in the pipeline. What does not survive
is structure finer than a block; the full-resolution files remain the record,
and the histogram shipped with each view is computed from them, not from the
preview.

Masked volumes (incisor, molar, CMPR: tissue values inside, exact zeros
outside) reserve quantized 0 for background and average only tissue voxels
into each block, so an edge block is not darkened by the air it overlaps.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import uuid
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .paths import REPO, SEGMENT, TOOLS

PREVIEW_VERSION = 3
HISTOGRAM_BINS = 256
# Voxels drawn for the histogram. A strided sample this size reproduces the
# histogram of a multi-GB volume to well within a bin's height.
HISTOGRAM_SAMPLE = 8_000_000
UNWRAP_ANGLES = 360


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _segment_io():
    """The segmentation package's loaders, so volumes read exactly as the pipeline reads them."""
    if str(SEGMENT) not in sys.path:
        sys.path.insert(0, str(SEGMENT))
    import data_io  # noqa: PLC0415
    return data_io


def load_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """A .nrrd/.tif volume as (Z, Y, X) plus its voxel size in mm, if recorded."""
    data_io = _segment_io()
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nrrd":
        volume, spacing = data_io.load_nrrd(path, return_spacing=True)
    elif ext in (".tif", ".tiff"):
        volume, spacing = data_io.load_tiff(path, return_spacing=True)
    else:
        raise ValueError(f"Not a volume file: {path}")
    if volume is None:
        raise FileNotFoundError(path)
    if volume.ndim != 3:
        raise ValueError(f"{path} is {volume.ndim}-D, expected a 3-D volume")
    return volume, spacing


# ---------------------------------------------------------------------------
# Geometry of the reduction
# ---------------------------------------------------------------------------

def _voxels(shape, factors) -> int:
    return math.prod(math.ceil(n / f) for n, f in zip(shape, factors))


def choose_factors(shape: tuple[int, ...], budget: int, z_first: bool = False,
                   max_z_first: int = 4) -> tuple[int, int, int]:
    """Integer per-axis block sizes that bring ``shape`` under ``budget`` voxels.

    Isotropic by default, so a preview keeps square pixels in every plane.
    ``z_first`` spends the budget on axis 0 before touching the others: a
    CMPR reslice is ~1500 slices of a ~150 px cross-section, and halving the
    number of slices costs far less than halving the cross-section.
    """
    shape = tuple(int(n) for n in shape)
    if _voxels(shape, (1, 1, 1)) <= budget:
        return 1, 1, 1
    if z_first:
        fz = 1
        while _voxels(shape, (fz, 1, 1)) > budget and fz < max_z_first:
            fz += 1
        k = 1
        while _voxels(shape, (fz, k, k)) > budget:
            k += 1
        return fz, k, k
    k = max(1, int((math.prod(shape) / budget) ** (1.0 / 3.0)))
    while _voxels(shape, (k, k, k)) > budget:
        k += 1
    return k, k, k


def nonzero_bbox(volume: np.ndarray, margin: int = 2, chunk: int = 16) -> tuple[slice, slice, slice] | None:
    """Bounding box of the non-zero voxels, padded by ``margin``.

    Walks the volume in slabs rather than building a full boolean copy, which
    for a whole-scan frame would be another gigabyte.
    """
    depth, height, width = volume.shape
    z_any = np.zeros(depth, dtype=bool)
    y_any = np.zeros(height, dtype=bool)
    x_any = np.zeros(width, dtype=bool)
    for z0 in range(0, depth, chunk):
        nz = volume[z0:z0 + chunk] != 0
        z_any[z0:z0 + chunk] = nz.any(axis=(1, 2))
        y_any |= nz.any(axis=(0, 2))
        x_any |= nz.any(axis=(0, 1))
    if not z_any.any():
        return None

    def span(mask: np.ndarray) -> slice:
        idx = np.flatnonzero(mask)
        return slice(max(int(idx[0]) - margin, 0), min(int(idx[-1]) + 1 + margin, mask.size))

    return span(z_any), span(y_any), span(x_any)


def block_reduce(volume: np.ndarray, factors: tuple[int, int, int], mode: str = "mean"
                 ) -> tuple[np.ndarray, np.ndarray | None]:
    """Reduce ``volume`` by ``factors`` blocks, one axis-0 slab at a time.

    mode "mean"    block mean (a partial block at the far edge divides by its
                   own voxel count, not the full block's)
         "nzmean"  mean of the non-zero voxels; also returns which blocks held any
         "frac"    fraction of the block that is non-zero (for masks)
    """
    fz, fy, fx = factors
    depth, height, width = volume.shape
    oz, oy, ox = math.ceil(depth / fz), math.ceil(height / fy), math.ceil(width / fx)
    out = np.zeros((oz, oy, ox), dtype=np.float32)
    has = np.zeros((oz, oy, ox), dtype=bool) if mode == "nzmean" else None

    # Voxels per block in Y/X, accounting for the partial blocks at the edges.
    ny = np.minimum(fy, height - np.arange(oy) * fy)
    nx = np.minimum(fx, width - np.arange(ox) * fx)
    in_plane = np.outer(ny, nx).astype(np.float32)

    for i in range(oz):
        slab = np.asarray(volume[i * fz:(i + 1) * fz])
        cz = slab.shape[0]
        padded = np.zeros((cz, oy * fy, ox * fx), dtype=np.float32)
        if mode == "frac":
            padded[:, :height, :width] = slab != 0
        else:
            padded[:, :height, :width] = slab
        blocks = padded.reshape(cz, oy, fy, ox, fx)
        if mode == "nzmean":
            count = (blocks != 0).sum(axis=(0, 2, 4))
            total = blocks.sum(axis=(0, 2, 4))
            out[i] = np.where(count > 0, total / np.maximum(count, 1), 0.0)
            has[i] = count > 0
        else:
            out[i] = blocks.sum(axis=(0, 2, 4)) / (in_plane * cz)
    return out, has


def value_range(volume: np.ndarray, masked: bool, chunk: int = 32) -> tuple[float, float]:
    """True [min, max] over the volume (over its non-zero voxels when masked)."""
    lo, hi = math.inf, -math.inf
    for z0 in range(0, volume.shape[0], chunk):
        slab = np.asarray(volume[z0:z0 + chunk])
        if masked:
            slab = slab[slab != 0]
        if slab.size:
            lo = min(lo, float(slab.min()))
            hi = max(hi, float(slab.max()))
    if not math.isfinite(lo):
        return 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def value_sample(volume: np.ndarray, masked: bool, cap: int = HISTOGRAM_SAMPLE) -> tuple[np.ndarray, int]:
    """A uniform strided sample of the volume's values, and the stride used."""
    step = 1
    while math.prod(math.ceil(n / step) for n in volume.shape) > cap * (8 if masked else 1):
        step += 1
    sample = np.asarray(volume[::step, ::step, ::step], dtype=np.float32)
    if masked:
        sample = sample[sample != 0]
    return sample.ravel(), step


def histogram(sample: np.ndarray, lo: float, hi: float, bins: int = HISTOGRAM_BINS) -> dict:
    counts, edges = np.histogram(sample, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def quantize(values: np.ndarray, lo: float, hi: float, has: np.ndarray | None = None) -> np.ndarray:
    """Map [lo, hi] to uint16. With ``has``, 0 is kept for background and tissue uses 1..65535."""
    span = hi - lo if hi > lo else 1.0
    if has is None:
        return np.clip(np.rint((values - lo) / span * 65535.0), 0, 65535).astype(np.uint16)
    q = np.clip(np.rint((values - lo) / span * 65534.0) + 1.0, 1, 65535).astype(np.uint16)
    q[~has] = 0
    return q


def write_gz(path: Path, array: np.ndarray, level: int = 3) -> int:
    """Raw little-endian bytes of ``array`` (C order), gzipped. Returns the compressed size."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    data = np.ascontiguousarray(array)
    if data.dtype.byteorder == ">":
        data = data.byteswap().view(data.dtype.newbyteorder("<"))
    with gzip.open(tmp, "wb", compresslevel=level) as handle:
        handle.write(data.tobytes())
    os.replace(tmp, path)
    return path.stat().st_size


def read_gz(path: Path, dtype, shape) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        return np.frombuffer(handle.read(), dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# CMPR unwrap
# ---------------------------------------------------------------------------

def unwrap_cmpr(volume: np.ndarray, n_angles: int = UNWRAP_ANGLES, step: float = 0.5) -> np.ndarray:
    """The brightest value along each ray from the slice centre, per slice and angle.

    Every CMPR slice is centred on the incisor's centerline, so rays cast from
    the centre cross the tooth wall once. The brightest sample along a ray is
    the enamel where there is enamel, so the (slice, angle) map unrolls the
    whole tooth's enamel layer into one image: position down the tooth one way,
    position around it the other. Angle 0 points along the slice's +X axis and
    increases towards +Y.
    """
    depth, height, width = volume.shape
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radii = np.arange(0.0, min(cy, cx), step)
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    coords = np.stack([
        (cy + np.outer(np.sin(theta), radii)).ravel(),
        (cx + np.outer(np.cos(theta), radii)).ravel(),
    ])
    out = np.zeros((depth, n_angles), dtype=np.float32)
    for z in range(depth):
        samples = ndi.map_coordinates(np.asarray(volume[z], dtype=np.float32), coords,
                                      order=1, mode="constant", cval=0.0)
        out[z] = samples.reshape(n_angles, radii.size).max(axis=1)
    return out


# ---------------------------------------------------------------------------
# Raw scans, read strided for the crop view
# ---------------------------------------------------------------------------

def _stride_for(shape, budget: int) -> int:
    step = 1
    while math.prod(math.ceil(n / step) for n in shape) > budget:
        step += 1
    return step


def read_isq_strided(path: Path, budget: int):
    """A strided sample of a Scanco .isq straight from disk, without reading it all.

    The pixel block is memory-mapped, so only every ``step``-th slice is paged
    in. The data offset is the file size minus the pixel bytes, the same rule
    convert_ISQ_NRRD.py uses: the header's own block count can undercount.
    """
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from convert_ISQ_NRRD import _HEADER_SIZE, _parse_isq_header, _spacing_mm  # noqa: PLC0415

    with open(path, "rb") as handle:
        header = _parse_isq_header(handle.read(_HEADER_SIZE))
    shape = (header["dim_z_pixels"], header["dim_y_pixels"], header["dim_x_pixels"])
    offset = path.stat().st_size - math.prod(shape) * 2
    mapped = np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=shape)
    step = _stride_for(shape, budget)
    sample = np.array(mapped[::step, ::step, ::step], dtype=np.float32)
    del mapped
    return sample, _spacing_mm(header), shape, step


def read_dicom_strided(path: Path, budget: int):
    data_io = _segment_io()
    from data_io.dicom import _slice_normal, _slice_position, find_dicom_files  # noqa: PLC0415
    import pydicom  # noqa: PLC0415

    files = find_dicom_files(str(path))
    if not files:
        raise ValueError(f"No DICOM files in {path}")
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    normal = _slice_normal(headers[0])
    order = sorted(range(len(files)), key=lambda i: _slice_position(headers[i], normal))
    first = pydicom.dcmread(files[order[0]])
    rows, cols = first.pixel_array.shape[-2:]
    shape = (len(files), int(rows), int(cols))
    step = _stride_for(shape, budget)
    picked = order[::step]
    sample = np.stack([
        np.asarray(pydicom.dcmread(files[i]).pixel_array, dtype=np.float32)[::step, ::step]
        for i in picked
    ])
    spacing = data_io.spacing_from_dicom_headers([headers[i] for i in order])
    return sample, spacing, shape, step


def read_bmp_strided(path: Path, budget: int):
    from PIL import Image  # noqa: PLC0415

    # The same slices load_bmp_stack converts: glob.glob skips dotfiles, so the
    # ._*.bmp AppleDouble sidecars macOS leaves on exFAT drives are not slices.
    files = sorted(p for p in path.glob("*.bmp")
                   if not p.name.startswith(".") and not p.name.lower().endswith("spr.bmp"))
    if not files:
        raise ValueError(f"No BMP slices in {path}")
    with Image.open(files[0]) as im:
        width, height = im.size
    shape = (len(files), height, width)
    step = _stride_for(shape, budget)
    slices = []
    for f in files[::step]:
        with Image.open(f) as im:
            slices.append(np.asarray(im.convert("F"), dtype=np.float32)[::step, ::step])
    return np.stack(slices), None, shape, step


# ---------------------------------------------------------------------------
# Building a view
# ---------------------------------------------------------------------------

def file_identity(path: str | Path) -> dict:
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}


def _thumbnail(q: np.ndarray, masked: bool, spacing_preview, out: Path) -> str | None:
    """A grey maximum projection along the thinnest axis, as a small PNG."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None
    axis = int(np.argmin(q.shape))
    mip = q.max(axis=axis).astype(np.float32)
    tissue = mip[mip > 0] if masked else mip.ravel()
    if tissue.size == 0:
        return None
    lo, hi = np.percentile(tissue, (1.0, 99.8))
    img = np.clip((mip - lo) / max(hi - lo, 1.0), 0, 1)
    img = (img * 255).astype(np.uint8)
    rest = [a for a in range(3) if a != axis]
    sy, sx = (spacing_preview[a] for a in rest)
    h, w = img.shape
    scale = 360.0 / max(h * sy, w * sx)
    size = (max(1, int(round(w * sx * scale))), max(1, int(round(h * sy * scale))))
    Image.fromarray(img).resize(size, Image.BILINEAR).save(out)
    return out.name


def build_view(out_dir: Path, *, view: str, base: dict, overlays: list[dict] | None = None,
               budget: int = 24_000_000, masked: bool = False, z_first: bool = False,
               unwrap: bool = False, crop_to_tissue: bool = True, source_kind: str | None = None,
               voxel_mm: tuple[float, float, float] | None = None, log=print) -> dict:
    """Build one view's preview files into ``out_dir`` and return its metadata.

    ``base`` is {"path": ..., "key": ...}. ``overlays`` are masks resampled onto
    the base's preview grid: [{"name", "path", "color"}], or with a "value", the
    voxels of a label map that hold it (one tooth of the molar step's teeth). ``source_kind`` reads
    a raw scan strided instead of loading it whole (isq, dicom, bmp).
    ``crop_to_tissue`` trims a masked volume to its non-zero box; the CMPR view
    turns it off, since its unwrap needs every slice centred on the centerline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(base["path"])
    counts: dict[str, int] = {}
    sampling = "block-mean"

    log(f"@@progress 0.05 Reading {base_path.name}")
    if source_kind in ("isq", "dicom", "bmp"):
        reader = {"isq": read_isq_strided, "dicom": read_dicom_strided, "bmp": read_bmp_strided}[source_kind]
        volume, spacing, full_shape, step = reader(base_path, budget)
        factors = (step, step, step)
        crop = tuple(slice(0, n) for n in full_shape)
        cropped = volume
        sampling = "strided"
    else:
        volume, spacing = load_volume(base_path)
        full_shape = volume.shape
        crop = tuple(slice(0, n) for n in full_shape)
        if masked and crop_to_tissue:
            box = nonzero_bbox(volume)
            if box is None:
                raise ValueError(f"{base_path.name} is empty: no non-zero voxels")
            crop = box
        cropped = volume[crop]
        factors = choose_factors(cropped.shape, budget, z_first=z_first)
    if spacing is None and voxel_mm is not None:
        spacing = tuple(float(v) for v in voxel_mm)
    spacing_known = spacing is not None
    spacing = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))

    log("@@progress 0.25 Measuring the intensity range")
    lo, hi = value_range(cropped, masked)
    sample, sampled_every = value_sample(cropped, masked)
    hist = histogram(sample, lo, hi)
    del sample
    if masked:
        counts[base.get("key", "base")] = int(sum(int(np.count_nonzero(cropped[z0:z0 + 32]))
                                                  for z0 in range(0, cropped.shape[0], 32)))

    log(f"@@progress 0.45 Downsampling by {factors}")
    if sampling == "strided":
        reduced, has = cropped.astype(np.float32), None
    else:
        reduced, has = block_reduce(cropped, factors, "nzmean" if masked else "mean")
    q = quantize(reduced, lo, hi, has if masked else None)
    del reduced, has
    write_gz(out_dir / "base.u16.gz", q)

    unwrap_meta = None
    if unwrap:
        log("@@progress 0.6 Unrolling the enamel layer")
        u = unwrap_cmpr(cropped)
        uq = quantize(u, lo, hi, u != 0)
        write_gz(out_dir / "unwrap.u16.gz", uq)
        unwrap_meta = {"file": "unwrap.u16.gz", "shape": list(uq.shape), "angles": UNWRAP_ANGLES}
        del u, uq
    del cropped, volume

    overlay_meta = []
    loaded_path, mask = None, None
    for i, ov in enumerate(overlays or []):
        log(f"@@progress {0.7 + 0.2 * i / max(len(overlays), 1):.2f} Resampling {ov['name']}")
        # Overlays cut from one label map (the teeth) share its file: read it once.
        if ov["path"] != loaded_path:
            loaded_path, mask = None, None
            try:
                mask, _ = load_volume(ov["path"])
            except (OSError, ValueError) as exc:
                log(f"Skipping overlay {ov['name']}: {exc}")
                continue
            loaded_path = ov["path"]
        if tuple(mask.shape) != tuple(full_shape):
            log(f"Skipping overlay {ov['name']}: shape {mask.shape} does not match {tuple(full_shape)}")
            continue
        region = mask[crop]
        if ov.get("value") is not None:
            region = region == ov["value"]
        counts[ov["name"]] = int(sum(int(np.count_nonzero(region[z0:z0 + 32]))
                                     for z0 in range(0, region.shape[0], 32)))
        frac, _ = block_reduce(region, factors, "frac")
        write_gz(out_dir / f"ov_{ov['name']}.u8.gz", np.rint(frac * 255).astype(np.uint8))
        overlay_meta.append({"name": ov["name"], "label": ov.get("label", ov["name"]),
                             "color": ov.get("color"), "file": f"ov_{ov['name']}.u8.gz",
                             "source": file_identity(ov["path"])})
        del region, frac
    del mask

    spacing_preview = [s * f for s, f in zip(spacing, factors)]
    thumb = _thumbnail(q, masked, spacing_preview, out_dir / "thumb.png")
    meta = {
        "version": PREVIEW_VERSION,
        "view": view,
        "base": {"key": base.get("key"), **file_identity(base_path)},
        "source_kind": source_kind,
        "sampling": sampling,
        "full_shape": [int(n) for n in full_shape],
        "crop": [[int(s.start), int(s.stop)] for s in crop],
        "factors": [int(f) for f in factors],
        "shape": [int(n) for n in q.shape],
        "spacing_mm": list(spacing),
        "spacing_known": spacing_known,
        "spacing_preview_mm": spacing_preview,
        "masked": bool(masked),
        "quant": {"lo": lo, "hi": hi, "offset": 1 if masked else 0, "scale": 65534 if masked else 65535},
        "histogram": {**hist, "sampled_every": sampled_every},
        "overlays": overlay_meta,
        "unwrap": unwrap_meta,
        "counts": counts,
        "thumb": thumb,
    }
    tmp = out_dir / f".view.{uuid.uuid4().hex[:6]}.json"
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    os.replace(tmp, out_dir / "view.json")
    log("@@progress 1.0 Preview ready")
    return meta


def suggest_crop(q: np.ndarray, meta: dict, margin_mm: float = 0.5) -> list[int]:
    """A crop box around the specimen, found on a view's preview.

    The same recipe as the converters' --autocrop -- Otsu split, keep the
    largest piece and those clustered with it so the holder and debris drop
    out but an incisor the threshold split from the bone does not (see
    ``largest_specimen_group``), pad by a margin -- run on the
    preview instead of the full scan, so it answers instantly.
    Returns (z0, z1, y0, y1, x0, x1) in full-resolution source indices.
    """
    from skimage.filters import threshold_otsu  # noqa: PLC0415
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from volume_pipeline import largest_specimen_group  # noqa: PLC0415

    factors = np.asarray(meta["factors"])
    spacing = np.asarray(meta["spacing_mm"], dtype=float)
    values = q.astype(np.float32)
    mask, _ = largest_specimen_group(values > threshold_otsu(values), spacing * factors)
    idx = np.argwhere(mask)
    if idx.size == 0:
        return [v for n in meta["full_shape"] for v in (0, n)]
    # One preview voxel stands for a whole block (or, strided, for the gap up
    # to the next sample), and the specimen can begin anywhere inside the
    # block before the first one that shows it -- so the box reaches back one
    # block on each side before the margin is added.
    lo, hi = idx.min(axis=0) - 1, idx.max(axis=0) + 2
    origin = np.asarray([c[0] for c in meta["crop"]])
    pad = np.ceil(margin_mm / np.maximum(spacing, 1e-9)).astype(int)
    full = np.asarray(meta["full_shape"])
    start = np.clip(origin + lo * factors - pad, 0, full)
    stop = np.clip(origin + hi * factors + pad, start + 1, full)
    return [int(v) for pair in zip(start, stop) for v in pair]


def _same_time(recorded, now: float) -> bool:
    """Whether two modification times are the same file's, as two systems report it.

    FAT-family drives keep times to 2 s, and a drive that stores local time
    reads whole hours off on a computer set to another zone.
    """
    diff = abs(float(recorded or 0) - now)
    off = diff % 1800  # time zones are whole or half hours apart
    return diff <= 2.0 or (diff <= 14 * 3600 + 2 and (off <= 2.0 or off >= 1798.0))


def view_is_current(meta: dict | None, base_path: Path | None, overlay_paths: dict[str, Path]) -> bool:
    """Whether a built view still matches the files it was built from."""
    if not meta or meta.get("version") != PREVIEW_VERSION or base_path is None:
        return False
    try:
        ident = file_identity(base_path)
    except OSError:
        return False
    b = meta.get("base") or {}
    # The file's name, not its full path: the same drive is /Volumes/LabDrive on a
    # Mac and E:\ on a PC, and its previews should not be rebuilt for that.
    if Path(str(b.get("path", "")).replace("\\", "/")).name != Path(ident["path"]).name or \
            b.get("size") != ident["size"] or not _same_time(b.get("mtime", 0), ident["mtime"]):
        return False
    built = {o["name"]: o.get("source") or {} for o in meta.get("overlays") or []}
    for name, path in overlay_paths.items():
        src = built.get(name)
        if not src:
            return False
        try:
            ident = file_identity(path)
        except OSError:
            continue
        if src.get("size") != ident["size"] or not _same_time(src.get("mtime", 0), ident["mtime"]):
            return False
    return True


__all__ = [
    "REPO", "build_view", "choose_factors", "block_reduce", "nonzero_bbox", "quantize",
    "read_gz", "suggest_crop", "unwrap_cmpr", "value_range", "view_is_current", "write_gz",
]
