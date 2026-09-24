"""
Convert a Scanco microCT .isq volume to a filtered .nrrd volume.

Usage:
    python convert_ISQ_NRRD.py <isq_file> [options]

This tool is the Scanco ISQ front end for the shared conversion pipeline in
``volume_pipeline.py``; ``convert_DICOM_NRRD.py`` is the DICOM front end.
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
                                   (true 3-D, numba-JIT + multi-threaded)
    4. Anisotropic diffusion    — edge-preserving smoothing: flattens noise
                                   within tissues but not across boundaries
                                   (optional, --aniso-iterations)
    5. Richardson-Lucy deconv   — recover detail lost to the imaging PSF
                                   (optional, --rl-iterations)
    6. TV (Chambolle) denoise   — piecewise-constant smoothing for cleaner
                                   thresholding boundaries (optional, --tv-weight)
    7. Unsharp mask             — high-pass boost that steepens boundary
                                   ramps (optional, --unsharp-amount)

Output:
    <stem>_prefiltered.nrrd — cropped/binned/reoriented but unfiltered
                              (optional, --save-raw)
    <stem>.nrrd          — the same volume, filtered, with geometry embedded
    <stem>_enhanced.nrrd — CLAHE-boosted viewing copy (optional, --enhance-contrast)
    <stem>_meta.json     — ISQ header metadata

Notes:
    - The ISQ header is Scanco's proprietary binary format. This reader
      parses the fixed 512-byte pre-header (documented by Scanco/SCANCO
      Medical AG and widely reverse-engineered in the microCT community)
      to recover the array dimensions and voxel spacing, then reads the
      raw 16-bit signed pixel data that follows.
    - Voxel spacing is derived from the physical dimensions and pixel
      dimensions stored in the header and written into the NRRD header so
      downstream tools (3D Slicer, ITK-SNAP) load it with correct geometry.
      ISQ carries no patient-orientation tags, so the volume is treated as
      an axial acquisition: columns (X) run along L, rows (Y) along P and
      slices (Z) along S — identical to what the DICOM converter produces
      for an axial series, and to its fallback for a series without
      ImageOrientationPatient. An ISQ scan and a DICOM series of the same
      specimen therefore overlay in Slicer/ITK-SNAP.
    - Raw stored pixel values (signed 16-bit scanner units) are preserved
      by default. Pass --apply-scaling to convert to linear attenuation
      units using the header's mu_scaling field, matching Scanco's own
      convention. (--apply-rescale is accepted as an alias, matching the
      DICOM converter's flag name.)
    - A few voxels per scan come off the reconstruction pinned to the int16
      rails (-32768 / 32767). Neither rail is a measurement: at mu_scaling
      4096 they sit at exactly -8.000 and +8.000 1/cm, the signature of a
      fixed-point limit. On the scan that motivated this the low-rail voxels
      ringed the rotation axis (11.9% of voxels within r<8 of it ran below
      -20000, against 0.000% elsewhere in the slice) and the high-rail voxels
      sat 1.76 mm outside the specimen in the mounting medium, flanked by
      impossible negative values — streak artifact, not dense material. They
      are replaced on read by the median of their unsaturated neighbours, so
      the fill matches whatever surrounds each one (--no-trim-saturated keeps
      them). Voxels off the rails keep their exact stored values, so
      densitometry is unaffected. Recorded under "_saturation_trim" in
      <stem>_meta.json.
    - Raw values also mean raw *background*: the air around the specimen is
      a band of low-amplitude counts, not zero. Pass --noise-floor LEVEL to
      flatten it to a constant before any filter runs. Voxels at or above
      LEVEL keep their exact stored values, so the specimen is untouched and
      densitometry still sees true counts. --noise-floor-method percentile
      reads LEVEL as a histogram percentile instead of an absolute count
      (portable across scans of differing gain), and --noise-floor-method
      otsu picks the specimen/air split automatically — aggressive, since it
      also removes low-density soft tissue and pulp. The level used and the
      fraction of voxels cleared are recorded in <stem>_meta.json.
    - NLM runs by default at --nlm-h 1.0, as in the DICOM converter. Pass
      --nlm-h 0 to skip it and get a plain read-and-write conversion, which
      is what this tool did before it gained the shared filter chain.
    - Pass --crop to interactively select a 3-D bounding box (napari) right
      after the ISQ is read, before any filtering runs. This speeds up every
      later step and both output volumes reflect the crop. The NRRD "space
      origin" is shifted by the crop offset, so a cropped volume still lands
      in the same physical position as the full scan.
    - Pass --target-voxel-um to bin the volume up to a coarser voxel size
      (e.g. --target-voxel-um 20). Per-axis integer bin factors are derived
      from the header's native spacing; because binning combines whole
      voxels, the closest achievable size is used and reported. Applied
      right after the crop, so the filters run on the reduced volume and
      every output file shrinks by the same factor. Block averaging also
      raises SNR by sqrt(N).
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
    - --save-dtype int16 halves file size versus the float32 default, and
      is lossless for an unfiltered, unbinned ISQ read since stored scanner
      values are signed 16-bit integers by construction.
    - --target-voxel-um cuts both write time and file size by the product
      of the resulting bin factors (2x per axis = 8x smaller).

Tuning the filters (blurry output, soft boundaries, enamel visibility) works
exactly as documented in convert_DICOM_NRRD.py — same flags, same defaults,
same stage order.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import struct
import sys
from pathlib import Path

import numpy as np

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
# ISQ header parsing
# ---------------------------------------------------------------------------

# Fixed 512-byte pre-header layout (little-endian), per Scanco's published
# ISQ format description. Offsets are in bytes from the start of the file.
#
# The file opens with a 16-byte ASCII magic ("CTDATA-HEADER_V1"), then a
# 12-byte gap (version//padding ints), then the field block proper. Reading
# the magic as a shorter string and starting the ints 12 bytes late is the
# classic way to mis-parse this header: it still "works" — every offset lands
# on a real int32 — but returns a neighbouring field, so the dimensions come
# back as small plausible-looking numbers (e.g. 3342 x 6 x 6 for a
# 3400 x 3400 x 557 scan) rather than failing outright.
_HEADER_SIZE = 512

_MAGIC = b"CTDATA-HEADER_V1"

_FIELDS = [
    # name, offset, struct format
    ("check",            0,   "16s"),
    ("data_type",        16,  "i"),
    ("nr_of_bytes",      20,  "i"),
    ("nr_of_blocks",     24,  "i"),
    ("patient_index",    28,  "i"),
    ("scanner_id",       32,  "i"),
    ("creation_date",    36,  "8s"),
    ("dim_x_pixels",     44,  "i"),
    ("dim_y_pixels",     48,  "i"),
    ("dim_z_pixels",     52,  "i"),
    ("dim_x_um",         56,  "i"),
    ("dim_y_um",         60,  "i"),
    ("dim_z_um",         64,  "i"),
    ("slice_thickness_um", 68, "i"),
    ("slice_increment_um", 72, "i"),
    ("slice_1_pos_um",   76,  "i"),
    ("min_data_value",   80,  "i"),
    ("max_data_value",   84,  "i"),
    ("mu_scaling",       88,  "i"),
    ("nr_of_samples",    92,  "i"),
    ("nr_of_projections", 96, "i"),
    ("scandist_um",      100, "i"),
    ("scanner_type",     104, "i"),
    ("sampletime_us",    108, "i"),
    ("index_measurement", 112, "i"),
    ("site",             116, "i"),
    ("reference_line_um", 120, "i"),
    ("recon_alg",        124, "i"),
    ("name",             128, "40s"),
    ("energy_ve",        168, "i"),
    ("intensity_ua",     172, "i"),
    ("header_data_offset_blocks", 508, "i"),
]


def _parse_isq_header(raw: bytes) -> dict:
    """Parse the fixed 512-byte ISQ pre-header into a dict of raw field values."""
    header = {}
    for name, offset, fmt in _FIELDS:
        (val,) = struct.unpack_from("<" + fmt, raw, offset)
        if isinstance(val, bytes):
            val = val.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
        header[name] = val
    return header


def read_isq(path: Path) -> tuple[np.ndarray, dict]:
    """
    Read a Scanco .isq file.

    Returns:
        (volume, header) where volume has shape (Z, Y, X), dtype int16, and
        header is the parsed metadata dict (raw units: um, scanner counts).
    """
    with open(path, "rb") as fh:
        raw_header = fh.read(_HEADER_SIZE)
        header = _parse_isq_header(raw_header)

        if raw_header[:len(_MAGIC)] != _MAGIC:
            log.warning(
                "Unexpected ISQ magic bytes (%r, expected %r) — file may not "
                "be a valid Scanco ISQ, or uses a header variant this parser "
                "doesn't recognize. Attempting to continue anyway.",
                raw_header[:len(_MAGIC)], _MAGIC,
            )

        dim_x = header["dim_x_pixels"]
        dim_y = header["dim_y_pixels"]
        dim_z = header["dim_z_pixels"]
        if dim_x <= 0 or dim_y <= 0 or dim_z <= 0:
            raise ValueError(
                f"Invalid dimensions parsed from ISQ header: "
                f"{dim_x} x {dim_y} x {dim_z}"
            )

        n_voxels = dim_x * dim_y * dim_z
        n_bytes = n_voxels * 2

        # Pixel data starts after (header_data_offset_blocks + 1) * 512 bytes.
        # The +1 accounts for the pre-header block itself.
        data_offset = (header["header_data_offset_blocks"] + 1) * _HEADER_SIZE

        # Cross-check that against the file: the pixel block runs to EOF, so
        # the true offset is (file size - pixel bytes). These disagree on real
        # scanner output — the block count can undercount the header by a
        # block or two — and trusting the stated offset then shears every
        # slice by a constant number of voxels, which reads as a diagonal
        # smear rather than an obvious failure. The file size is the
        # authority; the stated offset only has to be plausible.
        file_size = path.stat().st_size
        implied_offset = file_size - n_bytes
        if implied_offset < _HEADER_SIZE:
            raise ValueError(
                f"ISQ file is too small for its header dimensions: "
                f"{dim_x} x {dim_y} x {dim_z} needs {n_bytes:,} bytes of pixel "
                f"data but the file is only {file_size:,} bytes. The header "
                f"may have been parsed with the wrong field offsets."
            )
        if implied_offset != data_offset:
            log.warning(
                "ISQ header states a data offset of %d bytes (%d blocks) but "
                "the file size implies %d. Using %d, since the pixel block "
                "runs to end of file.",
                data_offset, header["header_data_offset_blocks"],
                implied_offset, implied_offset,
            )
            data_offset = implied_offset

        log.info(
            "ISQ '%s': %d x %d x %d voxels (X, Y, Z), data at offset %d.",
            header.get("name", ""), dim_x, dim_y, dim_z, data_offset,
        )
        fh.seek(data_offset)
        # np.fromfile reads straight into the destination array, unlike
        # fh.read() + np.frombuffer, which holds an extra full-size bytes
        # object alive for as long as the (read-only) view over it exists.
        flat = np.fromfile(fh, dtype="<i2", count=n_voxels)
        if flat.size < n_voxels:
            raise ValueError(
                f"ISQ file is truncated: expected {n_voxels * 2} bytes of "
                f"pixel data at offset {data_offset}, got {flat.size * 2}."
            )

    volume = flat.reshape(dim_z, dim_y, dim_x)
    return volume, header


def _spacing_mm(header: dict) -> tuple[float, float, float]:
    """Return (dz, dy, dx) in mm from the ISQ header's physical/pixel dims."""
    dim_x, dim_y, dim_z = header["dim_x_pixels"], header["dim_y_pixels"], header["dim_z_pixels"]
    dx = header["dim_x_um"] / dim_x / 1000.0
    dy = header["dim_y_um"] / dim_y / 1000.0
    # Prefer the explicit slice increment when present; fall back to the
    # physical Z extent divided by slice count.
    if header.get("slice_increment_um"):
        dz = header["slice_increment_um"] / 1000.0
    else:
        dz = header["dim_z_um"] / dim_z / 1000.0
    return dz, dy, dx


def _apply_scaling(volume: np.ndarray, header: dict) -> np.ndarray:
    """Convert raw scanner counts to linear attenuation units (cm^-1).

    Scanco's convention: attenuation = raw_value / mu_scaling. This is the
    ISQ counterpart of the DICOM converter's RescaleSlope/Intercept step —
    both turn stored values into physical units before anything else runs.
    """
    mu_scaling = header.get("mu_scaling") or 1
    if mu_scaling == 0:
        log.warning("mu_scaling is 0 in header — skipping scaling.")
        return volume.astype(np.float32)
    log.info("Applying scaling (mu_scaling=%d)…", mu_scaling)
    return volume.astype(np.float32) / float(mu_scaling)


def trim_saturated(
    volume: np.ndarray,
    *,
    radius: int = 3,
) -> tuple[np.ndarray, dict]:
    """Replace int16-rail voxels with the median of their unsaturated neighbours.

    Scanco reconstructions carry a handful of voxels pinned to the int16 rails
    (-32768 / 32767). Neither rail is a measurement. At ``mu_scaling`` 4096 they
    sit at exactly -8.000 and +8.000 1/cm -- round numbers, which is the
    signature of a fixed-point encoding limit rather than a physical one.

    Both clusters on the scan that motivated this were reconstruction artifacts:

    * The 13 voxels at the low rail sat 3-4 voxels from the rotation centre.
      Within r < 8 of that axis, 11.9% of voxels ran below -20000, against
      0.000% across the 7.8 million voxels elsewhere in the same slice. Every
      projection crosses the axis, so a sub-pixel misalignment concentrates
      there and the backprojection kernel's negative lobes undershoot into the
      rail. A negative attenuation is physically impossible regardless.
    * The 47 voxels at the high rail were 1.76 mm *outside* the specimen, in the
      mounting material, in a blob 6 slices thick with negative undershoot
      (-4086) a few voxels away from the +32767 peak. Nothing in enamel is that
      dense, and no real object is flanked by impossible values: that is ring
      artifact, not material.

    Sixty voxels in four billion is nothing for densitometry and everything for
    any statistic keyed to the *extremes*. One voxel at -32768 sets
    ``volume.min()``, and every contrast stretch, display window and min/max
    normalization downstream inherits it.

    Replacement is the median of the unsaturated voxels in a (2*radius+1) cube
    around each rail voxel, so the fill comes from that voxel's own
    neighbourhood -- local tissue if it sits in tissue, mounting medium if it
    sits in mounting medium -- rather than from a single global constant that
    would be wrong in one of those places. Voxels off the rails keep their exact
    stored values, so densitometry is untouched.

    Note what this does *not* do. It replaces the voxels that sit exactly on a
    rail, not the artifact around them. Both sites stay locally corrupted after
    the fix -- the rotation-axis window still holds 23 voxels below -20000 and
    the streak still holds 25 above 28000 -- because the ringing extends well
    past the clipped peak. This is deliberate: guessing at where an artifact
    ends would destroy real data, and the job here is only to stop a handful of
    impossible values from setting the volume's extremes. Removing the
    artifacts themselves is a separate problem (ring/streak correction), and
    both sites here sit outside the specimen anyway, so a crop discards them.

    Nor is this an overflow fix. A wide intensity span is normal for these
    scans and is not itself a defect: of four scans checked, all four spanned
    more than 32767 counts and only one had any saturation at all. Code that
    subtracts int16 extremes in the input dtype therefore overflows on clean
    scans too, and the fix for that belongs at the arithmetic, not here.

    Returns ``(volume, info)``; *volume* is returned unchanged when no voxel
    sits on a rail.
    """
    if not np.issubdtype(volume.dtype, np.integer):
        return volume, {"applied": False, "reason": "not an integer volume"}

    limits = np.iinfo(volume.dtype)
    at_low = volume == limits.min
    at_high = volume == limits.max
    n_low, n_high = int(at_low.sum()), int(at_high.sum())
    if not (n_low or n_high):
        return volume, {"applied": False, "n_low": 0, "n_high": 0}

    saturated = at_low | at_high
    coords = np.argwhere(saturated)

    volume = volume.copy()
    replaced, unresolved = 0, 0
    for idx in coords:
        window = tuple(
            slice(max(0, int(i) - radius), min(int(n), int(i) + radius + 1))
            for i, n in zip(idx, volume.shape)
        )
        patch = volume[window]
        clean = patch[(patch != limits.min) & (patch != limits.max)]
        if clean.size:
            volume[tuple(idx)] = int(np.median(clean))
            replaced += 1
        else:
            unresolved += 1

    if unresolved:
        # Every neighbour was itself on a rail. Rare enough that a global
        # fallback is fine, and leaving the rail in place would defeat the point.
        steps = tuple(slice(None, None, max(1, n // 192)) for n in volume.shape)
        sub = volume[steps]
        clean = sub[(sub != limits.min) & (sub != limits.max)]
        fallback = int(np.median(clean)) if clean.size else 0
        still = (volume == limits.min) | (volume == limits.max)
        volume[still] = fallback
        log.warning(
            "%d saturated voxel(s) had no unsaturated neighbour within radius "
            "%d — filled with the volume median (%d).",
            unresolved, radius, fallback,
        )

    log.info(
        "Replaced %d saturated voxel(s) (%d at %d, %d at %d) with local "
        "neighbourhood medians (radius %d); range is now [%d, %d].",
        n_low + n_high, n_low, limits.min, n_high, limits.max, radius,
        int(volume.min()), int(volume.max()),
    )
    return volume, {
        "applied": True,
        "n_low": n_low,
        "n_high": n_high,
        "method": "neighbour-median",
        "radius": int(radius),
        "unresolved": unresolved,
    }


def clamp_negatives(volume: np.ndarray, *, floor: int = 0) -> tuple[np.ndarray, dict]:
    """Clamp values below *floor* up to it.

    Reconstructed ISQ values are signed by construction: they are linear
    attenuation relative to a calibration baseline, not an amount of material.
    Filtered backprojection uses a kernel with negative lobes, air is calibrated
    to a non-zero offset (~2050 counts here) with photon noise straddling it,
    and beam hardening drives values below baseline next to dense structures.
    About 6.8% of raw voxels on the scan this was written for came off the
    scanner negative, and none of them mean "less than nothing is there".

    Clamping is therefore a presentation choice, not a correction: it makes the
    stored numbers monotonic in density, at the cost of flattening the lower
    half of the noise distribution onto a single value. That has consequences
    worth knowing before switching it on:

    * The air band becomes asymmetric. Its noise still scatters upward but no
      longer downward, so any mean, standard deviation or Gaussian fit over
      background is biased high afterwards.
    * A spike appears at *floor*, exactly as the rotation pad produces one.
      Statistics that assume a smooth histogram see a new mode.
    * It is not reversible. The commercial reference scan compared against this
      one arrived already clamped at 0, which is why it showed 0.0000%
      negatives -- that is the vendor discarding information, not a cleaner
      reconstruction.

    Nothing at or above *floor* is touched, so mineralised tissue -- the part
    that carries the enamel and dentine signal -- is bit-identical either way.
    """
    if not np.issubdtype(volume.dtype, np.integer) and not np.issubdtype(
        volume.dtype, np.floating
    ):
        return volume, {"applied": False, "reason": "unsupported dtype"}

    below = volume < floor
    n_below = int(below.sum())
    if not n_below:
        return volume, {"applied": False, "n_clamped": 0, "floor": int(floor)}

    original_min = float(volume.min())
    volume = np.where(below, volume.dtype.type(floor), volume)
    log.info(
        "Clamped %d voxel(s) (%.4f%%) below %d up to it; minimum was %g. "
        "Reconstruction values are signed by design, so this flattens the low "
        "half of the background noise rather than removing an error.",
        n_below, 100.0 * n_below / volume.size, floor, original_min,
    )
    return volume, {
        "applied": True,
        "n_clamped": n_below,
        "fraction": float(n_below) / float(volume.size),
        "floor": int(floor),
        "original_min": original_min,
    }


def read_isq_volume(
    isq_path: Path,
    *,
    apply_scaling: bool = False,
    trim_saturated_voxels: bool = True,
    clamp_negative: bool = False,
    negative_floor: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read an ISQ file into the pipeline's source tuple.

    Returns ``(volume [Z, Y, X], direction_matrix, origin, meta)``.

    Unlike the DICOM reader, the volume is left as int16 when --apply-scaling
    is off: every downstream stage promotes to float32 on demand, so keeping
    the stored dtype here halves peak memory on a full-resolution scan without
    changing a single output value.

    Saturated rail voxels are clamped on read (see ``trim_saturated``), before
    any crop or filter, so no later stage ever sees a value that is an artifact
    of the storage format rather than a measurement.

    With *clamp_negative*, values below *negative_floor* are then raised to it
    (see ``clamp_negatives``). Off by default: negative reconstruction values
    are expected rather than erroneous, and clamping them biases background
    statistics upward.
    """
    log.info("Reading %s…", isq_path)
    volume, header = read_isq(isq_path)

    direction_matrix = directions_from_spacing(*_spacing_mm(header))
    # ISQ records no patient position; voxel [0, 0, 0] is the coordinate
    # origin. A crop shifts this the same way it does for DICOM.
    origin = np.zeros(3)

    meta = dict(header)
    if trim_saturated_voxels:
        volume, trim_info = trim_saturated(volume)
        meta["_saturation_trim"] = trim_info

    # After the rail trim, so the saturated voxels are already real values and
    # are not all swept into the floor, and before any filter or geometry stage.
    if clamp_negative:
        volume, clamp_info = clamp_negatives(volume, floor=negative_floor)
        meta["_negative_clamp"] = clamp_info

    if apply_scaling:
        volume = _apply_scaling(volume, header)

    return volume, direction_matrix, origin, meta


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def convert(
    isq_path: Path,
    output_dir: Path,
    opts: PipelineOptions | None = None,
    *,
    trim_saturated_voxels: bool = True,
    clamp_negative: bool = False,
    negative_floor: int = 0,
) -> dict[str, Path]:
    """Convert *isq_path* into *output_dir* using the shared pipeline."""
    opts = opts or PipelineOptions()
    isq_path = isq_path.resolve()
    if clamp_negative:
        opts = dataclasses.replace(opts, output_floor=float(negative_floor))

    return convert_volume(
        output_dir=output_dir,
        stem=isq_path.stem,
        read_source=lambda: read_isq_volume(
            isq_path,
            apply_scaling=opts.apply_rescale,
            trim_saturated_voxels=trim_saturated_voxels,
            clamp_negative=clamp_negative,
            negative_floor=negative_floor,
        ),
        opts=opts,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a Scanco microCT .isq volume to a filtered .nrrd volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("isq_file", type=Path, help="Path to the .isq file")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: same directory as isq_file)")
    p.add_argument("--apply-scaling", "--apply-rescale", action="store_true",
                   dest="apply_rescale",
                   help="Convert raw scanner counts to linear attenuation units "
                        "(cm^-1) using the header's mu_scaling field, instead of "
                        "keeping raw signed 16-bit scanner values (the ISQ "
                        "counterpart of convert_DICOM_NRRD.py's --apply-rescale, "
                        "which is accepted here as an alias)")
    p.add_argument("--no-trim-saturated", action="store_false",
                   dest="trim_saturated",
                   help="Keep voxels pinned to the int16 rails (-32768 / 32767) "
                        "instead of replacing them with the median of their "
                        "unsaturated neighbours. Both rails are reconstruction "
                        "artifacts - the low one rings the rotation axis, the "
                        "high one is streak artifact - and a single one of them "
                        "sets the volume's min or max, which every contrast "
                        "stretch and min/max normalization downstream inherits")
    p.add_argument("--clamp-negative", action="store_true",
                   help="Raise every voxel below --negative-floor up to it. "
                        "Reconstructed values are signed by design - filtered "
                        "backprojection has negative lobes and air is calibrated "
                        "to a non-zero offset - so this is a presentation choice, "
                        "not a correction. Voxels at or above the floor are "
                        "untouched, so enamel and dentine are bit-identical")
    p.add_argument("--negative-floor", type=int, default=0, metavar="COUNTS",
                   help="Value --clamp-negative raises low voxels to. 0 keeps the "
                        "genuine low tail intact and moves ~0.1%% of voxels by a "
                        "median of ~230 counts. Clamping to the air mode instead "
                        "(~2050 here) would move them ~2280 counts and create a "
                        "discontinuity at zero, so 0 is the safer default")
    add_pipeline_arguments(p)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    setup_logging(args.verbose)

    isq_path: Path = args.isq_file
    if not isq_path.is_file():
        log.error("'%s' is not a file.", isq_path)
        sys.exit(1)

    output_dir = args.out if args.out else isq_path.parent

    outputs = convert(
        isq_path,
        output_dir,
        options_from_args(args, apply_rescale=args.apply_rescale),
        trim_saturated_voxels=args.trim_saturated,
        clamp_negative=args.clamp_negative,
        negative_floor=args.negative_floor,
    )
    report_outputs(outputs)


if __name__ == "__main__":
    main()
