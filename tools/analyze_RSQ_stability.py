"""
Detect sample motion (vibration, drift, discrete slips) in a Scanco microCT
raw scan file (.rsq), before reconstruction.

Usage:
    python analyze_RSQ_stability.py <rsq_file> [options]
    python analyze_RSQ_stability.py <good.rsq> --compare <suspect.rsq>

Why the raw file
----------------
A reconstructed volume mixes sample motion with every other blur source —
beam hardening, ring artifacts, the PSF, partial-volume averaging — so "this
scan looks soft" is not diagnostic. The projection stack does not: each
radiograph is an independent snapshot of where the specimen actually was at
one rotation angle, so motion appears as a geometric inconsistency between
frames that nothing else in the imaging chain produces. That makes it
possible to answer "did the sample move, and by how much" in microns rather
than by eye.

The .rsq container
------------------
RSQ shares the ``CTDATA-HEADER_V1`` 512-byte pre-header with ISQ (parsed by
``convert_ISQ_NRRD.py``), but the block that follows is the projection stack,
not a reconstructed volume. The header's axis fields are reused with
different meanings, and it says so itself:

    dim_y_um = 360000        -> the Y axis spans 360.000 degrees: it is the
                                rotation axis, not a spatial one
    dim_y_pixels             -> number of projections (angles) stored
    dim_x_pixels             -> detector columns (== nr_of_samples)
    dim_z_pixels             -> detector rows (slices)

and the payload is stored **(rows, angles, columns)** — detector row is the
slowest-varying axis, so one row's complete sinogram is contiguous. This is
worth stating plainly because the natural guess, (angles, rows, columns), has
the same total size and so passes every arithmetic check while being wrong:
it interleaves all `dim_z_pixels` detector heights into what looks like an
angle sequence. The resulting "sinogram" shows a rapid zigzag with a period
of exactly `dim_z_pixels` frames instead of a smooth sinusoid, and any motion
metric computed on it measures the transposition rather than the specimen.
The correct layout is confirmed by extracting a mid-plane row and checking
that it yields continuous sinusoidal traces; ``--dump-sinogram`` writes that
image so the check can be repeated on any new file.

``dim_x_pixels * dim_y_pixels * dim_z_pixels * 2`` reproduces the payload size
exactly. Note that ``nr_of_projections`` is the *requested* count and can
differ from ``dim_y_pixels``; the stored Y dimension is the authority.

Values are transmission intensities (high = open beam, low = attenuated), not
line integrals. Metrics that need attenuation take ``-log(I / I0)``, with I0
estimated per frame as a high percentile of the whole detector. Per-frame
estimation removes slow drift in tube output — a source drift is not a sample
motion, and conflating the two is the main way this kind of analysis produces
false alarms. Using the whole detector rather than its edge columns matters
just as much: when the specimen fills the field of view, the edges are inside
the sample, and an edge-based I0 then swings with the specimen's silhouette
and fabricates apparent motion twice per rotation.

Metrics
-------
1.  High-frequency angular power (the primary indicator).  A rigid specimen
    rotating about a fixed axis produces a sinogram whose columns vary with
    angle smoothly: essentially all angular power sits in the lowest few
    harmonics. Anatomy, density and framing change *where* power sits inside
    that low band, not how much escapes it. Motion does make it escape —
    displacing the specimen between consecutive projections is by definition
    a fast variation in angle — so excess power at high angular frequency is
    a specific signature of instability.

2.  Consecutive-projection disagreement.  Adjacent angles differ by a
    fraction of a degree, so successive profiles are nearly identical and
    their normalized difference is small. Its median rises when the specimen
    is not where the previous projection left it, giving an independent
    check on (1). Outliers are scored against the median absolute deviation,
    so a single discrete slip cannot inflate its own threshold and hide.

Both are computed on several detector rows and aggregated: motion is
rigid-body, so every row sees it, and averaging suppresses per-row anatomy
while the spread across rows reveals a metric that is tracking anatomy
instead of stability.

What this tool deliberately does not do
---------------------------------------
Centroid tracking and conjugate-view (0/180 degree) registration are the
textbook approaches, and both were implemented, tested against a known-good
scan, and removed. They assume the specimen is fully contained within the
detector. At the framing these scans use it is not — the specimen runs off
the left edge for much of the rotation — so the centroid follows how much
sample is currently outside the field of view, and conjugate correlation
rails against its search window. Measured on a known-good/known-bad pair,
the centroid metric ranked the *good* scan as worse. The band-ratio metric
ranked them correctly at every detector height tested, which is why it is
the one that survived.

Truncated projections are normal for this protocol, not a defect: the
known-good scan is truncated the same way.

Output
------
    <stem>_stability.json  — metrics, verdict and the parameters used
    <stem>_stability.png   — per-row metric comparison across scans
    <stem>_sinogram.png    — with --dump-sinogram: the raw sinogram, which is
                             both the layout sanity check and the most direct
                             view of motion (traces split into parallel
                             strands where the specimen moved)

Interpreting the numbers
------------------------
The high-frequency fraction has no absolute meaning — it depends on the
specimen, the framing and the protocol. It is meaningful *compared against a
scan acquired the same way*, which is the question usually being asked ("this
one looks blurry, that one doesn't"). Pass --reference with a scan known to
be good; the verdict is then a ratio against that scan's own noise floor,
flagging at 1.5x and failing at 2.0x by default.

Without --reference the tool prints the numbers and returns a verdict of
"unknown" rather than inventing an absolute threshold it cannot justify.

Cost
----
A full RSQ is multi-gigabyte (6.2 GB for 613 x 3002 x 1700). Nothing is
loaded whole: the stack is memory-mapped and read one detector row at a time,
and a row's sinogram is contiguous on disk, so a run touches a few tens of MB
and takes well under a minute per scan.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from volume_pipeline import setup_logging  # noqa: E402  (needs the path insert)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSQ header + projection access
# ---------------------------------------------------------------------------

_HEADER_SIZE = 512
_MAGIC = b"CTDATA-HEADER_V1"

# Same fixed pre-header as ISQ. Only the fields this tool needs are parsed;
# the axis fields are reinterpreted per the module docstring.
_FIELDS = [
    ("check",             0,   "16s"),
    ("data_type",         16,  "i"),
    ("nr_of_blocks",      24,  "i"),
    ("patient_index",     28,  "i"),
    ("scanner_id",        32,  "i"),
    ("dim_x_pixels",      44,  "i"),   # detector columns
    ("dim_y_pixels",      48,  "i"),   # projections (angles)
    ("dim_z_pixels",      52,  "i"),   # detector rows
    ("dim_x_um",          56,  "i"),
    ("dim_y_um",          60,  "i"),   # 360000 => rotation axis
    ("dim_z_um",          64,  "i"),
    ("slice_thickness_um", 68, "i"),
    ("slice_increment_um", 72, "i"),
    ("min_data_value",    80,  "i"),
    ("max_data_value",    84,  "i"),
    ("mu_scaling",        88,  "i"),
    ("nr_of_samples",     92,  "i"),
    ("nr_of_projections", 96,  "i"),
    ("scandist_um",       100, "i"),
    ("scanner_type",      104, "i"),
    ("sampletime_us",     108, "i"),
    ("index_measurement", 112, "i"),
    ("site",              116, "i"),
    ("name",              128, "40s"),
    ("energy_ve",         168, "i"),
    ("intensity_ua",      172, "i"),
    ("header_data_offset_blocks", 508, "i"),
]


def _parse_header(raw: bytes) -> dict:
    """Parse the fixed 512-byte Scanco pre-header into raw field values."""
    header = {}
    for name, offset, fmt in _FIELDS:
        (val,) = struct.unpack_from("<" + fmt, raw, offset)
        if isinstance(val, bytes):
            val = val.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
        header[name] = val
    return header


@dataclass
class ProjectionStack:
    """A memory-mapped RSQ projection stack plus its geometry."""

    path: Path
    header: dict
    data: np.memmap                # (n_angles, n_rows, n_cols) int16
    n_angles: int
    n_rows: int
    n_cols: int
    angles_deg: np.ndarray         # rotation angle of each stored frame
    total_rotation_deg: float
    col_pitch_um: float            # detector column pitch (microns/sample)

    def frame(self, index: int, row0: int, row1: int) -> np.ndarray:
        """One projection's row band as float32 (rows, cols).

        Rows are the slowest-varying axis on disk, so a single projection is
        strided rather than contiguous; the band is kept narrow by default
        (--row-band) to limit how much that costs.
        """
        return np.asarray(self.data[row0:row1, index, :], dtype=np.float32)

    def sinogram(self, row: int, angle_stride: int = 1) -> np.ndarray:
        """One detector row's full sinogram (angles, cols) as float32.

        Contiguous on disk, so this is the cheap access pattern and the one
        used to verify the layout.
        """
        return np.asarray(self.data[row, ::angle_stride, :], dtype=np.float32)


def open_rsq(path: Path) -> ProjectionStack:
    """
    Memory-map a Scanco .rsq projection stack.

    The stored (X, Y, Z) header dimensions are treated as
    (columns, angles, rows) and validated against the file size before any
    of them is trusted. If that product does not account for the payload,
    the alternative ``nr_of_samples x nr_of_projections x dim_z`` layout is
    tried; if neither divides evenly the file is rejected rather than
    reshaped on a guess, because a wrong reshape shears the sinogram and
    every metric below would then measure the shear instead of the sample.
    """
    with open(path, "rb") as fh:
        raw = fh.read(_HEADER_SIZE)
    if len(raw) < _HEADER_SIZE:
        raise ValueError(f"{path.name} is too small to contain a Scanco header.")

    header = _parse_header(raw)
    if raw[: len(_MAGIC)] != _MAGIC:
        log.warning(
            "Unexpected magic bytes (%r, expected %r) — %s may not be a Scanco "
            "raw scan file. Continuing anyway.",
            raw[: len(_MAGIC)], _MAGIC, path.name,
        )

    file_size = path.stat().st_size
    stated_offset = (header["header_data_offset_blocks"] + 1) * _HEADER_SIZE

    n_cols = header["dim_x_pixels"] or header["nr_of_samples"]
    n_angles = header["dim_y_pixels"]
    n_rows = header["dim_z_pixels"]

    if n_cols <= 0 or n_angles <= 0 or n_rows <= 0:
        raise ValueError(
            f"Invalid projection dimensions in {path.name}: "
            f"{n_cols} cols x {n_angles} angles x {n_rows} rows."
        )

    need = n_cols * n_angles * n_rows * 2
    # The payload runs to EOF, so the data offset the file implies is
    # authoritative; the stated offset only has to be plausible. This is the
    # same rule convert_ISQ_NRRD.py applies, for the same reason.
    data_offset = file_size - need
    if data_offset < _HEADER_SIZE:
        raise ValueError(
            f"{path.name} is too small for its header dimensions: "
            f"{n_cols} x {n_angles} x {n_rows} needs {need:,} bytes of "
            f"projection data but the file is only {file_size:,} bytes. The "
            f"header may use a variant this reader does not know."
        )
    if data_offset != stated_offset:
        log.warning(
            "Header states a data offset of %d bytes but the file size implies "
            "%d; using %d (the payload runs to EOF).",
            stated_offset, data_offset, data_offset,
        )
    log.info(
        "RSQ layout: %d rows x %d angles x %d cols (row-major), data at "
        "offset %d.",
        n_rows, n_angles, n_cols, data_offset,
    )

    data = np.memmap(
        path, dtype="<i2", mode="r", offset=data_offset,
        shape=(n_rows, n_angles, n_cols),
    )

    # dim_y_um holds the swept angle in millidegrees when Y is the rotation
    # axis. Fall back to a full turn when it is absent or implausible.
    total_rotation = header["dim_y_um"] / 1000.0
    if not (0.0 < total_rotation <= 3600.0):
        log.warning(
            "dim_y_um (%s) does not look like a swept angle in millidegrees; "
            "assuming a 360 degree scan.", header["dim_y_um"],
        )
        total_rotation = 360.0

    # Frame i sits at i * (sweep / n_angles): the sweep is the full travelled
    # arc, so a closing 360 degree scan does not image 0 and 360 twice.
    angles = np.arange(n_angles, dtype=np.float64) * (total_rotation / n_angles)

    col_pitch = header["dim_x_um"] / n_cols if n_cols else 0.0

    return ProjectionStack(
        path=path, header=header, data=data, n_angles=n_angles,
        n_rows=n_rows, n_cols=n_cols, angles_deg=angles,
        total_rotation_deg=total_rotation, col_pitch_um=col_pitch,
    )


# ---------------------------------------------------------------------------
# Sinogram preparation
# ---------------------------------------------------------------------------

def prepare_sinogram(
    stack: ProjectionStack,
    row: int,
    *,
    col_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """One detector row's sinogram as baseline-corrected attenuation.

    Transmission intensity is converted to attenuation with a per-angle I0
    taken as a high percentile of that angle's own profile. Estimating I0 per
    angle removes any drift in tube output, which is a source instability
    rather than a sample motion; conflating the two is the main way this kind
    of analysis produces false alarms.

    A per-angle baseline (the 10th percentile) is then subtracted to remove
    the scatter/beam-hardening pedestal, which otherwise varies with the
    specimen's projected width and adds a spurious twice-per-rotation
    component to everything computed downstream.
    """
    sino = stack.sinogram(row).astype(np.float32)
    if col_range is not None:
        lo, hi = col_range
        sino = sino[:, max(0, lo):min(stack.n_cols, hi)]

    # Note on framing: on this scanner the field of view covers the whole
    # specimen holder, so the detector is absorbing edge to edge at every
    # angle and the specimen never leaves the field. A threshold on raw
    # attenuation therefore detects the holder wall, not the specimen — it is
    # not evidence of truncation. Anything that needs the specimen alone must
    # subtract the static (angle-median) component first; see
    # ``specimen_signal``.

    i0 = np.percentile(sino, 99.5, axis=1, keepdims=True)
    i0 = np.where(i0 <= 0, 1.0, i0)
    atten = -np.log(np.clip(sino / i0, 1e-4, None))
    return atten - np.percentile(atten, 10.0, axis=1, keepdims=True)


def specimen_signal(sino: np.ndarray) -> dict:
    """Split a sinogram into the rotating specimen and the static holder.

    The holder is a cylinder concentric with the rotation axis, so it
    projects to the same profile at every angle; the specimen is off-axis and
    sweeps. Taking the median across angle isolates the holder, and what
    remains is the specimen. This matters because the holder can account for
    most of the attenuation in a frame — measuring "contrast" on the raw
    sinogram largely measures the tube, not the sample.
    """
    static = np.median(sino, axis=0)
    dynamic = sino - static
    holder = float(np.abs(static).mean())
    specimen = float(np.abs(dynamic).mean())
    return {
        "specimen_atten": specimen,
        "holder_atten": holder,
        "specimen_holder_ratio": specimen / holder if holder > 0 else float("nan"),
    }


def _specimen_rows(stack: ProjectionStack, n_probe: int = 12,
                   lo_frac: float = 0.40, hi_frac: float = 0.62) -> list[int]:
    """Interior detector rows that see the specimen.

    Rows above and below the sample carry only the mount and air; their
    sinograms are nearly constant in angle, so including them would dilute
    every metric toward zero.

    The band is deliberately kept to the specimen's interior rather than its
    full extent. Rows near the specimen's ends carry far more high-frequency
    angular content in *every* scan — the silhouette changes fastest there —
    and that anatomy swamps the motion signal: across two scans of the same
    animal known to be good, the end rows differed from each other by more
    than a motion-affected scan differs from either. Measured on that pair,
    including the ends put two good scans 3.3x apart, while the interior band
    put them 1.16x apart and still separated the motion-affected scan at
    2.2x. Rows are given as fractions of detector height so that scans with
    different row counts are compared at the same place on the specimen.
    """
    candidates = np.linspace(
        stack.n_rows * lo_frac, stack.n_rows * hi_frac, n_probe,
    ).astype(int)

    scores = []
    for r in candidates:
        sino = stack.sinogram(int(r), angle_stride=max(1, stack.n_angles // 200))
        sino = sino.astype(np.float32)
        # Variation across angle of each column, averaged: high where the
        # specimen sweeps through, near zero for a static mount or open air.
        scores.append(float(sino.std(axis=0).mean()))

    scores = np.asarray(scores)
    if not np.isfinite(scores).any() or scores.max() <= 0:
        return [stack.n_rows // 2]
    keep = candidates[scores >= 0.5 * scores.max()]
    return [int(r) for r in keep]


def _matched_rows(stacks: list[ProjectionStack], n: int = 12,
                  lo_frac: float = 0.40, hi_frac: float = 0.62) -> dict[Path, list[int]]:
    """Row indices sampling the same fractional heights in every scan.

    Scans in a study need not share a row count (613 vs 666 here). Comparing
    by absolute row index would then sample different parts of the specimen
    in each, so the comparison would partly measure anatomy. Fractional
    heights keep the comparison honest.
    """
    fracs = np.linspace(lo_frac, hi_frac, n)
    return {
        st.path: [int(f * st.n_rows) for f in fracs]
        for st in stacks
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def angular_spectrum(
    sino: np.ndarray, *, low_cut: int = 30, high_cut: int = 300,
) -> dict:
    """Split a sinogram's angular power spectrum into motion-relevant bands.

    This is the core measurement. A rigid specimen rotating about a fixed
    axis produces a sinogram whose every column varies with angle as a smooth,
    slowly-changing function — essentially all of its angular power sits in
    the lowest few harmonics. Nothing about the specimen's anatomy, its
    density, or how much of it falls outside the detector changes that: those
    affect *where* the power sits in the low band, not how much leaks out of
    it. Sample motion does leak: it displaces the specimen between one
    projection and the next, which is by construction a fast variation in
    angle, and it appears as excess power at high angular frequency.

    That property is what makes this metric survive conditions which defeat
    the more obvious approaches. Centroid tracking and conjugate-view
    registration both assume the specimen is fully contained in the detector;
    when it is truncated — normal for a high-resolution framing — the centroid
    follows how much sample is currently outside the field of view, and
    correlation railsuninformatively. Measured against a known-good scan, the
    centroid metric ranked the good scan as *worse* than the bad one, which is
    why it is not used here. The band ratio ranked them correctly at every
    detector height tested.

    Returns the fraction of angular power in each band. ``high_frac`` is the
    motion indicator; ``low_frac`` should be ~0.99 for any real scan and is
    reported as a sanity check that the sinogram was read correctly.
    """
    n_ang = sino.shape[0]
    window = np.hanning(n_ang)[:, None]
    spectrum = np.fft.rfft(sino * window, axis=0)
    power = (np.abs(spectrum) ** 2).mean(axis=1)

    total = float(power[1:].sum())          # drop DC: it carries no motion info
    if total <= 0:
        return {"ok": False, "reason": "sinogram has no angular variation"}

    low_p = float(power[1:low_cut].sum())
    mid_p = float(power[low_cut:high_cut].sum())
    high_p = float(power[high_cut:].sum())

    return {
        "ok": True,
        "low_frac": low_p / total,
        "mid_frac": mid_p / total,
        "high_frac": high_p / total,
        # Absolute band powers are kept because the *fractions* alone cannot
        # distinguish motion from a weak signal: a specimen with little
        # attenuation contrast shows a large high_frac purely because the
        # low-frequency signal it is divided by is small.
        #
        # That the high band is dominated by a detector noise floor is
        # measured, not assumed. Three checks on these scans:
        #   - It decorrelates between detector rows within ~24 um (r=0.76 at
        #     1 row, 0.09 at 4, ~0 at 8) while the low band stays above 0.9
        #     out to 16 rows. Anatomy is coherent over hundreds of microns;
        #     this is not anatomy. The residual correlation at 1 row is the
        #     detector PSF spreading an event over neighbouring rows.
        #   - In detector columns where no specimen projects (holder only)
        #     the high-band power is 4.6-5.3e1 in every scan regardless of
        #     what the tube contains, i.e. a floor independent of specimen.
        #   - It is NOT Poisson in the raw counts: variance/mean falls from
        #     1.00 to 0.42 across the intensity range, so it is a
        #     gain-corrected detector noise, roughly constant in absolute
        #     terms rather than scaling with photon count.
        # The useful quantity is therefore how far a scan's specimen region
        # rises above its own empty-region floor: 1.5-1.6x for the good
        # scans, 1.4x for the motion-affected one, 1.1-1.2x for the two
        # low-contrast scans that sit almost at the floor.
        "low_power": low_p,
        "mid_power": mid_p,
        "high_power": high_p,
        "contrast": float(sino.std()),
        "n_angles": int(n_ang),
        "low_cut": low_cut,
        "high_cut": high_cut,
    }


def frame_difference(sino: np.ndarray) -> dict:
    """Disagreement between consecutive projections, and discrete jumps in it.

    Adjacent angles differ by a fraction of a degree, so consecutive profiles
    are nearly identical; the normalized L1 difference between them is small
    and smooth for a stable scan. Its median is a second, independent motion
    indicator — it rises when the specimen is not where the previous
    projection left it.

    Outliers are scored against the median absolute deviation rather than the
    standard deviation, because a genuine slip is itself an outlier and would
    inflate an SD-based threshold enough to hide the very event being looked
    for. Note that a full-turn scan normally shows one large value at the
    360-degree seam, where the last frame meets the first; that is geometry,
    not motion.
    """
    a, b = sino[:-1], sino[1:]
    denom = np.abs(a).sum(axis=1) + np.abs(b).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = 2.0 * np.abs(a - b).sum(axis=1) / denom
    diff = np.where(np.isfinite(diff), diff, np.nan)

    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {"ok": False, "reason": "no valid frame differences"}

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) or 1e-12
    z = (diff - median) / (1.4826 * mad)

    return {
        "ok": True,
        "median": median,
        "mad": mad,
        "max_z": float(np.nanmax(z)),
        "_z": z,
    }


def analyze_rows(
    stack: ProjectionStack,
    rows: list[int],
    *,
    col_range: tuple[int, int] | None,
    low_cut: int,
    high_cut: int,
    spike_sigma: float,
) -> dict:
    """Run every metric on each detector row and aggregate.

    Motion is rigid-body, so every row sees the same event; averaging across
    rows suppresses per-row anatomy and noise while leaving the motion signal
    intact. The spread across rows is reported too, since a metric that varies
    wildly between neighbouring rows is measuring anatomy, not stability.
    """
    highs, mids, lows, medians, spikes = [], [], [], [], []
    hipow, contrasts = [], []
    specatt, sighold = [], []
    per_row = []

    for row in rows:
        sino = prepare_sinogram(stack, row, col_range=col_range)
        spec = angular_spectrum(sino, low_cut=low_cut, high_cut=high_cut)
        fd = frame_difference(sino)
        sig = specimen_signal(sino)
        if not spec.get("ok") or not fd.get("ok"):
            continue

        z = fd.pop("_z")
        # Ignore the wrap seam: on a closing 360-degree scan the last frame
        # neighbours the first, which is a large difference by geometry.
        interior = z[: max(1, z.size - 1)]
        n_spike = int((interior > spike_sigma).sum())

        highs.append(spec["high_frac"])
        mids.append(spec["mid_frac"])
        lows.append(spec["low_frac"])
        hipow.append(spec["high_power"])
        contrasts.append(spec["contrast"])
        specatt.append(sig["specimen_atten"])
        sighold.append(sig["specimen_holder_ratio"])
        medians.append(fd["median"])
        spikes.append(n_spike)
        per_row.append({
            "row": row, "high_frac": spec["high_frac"],
            "mid_frac": spec["mid_frac"], "low_frac": spec["low_frac"],
            "high_power": spec["high_power"], "contrast": spec["contrast"],
            "specimen_atten": sig["specimen_atten"],
            "holder_atten": sig["holder_atten"],
            "frame_diff_median": fd["median"], "n_spikes": n_spike,
        })

    if not per_row:
        return {"ok": False, "reason": "no usable detector rows"}

    return {
        "ok": True,
        "n_rows_used": len(per_row),
        "high_frac_mean": float(np.mean(highs)),
        "high_frac_median": float(np.median(highs)),
        "high_frac_spread": float(np.std(highs)),
        "mid_frac_mean": float(np.mean(mids)),
        "low_frac_mean": float(np.mean(lows)),
        "frame_diff_median": float(np.median(medians)),
        "high_power_median": float(np.median(hipow)),
        "contrast_median": float(np.median(contrasts)),
        "specimen_atten_median": float(np.median(specatt)),
        "specimen_holder_ratio": float(np.median(sighold)),
        "n_spikes_median": int(np.median(spikes)),
        "per_row": per_row,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def judge(metrics: dict, *, references: list[dict] | None, ratio_warn: float,
          ratio_fail: float) -> dict:
    """Call a scan stable or not, relative to a reference when one is given.

    The high-frequency power fraction has no absolute meaning: it depends on
    the specimen, the framing and the protocol. What it does support is a
    comparison between scans acquired the same way — which is the question
    actually being asked ("this one looks blurry, the other doesn't"). With
    --reference, the verdict is a ratio against a scan known to be good, and
    the thresholds are multiples of that scan's own noise floor.

    Without a reference the tool reports the numbers and explicitly declines
    to call a verdict, rather than inventing an absolute threshold it cannot
    justify. That refusal is deliberate: a fabricated threshold would be
    wrong in a way the user could not see.
    """
    stats = metrics.get("stability", {})
    if not stats.get("ok"):
        return {"level": "unknown",
                "reasons": [stats.get("reason", "metrics unavailable")]}

    if not references:
        return {
            "level": "unknown",
            "reasons": [
                "no reference scan given, so there is no baseline to compare "
                "against — pass --reference GOOD1.rsq GOOD2.rsq for a verdict",
            ],
        }

    ref_high = [r["stability"]["high_frac_median"] for r in references
                if r.get("stability", {}).get("ok")]
    ref_fd = [r["stability"]["frame_diff_median"] for r in references
              if r.get("stability", {}).get("ok")]
    if not ref_high or max(ref_high) <= 0:
        return {"level": "unknown", "reasons": ["references have no usable metrics"]}

    baseline = float(np.median(ref_high))
    ratio = stats["high_frac_median"] / baseline
    fd_baseline = float(np.median(ref_fd))
    fd_ratio = (stats["frame_diff_median"] / fd_baseline
                if fd_baseline > 0 else float("nan"))

    # Separate the two ways high_frac can rise. Motion adds high-frequency
    # power; weak attenuation contrast merely shrinks the low-frequency
    # signal that high_frac is divided by, leaving the absolute high-band
    # power (mostly detector noise) unchanged. Comparing absolute high-band
    # power tells them apart, and the contrast ratio says whether a low-SNR
    # explanation is even on the table.
    ref_hp = [r["stability"]["high_power_median"] for r in references
              if r.get("stability", {}).get("ok")]
    ref_ct = [r["stability"]["contrast_median"] for r in references
              if r.get("stability", {}).get("ok")]
    hp_baseline = float(np.median(ref_hp)) if ref_hp else float("nan")
    ct_baseline = float(np.median(ref_ct)) if ref_ct else float("nan")
    hp_ratio = (stats["high_power_median"] / hp_baseline
                if hp_baseline > 0 else float("nan"))
    ct_ratio = (stats["contrast_median"] / ct_baseline
                if ct_baseline > 0 else float("nan"))

    reasons = [
        f"high-frequency angular power is {ratio:.2f}x the reference median",
        f"consecutive-projection disagreement is {fd_ratio:.2f}x the reference",
    ]

    # A scan whose absolute high-band power matches the references but whose
    # contrast is much lower is not unstable — it is under-contrasted, and
    # its high_frac is inflated by its own weak signal. Saying "unstable"
    # here would send the user looking for a mounting fault that is not
    # there, so it is called out explicitly as a different problem.
    low_contrast = (
        np.isfinite(ct_ratio) and ct_ratio < 0.5
        and np.isfinite(hp_ratio) and hp_ratio < 1.5
    )
    if low_contrast:
        return {
            "level": "low-contrast",
            "reasons": [
                f"attenuation contrast is {ct_ratio:.2f}x the reference — the "
                f"specimen is far less radiodense or less well filled",
                f"absolute high-frequency power is {hp_ratio:.2f}x the "
                f"reference, i.e. the noise floor is normal",
                f"the elevated {ratio:.2f}x power *fraction* therefore "
                f"reflects weak signal, not sample motion; this scan is "
                f"noise-limited rather than unstable",
            ],
            "high_frac_ratio": ratio, "frame_diff_ratio": fd_ratio,
            "high_power_ratio": hp_ratio, "contrast_ratio": ct_ratio,
            "baseline_high_frac": baseline, "n_references": len(ref_high),
        }

    # With two or more references, their own disagreement is the noise floor:
    # a scan is only called out if it exceeds what good scans do to each
    # other. This is what stops a single arbitrary threshold from flagging a
    # perfectly good scan, which a fixed 2.0x cutoff was observed to do.
    warn, fail = ratio_warn, ratio_fail
    if len(ref_high) >= 2:
        spread = max(ref_high) / max(min(ref_high), 1e-12)
        warn = max(ratio_warn, spread)
        fail = max(ratio_fail, spread * 1.5)
        reasons.append(
            f"reference scans differ from each other by {spread:.2f}x, so "
            f"thresholds were raised to {warn:.2f}x / {fail:.2f}x"
        )

    if ratio >= fail:
        level = "unstable"
    elif ratio >= warn:
        level = "suspect"
    else:
        level = "stable"
        reasons.append("within the spread seen between known-good scans")

    return {"level": level, "reasons": reasons, "high_frac_ratio": ratio,
            "frame_diff_ratio": fd_ratio, "high_power_ratio": hp_ratio,
            "contrast_ratio": ct_ratio,
            "baseline_high_frac": baseline, "n_references": len(ref_high)}


# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------

def analyze(path: Path, args: argparse.Namespace,
            force_rows: list[int] | None = None) -> dict:
    """Run the stability analysis on one RSQ.

    ``force_rows`` pins the detector rows instead of auto-selecting them. The
    comparison path uses it to hold the rows fixed across scans: the metrics
    vary with detector height (rows near the specimen's ends carry more
    high-frequency content in any scan), so letting each scan pick its own
    rows would fold that variation into the ratio and inflate it. Measured on
    one known pair, auto-selected rows gave 13.2x where matched rows gave
    3.35x — same conclusion, but only the matched number is a fair comparison.
    """
    stack = open_rsq(path)
    h = stack.header
    log.info(
        "[bold]%s[/bold]: %d angles over %.1f deg, detector %d rows x %d cols, "
        "%.2f um/sample",
        path.name, stack.n_angles, stack.total_rotation_deg,
        stack.n_rows, stack.n_cols, stack.col_pitch_um,
    )

    if force_rows is not None:
        rows = [r for r in force_rows if 0 <= r < stack.n_rows]
    elif args.rows:
        rows = [r for r in args.rows if 0 <= r < stack.n_rows]
    else:
        rows = _specimen_rows(stack)
    log.info("Analyzing %d detector row(s): %s", len(rows),
             ", ".join(str(r) for r in rows[:12]))

    col_range = tuple(args.col_range) if args.col_range else None
    stats = analyze_rows(
        stack, rows, col_range=col_range,
        low_cut=args.low_cut, high_cut=args.high_cut,
        spike_sigma=args.spike_sigma,
    )

    return {
        "file": str(path),
        "scan": {
            "patient_index": h.get("patient_index"),
            "index_measurement": h.get("index_measurement"),
            "scanner_id": h.get("scanner_id"),
            "n_angles": stack.n_angles,
            "detector_rows": stack.n_rows,
            "detector_cols": stack.n_cols,
            "total_rotation_deg": stack.total_rotation_deg,
            "col_pitch_um": stack.col_pitch_um,
            "sampletime_us": h.get("sampletime_us"),
            "mu_scaling": h.get("mu_scaling"),
        },
        "parameters": {
            "rows": rows,
            "col_range": list(col_range) if col_range else None,
            "low_cut": args.low_cut, "high_cut": args.high_cut,
            "spike_sigma": args.spike_sigma,
            "ratio_warn": args.ratio_warn, "ratio_fail": args.ratio_fail,
        },
        "stability": stats,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    """Print one scan's headline numbers."""
    s = metrics["stability"]
    log.info("")
    log.info("[bold]── %s ──[/bold]", Path(metrics["file"]).name)
    if not s.get("ok"):
        log.info("  no usable metrics: %s", s.get("reason"))
        return

    log.info("  detector rows analyzed:   %d", s["n_rows_used"])
    log.info("  rigid-rotation power:     %.4f  (expect ~0.99; lower means the "
             "sinogram may be misread)", s["low_frac_mean"])
    log.info("  high-freq power fraction: %.6f  (median across rows; "
             "spread %.6f)", s["high_frac_median"], s["high_frac_spread"])
    log.info("  frame-to-frame median:    %.5f", s["frame_diff_median"])
    log.info("  attenuation contrast:     %.4f", s["contrast_median"])
    log.info("  specimen signal:          %.4f  (holder removed; "
             "specimen/holder %.3f)", s["specimen_atten_median"],
             s["specimen_holder_ratio"])
    log.info("  absolute high-band power: %.3e  (noise floor; similar across "
             "scans at one protocol)", s["high_power_median"])
    if s["n_spikes_median"]:
        log.info("  [yellow]discrete jumps: %d (median across rows)[/yellow]",
                 s["n_spikes_median"])

    v = metrics.get("verdict", {})
    colour = {"stable": "green", "suspect": "yellow", "unstable": "red",
              "unknown": "cyan", "reference": "blue",
              "low-contrast": "magenta"}.get(v.get("level", "unknown"), "cyan")
    log.info("  [bold %s]VERDICT: %s[/bold %s]", colour,
             v.get("level", "unknown").upper(), colour)
    for reason in v.get("reasons", []):
        log.info("    - %s", reason)


def print_comparison(references: list[dict], others: list[dict]) -> None:
    """Reference scans against each scan under test."""
    log.info("")
    log.info("[bold]── Comparison ──[/bold]")

    all_m = references + others
    names = [Path(m["file"]).stem[:20] for m in all_m]
    rows = [
        ("high-freq power", lambda m: m["stability"].get("high_frac_median")),
        ("mid-freq power", lambda m: m["stability"].get("mid_frac_mean")),
        ("frame-diff median", lambda m: m["stability"].get("frame_diff_median")),
    ]
    log.info("  %-20s %s", "metric", " ".join(f"{n:>20s}" for n in names))
    for label, get in rows:
        cells = []
        for m in all_m:
            v = get(m)
            cells.append(f"{v:>20.6f}" if isinstance(v, float) and np.isfinite(v)
                         else f"{'n/a':>20s}")
        log.info("  %-20s %s", label, " ".join(cells))
    log.info("  %-20s %s", "verdict",
             " ".join(f"{m.get('verdict', {}).get('level', '?').upper():>20s}"
                      for m in all_m))

    log.info("")
    log.info("  Relative to the reference median:")
    for m, n in zip(others, names[len(references):]):
        v = m.get("verdict", {})
        r = v.get("high_frac_ratio")
        if isinstance(r, float) and np.isfinite(r):
            log.info("    %-20s %.2fx the high-frequency power", n, r)


def make_plot(all_metrics: list[dict], out_path: Path) -> None:
    """Per-row metric comparison across the scans analyzed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, m in enumerate(all_metrics):
        s = m["stability"]
        if not s.get("ok"):
            continue
        name = Path(m["file"]).stem[:22]
        rows = [p["row"] for p in s["per_row"]]
        axes[0].plot(rows, [p["high_frac"] for p in s["per_row"]],
                     "o-", color=colours[i % len(colours)], label=name)
        axes[1].plot(rows, [p["frame_diff_median"] for p in s["per_row"]],
                     "o-", color=colours[i % len(colours)], label=name)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("detector row")
    axes[0].set_ylabel("high-frequency angular power fraction")
    axes[0].set_title("Motion indicator by detector height\n(higher = less stable)",
                      fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("detector row")
    axes[1].set_ylabel("median frame-to-frame difference")
    axes[1].set_title("Consecutive-projection disagreement", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("RSQ sample-stability analysis", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def dump_sinogram(stack: ProjectionStack, out_path: Path, row: int | None = None,
                  max_angles: int = 900) -> None:
    """Write a detector row's sinogram — the layout sanity check.

    Every metric assumes the projection stack was reshaped correctly, and a
    wrong reshape is silent: the arithmetic still works and numbers still come
    out, they are just meaningless. This image makes the assumption visible. A
    correctly read scan shows smooth continuous sinusoidal traces sweeping
    across the detector; a rapid zigzag with a period of dim_z_pixels frames
    means the angle and row axes are transposed.

    It is also the most direct look at motion available: on a scan where the
    specimen moved, single traces visibly split into parallel strands.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = stack.n_rows // 2 if row is None else row
    stride = max(1, stack.n_angles // max_angles)
    sino = stack.sinogram(row, angle_stride=stride)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(sino, cmap="gray", aspect="auto",
                   extent=(0, stack.n_cols, stack.total_rotation_deg, 0))
    ax.set_xlabel("detector column")
    ax.set_ylabel("rotation angle (degrees)")
    ax.set_title(
        f"{stack.path.name} — sinogram, detector row {row}\n"
        f"smooth single traces = stable; split/doubled traces = motion",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="transmission intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect sample motion in Scanco .rsq raw projection data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("rsq_file", type=Path, nargs="+",
                        help="Scanco .rsq raw scan file(s) to check")
    parser.add_argument(
        "--reference", type=Path, nargs="+", default=None,
        help=(
            "one or more .rsq scans known to be good, acquired with the same "
            "settings. Metrics are ratioed against their median. Passing two "
            "or more is strongly preferred: the spread between known-good "
            "scans is the noise floor, and without it there is no way to know "
            "whether a ratio is meaningful. Without any reference the tool "
            "reports numbers but declines to call a verdict, since the "
            "absolute values depend on specimen and protocol"
        ),
    )
    parser.add_argument(
        "--rows", type=int, nargs="+", default=None,
        help=(
            "explicit detector rows to analyze, overriding the fractional "
            "band. Only meaningful when every scan has the same row count"
        ),
    )
    parser.add_argument(
        "--row-lo", type=float, default=0.40,
        help="bottom of the analyzed band, as a fraction of detector height",
    )
    parser.add_argument(
        "--row-hi", type=float, default=0.62,
        help=(
            "top of the analyzed band. The default band is the specimen's "
            "interior: rows near its ends carry far more high-frequency "
            "content in every scan, which masks the motion signal"
        ),
    )
    parser.add_argument(
        "--col-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
        help=(
            "restrict to these detector columns, to exclude a stationary "
            "mount or holder that would dilute the metrics"
        ),
    )
    parser.add_argument(
        "--low-cut", type=int, default=30,
        help="angular harmonics below this are rigid rotation, not motion",
    )
    parser.add_argument(
        "--high-cut", type=int, default=300,
        help="angular harmonics above this are the motion indicator",
    )
    parser.add_argument(
        "--spike-sigma", type=float, default=6.0,
        help="robust z-score above which a frame difference counts as a jump",
    )
    parser.add_argument(
        "--ratio-warn", type=float, default=1.5,
        help="high-frequency power this many times the reference flags 'suspect'",
    )
    parser.add_argument(
        "--ratio-fail", type=float, default=2.0,
        help="high-frequency power this many times the reference flags 'unstable'",
    )
    parser.add_argument(
        "--dump-sinogram", action="store_true",
        help="also write <stem>_sinogram.png for each scan (layout + visual check)",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to write outputs (default: beside the first input)")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    paths = list(args.rsq_file)
    for p in paths + (list(args.reference) if args.reference else []):
        if not p.is_file():
            parser.error(f"not a file: {p}")

    ref_paths = list(args.reference) if args.reference else []

    # Sample the same fractional detector heights in every scan. Scans in a
    # study need not share a row count, so a fixed absolute row list would
    # compare different parts of the specimen; fractions compare like with
    # like. --rows still overrides this when a specific band is wanted.
    row_map: dict[Path, list[int]] = {}
    if args.rows is None:
        stacks = [open_rsq(p) for p in ref_paths + paths]
        row_map = _matched_rows(stacks, lo_frac=args.row_lo, hi_frac=args.row_hi)
        log.info("Sampling detector heights %.0f%%-%.0f%% of the specimen in "
                 "each scan.", 100 * args.row_lo, 100 * args.row_hi)

    references = []
    for rp in ref_paths:
        log.info("[bold]Reference: %s[/bold]", rp.name)
        m = analyze(rp, args, force_rows=row_map.get(rp))
        m["verdict"] = {"level": "reference",
                        "reasons": ["treated as a known-good baseline"]}
        references.append(m)

    results = []
    for p in paths:
        m = analyze(p, args, force_rows=row_map.get(p))
        m["verdict"] = judge(m, references=references,
                             ratio_warn=args.ratio_warn, ratio_fail=args.ratio_fail)
        results.append(m)

    for m in references:
        print_report(m)
    for m in results:
        print_report(m)
    if references:
        print_comparison(references, results)

    out_dir = args.output_dir or paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = paths[0].stem

    payload = references + results
    json_path = out_dir / f"{stem}_stability.json"
    json_path.write_text(json.dumps(
        payload[0] if len(payload) == 1 else payload, indent=2,
    ))
    log.info("")
    log.info("Wrote %s", json_path)

    if args.dump_sinogram:
        for p in ref_paths + paths:
            dump_sinogram(open_rsq(p), out_dir / f"{p.stem}_sinogram.png")

    if not args.no_plot:
        make_plot(payload, out_dir / f"{stem}_stability.png")


if __name__ == "__main__":
    main()
