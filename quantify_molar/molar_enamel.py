"""
Molar enamel quantification: how much of each molar is enamel.

Usage:
    python molar_enamel.py -i ".../Segments/P52_Odam-Cre_WT_1" -o results --log
    python molar_enamel.py -i ".../Segments" -o results        # every sample folder
    python molar_enamel.py -i ".../volumes/<sample>_molar_volume.nrrd" -o results

Reads the molar volume written by the segmentation pipeline -- scanner
intensities inside the molar mask, zero everywhere else -- and splits it into
enamel and the rest of the tooth (dentin, cementum, pulp lining). The headline
number is the enamel fraction: enamel volume over total molar volume.

Where enamel starts
-------------------
The enamel is the molar's voxels between its two faces -- the outer surface
and the enamel-dentin junction -- each placed halfway between the materials
on either side, at levels read where the face is
(``enamel_surfaces.enamel_layer``): every voxel follows the intensity uphill
to the enamel's local peak, and the dentin under the enamel is read just past
the junction, along the jaw. These are the surfaces the thickness is measured
between, so the enamel's volume and its thickness agree.

One threshold for the whole molar cannot do this on a blurred scan. A thin
layer never reaches the enamel's full brightness, and a scan can dim along the
jaw: on P82_Odam-Cre_WT_2 (SkyScan, 8 um) the enamel's local peak runs from
158 to 225 (5th-95th percentile), and M1's median is 188. The histogram's
threshold landed there, at 188, and the mask held 7.3% of the molar against
16.8% on its littermate WT_1; between the halfway faces they read 18.5% and
19.5%.

The molar histogram still sets the starting point: its dentin and enamel
peaks, either side of the valley below the enamel's peak on the LOG histogram
(``segmentation.molar_bone._find_enamel_threshold``, the threshold the
segmentation finds the molars with). Enamel is a few percent of the voxels,
too small to register as a peak on a linear histogram, but a second mode on a
log one; where it shows none, the threshold falls back to the histogram's
knee, and ``threshold.method`` says so.

The enamel is found by its shape, as a layer brighter than the dentin it
covers, not by its density. A knockout whose enamel is dimmer keeps its
volume, and the difference shows in its density. ``--mask threshold`` takes
every voxel at or above the histogram's threshold instead, and an absolute
``--threshold`` for the whole cohort (on calibrated scans) measures the volume
above one density.

Enamel density
--------------
The enamel's intensities are summarized by their mean and a lower and upper
percentile (5th and 95th by default, ``--percentiles``), in the volume's own
units and, when the scanner's hydroxyapatite calibration is known, in
mg HA/cm³:

    density = value / mu_scaling * slope + intercept

A Scanco scanner writes that calibration into every ISQ it reconstructs.
``--calibration`` reads it from the ISQ itself, from the converter's
``<stem>_meta.json`` (which copies it and records whether the volume was
rescaled to 1/cm), or from a JSON holding ``slope`` and ``intercept``;
``--ha-slope``/``--ha-intercept`` give one by hand. The conversion is linear,
so the converted mean and percentiles are exactly the raw ones converted.

The lower percentile leans on the partial-volume voxels at the enamel's
surface and at the dentin junction, which read darker than the enamel they
belong to. And the conversion is only as good as the values it is given: the
converter's denoising and sharpening filters move voxel values a little, so
compare densities between scans converted with one recipe.

Enamel thickness
----------------
Between the enamel's outer surface and its junction with the dentin, each
placed at its own halfway intensity, measured in 3-D along lines through
enamel (``enamel.thickness_um``, per tooth too; see ``enamel_surfaces``). The
area of the enamel-dentin junction comes with it. The background outside the
enamel is read from the unmasked scan -- ``--scan``, or the segmentation's
reoriented volume found in the sample folder -- since the molar volume is zero
there.

Teeth
-----
Neighbouring molars touch enamel to enamel (M1 and M2 already do by P22), but
their dentin never touches. The teeth are split from their dentin cores
(``split_teeth``): the cut between two touching teeth follows the seam of their
contact where the scan shows one, and otherwise runs midway between their
enamel-dentin junctions. They are named by size (M1 > M2 > M3), and the area
each pair of neighbours shares, or the gap between them, is reported.

Outputs, per sample, in the output directory:
    <sample>_molar_enamel.json          summary, per-tooth table, contacts, threshold provenance
    <sample>_molar_enamel_teeth.csv     one row per tooth (M1, M2, M3)
    <sample>_molar_enamel_profile.csv   per-slice enamel fraction along the jaw
    <sample>_enamel_mask.nrrd           the enamel voxels, uint8, same frame as the input
    <sample>_tooth_labels.nrrd          the teeth by number (1 = M1, 2 = M2, 3 = M3), same frame
    <sample>_molar_enamel.png           histogram, profile and per-tooth figure
    <sample>_thickness_map.html         the two-surface thickness in 3-D, to turn and read
    <sample>_thickness_map.png          the same map from above and from one side
and, in batch mode, molar_enamel_summary.csv with one row per sample.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

# The segmentation package uses top-level imports (``from data_io import``), so
# its own directory has to be importable, not just the repository root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "segment_mandible"))

# The converters' directory holds the HA calibration reader. Appended rather
# than inserted, so the segmentation's own modules keep precedence.
sys.path.append(os.path.join(_REPO, "tools"))

import enamel_patches  # noqa: E402  (this directory, importable as the script's own)
import enamel_surfaces  # noqa: E402  (this directory, importable as the script's own)
from data_io import load_nrrd, load_tiff, save_mask, save_volume  # noqa: E402
from scanco_calibration import find_calibration, to_density  # noqa: E402
from segmentation.molar_bone import (  # noqa: E402
    _ENAMEL_THRESH_NBINS,
    _ENAMEL_THRESH_SMOOTH_SIGMA,
    _MIN_ENAMEL_SEED_SIZE,
    _find_enamel_threshold,
)

logger = logging.getLogger(__name__)

# 2: enamel.density and the calibration block.
# 3: enamel.thickness_um is measured between the enamel's two surfaces (it was
#    the enamel mask's distance-map ridge), per tooth too, with the area of
#    the enamel-dentin junction.
# 4: teeth are split from their dentin cores (eroding the whole mask never
#    parted enamel that touches), saved as <sample>_tooth_labels.nrrd, with the
#    contacts between them and each crown's extent along the jaw; a contact is
#    each tooth's outer surface for the thickness.
# 5: the enamel lies between its two halfway faces, their levels read along
#    the tooth (it was every voxel above one threshold), described in the
#    ``enamel_mask`` block; the thickness surfaces sit at the same local
#    levels; ``threshold.method`` names the way the histogram's threshold was
#    found (log-valley, knee, ...), not the option asked for.
# 6: ``enamel.patches``: patches of enamel thinner, or less dense, than the
#    enamel around them on the same tooth, per tooth (``enamel_patches``), with
#    the density of the enamel's core clear of its blurred faces;
#    ``thickness_um`` gains the scan's blur at the outer face and each tooth's
#    share of junction points measured.
VERSION = 7   # 7: thickness mean and SD per tooth, measurement lines in the map

# How the enamel mask is made: between the enamel's two halfway faces, at
# levels read where they are, or every voxel at or above the threshold.
MASKS = ("local", "threshold")

# Intensities outside these percentiles of the molar are dropped before the
# histogram is built. One saturated streak or a metal fleck otherwise stretches
# the 128 bins until dentin and enamel share a handful of them. Dropped, not
# clipped: clipping would pile the outliers into the last bin, where they would
# pose as the brightest peak and be taken for enamel.
_HISTOGRAM_PERCENTILES = (0.01, 99.99)

# A tooth smaller than this share of the molar mask is a fragment -- a root tip
# cut off by the background trim, a speck of debris -- not a tooth of its own.
_MIN_TOOTH_FRACTION = 0.02

# Mouse hemimandibles carry three molars. Fewer components than this means
# neighbouring teeth touch and share one component.
_EXPECTED_TEETH = 3

# Erosion depth, in voxels, tried in turn on the dentin (the molar without its
# enamel) until each tooth's core stands apart. Only a voxel or two of blur and
# seam between the enamel layers joins two teeth's dentin: on the twelve P22
# TGM2 scans the cores parted at one or two voxels and stayed three through six.
_MAX_SPLIT_EROSION = 8

# A voxel with enamel no further than this, in voxels, on both sides along an
# axis lies between enamel layers -- the seam of a contact, a fissure's floor --
# however dentin-dim it reads, and never seeds a tooth. The seam of the M1-M2
# contact on P22_TGM2_KO_F_1 is 5 voxels across and reads 10,700-20,000, below
# the enamel threshold; seeded as dentin, it flooded M2's enamel from inside
# the contact and cut M2's enamel in half.
_SEAM_REACH = 6

# A dentin core has to be thicker than this, in voxels, somewhere: two more
# erosions must leave something of it. The blurred rim over the enamel and the
# seam of a contact are dentin-dim too, but no thicker than the blur (across a
# blurred edge, the enamel threshold and the mask's floor lie about two voxels
# apart), so a rim that closes on itself is not taken for a tooth.
_MIN_CORE_DEPTH = 2

# Compactness of the watershed that grows the cores back over the tooth: what
# a voxel of distance costs, in enamel-dentin contrasts, against how much
# dimmer than enamel a voxel is. The floods cross enamel at one pace from every
# junction, so a contact without a seam is cut midway between the two
# junctions. A seam dimmer than the enamel pulls the cut to it; blurred as a
# scan blurs it, it holds the cut within a voxel of its crest, on the
# midline's side (phantoms; 0.1-1.4 voxels from the seam's dimmest point on
# the 13 P22 TGM2 scans). On P22_TGM2_WT_M_4, 0.02 and 0.05 cut all but 569 of
# its 5.5 M voxels alike; the larger keeps the cut steadier where the enamel's
# own density varies (its interior spread is 0.15 of the contrast).
_SPLIT_COMPACTNESS = 0.05

# A crown's first and last slice along the jaw hold at least this share of the
# enamel in its fullest slice: the extent a reader would mark by eye, without a
# stray voxel or two past a cusp tip stretching it.
_CROWN_FLOOR = 0.005

# Smoothing, in voxels, of the intensities that watershed floods.
_SPLIT_SMOOTH_SIGMA = 1.0

# How far past each tooth's dentin, in voxels, the enamel counts as bright
# whatever it reads. The junction's blurred step leaves the first two voxels
# of enamel dim (blur sigma ~1.2 voxels on the scans, plus the smoothing), and
# a flood held there would let whichever tooth's flood cleared its own step
# first run on past the midline before the other had started.
_SPLIT_JUNCTION_VOXELS = 3

# Histogram shipped in the JSON for plotting (not the one the threshold uses).
_REPORT_BINS = 256

# The lower and upper percentile of the enamel's density that are reported.
_DENSITY_PERCENTILES = (5.0, 95.0)

# A converted enamel mean outside this range (mg HA/cm³) is a units mistake --
# a volume already in 1/cm divided by mu_scaling again, a calibration from
# another scanner -- rather than a phenotype. P22_WT_1's enamel reads about
# 1930 and its dentin 1180; the range is kept wide enough for enamel far less
# mineralized than that.
_PLAUSIBLE_ENAMEL_MG_HA = (500.0, 3500.0)

# Margin, in voxels, kept around the molar's box for the thickness:
# the outer surface lies just outside the mask, its far side is read four
# voxels further out, and the background up to eight.
_SURFACE_MARGIN = 12

# Top of the thickness map's colour scale, in um. One value across a cohort
# keeps colours comparable; P22 molar enamel runs to about 100 um.
_THICKNESS_MAX_UM = 120.0


@dataclass
class Tooth:
    """One molar's share of the measurement."""

    name: str
    voxels: int
    volume_mm3: float
    enamel_voxels: int
    enamel_volume_mm3: float
    enamel_fraction: float
    enamel_median: float | None
    dentin_median: float | None
    z_start: int
    z_end: int
    centroid_zyx: list[float] = field(default_factory=list)
    # The crown -- the tooth's enamel -- along the jaw (axis 0), in the input's z:
    # its lowest and highest slice.
    crown_z_start: int | None = None
    crown_z_end: int | None = None
    # The molar's landmark span, as M1-M3 are marked by hand: its crown, less the
    # slices where the crown in front of it (toward M1) still shows, so that
    # neighbours meet without overlapping. Start is the end toward the front of
    # the row. See ``landmark_spans``.
    landmark_z_start: int | None = None
    landmark_z_end: int | None = None
    # Enamel thickness, from the enamel-dentin junction (um): its median, mean
    # and standard deviation over the measured points, its middle half, and
    # that junction's area.
    thickness_median_um: float | None = None
    thickness_mean_um: float | None = None
    thickness_sd_um: float | None = None
    thickness_p25_um: float | None = None
    thickness_p75_um: float | None = None
    junction_area_mm2: float | None = None
    # The mean thickness over the junction's linear size (the square root of
    # its area), both in um: a pure number, the same for a tooth of the same
    # shape at any size (``per_sqrt_junction_area``).
    thickness_mean_per_sqrt_junction: float | None = None
    # The share of the junction's points whose thickness was measured.
    thickness_measured: float | None = None
    # Patches of enamel thinner, or less dense, than the enamel around them
    # (``enamel_patches``): the share of the judged junction they cover, how
    # many there are per mm2 judged, how much thinner or less dense they are
    # on average (a share), and the one-sided spread that needs no threshold.
    thin_patch_share: float | None = None
    thin_patches_per_mm2: float | None = None
    thin_patch_depth: float | None = None
    thickness_shortfall: float | None = None
    low_density_patch_share: float | None = None
    low_density_patches_per_mm2: float | None = None
    low_density_patch_depth: float | None = None
    density_shortfall: float | None = None
    # The enamel's core density, read clear of its blurred faces (raw, and in
    # mg HA/cm3 when calibrated).
    core_density_raw: float | None = None
    core_density_mgHA: float | None = None


# ---------------------------------------------------------------------------
# Loading and naming
# ---------------------------------------------------------------------------

def load_volume(path: str) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Read a molar volume as (Z, Y, X) with its voxel size, if the file has one."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nrrd":
        volume, spacing = load_nrrd(path, return_spacing=True)
    elif ext in (".tif", ".tiff"):
        volume, spacing = load_tiff(path, return_spacing=True)
    else:
        raise SystemExit(f"Unsupported molar volume format {ext!r} (expected .nrrd or .tif)")
    if volume is None:
        raise SystemExit(f"Could not read {path}")
    if spacing is None:
        spacing = _spacing_from_sidecar(path)
    return volume, spacing


def _spacing_from_sidecar(path: str) -> tuple[float, float, float] | None:
    """Voxel size from the segmentation's <sample>_voxel_size.json, if present.

    TIFFs written without resolution tags carry no spacing, but the pipeline
    records it alongside every sample in plain JSON.
    """
    sample_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for candidate in glob.glob(os.path.join(sample_dir, "*voxel_size.json")):
        try:
            with open(candidate, encoding="utf-8") as handle:
                record = json.load(handle)
            if record.get("is_default"):
                continue
            dz, dy, dx = (float(v) for v in record["spacing_mm_zyx"])
            logger.info("Voxel size %s mm from %s", (dz, dy, dx), candidate)
            return dz, dy, dx
        except (OSError, KeyError, TypeError, ValueError):
            continue
    return None


def derive_sample_name(path: str) -> str:
    """The specimen name from a molar volume path.

    The pipeline writes ``<Sample>/volumes/<Sample>_molar_volume.nrrd``; the
    legacy layout wrote a bare ``molar_volume.tif``, whose name lives in the
    enclosing sample folder instead.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in ("molar_volume", "volume"):
        parent = os.path.dirname(os.path.abspath(path))
        if os.path.basename(parent) in ("volumes", "masks"):
            parent = os.path.dirname(parent)
        return os.path.basename(parent) or stem
    for suffix in ("_molar_volume", "_volume"):
        if stem.endswith(suffix) and stem[: -len(suffix)]:
            return stem[: -len(suffix)]
    return stem


MOLAR_PATTERNS = (
    "volumes/*_molar_volume.nrrd",
    "volumes/*_molar_volume.tif",
    "*_molar_volume.nrrd",
    "*_molar_volume.tif",
    "volumes/molar_volume.nrrd",
    "volumes/molar_volume.tif",
)


def find_molar_volume(sample_dir: str) -> str | None:
    """Locate a sample folder's molar volume, newest layout first."""
    for pattern in MOLAR_PATTERNS:
        matches = sorted(glob.glob(os.path.join(sample_dir, pattern)))
        matches = [m for m in matches if not os.path.basename(m).startswith("._")]
        if matches:
            return matches[0]
    return None


SCAN_PATTERNS = ("*_reoriented_volume.nrrd", "*_reoriented_volume.tif")


def find_scan(volume_path: str) -> str | None:
    """The unmasked scan a molar volume was cut from, if it lies in the same sample folder.

    The segmentation writes ``<sample>/<sample>_reoriented_volume.nrrd`` beside
    ``<sample>/volumes/``. The studio may write it beside the raw scan instead,
    and passes it with ``--scan``.
    """
    folder = os.path.dirname(os.path.abspath(volume_path))
    if os.path.basename(folder) in ("volumes", "masks"):
        folder = os.path.dirname(folder)
    for pattern in SCAN_PATTERNS:
        matches = sorted(m for m in glob.glob(os.path.join(folder, pattern))
                         if not os.path.basename(m).startswith("._"))
        if matches:
            return matches[0]
    return None


def discover_samples(input_path: str) -> list[tuple[str, str]]:
    """Resolve ``--input`` into (sample_name, molar_volume_path) pairs."""
    if os.path.isfile(input_path):
        return [(derive_sample_name(input_path), input_path)]
    if not os.path.isdir(input_path):
        raise SystemExit(f"Input path does not exist: {input_path}")

    direct = find_molar_volume(input_path)
    if direct is not None:
        return [(derive_sample_name(direct), direct)]

    samples = []
    for name in sorted(os.listdir(input_path)):
        sample_dir = os.path.join(input_path, name)
        if not os.path.isdir(sample_dir) or name.startswith((".", "_")):
            continue
        volume_path = find_molar_volume(sample_dir)
        if volume_path is None:
            logger.warning("No molar volume in %s -- skipping", sample_dir)
            continue
        samples.append((derive_sample_name(volume_path), volume_path))
    if not samples:
        raise SystemExit(f"No molar volumes found under {input_path}")
    return samples


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

def _histogram_values(values: np.ndarray) -> np.ndarray:
    """Molar intensities with the extreme outliers dropped (see _HISTOGRAM_PERCENTILES)."""
    lo, hi = np.percentile(values, _HISTOGRAM_PERCENTILES)
    kept = values[(values >= lo) & (values <= hi)]
    return kept if kept.size else values


def _smoothed_histogram(values: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, gaussian_filter1d(counts.astype(np.float64), sigma=_ENAMEL_THRESH_SMOOTH_SIGMA)


def histogram_modes(values: np.ndarray, threshold: float,
                    nbins: int = _ENAMEL_THRESH_NBINS) -> tuple[float | None, float | None]:
    """The dentin and enamel peaks either side of the threshold.

    Dentin is the most prominent peak below it -- the bulk of the tooth. The
    enamel mode is simply the tallest bin above it; there may be no clean peak
    there on a hypomineralized tooth, and the tallest bin is still the honest
    answer to "where does most of the enamel sit".
    """
    centres, smoothed = _smoothed_histogram(values, nbins)
    peaks, props = find_peaks(smoothed, prominence=max(float(smoothed.max()) * 0.05, 1e-8))
    below = [(p, prom) for p, prom in zip(peaks, props["prominences"]) if centres[p] < threshold]
    dentin = float(centres[max(below, key=lambda pp: pp[1])[0]]) if below else None
    above = np.flatnonzero(centres >= threshold)
    enamel = float(centres[above[np.argmax(smoothed[above])]]) if above.size else None
    return dentin, enamel


def find_threshold(hist_values: np.ndarray, method: str = "log-valley",
                   fixed: float | None = None, nbins: int = _ENAMEL_THRESH_NBINS,
                   info: dict | None = None) -> float:
    """The molar histogram's enamel threshold: between its dentin and enamel peaks.

    ``hist_values`` are the molar's intensities with outliers already dropped
    (``_histogram_values``). ``info``, when given, is told how the threshold
    was found (``info["method"]``): "fixed", "otsu", or for the log-valley
    method the way it went -- "log-valley", or "knee" where the log histogram
    shows no enamel peak.
    """
    info = {} if info is None else info
    if fixed is not None:
        logger.info("Enamel threshold fixed at %g", fixed)
        info["method"] = "fixed"
        return float(fixed)
    if method == "otsu":
        threshold = float(threshold_otsu(hist_values))
        logger.info("Enamel threshold (Otsu) at %g", threshold)
        info["method"] = "otsu"
        return threshold
    if method != "log-valley":
        raise ValueError(f"Unknown threshold method {method!r}")
    return float(_find_enamel_threshold(hist_values, nbins=nbins, info=info))


# ---------------------------------------------------------------------------
# The enamel
# ---------------------------------------------------------------------------

def _surface_frame(box: tuple[slice, ...], shape: tuple[int, ...]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """The molar's box widened by ``_SURFACE_MARGIN`` within the volume, and where the box lies in it."""
    frame = tuple(slice(max(s.start - _SURFACE_MARGIN, 0), min(s.stop + _SURFACE_MARGIN, n))
                  for s, n in zip(box, shape))
    inner = tuple(slice(s.start - f.start, s.stop - f.start) for s, f in zip(box, frame))
    return frame, inner


def _widen(a: np.ndarray, frame: tuple[slice, ...], inner: tuple[slice, ...]) -> np.ndarray:
    """``a``, cropped to the molar's box, set into the wider frame."""
    out = np.zeros(tuple(f.stop - f.start for f in frame), dtype=a.dtype)
    out[inner] = a
    return out


@dataclass
class EnamelFound:
    """The enamel in the molar's box, and how it was found."""

    enamel: np.ndarray
    specks: int
    threshold: float
    threshold_method: str
    dentin_mode: float | None
    enamel_mode: float | None
    record: dict                    # the summary's ``enamel_mask`` block
    dentin_z: np.ndarray | None     # local mask: the dentin under the enamel per slice of the surface frame
    enamel_ref: float | None = None  # the enamel's brightness the faces were read against
    warnings: list[str] = field(default_factory=list)


def find_enamel(volume: np.ndarray, vol: np.ndarray, molar: np.ndarray, box: tuple[slice, ...],
                hist_values: np.ndarray, *, scan: np.ndarray | None = None, mask: str = "local",
                method: str = "log-valley", fixed_threshold: float | None = None,
                min_enamel_voxels: int = _MIN_ENAMEL_SEED_SIZE) -> EnamelFound:
    """The molar's enamel, in its box.

    ``vol`` and ``molar`` are the molar volume and its mask cropped to
    ``box``, ``hist_values`` its intensities for the histogram. The
    histogram's threshold, and its dentin and enamel peaks either side, come
    first. With ``mask`` "local" the enamel is then the molar's voxels between
    its two halfway faces (``enamel_surfaces.enamel_layer``), read on the
    unmasked ``scan`` when there is one: the molar volume is zero beyond its
    mask, which costs the enamel its blurred outer rim wherever the mask ends
    close to it (2.7% of P82_Odam-Cre_WT_2's enamel). With "threshold", or a
    ``fixed_threshold``, it is every molar voxel at or above the threshold.
    Specks under ``min_enamel_voxels`` go back to the dentin either way.
    """
    if mask not in MASKS:
        raise ValueError(f"Unknown enamel mask {mask!r}")
    warnings: list[str] = []
    info: dict = {}
    threshold = find_threshold(hist_values, method=method, fixed=fixed_threshold, info=info)
    dentin_mode, enamel_mode = histogram_modes(hist_values, threshold)
    if fixed_threshold is not None:
        mask = "threshold"
    if mask == "local" and (dentin_mode is None or enamel_mode is None or not dentin_mode < enamel_mode):
        warnings.append("The molar histogram has no dentin and enamel peak either side of its threshold to read "
                        "the enamel's faces from: the enamel is every voxel at or above the threshold")
        mask = "threshold"

    # The enamel's brightness to read its faces against: the histogram's
    # enamel peak, or the brightest 5% of the molar when that is brighter. A
    # histogram with no enamel peak, not even past its knee, puts its "peak"
    # on the threshold itself: on P42_C67B6_WT_1 at 139, just above the dentin
    # (126) and under most of the enamel (local peaks median 174, the
    # molar's 95th percentile 177). Enamel is 15-35% of a molar, so its
    # brightest 5% is enamel.
    enamel_ref = enamel_mode
    if enamel_mode is not None:
        enamel_ref = max(float(enamel_mode), float(np.percentile(hist_values, 95)))

    dentin_z = None
    if mask == "local":
        frame, inner = _surface_frame(box, volume.shape)
        image = np.asarray((scan if scan is not None else volume)[frame], dtype=np.float32)
        molar_f = _widen(molar, frame, inner)
        background = None
        if scan is not None:
            # Read in a shell beyond the enamel the threshold finds: the
            # background does not depend on where exactly the enamel ends.
            rough = _widen(molar & (vol >= threshold), frame, inner)
            background = enamel_surfaces.background_level(
                image, molar_f, ndi.distance_transform_cdt(~rough, metric="chessboard"))
            del rough
        # Peaks as close as the dentin's own texture are no enamel and dentin
        # to read the faces from (a molar that is mostly enamel, on a phantom).
        if enamel_ref - dentin_mode < enamel_surfaces.contrast_floor(dentin_mode, background or 0.0):
            warnings.append(f"The molar's enamel ({enamel_ref:.5g}) stands too little above its dentin peak "
                            f"({dentin_mode:.5g}) to read the enamel's faces from: the enamel is every voxel at or "
                            "above the threshold")
            mask = "threshold"
    if mask == "local":
        enamel_f, levels, record = enamel_surfaces.enamel_layer(
            image, molar_f, background=0.0 if background is None else background, dentin_level=dentin_mode,
            enamel_level=enamel_ref, background_from_scan=background is not None)
        record["enamel_reference"] = enamel_ref
        enamel = enamel_f[inner]
        dentin_z = levels.dentin_z
        del image, molar_f, enamel_f, levels
    else:
        enamel = molar & (vol >= threshold)
        record = {"method": "threshold", "threshold": float(threshold)}

    labels, _ = _significant_components(enamel, min_enamel_voxels)
    specks = int(enamel.sum()) - int((labels > 0).sum())
    enamel = labels > 0
    del labels
    if specks:
        logger.info("Dropped %d enamel voxel(s) in specks under %d voxels", specks, min_enamel_voxels)
    return EnamelFound(enamel=enamel, specks=specks, threshold=float(threshold),
                       threshold_method=info.get("method", method), dentin_mode=dentin_mode,
                       enamel_mode=enamel_mode, record=record, dentin_z=dentin_z, enamel_ref=enamel_ref,
                       warnings=warnings)


# ---------------------------------------------------------------------------
# Teeth
# ---------------------------------------------------------------------------

def _significant_components(mask: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int]:
    labels, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return labels, 0
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    keep = np.flatnonzero(sizes >= min_voxels)
    relabelled = np.zeros(n + 1, dtype=np.int32)
    relabelled[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    return relabelled[labels], int(keep.size)


def _between_enamel(enamel: np.ndarray, reach: int) -> np.ndarray:
    """The non-enamel voxels with enamel within ``reach`` voxels on both sides along some axis."""
    between = np.zeros(enamel.shape, dtype=bool)
    for axis in range(enamel.ndim):
        n = enamel.shape[axis]
        ahead = np.zeros(enamel.shape, dtype=bool)
        behind = np.zeros(enamel.shape, dtype=bool)
        for k in range(1, min(reach, n - 1) + 1):
            near = tuple(slice(0, n - k) if i == axis else slice(None) for i in range(enamel.ndim))
            far = tuple(slice(k, n) if i == axis else slice(None) for i in range(enamel.ndim))
            ahead[near] |= enamel[far]
            behind[far] |= enamel[near]
        between |= ahead & behind
    return between & ~enamel


def _dentin_cores(dentin: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int]:
    """The pieces of ``dentin`` of at least ``min_voxels`` that are thicker than a blur band somewhere."""
    labels, n = _significant_components(dentin, min_voxels)
    if not n:
        return labels, 0
    deep = ndi.binary_erosion(dentin, structure=ndi.generate_binary_structure(3, 1), iterations=_MIN_CORE_DEPTH)
    thick = np.unique(labels[deep])
    thick = thick[thick > 0]
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[thick] = np.arange(1, thick.size + 1, dtype=np.int32)
    return lut[labels], int(thick.size)


def _keep_largest(labels: np.ndarray, n: int, keep: int) -> np.ndarray:
    """The ``keep`` largest of labels 1..n, renumbered 1..keep; the rest become 0."""
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[np.argsort(sizes)[::-1][:keep]] = np.arange(1, keep + 1, dtype=np.int32)
    return lut[labels]


def split_contrast(dentin_mode: float | None, enamel_mode: float | None, values: np.ndarray) -> float:
    """The enamel-dentin contrast the split's compactness is scaled by.

    The spread of the molar's intensities stands in when either peak is missing.
    """
    if dentin_mode is not None and enamel_mode is not None and enamel_mode > dentin_mode:
        return float(enamel_mode - dentin_mode)
    return float(np.std(values)) or 1.0


def split_teeth(molar: np.ndarray, enamel: np.ndarray, image: np.ndarray, enamel_level: float,
                contrast: float, expected: int = _EXPECTED_TEETH,
                max_erosion: int = _MAX_SPLIT_EROSION) -> tuple[np.ndarray, str]:
    """Label the molar mask one tooth per label.

    Neighbouring molars touch enamel to enamel -- M1 and M2 already do by P22
    -- and a contact is as thick and bright as the rest of the crown, so
    neither a threshold nor eroding the whole tooth parts them (eight voxels of
    erosion left M1 and M2 joined on every P22 TGM2 scan). Their dentin never
    touches. The dentin is the molar without its enamel and without what lies
    between enamel layers -- a contact's seam, which reads as dim as dentin --
    and only a voxel or two of blur still joins it from tooth to tooth, so it
    is eroded a voxel at a time until the expected number of cores stands
    apart.

    Each core, grown back through the dentin to its enamel-dentin junction,
    then seeds a compact watershed of the whole tooth on how much dimmer than
    enamel (``enamel_level``) each voxel is, in contrasts (``contrast``, the
    enamel-dentin difference). Every flood starts at once, and through enamel
    only its distance from its own junction counts, so where two teeth meet
    they stop at the seam if the contact shows one (a line dimmer than
    enamel), within a voxel of it, and otherwise midway between the two
    junctions: off the true contact by half the difference where the two
    enamel layers differ in thickness. (Flooding the inverted intensities
    instead would let whichever flood first clears the dim blur at its own
    junction run on downhill through the enamel before the other has started.)

    ``image`` holds the molar's intensities, in ``molar``'s frame. Returns
    (labels, how), where ``how`` records the erosion depth the cores stood
    apart at; "components" when there was no dentin to seed from.
    """
    structure = ndi.generate_binary_structure(3, 1)
    min_voxels = max(int(_MIN_TOOTH_FRACTION * int(molar.sum())), 1)
    # The dentin: not enamel, and not between enamel layers, where a contact's seam lies.
    dentin = molar & ~enamel & ~_between_enamel(enamel, _SEAM_REACH)
    cores = dentin
    seeds, n, depth = None, 0, 0
    for d in range(max_erosion + 1):
        if d:
            cores = ndi.binary_erosion(cores, structure=structure)
        if not cores.any():
            break
        found, n_found = _dentin_cores(cores, max(min_voxels // 4, 1))
        if n_found > n:
            seeds, n, depth = found, n_found, d
        if n_found >= expected:
            break
    del cores
    if seeds is None:
        labels, _ = _significant_components(molar, min_voxels)
        return labels, "components"
    if n > expected:
        logger.info("%d dentin cores where %d teeth were expected: the %d largest seed the teeth",
                    n, expected, expected)
        seeds, n = _keep_largest(seeds, n, expected), expected
    elif n < expected:
        logger.warning("Found %d dentin core(s) where %d teeth were expected, even after %d voxels "
                       "of erosion", n, expected, max_erosion)

    # The cores, grown back through the dentin as far as they were eroded.
    if depth:
        reach = ndi.binary_dilation(seeds > 0, structure=structure, iterations=depth, mask=dentin)
        seeds = watershed(np.zeros(molar.shape, dtype=np.uint8), markers=seeds, mask=reach, connectivity=1)
        del reach
    smooth = ndi.gaussian_filter(np.asarray(image, dtype=np.float32), _SPLIT_SMOOTH_SIGMA)
    dimness = np.maximum(float(enamel_level) - smooth, 0.0) / float(contrast)
    del smooth
    # Every flood starts at once, and none is held by its own junction's blur.
    dimness[ndi.binary_dilation(seeds > 0, structure=structure, iterations=_SPLIT_JUNCTION_VOXELS)] = 0.0
    labels = watershed(dimness, markers=seeds, mask=molar, compactness=_SPLIT_COMPACTNESS)
    logger.info("Teeth split from %d dentin cores, apart after %d voxel(s) of erosion", n, depth)
    return labels.astype(np.int32), f"dentin-{depth}"


def name_teeth(labels: np.ndarray, n: int) -> dict[int, str]:
    """Map label -> M1/M2/M3, ordered along the jaw.

    Teeth are ordered by centroid along axis 0, the long axis after the
    pipeline's reorientation, but that axis can run either way along the jaw.
    Mouse molars shrink from front to back (M1 > M2 > M3), so whichever end
    holds the larger tooth is M1's end.
    """
    if n == 0:
        return {}
    ids = np.arange(1, n + 1)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    centroids = ndi.center_of_mass(labels > 0, labels, ids)
    order = sorted(ids, key=lambda i: centroids[i - 1][0])
    if len(order) > 1 and sizes[order[-1] - 1] > sizes[order[0] - 1]:
        order = order[::-1]
    prefix = "M" if n <= _EXPECTED_TEETH else "T"
    return {int(label): f"{prefix}{rank}" for rank, label in enumerate(order, start=1)}


def number_teeth(labels: np.ndarray, names: dict[int, str]) -> tuple[np.ndarray, dict[int, str]]:
    """Relabel so each tooth's label is its number (M1 -> 1): the saved map then needs no key."""
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for label, name in names.items():
        lut[label] = int(name[1:])
    return lut[labels], {int(name[1:]): name for name in names.values()}


def crown_extent(enamel_per_slice: np.ndarray) -> tuple[int, int] | None:
    """First and last slice of a tooth's crown, from its enamel voxels per slice along the jaw.

    The crown runs over the slices holding at least ``_CROWN_FLOOR`` of its
    fullest slice's enamel, so a stray voxel or two past a cusp does not
    stretch it. None for a tooth with no enamel.
    """
    peak = int(enamel_per_slice.max()) if enamel_per_slice.size else 0
    if peak == 0:
        return None
    held = np.flatnonzero(enamel_per_slice >= max(_CROWN_FLOOR * peak, 1))
    return int(held[0]), int(held[-1])


def landmark_spans(teeth: list[Tooth | Crown]) -> None:
    """Set each molar's landmark span along the jaw from the crowns, M1 first.

    Where the contact between two teeth is oblique their crowns overlap in z,
    and the landmarks, as marked by hand, give those slices to the tooth in
    front: a molar's span is its crown less the span of the tooth before it.
    On P22_TGM2_KO_F_1 that puts M1-M3 at 975-1216, 835-974 and 722-828
    against 973-1219, 832-970 and 715-829 marked by eye; M2's crown runs on to
    1000, beside M1's. Start is the end toward the front of the row.
    """
    row = sorted((t for t in teeth if t.crown_z_start is not None), key=lambda t: int(t.name[1:]))
    # Which way the row runs along z, from M1 toward the teeth behind it.
    middle = [(t.crown_z_start + t.crown_z_end) / 2 for t in row]
    falling = len(row) > 1 and middle[1] < middle[0]
    previous = None
    for tooth in row:
        lo, hi = tooth.crown_z_start, tooth.crown_z_end
        if previous is not None:
            if falling:
                hi = min(hi, previous[0] - 1)
            else:
                lo = max(lo, previous[1] + 1)
        if lo > hi:                      # wholly beside the tooth in front: no span of its own
            continue
        tooth.landmark_z_start, tooth.landmark_z_end = (hi, lo) if falling else (lo, hi)
        previous = (lo, hi)


@dataclass
class Crown:
    """One molar's crown along the jaw and its landmark span, as on ``Tooth``, without the measurements."""

    name: str
    crown_z_start: int | None
    crown_z_end: int | None
    landmark_z_start: int | None = None
    landmark_z_end: int | None = None


def molar_crowns(volume: np.ndarray, method: str = "log-valley", fixed_threshold: float | None = None,
                 min_enamel_voxels: int = _MIN_ENAMEL_SEED_SIZE, mask: str = "local",
                 scan: np.ndarray | None = None) -> tuple[list[Crown], str]:
    """Each molar's crown along the jaw and its landmark span, from the molar step's own split.

    The enamel, the split and the names are ``quantify``'s, and nothing is
    measured, so M1-M3 can be placed from a molar volume in well under a
    minute without the molar step's thickness. Without the ``scan`` the
    enamel loses its blurred outer rim where the molar mask ends close to it,
    which moved no crown's end by more than a slice on P82_Odam-Cre_WT_2.
    Returns (the crowns, M1 first; how the teeth were split).
    """
    molar_full = volume != 0
    if not molar_full.any():
        raise ValueError("The molar volume is empty: no non-zero voxels")
    box = ndi.find_objects(molar_full.astype(np.uint8))[0]
    vol = np.asarray(volume[box], dtype=np.float32)
    molar = np.asarray(molar_full[box], dtype=bool)
    del molar_full

    hist_values = _histogram_values(vol[molar])
    found = find_enamel(volume, vol, molar, box, hist_values, scan=scan, mask=mask, method=method,
                        fixed_threshold=fixed_threshold, min_enamel_voxels=min_enamel_voxels)
    enamel = found.enamel
    labels, how = split_teeth(molar, enamel, vol, found.threshold if found.enamel_mode is None else found.enamel_mode,
                              split_contrast(found.dentin_mode, found.enamel_mode, hist_values))
    n_teeth = int(labels.max())
    labels, names = number_teeth(labels, name_teeth(labels, n_teeth))

    crowns = []
    for label in range(1, n_teeth + 1):
        crown = crown_extent(((labels == label) & enamel).sum(axis=(1, 2)))
        crowns.append(Crown(name=names[label],
                            crown_z_start=None if crown is None else int(box[0].start + crown[0]),
                            crown_z_end=None if crown is None else int(box[0].start + crown[1])))
    crowns.sort(key=lambda c: int(c.name[1:]))
    landmark_spans(crowns)
    return crowns, how


def _share_a_face(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether a voxel of ``first`` shares a face with a voxel of ``second``."""
    for axis in range(first.ndim):
        lo = tuple(slice(None, -1) if i == axis else slice(None) for i in range(first.ndim))
        hi = tuple(slice(1, None) if i == axis else slice(None) for i in range(first.ndim))
        if np.any(first[lo] & second[hi]) or np.any(second[lo] & first[hi]):
            return True
    return False


def _contact_area(first: np.ndarray, second: np.ndarray, voxel_um: np.ndarray) -> float:
    """Area (mm2) of the surface two touching teeth share.

    The zero level of the difference of their masks, each smoothed by a voxel,
    kept where the two together fill the space: between the teeth, not along
    their outer surfaces, where both fall away together.
    """
    from skimage.measure import marching_cubes

    meeting = first & ndi.binary_dilation(second)
    box = tuple(slice(max(s.start - 4, 0), s.stop + 4) for s in ndi.find_objects(meeting.astype(np.uint8))[0])
    a = ndi.gaussian_filter(first[box].astype(np.float32), 1.0)
    b = ndi.gaussian_filter(second[box].astype(np.float32), 1.0)
    try:
        verts, faces, _, _ = marching_cubes(a - b, level=0.0)
    except (ValueError, RuntimeError):
        return 0.0
    inside = ndi.map_coordinates(a + b, verts[faces].mean(axis=1).T, order=1) >= 0.5
    return float(enamel_surfaces._triangle_areas(verts, faces[inside], voxel_um).sum())  # noqa: SLF001


def tooth_contacts(labels: np.ndarray, names: dict[int, str],
                   spacing: tuple[float, float, float]) -> list[dict]:
    """Where neighbouring teeth meet: the area of the surface they share, or the gap between them.

    ``labels`` are tooth numbers (``number_teeth``); neighbours are consecutive
    numbers, M1-M2 and M2-M3. The area is where the two are fused in the
    scan -- touching, or closer than its blur resolves. The gap is measured
    between the molar masks, between voxel faces; those masks reach about a
    voxel into the blur beyond the enamel, so it reads a voxel or two short
    of the space between the enamel surfaces.
    """
    voxel_um = np.asarray(spacing, dtype=np.float64) * 1000.0
    boxes = ndi.find_objects(labels)
    numbers = sorted(n for n in names if n <= len(boxes) and boxes[n - 1] is not None)
    contacts = []
    for a, b in zip(numbers[:-1], numbers[1:]):
        box = tuple(slice(max(min(p.start, q.start) - 1, 0), max(p.stop, q.stop) + 1)
                    for p, q in zip(boxes[a - 1], boxes[b - 1]))
        sub = labels[box]
        first, second = sub == a, sub == b
        contact = {"teeth": [names[a], names[b]], "touching": _share_a_face(first, second),
                   "contact_area_mm2": 0.0, "gap_um": 0.0}
        if contact["touching"]:
            contact["contact_area_mm2"] = _contact_area(first, second, voxel_um)
        else:
            nearest = float(ndi.distance_transform_edt(~first, sampling=voxel_um)[second].min())
            contact["gap_um"] = max(nearest - float(voxel_um.mean()), 0.0)
        contacts.append(contact)
    return contacts


# ---------------------------------------------------------------------------
# Thickness
# ---------------------------------------------------------------------------

def enamel_thickness(volume: np.ndarray, scan: np.ndarray | None, box: tuple[slice, ...],
                      molar: np.ndarray, enamel: np.ndarray, labels: np.ndarray | None,
                      names: dict[int, str], spacing: tuple[float, float, float],
                      dentin_mode: float | None, enamel_mode: float | None,
                      display: bool = True, dentin_z: np.ndarray | None = None,
                      patches: enamel_patches.Params | None = enamel_patches.Params(),
                      points_out: dict | None = None) -> tuple[dict | None, dict | None, list[str]]:
    """Thickness between the enamel's outer surface and its junction with the dentin,
    and the junction's area.

    See ``enamel_surfaces``. ``molar``, ``enamel`` and ``labels`` are cropped
    to ``box``; the measurement needs a margin around them, and the scan's own
    intensities there when it is given (the molar volume is zero outside its
    mask). ``dentin_z`` is the dentin under the enamel as the enamel mask read
    it, per slice of the same frame. ``patches`` is how the patches of thinner
    or less dense enamel are found (None: not looked for); ``points_out`` is
    filled with each junction point's measurements.
    """
    frame, inner = _surface_frame(box, volume.shape)
    image = np.asarray((scan if scan is not None else volume)[frame], dtype=np.float32)
    return enamel_surfaces.measure(
        image, _widen(molar, frame, inner), _widen(enamel, frame, inner), spacing, dentin_level=dentin_mode,
        enamel_level=enamel_mode, from_scan=scan is not None,
        labels=_widen(labels, frame, inner) if labels is not None else None, names=names,
        display=display, dentin_z=dentin_z, patches=patches, origin=[f.start for f in frame],
        points_out=points_out)


def patches_in_mg_ha(block: dict, calibration: dict | None) -> None:
    """Add the less dense patches' depth and the core density in mg HA/cm³ to ``block``, in place.

    The depth is a difference, so the calibration's offset and intercept drop
    out of it; the core density converts like any value.
    """
    entry = block.get("low_density")
    if not entry:
        return
    for part in [entry, *entry.get("teeth", [])]:
        depth, core = part.get("depth_raw"), part.get("core_density_raw")
        part["depth_mg_ha"] = (float(depth / calibration["divisor"] * calibration["slope"])
                               if calibration is not None and depth is not None else None)
        part["core_density_mg_ha"] = (float(to_density(core, calibration))
                                      if calibration is not None and core is not None else None)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def enamel_density(values: np.ndarray, percentiles: tuple[float, float] = _DENSITY_PERCENTILES,
                   calibration: dict | None = None) -> dict:
    """Mean and lower/upper percentile of the enamel's intensities, raw and in mg HA/cm³.

    The calibration is linear with a positive slope, so it carries the mean and
    every percentile across unchanged in rank: converting the raw summary is
    the same as summarizing the converted voxels.
    """
    out = {"percentiles": [float(p) for p in percentiles], "raw": None, "mg_ha_cm3": None}
    if not values.size:
        return out
    low, high = np.percentile(values, percentiles)
    raw = {"mean": float(values.mean(dtype=np.float64)), "low": float(low), "high": float(high)}
    out["raw"] = raw
    if calibration is not None:
        out["mg_ha_cm3"] = {k: float(to_density(v, calibration)) for k, v in raw.items()}
    return out


def _ordinal(p: float) -> str:
    """5 -> '5th', 1 -> '1st', 2.5 -> '2.5th'."""
    if p != int(p):
        return f"{p:g}th"
    n = int(p)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def quantify(volume: np.ndarray, spacing: tuple[float, float, float] | None, *,
             mask: np.ndarray | None = None, method: str = "log-valley",
             fixed_threshold: float | None = None,
             min_enamel_voxels: int = _MIN_ENAMEL_SEED_SIZE,
             split: bool = True, thickness: bool = True,
             calibration: dict | None = None,
             percentiles: tuple[float, float] = _DENSITY_PERCENTILES,
             scan: np.ndarray | None = None, thickness_map: bool = True,
             enamel_mask: str = "local",
             patches: enamel_patches.Params | None = enamel_patches.Params(),
             save_points: bool = False) -> tuple[dict, dict]:
    """Measure one molar volume.

    ``calibration`` (from ``scanco_calibration.find_calibration``) converts the
    enamel density to mg HA/cm³; without it the density is reported raw only.

    ``scan`` is the unmasked scan the volume was cut from, same frame; the
    enamel's outer face and the thickness read the background outside the
    enamel from it. ``thickness_map`` keeps the meshes for the thickness map.
    ``enamel_mask`` is how the enamel is found (``MASKS``; ``find_enamel``),
    ``method`` the histogram threshold it starts from. ``patches`` is how the
    patches of enamel thinner or less dense than the enamel around them are
    found (``enamel_patches``; None: not looked for); ``save_points`` keeps
    every junction point's measurements in ``arrays["junction_points"]``.

    Returns (summary, arrays): ``summary`` is the JSON-ready record; ``arrays``
    holds the per-slice profile, the enamel mask and the teeth by number (None
    without the split) with their crop box for writing, and the thickness
    map's meshes.
    """
    if scan is not None and scan.shape != volume.shape:
        raise ValueError(f"The scan's shape {scan.shape} does not match the molar volume's {volume.shape}")
    warnings: list[str] = []
    if spacing is None:
        warnings.append("No voxel size in the file or a sidecar: volumes are in voxels (1 mm assumed)")
        spacing = (1.0, 1.0, 1.0)
    spacing = tuple(float(s) for s in spacing)
    voxel_mm3 = float(np.prod(spacing))

    molar_full = mask if mask is not None else volume != 0
    if not molar_full.any():
        raise ValueError("The molar volume is empty: no non-zero voxels")

    # Everything below runs on the molar's bounding box: the saved volume is
    # the whole scan's frame, and the molars fill a small corner of it.
    box = ndi.find_objects(molar_full.astype(np.uint8))[0]
    vol = np.asarray(volume[box], dtype=np.float32)
    molar = np.asarray(molar_full[box], dtype=bool)
    del molar_full

    hist_values = _histogram_values(vol[molar])
    found = find_enamel(volume, vol, molar, box, hist_values, scan=scan, mask=enamel_mask, method=method,
                        fixed_threshold=fixed_threshold, min_enamel_voxels=min_enamel_voxels)
    warnings.extend(found.warnings)
    threshold, dentin_mode, enamel_mode = found.threshold, found.dentin_mode, found.enamel_mode
    enamel, specks = found.enamel, found.specks

    n_total = int(molar.sum())
    n_enamel = int(enamel.sum())
    if n_enamel == 0:
        warnings.append("No enamel found: enamel fraction is 0")
    rest = molar & ~enamel
    enamel_values = vol[enamel]
    rest_values = vol[rest]

    enamel_median = float(np.median(enamel_values)) if n_enamel else None
    ratio = (
        enamel_median / dentin_mode
        if enamel_median is not None and dentin_mode is not None and dentin_mode > 0 else None
    )
    if dentin_mode is not None and dentin_mode <= 0:
        warnings.append("Dentin sits at or below zero intensity, so the density ratio is undefined; "
                        "use enamel_minus_dentin")

    density = enamel_density(enamel_values, percentiles, calibration)
    converted = density["mg_ha_cm3"]
    lo_ok, hi_ok = _PLAUSIBLE_ENAMEL_MG_HA
    if calibration is not None and converted is not None and not lo_ok <= converted["mean"] <= hi_ok:
        warnings.append(
            f"Enamel converts to {converted['mean']:.0f} mg HA/cm³, which no enamel reads: check that "
            f"the calibration belongs to this scan and that the volume holds the scanner's values "
            f"(divided by {calibration['divisor']:g} before calibrating)")

    # Teeth.
    teeth: list[Tooth] = []
    contacts: list[dict] = []
    how = "none"
    labels, names = None, {}
    if split:
        labels, how = split_teeth(molar, enamel, vol, threshold if enamel_mode is None else enamel_mode,
                                  split_contrast(dentin_mode, enamel_mode, hist_values))
        n_teeth = int(labels.max())
        labels, names = number_teeth(labels, name_teeth(labels, n_teeth))
        contacts = tooth_contacts(labels, names, spacing)
        if n_teeth != _EXPECTED_TEETH:
            warnings.append(f"{n_teeth} {'tooth' if n_teeth == 1 else 'teeth'} found where {_EXPECTED_TEETH} were expected; "
                            "per-tooth names may not match M1-M3")
        ids = np.arange(1, n_teeth + 1)
        if n_teeth:
            sizes = ndi.sum_labels(molar, labels, ids)
            enamel_sizes = ndi.sum_labels(enamel, labels, ids)
            centroids = ndi.center_of_mass(molar, labels, ids)
            extents = ndi.find_objects(labels)
            for i, label in enumerate(ids):
                in_tooth = labels == label
                t_enamel = vol[in_tooth & enamel]
                t_rest = vol[in_tooth & ~enamel]
                z_extent = extents[i][0]
                crown = crown_extent((in_tooth & enamel).sum(axis=(1, 2)))
                teeth.append(Tooth(
                    name=names[int(label)],
                    voxels=int(sizes[i]),
                    volume_mm3=float(sizes[i]) * voxel_mm3,
                    enamel_voxels=int(enamel_sizes[i]),
                    enamel_volume_mm3=float(enamel_sizes[i]) * voxel_mm3,
                    enamel_fraction=float(enamel_sizes[i]) / float(sizes[i]) if sizes[i] else 0.0,
                    enamel_median=float(np.median(t_enamel)) if t_enamel.size else None,
                    dentin_median=float(np.median(t_rest)) if t_rest.size else None,
                    z_start=int(box[0].start + z_extent.start),
                    z_end=int(box[0].start + z_extent.stop - 1),
                    centroid_zyx=[float(c + s.start) for c, s in zip(centroids[i], box)],
                    crown_z_start=None if crown is None else int(box[0].start + crown[0]),
                    crown_z_end=None if crown is None else int(box[0].start + crown[1]),
                ))
            teeth.sort(key=lambda t: int(t.name[1:]))
            landmark_spans(teeth)
        tooth_of_slice = _dominant_tooth_per_slice(labels, names)
    else:
        tooth_of_slice = [""] * molar.shape[0]

    # Thickness between the enamel's two surfaces, and the junction's area, per tooth too.
    thickness_summary, thickness_meshes, patch_summary = None, None, None
    points = {} if save_points else None
    if thickness and n_enamel:
        thickness_summary, thickness_meshes, notes = enamel_thickness(
            volume, scan, box, molar, enamel, labels, names, spacing, dentin_mode, found.enamel_ref,
            display=thickness_map, dentin_z=found.dentin_z, patches=patches, points_out=points)
        warnings.extend(notes)
        if thickness_summary is not None:
            patch_summary = thickness_summary.pop("patches", None)
        if patch_summary is not None:
            patches_in_mg_ha(patch_summary, calibration)
        per_tooth = {t["name"]: t for t in (thickness_summary or {}).get("teeth", [])}
        by_kind = {kind: {t["name"]: t for t in ((patch_summary or {}).get(kind) or {}).get("teeth", [])}
                   for kind in ("thin", "low_density")}
        for tooth in teeth:
            stats = per_tooth.get(tooth.name)
            if stats:
                tooth.thickness_median_um = stats["median"]
                tooth.thickness_mean_um = stats.get("mean")
                tooth.thickness_sd_um = stats.get("sd")
                tooth.thickness_p25_um = stats["p25"]
                tooth.thickness_p75_um = stats["p75"]
                tooth.junction_area_mm2 = stats["junction_area_mm2"]
                tooth.thickness_mean_per_sqrt_junction = per_sqrt_junction_area(stats.get("mean"),
                                                                                 stats["junction_area_mm2"])
                tooth.thickness_measured = stats.get("share_measured")
            thin = by_kind["thin"].get(tooth.name)
            if thin:
                tooth.thin_patch_share = thin["share"]
                tooth.thin_patches_per_mm2 = thin["patches_per_mm2"]
                tooth.thin_patch_depth = thin["depth"]
                tooth.thickness_shortfall = thin["shortfall"]
            low = by_kind["low_density"].get(tooth.name)
            if low:
                tooth.low_density_patch_share = low["share"]
                tooth.low_density_patches_per_mm2 = low["patches_per_mm2"]
                tooth.low_density_patch_depth = low["depth"]
                tooth.density_shortfall = low["shortfall"]
                tooth.core_density_raw = low.get("core_density_raw")
                tooth.core_density_mgHA = low.get("core_density_mg_ha")
    teeth_map = labels.astype(np.uint8) if labels is not None else None
    del labels

    # Profile along the long axis.
    molar_per_z = molar.sum(axis=(1, 2))
    enamel_per_z = enamel.sum(axis=(1, 2))
    enamel_sum_z = np.where(enamel, vol, 0.0).sum(axis=(1, 2))
    profile = []
    for i in range(molar.shape[0]):
        m, e = int(molar_per_z[i]), int(enamel_per_z[i])
        if m == 0:
            continue
        z = int(box[0].start + i)
        profile.append({
            "z": z,
            "z_mm": round(z * spacing[0], 5),
            "tooth": tooth_of_slice[i],
            "molar_voxels": m,
            "enamel_voxels": e,
            "enamel_fraction": round(e / m, 6),
            "enamel_mean_intensity": round(float(enamel_sum_z[i]) / e, 4) if e else "",
        })

    counts, edges = np.histogram(hist_values, bins=_REPORT_BINS)

    summary = {
        "version": VERSION,
        "spacing_mm_zyx": list(spacing),
        "voxel_volume_mm3": voxel_mm3,
        "bbox_zyx": [[int(s.start), int(s.stop)] for s in box],
        "threshold": {
            "value": float(threshold),
            "method": found.threshold_method,
            "dentin_mode": dentin_mode,
            "enamel_mode": enamel_mode,
        },
        "enamel_mask": found.record,
        "total": {"voxels": n_total, "volume_mm3": n_total * voxel_mm3},
        "enamel": {
            "voxels": n_enamel,
            "volume_mm3": n_enamel * voxel_mm3,
            "fraction": n_enamel / n_total,
            "median_intensity": enamel_median,
            "mean_intensity": float(enamel_values.mean()) if n_enamel else None,
            "density": density,
            "thickness_um": thickness_summary,
            "patches": patch_summary,
            "specks_dropped": specks,
        },
        "calibration": dict(calibration) if calibration is not None else None,
        "dentin": {
            "voxels": int(rest.sum()),
            "volume_mm3": int(rest.sum()) * voxel_mm3,
            "median_intensity": float(np.median(rest_values)) if rest_values.size else None,
            "mode": dentin_mode,
        },
        "enamel_to_dentin_volume": (n_enamel / int(rest.sum())) if rest.any() else None,
        "enamel_to_dentin_density": ratio,
        "enamel_minus_dentin": (enamel_median - dentin_mode)
        if enamel_median is not None and dentin_mode is not None else None,
        "teeth_split": how,
        "n_teeth": len(teeth),
        "teeth": [asdict(t) for t in teeth],
        "contacts": contacts,
        "histogram": {
            "edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
        "warnings": warnings,
    }
    for message in warnings:
        logger.warning(message)
    arrays = {"profile": profile, "enamel": enamel, "teeth": teeth_map, "box": box,
              "thickness_meshes": thickness_meshes, "junction_points": points or None}
    return summary, arrays


def _dominant_tooth_per_slice(labels: np.ndarray, names: dict[int, str]) -> list[str]:
    """For each z, the name of the tooth holding most of that slice's voxels."""
    n = int(labels.max())
    if n == 0:
        return [""] * labels.shape[0]
    out = []
    for z in range(labels.shape[0]):
        counts = np.bincount(labels[z].ravel(), minlength=n + 1)
        counts[0] = 0
        out.append(names.get(int(counts.argmax()), "") if counts.any() else "")
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

TEETH_COLUMNS = ("sample", "name", "voxels", "volume_mm3", "enamel_voxels", "enamel_volume_mm3",
                 "enamel_fraction", "enamel_median", "dentin_median", "z_start", "z_end",
                 "crown_z_start", "crown_z_end", "landmark_z_start", "landmark_z_end",
                 "thickness_median_um", "thickness_mean_um", "thickness_sd_um", "thickness_p25_um",
                 "thickness_p75_um", "junction_area_mm2", "thickness_mean_per_sqrt_junction",
                 "thickness_measured", "thin_patch_share", "thin_patches_per_mm2", "thin_patch_depth",
                 "thickness_shortfall", "low_density_patch_share", "low_density_patches_per_mm2",
                 "low_density_patch_depth", "density_shortfall", "core_density_raw", "core_density_mgHA")
PROFILE_COLUMNS = ("z", "z_mm", "tooth", "molar_voxels", "enamel_voxels", "enamel_fraction",
                   "enamel_mean_intensity")
SUMMARY_COLUMNS = ("sample", "total_volume_mm3", "enamel_volume_mm3", "dentin_volume_mm3",
                   "enamel_fraction", "enamel_to_dentin_density", "enamel_minus_dentin",
                   "enamel_median", "enamel_mean_mgHA", "enamel_low_mgHA", "enamel_high_mgHA",
                   "enamel_mean_raw", "enamel_low_raw", "enamel_high_raw", "density_percentiles",
                   "calibration", "dentin_mode", "threshold", "threshold_method",
                   "enamel_mask", "junction_level", "dentin_under_enamel",
                   "thickness_median_um", "thickness_mean_um", "thickness_sd_um", "thickness_p25_um",
                   "thickness_p75_um", "thickness_measured", "thickness_background_from", "junction_area_mm2",
                   "thickness_mean_per_sqrt_junction", "n_teeth", "M1_enamel_fraction", "M2_enamel_fraction", "M3_enamel_fraction",
                   "M1_thickness_um", "M2_thickness_um", "M3_thickness_um",
                   "M1_thickness_mean_um", "M2_thickness_mean_um", "M3_thickness_mean_um",
                   "M1_thickness_sd_um", "M2_thickness_sd_um", "M3_thickness_sd_um",
                   "M1_thickness_mean_per_sqrt_junction", "M2_thickness_mean_per_sqrt_junction",
                   "M3_thickness_mean_per_sqrt_junction",
                   "M1_junction_area_mm2", "M2_junction_area_mm2", "M3_junction_area_mm2",
                   "M1_M2_contact_mm2", "M2_M3_contact_mm2", "M1_M2_gap_um", "M2_M3_gap_um",
                   "edge_blur_um", "thin_judged_share", "thin_patch_share", "thin_patches_per_mm2",
                   "thin_patch_depth", "thin_missing", "thickness_shortfall",
                   "low_density_judged_share", "low_density_patch_share", "low_density_patches_per_mm2",
                   "low_density_patch_depth", "low_density_patch_depth_mgHA", "density_shortfall",
                   "core_density_raw", "core_density_mgHA",
                   "M1_thickness_measured", "M2_thickness_measured", "M3_thickness_measured",
                   "M1_thin_patch_share", "M2_thin_patch_share", "M3_thin_patch_share",
                   "M1_low_density_patch_share", "M2_low_density_patch_share", "M3_low_density_patch_share")


def density_columns(summary: dict) -> dict:
    """The enamel density as flat CSV columns: raw and mg HA/cm³, low/high percentile named apart."""
    density = summary["enamel"].get("density") or {}
    raw = density.get("raw") or {}
    mg = density.get("mg_ha_cm3") or {}
    percentiles = density.get("percentiles")
    calibration = summary.get("calibration") or {}
    return {
        "enamel_mean_mgHA": mg.get("mean"), "enamel_low_mgHA": mg.get("low"), "enamel_high_mgHA": mg.get("high"),
        "enamel_mean_raw": raw.get("mean"), "enamel_low_raw": raw.get("low"), "enamel_high_raw": raw.get("high"),
        "density_percentiles": f"{percentiles[0]:g}/{percentiles[1]:g}" if percentiles else None,
        "calibration": os.path.basename(calibration["source"]) if calibration.get("source") else None,
    }


def patch_columns(block: dict | None) -> dict:
    """The patches of the whole row as flat CSV columns; empty for summaries without them."""
    thin = (block or {}).get("thin") or {}
    low = (block or {}).get("low_density") or {}
    return {
        "thin_judged_share": thin.get("share_judged"), "thin_patch_share": thin.get("share"),
        "thin_patches_per_mm2": thin.get("patches_per_mm2"), "thin_patch_depth": thin.get("depth"),
        "thin_missing": thin.get("missing"), "thickness_shortfall": thin.get("shortfall"),
        "low_density_judged_share": low.get("share_judged"), "low_density_patch_share": low.get("share"),
        "low_density_patches_per_mm2": low.get("patches_per_mm2"), "low_density_patch_depth": low.get("depth"),
        "low_density_patch_depth_mgHA": low.get("depth_mg_ha"), "density_shortfall": low.get("shortfall"),
        "core_density_raw": low.get("core_density_raw"), "core_density_mgHA": low.get("core_density_mg_ha"),
    }


def per_sqrt_junction_area(thickness_um: float | None, area_mm2: float | None) -> float | None:
    """Mean thickness over the square root of the enamel-dentin junction's area, both in um.

    The square root of the area is the junction's linear size, so the ratio
    has no unit and does not change when a tooth is scaled up in every
    direction; 0.06 means the enamel is on average 6% as thick as its junction
    is across. For the whole row the area is the three junctions together,
    which is no one tooth's size: the per-tooth values are the proportion.
    """
    if thickness_um is None or not area_mm2 or area_mm2 <= 0:
        return None
    return float(thickness_um) / (float(np.sqrt(area_mm2)) * 1000.0)


def summary_row(sample: str, summary: dict) -> dict:
    """Flatten one sample's summary into a summary-CSV row."""
    thickness = summary["enamel"].get("thickness_um") or {}
    # Summaries before version 5 have no block: their enamel was one threshold.
    found = summary.get("enamel_mask") or {"method": "threshold"}
    row = {
        "enamel_mask": found.get("method"),
        "junction_level": (found.get("junction_level") or {}).get("median"),
        "dentin_under_enamel": (found.get("dentin_under_enamel") or {}).get("median"),
        **density_columns(summary),
        "thickness_p25_um": thickness.get("p25"),
        "thickness_p75_um": thickness.get("p75"),
        "thickness_measured": thickness.get("share_measured"),
        "thickness_background_from": (thickness.get("levels") or {}).get("background_from"),
        "junction_area_mm2": thickness.get("junction_area_mm2"),
        "thickness_mean_per_sqrt_junction": per_sqrt_junction_area(thickness.get("mean"),
                                                                   thickness.get("junction_area_mm2")),
        "sample": sample,
        "total_volume_mm3": summary["total"]["volume_mm3"],
        "enamel_volume_mm3": summary["enamel"]["volume_mm3"],
        "dentin_volume_mm3": summary["dentin"]["volume_mm3"],
        "enamel_fraction": summary["enamel"]["fraction"],
        "enamel_to_dentin_density": summary["enamel_to_dentin_density"],
        "enamel_minus_dentin": summary["enamel_minus_dentin"],
        "enamel_median": summary["enamel"]["median_intensity"],
        "dentin_mode": summary["dentin"]["mode"],
        "threshold": summary["threshold"]["value"],
        "threshold_method": summary["threshold"]["method"],
        "thickness_mean_um": thickness.get("mean"),
        "thickness_sd_um": thickness.get("sd"),
        "thickness_median_um": thickness.get("median"),
        "n_teeth": summary["n_teeth"],
        "edge_blur_um": thickness.get("edge_blur_um"),
        **patch_columns(summary["enamel"].get("patches")),
    }
    for tooth in summary["teeth"]:
        for key, value in ((f"{tooth['name']}_enamel_fraction", tooth["enamel_fraction"]),
                           (f"{tooth['name']}_thickness_um", tooth.get("thickness_median_um")),
                           (f"{tooth['name']}_thickness_mean_um", tooth.get("thickness_mean_um")),
                           (f"{tooth['name']}_thickness_sd_um", tooth.get("thickness_sd_um")),
                           (f"{tooth['name']}_thickness_mean_per_sqrt_junction",
                            tooth.get("thickness_mean_per_sqrt_junction")),
                           (f"{tooth['name']}_junction_area_mm2", tooth.get("junction_area_mm2")),
                           (f"{tooth['name']}_thickness_measured", tooth.get("thickness_measured")),
                           (f"{tooth['name']}_thin_patch_share", tooth.get("thin_patch_share")),
                           (f"{tooth['name']}_low_density_patch_share", tooth.get("low_density_patch_share"))):
            if key in SUMMARY_COLUMNS:
                row[key] = value
    for contact in summary.get("contacts") or []:
        pair = "_".join(contact["teeth"])
        for key, value in ((f"{pair}_contact_mm2", contact["contact_area_mm2"]),
                           (f"{pair}_gap_um", contact["gap_um"])):
            if key in SUMMARY_COLUMNS:
                row[key] = value
    return {k: ("" if row.get(k) is None else row.get(k, "")) for k in SUMMARY_COLUMNS}


def _write_rows(path: str, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(sample: str, summary: dict, arrays: dict, out_dir: str,
                  full_shape: tuple[int, ...], plot: bool = True,
                  thickness_max: float = _THICKNESS_MAX_UM) -> dict[str, str]:
    """Write every per-sample file; returns {kind: path}."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    json_path = os.path.join(out_dir, f"{sample}_molar_enamel.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    written["summary"] = json_path

    teeth_path = os.path.join(out_dir, f"{sample}_molar_enamel_teeth.csv")
    _write_rows(teeth_path, TEETH_COLUMNS, [{"sample": sample, **t} for t in summary["teeth"]])
    written["teeth"] = teeth_path

    profile_path = os.path.join(out_dir, f"{sample}_molar_enamel_profile.csv")
    _write_rows(profile_path, PROFILE_COLUMNS, arrays["profile"])
    written["profile"] = profile_path

    # The mask goes back into the input's full frame, so it overlays the molar
    # volume and the reoriented scan voxel for voxel.
    full = np.zeros(full_shape, dtype=np.uint8)
    full[arrays["box"]] = arrays["enamel"]
    written["mask"] = save_mask(full, "", f"{sample}_enamel_mask", base_dir=out_dir,
                                output_format="nrrd", spacing=tuple(summary["spacing_mm_zyx"]))
    del full

    # The teeth by number (1 = M1, 2 = M2, 3 = M3) in the same frame. Written as
    # a volume: the mask writer would binarize them.
    if arrays.get("teeth") is not None:
        full = np.zeros(full_shape, dtype=np.int16)
        full[arrays["box"]] = arrays["teeth"]
        written["tooth_labels"] = save_volume(full, "", f"{sample}_tooth_labels", base_dir=out_dir,
                                              output_format="nrrd", spacing=tuple(summary["spacing_mm_zyx"]))
        del full

    if plot:
        try:
            written["figure"] = make_plot(sample, summary, arrays["profile"], out_dir)
        except Exception as exc:  # noqa: BLE001 -- a figure must never fail the measurement
            logger.warning("Could not draw the figure: %s", exc)

    # Every junction point's measurements, for running the patch stage again
    # (``enamel_patches.py sweep``) without measuring.
    points = arrays.get("junction_points")
    if points:
        points_path = os.path.join(out_dir, f"{sample}_junction_points.npz")
        np.savez_compressed(points_path, version=np.int64(VERSION), **points)
        written["junction_points"] = points_path

    meshes = arrays.get("thickness_meshes")
    thickness = summary["enamel"].get("thickness_um")
    if meshes and thickness:
        try:
            written["thickness_map"] = enamel_surfaces.write_viewer(
                os.path.join(out_dir, f"{sample}_thickness_map.html"), meshes,
                sample=sample, summary={**thickness, "patches": summary["enamel"].get("patches")},
                vmax=thickness_max)
            if plot:
                written["thickness_figure"] = enamel_surfaces.write_figure(
                    os.path.join(out_dir, f"{sample}_thickness_map.png"), meshes,
                    sample=sample, summary=thickness, vmax=thickness_max)
        except Exception as exc:  # noqa: BLE001 -- a map must never fail the measurement
            logger.warning("Could not draw the thickness map: %s", exc)
    return written


def make_plot(sample: str, summary: dict, profile: list[dict], out_dir: str) -> str:
    """Histogram with where the enamel starts, the along-jaw profile, and per-tooth fractions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    hist = summary["histogram"]
    edges = np.asarray(hist["edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    ax = axes[0]
    ax.fill_between(centres, np.maximum(hist["counts"], 1), step="mid", color="#9ec5f4")
    ax.set_yscale("log")
    junction = (summary.get("enamel_mask") or {}).get("junction_level")
    if junction:
        # Between the halfway faces, the enamel starts at a level of its own
        # along the junction: its middle 90% and its median.
        ax.axvspan(junction["p5"], junction["p95"], color="#e34948", alpha=0.12, lw=0)
        ax.axvline(junction["median"], color="#e34948", lw=1.5,
                   label="enamel-dentin junction level (5th–95th percentile)")
    else:
        ax.axvline(summary["threshold"]["value"], color="#e34948", lw=1.5, label="enamel threshold")
    if summary["threshold"]["dentin_mode"] is not None:
        ax.axvline(summary["threshold"]["dentin_mode"], color="#52514e", lw=1, label="dentin peak")
    density = summary["enamel"].get("density") or {}
    raw = density.get("raw")
    if raw:
        low_p, high_p = density["percentiles"]
        ax.axvspan(raw["low"], raw["high"], color="#2a78d6", alpha=0.12, lw=0,
                   label=f"enamel {_ordinal(low_p)}–{_ordinal(high_p)} percentile")
        ax.axvline(raw["mean"], color="#2a78d6", lw=1, ls="--", label="enamel mean")
    calibration = summary.get("calibration")
    if calibration:
        top = ax.secondary_xaxis("top", functions=(
            lambda v: to_density(v, calibration),
            lambda d: (d - calibration["intercept"]) / calibration["slope"] * calibration["divisor"]
            - calibration.get("offset", 0.0)))
        top.set_xlabel("mg HA/cm³")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Voxels (log)")
    ax.set_title("Molar intensity histogram")
    ax.legend(frameon=False, fontsize="small")

    ax = axes[1]
    if profile:
        z = [p["z_mm"] for p in profile]
        ax.plot(z, [p["enamel_fraction"] for p in profile], color="#2a78d6", lw=2)
    ax.set_xlabel("Position along the jaw (mm)")
    ax.set_ylabel("Enamel fraction of slice")
    ax.set_title("Enamel along the molar row")

    ax = axes[2]
    teeth = summary["teeth"]
    if teeth:
        ax.bar([t["name"] for t in teeth], [t["enamel_fraction"] for t in teeth],
               color="#2a78d6", width=0.5)
    ax.axhline(summary["enamel"]["fraction"], color="#52514e", lw=1, label="whole molar row")
    ax.set_ylabel("Enamel volume / tooth volume")
    ax.set_title("Per tooth")
    ax.legend(frameon=False)

    converted = density.get("mg_ha_cm3")
    fig.suptitle(f"{sample} — enamel {100 * summary['enamel']['fraction']:.1f}% of "
                 f"{summary['total']['volume_mm3']:.3f} mm³"
                 + (f", mean density {converted['mean']:.0f} mg HA/cm³" if converted else ""))
    fig.tight_layout()
    path = os.path.join(out_dir, f"{sample}_molar_enamel.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_calibration(args: argparse.Namespace) -> tuple[dict | None, list[str]]:
    """The HA calibration every sample of this run is converted with, and why there is none.

    A calibration by hand wins over one read from files; ``--mu-scaling`` wins
    over the divisor the files imply.
    """
    if args.ha_slope is not None:
        divisor = float(args.mu_scaling or 1)
        return {"slope": args.ha_slope, "intercept": args.ha_intercept, "units": "mg HA/ccm",
                "mu_scaling": divisor, "divisor": divisor, "offset": 0.0, "rescaled": False,
                "source": "command line"}, []
    if not args.calibration:
        return None, []
    calibration, notes = find_calibration(args.calibration)
    if calibration is None:
        return None, notes
    if args.mu_scaling:
        calibration["divisor"] = float(args.mu_scaling)
    return calibration, notes


def run_sample(sample: str, volume_path: str, out_dir: str, args: argparse.Namespace,
               calibration: dict | None = None, calibration_notes: list[str] | None = None) -> dict:
    logger.info("Measuring %s (%s)", sample, volume_path)
    volume, spacing = load_volume(volume_path)
    if args.voxel_size is not None:
        spacing = tuple(args.voxel_size) * (3 if len(args.voxel_size) == 1 else 1)
    mask = None
    if args.mask:
        mask_volume, _ = load_volume(args.mask)
        if mask_volume.shape != volume.shape:
            raise SystemExit(f"--mask shape {mask_volume.shape} does not match the volume's {volume.shape}")
        mask = mask_volume > 0
        del mask_volume

    # The scan around the tooth: the thickness reads the background from it, and
    # so does the enamel's outer face.
    scan, scan_path = None, None
    if not args.no_thickness or (args.enamel_mask == "local" and args.threshold is None):
        scan_path = args.scan or find_scan(volume_path)
        if scan_path:
            logger.info("Scan around the tooth: %s", scan_path)
            scan, _ = load_volume(scan_path)
            if scan.shape != volume.shape:
                logger.warning("The scan's shape %s does not match the molar volume's %s: not used",
                               scan.shape, volume.shape)
                scan, scan_path = None, None

    summary, arrays = quantify(
        volume, spacing, mask=mask, method=args.method, fixed_threshold=args.threshold,
        min_enamel_voxels=args.min_enamel_voxels, split=not args.no_split,
        thickness=not args.no_thickness, calibration=calibration,
        percentiles=tuple(args.percentiles),
        scan=scan, thickness_map=not args.no_thickness_map, enamel_mask=args.enamel_mask,
        patches=patch_params(args), save_points=args.save_points,
    )
    del scan
    if args.calibration and calibration is None:
        message = ("No HA calibration in " + ", ".join(os.path.basename(p) for p in args.calibration)
                   + ": enamel density is reported in raw intensity only")
        summary["warnings"].append(message)
        logger.warning(message)
    if calibration_notes:
        summary["calibration_notes"] = list(calibration_notes)
    summary.update({
        "sample": sample,
        "input": os.path.abspath(volume_path),
        "mask": os.path.abspath(args.mask) if args.mask else None,
        "scan": os.path.abspath(scan_path) if scan_path else None,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    full_shape = volume.shape
    del volume, mask
    written = write_outputs(sample, summary, arrays, out_dir, full_shape, plot=not args.no_plot,
                            thickness_max=args.thickness_max)

    enamel = summary["enamel"]
    found = summary["enamel_mask"]
    junction = found.get("junction_level")
    how = (f"between its halfway faces, junction at {junction['median']:.5g}" if junction
           else f"threshold {summary['threshold']['value']:g}")
    logger.info(
        "%s: enamel %.4f mm³ of %.4f mm³ total -> %.2f%% (%s; histogram threshold %g, %s); %d teeth",
        sample, enamel["volume_mm3"], summary["total"]["volume_mm3"], 100 * enamel["fraction"], how,
        summary["threshold"]["value"], summary["threshold"]["method"], summary["n_teeth"],
    )
    density = enamel.get("density") or {}
    if density.get("raw"):
        low_p, high_p = density["percentiles"]
        for units, key in (("mg HA/cm³", "mg_ha_cm3"), ("raw", "raw")):
            stats = density.get(key)
            if stats:
                logger.info("  enamel density (%s): mean %.6g, %s-%s percentile %.6g-%.6g", units,
                            stats["mean"], _ordinal(low_p), _ordinal(high_p), stats["low"], stats["high"])
    for tooth in summary["teeth"]:
        thick, mean, sd = tooth.get("thickness_median_um"), tooth.get("thickness_mean_um"), tooth.get("thickness_sd_um")
        thin, low = tooth.get("thin_patch_share"), tooth.get("low_density_patch_share")
        spread = f" (mean {mean:.1f} +/- {sd:.1f})" if mean is not None and sd is not None else ""
        logger.info("  %-3s %.4f mm³, enamel %.2f%%%s%s%s", tooth["name"], tooth["volume_mm3"],
                    100 * tooth["enamel_fraction"],
                    f", thickness median {thick:.1f} um{spread}" if thick is not None else "",
                    f", thin patches {100 * thin:.2f}%" if thin is not None else "",
                    f", less dense patches {100 * low:.2f}%" if low is not None else "")
    for contact in summary.get("contacts") or []:
        pair = "-".join(contact["teeth"])
        if contact["touching"]:
            logger.info("  %s touch over %.4f mm2", pair, contact["contact_area_mm2"])
        else:
            logger.info("  %s %.1f um apart", pair, contact["gap_um"])
    for kind, path in written.items():
        logger.info("  wrote %s: %s", kind, path)
    return summary


def patch_params(args: argparse.Namespace) -> enamel_patches.Params | None:
    """How the patches are found, from the command line; None when they are not looked for."""
    if args.no_patches:
        return None
    return replace(enamel_patches.Params(), around_um=args.patch_around_um, thinner_by=args.thinner_by,
                   less_by=args.less_dense_by, min_area_mm2=args.patch_min_area_mm2, margin_um=args.patch_margin_um)


def setup_logging(log_dir: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(os.path.join(log_dir, f"molar_enamel_{stamp}.log")))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify molar enamel as a share of total molar volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="A molar volume (.nrrd/.tif), a sample folder, or a directory of sample folders")
    parser.add_argument("--output", "-o", default="./molar_results", help="Output directory")
    parser.add_argument("--mask", default=None,
                        help="Molar mask to use instead of the volume's non-zero voxels (single sample only)")
    parser.add_argument("--enamel-mask", choices=MASKS, default="local",
                        help="The enamel: 'local', the voxels between its outer surface and its junction "
                             "with the dentin, each halfway between the materials either side at levels read "
                             "along the tooth; 'threshold', every voxel at or above the histogram's threshold")
    parser.add_argument("--method", choices=("log-valley", "otsu"), default="log-valley",
                        help="The molar histogram's threshold: the segmentation's log-histogram valley, or Otsu. "
                             "Its dentin and enamel peaks either side are where the local mask starts")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Absolute enamel threshold for every sample: the enamel is every voxel at or above "
                             "it, the volume above one density across a cohort of calibrated scans")
    parser.add_argument("--min-enamel-voxels", type=int, default=_MIN_ENAMEL_SEED_SIZE,
                        help="Enamel specks smaller than this are counted as dentin")
    parser.add_argument("--voxel-size", type=float, nargs="+", default=None, metavar="MM",
                        help="Override the voxel size in mm: one value, or dz dy dx")
    parser.add_argument("--percentiles", type=float, nargs=2, default=list(_DENSITY_PERCENTILES),
                        metavar=("LOW", "HIGH"), help="Lower and upper percentile of the enamel density")
    parser.add_argument("--calibration", nargs="+", default=None, metavar="PATH",
                        help="Where the scanner's HA calibration is, tried in order: the converter's "
                             "<stem>_meta.json, the Scanco .isq, or a JSON with slope and intercept. "
                             "Converts the enamel density to mg HA/cm³ for every sample of the run")
    parser.add_argument("--ha-slope", type=float, default=None,
                        help="HA calibration by hand, overriding --calibration: "
                             "mg HA/cm³ = value / mu_scaling * slope + intercept")
    parser.add_argument("--ha-intercept", type=float, default=None, help="Intercept for --ha-slope, mg HA/cm³")
    parser.add_argument("--mu-scaling", type=float, default=None,
                        help="Divide stored values by this before calibrating: 4096 for raw Scanco counts, "
                             "1 for a volume already in 1/cm. Default: what the calibration implies "
                             "(1 for --ha-slope)")
    parser.add_argument("--no-split", action="store_true", help="Skip the per-tooth split")
    parser.add_argument("--no-thickness", action="store_true",
                        help="Skip the enamel thickness, the junction's area and the thickness map")
    parser.add_argument("--scan", default=None,
                        help="The unmasked scan the molar volume was cut from (the segmentation's "
                             "<sample>_reoriented_volume.nrrd). The enamel's outer face and the thickness "
                             "read the background outside the enamel from it. Found in the sample folder "
                             "when left out (single sample only)")
    parser.add_argument("--no-thickness-map", action="store_true",
                        help="Measure the thickness but write no map")
    parser.add_argument("--thickness-max", type=float, default=_THICKNESS_MAX_UM, metavar="UM",
                        help="Top of the thickness map's colour scale, in um. Keep one value across a cohort "
                             "so the colours compare")
    defaults = enamel_patches.Params()
    parser.add_argument("--no-patches", action="store_true",
                        help="Skip the patches of enamel thinner, or less dense, than the enamel around them")
    parser.add_argument("--patch-around-um", type=float, default=defaults.around_um, metavar="UM",
                        help="How far around each point its enamel is compared with: the sigma of the "
                             "Gaussian 'around' is read over, in um")
    parser.add_argument("--thinner-by", type=float, default=defaults.thinner_by, metavar="SHARE",
                        help="A thin patch is at least this share thinner than the enamel around it")
    parser.add_argument("--less-dense-by", type=float, default=defaults.less_by, metavar="SHARE",
                        help="A less dense patch misses at least this share of the mineral around it "
                             "(its core density above the background)")
    parser.add_argument("--patch-min-area-mm2", type=float, default=defaults.min_area_mm2, metavar="MM2",
                        help="The smallest patch, with the holes it encloses")
    parser.add_argument("--patch-margin-um", type=float, default=defaults.margin_um, metavar="UM",
                        help="Leave unjudged the enamel this close to its edge -- the cervical edge, or a bare area "
                             "of the crown -- where it tapers to nothing")
    parser.add_argument("--save-points", action="store_true",
                        help="Also write <sample>_junction_points.npz: every junction point's measurements, for "
                             "running the patch stage again (enamel_patches.py sweep)")
    parser.add_argument("--no-plot", action="store_true", help="Skip the figure")
    parser.add_argument("--per-sample-dir", action="store_true",
                        help="Write each sample into its own <output>/<sample>/ subdirectory")
    parser.add_argument("--log", "-l", action="store_true", help="Also write a log file to the output directory")
    args = parser.parse_args(argv)
    if args.voxel_size is not None and len(args.voxel_size) not in (1, 3):
        parser.error("--voxel-size takes one value or three")
    low, high = args.percentiles
    if not 0 <= low < high <= 100:
        parser.error("--percentiles takes LOW < HIGH, both within 0-100")
    if (args.ha_slope is None) != (args.ha_intercept is None):
        parser.error("--ha-slope and --ha-intercept go together")
    if args.ha_slope is not None and args.ha_slope <= 0:
        parser.error("--ha-slope must be positive")
    if args.mu_scaling is not None and args.mu_scaling <= 0:
        parser.error("--mu-scaling must be positive")
    if args.thickness_max <= 0:
        parser.error("--thickness-max must be positive")
    if args.patch_around_um <= 0 or args.patch_min_area_mm2 <= 0 or args.patch_margin_um < 0:
        parser.error("--patch-around-um and --patch-min-area-mm2 must be positive, --patch-margin-um not negative")
    if not 0 < args.thinner_by < 1 or not 0 < args.less_dense_by < 1:
        parser.error("--thinner-by and --less-dense-by are shares between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    setup_logging(args.output if args.log else None)

    samples = discover_samples(args.input)
    if args.mask and len(samples) > 1:
        raise SystemExit("--mask applies to a single sample, but the input holds several")
    if args.scan and len(samples) > 1:
        raise SystemExit("--scan applies to a single sample, but the input holds several; "
                         "each sample's scan is found in its own folder")
    logger.info("Found %d sample(s): %s", len(samples), ", ".join(n for n, _ in samples))

    calibration, notes = resolve_calibration(args)
    for note in notes:
        logger.info("Calibration: %s", note)
    if calibration is not None:
        logger.info("HA calibration from %s: mg HA/cm³ = (value + %g) / %g x %.6g + %.6g",
                    calibration["source"], calibration["offset"], calibration["divisor"],
                    calibration["slope"], calibration["intercept"])

    rows, failed = [], []
    for sample, path in samples:
        out_dir = os.path.join(args.output, sample) if (args.per_sample_dir or len(samples) > 1) else args.output
        try:
            rows.append(summary_row(sample, run_sample(sample, path, out_dir, args, calibration, notes)))
        except (ValueError, SystemExit) as exc:
            logger.error("%s failed: %s", sample, exc)
            failed.append(sample)

    if len(samples) > 1 and rows:
        summary_path = os.path.join(args.output, "molar_enamel_summary.csv")
        _write_rows(summary_path, SUMMARY_COLUMNS, rows)
        logger.info("Wrote %s", summary_path)
    if failed:
        logger.error("Failed: %s", ", ".join(failed))
    return 1 if failed or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
