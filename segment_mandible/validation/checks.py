"""
Per-step plausibility checks for the incisor segmentation.

Each ``check_*`` function inspects one pipeline stage and returns a
:class:`CheckReport` -- a list of named pass/warn/fail results with the measured
number attached, so a log line says *why* something is suspect rather than just
that it is.

What these checks are for
-------------------------
The incisor segmentation is a chain of greedy, per-slice decisions: pick the
brightest region on the seed slice, then follow the nearest centroid forward.
Every link can go wrong quietly. The seed can land on a molar cusp, the
centroid chase can hop onto bone where the two touch, and the final region
grow can leak through a partial-volume bridge and return the entire
hemimandible as "incisor". All three produce a mask that looks like *a*
structure -- the failure is only obvious when you know the shape a mouse
incisor is supposed to have.

So the checks are geometric and statistical, not comparisons against a
reference. A mouse incisor is:

- **Small** relative to the mandible -- a few percent of the foreground, never
  most of it. (Catches "it picked up everything".)
- **Long and thin** -- it runs nearly the full anterior-posterior length of the
  hemimandible as a single continuous tube, so it is present on most Z slices
  and its extent along Z dwarfs its cross-section.
- **Connected** -- one component, not a scatter of fragments.
- **Smooth** -- its cross-sectional area and centroid drift gradually from
  slice to slice. A step change means the mask jumped onto a neighbouring
  structure.
- **Dense** -- enamel is the brightest tissue in the scan, so the mask's mean
  intensity should sit above the surrounding bone's.

Thresholds
----------
The limits below are deliberately loose. They are set to catch gross failure,
not to enforce a tight morphometric prior: a warning should mean "open this one
in napari", and a failure should mean "this output is not usable". Every
threshold is a keyword argument so a study with unusual anatomy (a young mouse,
a knockout with a stunted incisor) can widen them per call rather than editing
this file. Where a number came from an anatomical measurement rather than
convention, the comment says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

# The in-plane voxel pitch the centroid-jump limit was calibrated at, used to
# express that limit as a physical distance.
_CENTROID_REFERENCE_PITCH_MM = 0.024


class Severity(Enum):
    """How badly a single check failed."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    """One measurement and the verdict on it."""

    name: str
    severity: Severity
    message: str
    value: float | None = None

    @property
    def ok(self) -> bool:
        return self.severity is Severity.PASS


@dataclass
class CheckReport:
    """
    The results of every check run on one pipeline stage.

    ``failed`` is the signal a caller acts on: a FAIL means the stage produced
    something that cannot be right, whereas a WARN means it is merely unusual
    and worth a look.
    """

    stage: str
    results: list[CheckResult] = field(default_factory=list)

    def add(
        self,
        name: str,
        severity: Severity,
        message: str,
        value: float | None = None,
    ) -> None:
        self.results.append(CheckResult(name, severity, message, value))

    def add_if(
        self,
        name: str,
        *,
        value: float,
        fail_when: bool,
        warn_when: bool = False,
        message: str,
    ) -> None:
        """Record one check, picking the severity from two precomputed predicates."""
        if fail_when:
            severity = Severity.FAIL
        elif warn_when:
            severity = Severity.WARN
        else:
            severity = Severity.PASS
        self.add(name, severity, message, value)

    @property
    def failed(self) -> bool:
        return any(r.severity is Severity.FAIL for r in self.results)

    @property
    def warned(self) -> bool:
        return any(r.severity is Severity.WARN for r in self.results)

    @property
    def problems(self) -> list[CheckResult]:
        """Every non-passing result, worst first."""
        order = {Severity.FAIL: 0, Severity.WARN: 1}
        return sorted(
            (r for r in self.results if not r.ok),
            key=lambda r: order[r.severity],
        )

    def log(self, log: logging.Logger | None = None) -> "CheckReport":
        """
        Emit the report, then return self so it can be chained onto a call.

        Passing checks are logged at DEBUG: on a good run the log should say
        the stage passed and nothing more, and only a problem should be loud.
        """
        log = log or logger
        for result in self.results:
            if result.severity is Severity.FAIL:
                log.error("[%s] %s: %s", self.stage, result.name, result.message)
            elif result.severity is Severity.WARN:
                log.warning("[%s] %s: %s", self.stage, result.name, result.message)
            else:
                log.debug("[%s] %s: %s", self.stage, result.name, result.message)

        if self.failed:
            log.error(
                "%s check FAILED (%d problem(s)) — the output of this step is "
                "very likely wrong; inspect before using it.",
                self.stage, len(self.problems),
            )
        elif self.warned:
            log.warning(
                "%s check passed with %d warning(s) — worth an eyeball.",
                self.stage, len(self.problems),
            )
        else:
            log.info("%s check passed (%d checks)", self.stage, len(self.results))
        return self


# ---------------------------------------------------------------------------
# Shared measurements
# ---------------------------------------------------------------------------

def _mask_stats(mask: np.ndarray) -> dict:
    """
    Per-slice geometry of a 3D mask, measured once and reused by several checks.

    Everything here is along axis 0, which after reorientation is the
    anterior-posterior axis the incisor runs along.
    """
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    occupied = np.flatnonzero(counts)
    return {
        "counts": counts,
        "occupied": occupied,
        "total": int(counts.sum()),
    }


def _largest_component_fraction(mask: np.ndarray) -> tuple[float, int]:
    """
    Fraction of the mask held by its biggest 26-connected component, and the
    component count.

    A clean incisor is one component. A fraction well below 1 means the mask
    shattered, which usually means the centroid chase lost the tooth partway
    along and picked up something else.
    """
    total = int(np.count_nonzero(mask))
    if total == 0:
        return 0.0, 0

    structure = ndi.generate_binary_structure(3, 3)
    labels, n = ndi.label(mask, structure=structure)  # type: ignore[arg-type]
    if n == 0:
        return 0.0, 0
    counts = np.bincount(labels.ravel())[1:]
    return float(counts.max()) / total, int(n)


# ---------------------------------------------------------------------------
# Stage checks
# ---------------------------------------------------------------------------

def check_loaded_volume(
    volume: np.ndarray,
    *,
    min_shape: int = 16,
    max_constant_fraction: float = 0.995,
) -> CheckReport:
    """
    Validate a freshly loaded volume before any processing touches it.

    These are the cheapest checks in the pipeline and they catch the most
    expensive mistakes: a half-loaded series, a stack assembled in the wrong
    axis order, or an all-background volume will otherwise run through an hour
    of denoising before failing somewhere less obvious.

    Parameters:
        min_shape: Smallest plausible size for any axis, in voxels. A volume
            thinner than this is a partial load, not a scan.
        max_constant_fraction: If more than this fraction of voxels share one
            value, the volume carries no structure -- an empty or saturated
            scan.
    """
    report = CheckReport("Loaded volume")

    if volume.ndim != 3:
        report.add(
            "dimensionality", Severity.FAIL,
            f"expected a 3D (Z, Y, X) volume, got {volume.ndim}D with shape {volume.shape}",
            float(volume.ndim),
        )
        return report

    smallest = int(min(volume.shape))
    report.add_if(
        "shape",
        value=float(smallest),
        fail_when=smallest < min_shape,
        message=(
            f"volume shape {volume.shape}"
            + (
                f" — axis of only {smallest} voxels is too thin to be a complete "
                f"scan (expected >= {min_shape})"
                if smallest < min_shape else ""
            )
        ),
    )

    finite = np.isfinite(volume) if np.issubdtype(volume.dtype, np.floating) else None
    if finite is not None and not finite.all():
        bad = int(np.count_nonzero(~finite))
        report.add(
            "finite values", Severity.FAIL,
            f"{bad:,} non-finite voxel(s) (NaN/Inf) in the loaded volume",
            float(bad),
        )

    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax <= vmin:
        report.add(
            "dynamic range", Severity.FAIL,
            f"volume is entirely one value ({vmin:g}) — nothing was loaded",
            0.0,
        )
        return report

    report.add("dynamic range", Severity.PASS, f"intensity range [{vmin:g}, {vmax:g}]", vmax - vmin)

    # A scan that is almost entirely one value has no specimen in it. Sampled
    # rather than measured in full: a multi-GB bincount to answer a yes/no
    # question is not worth the memory.
    sample = volume.ravel()[:: max(1, volume.size // 1_000_000)]
    modal_fraction = float(np.bincount(
        np.searchsorted(np.unique(sample), sample)
    ).max()) / sample.size
    report.add_if(
        "structure",
        value=modal_fraction,
        fail_when=modal_fraction > max_constant_fraction,
        warn_when=modal_fraction > 0.98,
        message=(
            f"{modal_fraction:.1%} of sampled voxels share a single value"
            + (" — the volume looks empty or saturated"
               if modal_fraction > 0.98 else "")
        ),
    )
    return report


def check_preprocessed_volume(
    preprocessed: np.ndarray,
    *,
    min_foreground_fraction: float = 0.001,
    max_foreground_fraction: float = 0.60,
) -> CheckReport:
    """
    Validate the volume after background removal and island cleanup.

    Background removal sets a threshold from the histogram, so it fails in two
    directions: too high and the specimen is erased, too low and the whole
    field of view survives as "foreground" -- which then makes every downstream
    Otsu and region grow meaningless.

    Parameters:
        min_foreground_fraction: Below this share of the volume, the threshold
            ate the specimen.
        max_foreground_fraction: Above it, background survived thresholding. A
            hemimandible in a typical field of view is well under half the
            voxels; 60% leaves generous room for a tightly-cropped scan.
    """
    report = CheckReport("Background removal")

    foreground = preprocessed > 0
    fraction = float(np.count_nonzero(foreground)) / foreground.size

    report.add_if(
        "foreground fraction",
        value=fraction,
        fail_when=fraction < min_foreground_fraction or fraction > max_foreground_fraction,
        warn_when=fraction > 0.40,
        message=(
            f"{fraction:.2%} of voxels survived background removal"
            + (
                " — threshold erased the specimen"
                if fraction < min_foreground_fraction
                else " — threshold left background in; downstream Otsu will be unreliable"
                if fraction > max_foreground_fraction
                else " — higher than a typical hemimandible scan"
                if fraction > 0.40
                else ""
            )
        ),
    )

    if fraction <= 0:
        return report

    # Islands were already removed, so the specimen should dominate what is
    # left. A small largest-component share means the threshold is passing
    # noise that survived the size filter.
    largest, n_components = _largest_component_fraction(foreground)
    report.add_if(
        "connectivity",
        value=largest,
        fail_when=largest < 0.30,
        warn_when=largest < 0.70,
        message=(
            f"largest component holds {largest:.1%} of the foreground "
            f"across {n_components} component(s)"
            + (" — the specimen is not dominant; likely retained noise or a "
               "second object in the field of view"
               if largest < 0.70 else "")
        ),
    )
    return report


def check_reorientation(
    volume: np.ndarray,
    record: dict | None,
    *,
    min_elongation: float = 1.2,
) -> CheckReport:
    """
    Validate the canonical orientation.

    Reorientation is what lets every later step assume "axis 0 runs along the
    tooth, tip at high Z". When tip detection fails it returns the volume
    unrotated and only logs a warning, and the incisor segmentation then walks
    the wrong axis -- producing a mask that fails every downstream check for
    reasons that have nothing to do with the incisor. Checking it here names
    the real cause.

    Parameters:
        min_elongation: Axis 0 should be the longest, by at least this ratio
            over the mean cross-section. Equal axes mean the permutation had
            nothing to work with.
    """
    report = CheckReport("Reorientation")

    shape = volume.shape
    cross_section = (shape[1] + shape[2]) / 2.0
    elongation = shape[0] / cross_section if cross_section else 0.0

    report.add_if(
        "long axis",
        value=elongation,
        fail_when=shape[0] < max(shape[1], shape[2]),
        warn_when=elongation < min_elongation,
        message=(
            f"axis 0 is {shape[0]} voxels vs cross-section {shape[1]}x{shape[2]} "
            f"(elongation {elongation:.2f})"
            + (" — axis 0 is not the longest; the anterior-posterior axis was "
               "not identified and later steps will walk the wrong axis"
               if shape[0] < max(shape[1], shape[2]) else "")
        ),
    )

    report.add_if(
        "flat cross-section",
        value=float(shape[2] - shape[1]),
        fail_when=False,
        warn_when=shape[1] > shape[2],
        message=(
            f"cross-section X={shape[2]} Y={shape[1]}"
            + (" — expected X >= Y (incisor lying flat); the axis 1/2 swap "
               "did not take effect" if shape[1] > shape[2] else "")
        ),
    )

    # reorient_mandible() writes a key per transform it applied. No 'perm',
    # 'flip0' or 'flip1' at all means it bailed before the tip-driven steps.
    if record is not None:
        oriented = any(k in record for k in ("perm", "flip0", "flip1", "swap12"))
        report.add_if(
            "tip detection",
            value=float(len(record)),
            fail_when=False,
            warn_when=not oriented,
            message=(
                f"reorientation applied: {sorted(record) or 'none'}"
                + (" — no orientation transform was recorded; the incisor tip "
                   "was probably not detected, so the volume may be in an "
                   "arbitrary orientation" if not oriented else "")
            ),
        )
    return report


def check_isolation_volume(
    isolation_volume: np.ndarray,
    foreground_region: np.ndarray,
    *,
    min_retained_fraction: float = 0.01,
    max_retained_fraction: float = 0.80,
) -> CheckReport:
    """
    Validate the conservative isolation step that strips bone from around the
    incisor before the real segmentation runs.

    This step subtracts a dilated bone mask, so it fails by subtracting too
    much (the incisor goes with the bone and the next step has no seed) or too
    little (the corridor still contains the bone the incisor touches, which is
    exactly what lets the region grow leak).

    Both counts are taken *within the thresholded foreground*, which is the
    only region the step can act on. Measuring against the whole CLAHE volume
    instead makes the ratio meaningless: CLAHE is a local contrast transform,
    so it lifts air off exact zero across the entire array, and a `> 0` count
    over it returns essentially every voxel in the scan. Bone removal on a
    mouse hemimandible strips a few percent of that denominator no matter how
    well it worked, so the check warned on every run -- a warning that fires
    unconditionally trains the reader to ignore it, which costs more than not
    having the check at all.

    Parameters:
        foreground_region: Boolean mask of the bone-plus-incisor foreground
            isolation was applied to -- the thresholded region, before the
            bone mask was subtracted from it. Voxels outside it are background
            in both volumes and belong in neither count.
        min_retained_fraction: Share of that foreground that must survive.
        max_retained_fraction: Above this, bone removal barely did anything.
    """
    report = CheckReport("Incisor isolation")

    foreground_region = foreground_region.astype(bool, copy=False)
    before = int(np.count_nonzero(foreground_region))
    after = int(np.count_nonzero((isolation_volume > 0) & foreground_region))

    if before == 0:
        report.add(
            "input foreground", Severity.FAIL,
            "the preprocessed volume handed to isolation is empty", 0.0,
        )
        return report

    # No "retained > 1" guard here, unlike the earlier version of this check:
    # the numerator is intersected with the same region that forms the
    # denominator, so it is a subset by construction and the ratio cannot
    # exceed 1. The old check needed that guard precisely because its two
    # counts came from different regions.
    retained = after / before

    report.add_if(
        "retained fraction",
        value=retained,
        fail_when=retained < min_retained_fraction,
        warn_when=retained > max_retained_fraction,
        message=(
            f"{retained:.1%} of the foreground survived bone removal "
            f"({after:,} of {before:,} voxels)"
            + (
                " — almost nothing is left for the incisor pass to seed on"
                if retained < min_retained_fraction
                else " — bone removal barely reduced the volume, so the incisor "
                     "pass may leak into surrounding bone"
                if retained > max_retained_fraction
                else ""
            )
        ),
    )
    return report


def check_incisor_seed(
    seed_slice_index: int | None,
    seed_area: int,
    depth: int,
    *,
    max_seed_area_fraction: float = 0.25,
    slice_area: int | None = None,
) -> CheckReport:
    """
    Validate the seed that the 3D region grow starts from.

    Everything downstream inherits this choice, and a wrong seed is silent: the
    grow will happily produce a full, connected, plausible-looking mask of the
    wrong structure. Two things are worth knowing straight away -- that a seed
    was found at all, and that it is a tooth cross-section rather than a slab
    of bone.

    Parameters:
        seed_slice_index: Z index the seed was taken from, or None if no seed
            was found.
        seed_area: Area of the seed region in voxels.
        depth: Number of slices in the volume.
        slice_area: Voxels per slice, used to judge the seed's relative size.
        max_seed_area_fraction: A seed covering more than this share of its
            slice is not a tooth cross-section.
    """
    report = CheckReport("Incisor seed")

    if seed_slice_index is None or seed_area <= 0:
        report.add(
            "seed found", Severity.FAIL,
            "no slice produced a region bright enough to seed the incisor — "
            "the intensity floor is too high for this scan, or the incisor is "
            "absent from the volume",
            0.0,
        )
        return report

    report.add(
        "seed found", Severity.PASS,
        f"seeded on slice {seed_slice_index} with {seed_area:,} voxels",
        float(seed_area),
    )

    # The seed is taken from the posterior end, where the incisor's apical
    # opening sits well inside the jaw -- so it should be a small, compact
    # cross-section, nothing like a slice-spanning sheet of bone.
    if slice_area:
        fraction = seed_area / slice_area
        report.add_if(
            "seed size",
            value=fraction,
            fail_when=fraction > max_seed_area_fraction,
            warn_when=fraction > 0.10,
            message=(
                f"seed covers {fraction:.2%} of its slice"
                + (" — far too large for an incisor cross-section; the seed "
                   "has almost certainly landed on bone"
                   if fraction > max_seed_area_fraction else "")
            ),
        )

    if depth:
        position = seed_slice_index / depth
        report.add_if(
            "seed position",
            value=position,
            fail_when=False,
            warn_when=position < 0.25,
            message=(
                f"seed sits at {position:.0%} along the anterior-posterior axis"
                + (" — expected the seed near the posterior end (high Z); a "
                   "low-Z seed means the sweep ran a long way before finding "
                   "anything" if position < 0.25 else "")
            ),
        )
    return report


def check_incisor_mask(
    mask: np.ndarray,
    preprocessed: np.ndarray,
    *,
    spacing: tuple[float, float, float] | None = None,
    max_foreground_fraction: float = 0.35,
    min_foreground_fraction: float = 0.005,
    min_slice_coverage: float = 0.40,
    min_elongation: float = 3.0,
    min_component_fraction: float = 0.95,
    max_area_jump: float = 3.0,
    max_centroid_jump_voxels: float = 15.0,
    min_slice_voxels: int = 20,
) -> CheckReport:
    """
    Validate the finished incisor mask -- the main guard on the pipeline.

    Ordered roughly by how badly each failure mis-states the result. The
    "it picked up everything" case is first because it is both the most common
    failure and the one whose output looks most complete.

    Parameters:
        max_foreground_fraction: The incisor's share of the segmented
            foreground. A mouse hemimandible is mostly bone, so the incisor is
            a minority of it; anything above a third means the mask has
            swallowed surrounding structure.
        min_foreground_fraction: Below this the mask is a fragment, not a tooth.
        min_slice_coverage: The incisor runs nearly the full length of the
            hemimandible, so it should appear on most slices. A mask covering
            well under half the slices stopped partway.
        min_elongation: Z extent over mean cross-sectional extent, both taken
            from the mask's bounding box. Note what this does and does not
            measure: the mouse incisor arcs several millimetres laterally along
            its length, and that curvature widens the bounding box without the
            tube itself being any thicker. Measured on six P82 scans, masks
            whose slenderness along their own curved axis is 16-18 score only
            2.3-2.8 here. So a warning at this check means "not a straight
            rod", which a healthy incisor legitimately is not; it is worth
            reading alongside the area-stability and single-component results
            rather than on its own. A genuine blob still scores near 1 and
            fails.
        min_component_fraction: Share of the mask in its largest connected
            component. The incisor is a single structure.
        max_area_jump: Largest tolerated slice-to-slice ratio in
            cross-sectional area. A step change means the mask jumped onto a
            neighbouring structure.
        max_centroid_jump_voxels: Largest tolerated slice-to-slice centroid
            movement, given in voxels at a 24 um pitch. The incisor curves
            gently; it never teleports. When ``spacing`` is known the *measured*
            jump is converted to mm and compared against this limit in mm, so
            the same anatomy gives the same verdict whether it was sampled at
            8 um or 24 um.
        min_slice_voxels: Slices with fewer voxels than this are left out of
            the area-stability and trajectory measurements. Both are ratios or
            centroids, and both stop meaning anything on the few-voxel slices
            at the tooth's tapering ends -- where a change of four voxels reads
            as a multi-fold jump and a centroid is set by whichever voxel
            survives. Raise it if a study's masks taper over a longer run.
    """
    report = CheckReport("Incisor mask")

    mask = mask.astype(bool, copy=False)
    stats = _mask_stats(mask)
    total = stats["total"]

    if total == 0:
        report.add(
            "non-empty", Severity.FAIL,
            "the incisor mask is empty — nothing was segmented", 0.0,
        )
        return report

    foreground = int(np.count_nonzero(preprocessed > 0))
    fraction = total / foreground if foreground else 0.0

    # --- 1. Did it pick up everything? ---
    report.add_if(
        "share of foreground",
        value=fraction,
        fail_when=fraction > max_foreground_fraction or fraction < min_foreground_fraction,
        warn_when=fraction > 0.25,
        message=(
            f"incisor is {fraction:.1%} of the segmented foreground "
            f"({total:,} of {foreground:,} voxels)"
            + (
                " — the mask has grown into the surrounding bone; it is not "
                "an incisor alone"
                if fraction > max_foreground_fraction
                else " — the mask is far too small to be a whole incisor"
                if fraction < min_foreground_fraction
                else " — larger than a typical incisor; check for bone leakage"
                if fraction > 0.25
                else ""
            )
        ),
    )

    # --- 2. One connected structure ---
    largest, n_components = _largest_component_fraction(mask)
    report.add_if(
        "single component",
        value=largest,
        fail_when=largest < 0.80,
        warn_when=largest < min_component_fraction,
        message=(
            f"largest component holds {largest:.1%} of the mask "
            f"({n_components} component(s))"
            + (" — the incisor is one continuous tooth; a fragmented mask means "
               "the slice-to-slice tracking lost it"
               if largest < min_component_fraction else "")
        ),
    )

    # --- 3. Runs the length of the jaw ---
    occupied = stats["occupied"]
    depth = mask.shape[0]
    coverage = len(occupied) / depth if depth else 0.0
    report.add_if(
        "slice coverage",
        value=coverage,
        fail_when=coverage < 0.20,
        warn_when=coverage < min_slice_coverage,
        message=(
            f"present on {len(occupied)} of {depth} slices ({coverage:.0%})"
            + (" — the incisor spans nearly the whole hemimandible, so a mask "
               "this short stopped tracking partway along"
               if coverage < min_slice_coverage else "")
        ),
    )

    # A gap in the middle of the span is worse than a short span: it means the
    # tracker dropped the tooth and re-acquired it somewhere else.
    if len(occupied) > 1:
        span = int(occupied[-1] - occupied[0]) + 1
        gap_fraction = 1.0 - (len(occupied) / span)
        report.add_if(
            "continuity",
            value=gap_fraction,
            fail_when=gap_fraction > 0.25,
            warn_when=gap_fraction > 0.05,
            message=(
                f"{gap_fraction:.1%} of slices inside the mask's Z span are empty"
                + (" — the mask is interrupted; tracking dropped the tooth and "
                   "picked it up again later" if gap_fraction > 0.05 else "")
            ),
        )

    # --- 4. Long and thin, in physical units where we have them ---
    coords = np.argwhere(mask)
    extents = coords.max(axis=0) - coords.min(axis=0) + 1
    if spacing is not None:
        extents_mm = np.asarray(extents, dtype=float) * np.asarray(spacing, dtype=float)
    else:
        extents_mm = np.asarray(extents, dtype=float)
    cross = float(extents_mm[1:].mean())
    elongation = float(extents_mm[0]) / cross if cross else 0.0
    report.add_if(
        "elongation",
        value=elongation,
        fail_when=elongation < 1.5,
        warn_when=elongation < min_elongation,
        message=(
            f"Z extent {extents_mm[0]:.3g} over mean cross-section {cross:.3g} "
            f"= {elongation:.2f}"
            + (" — the incisor is a long thin tube; this mask is too blocky to "
               "be one" if elongation < min_elongation else "")
        ),
    )

    # --- 5. Smooth cross-section and trajectory ---
    counts = stats["counts"]
    present = counts[occupied]

    # Measure area stability only where the cross-section is big enough for a
    # ratio to carry information. At the tooth's tapering tips a slice holds a
    # handful of voxels, and there a change of four or five voxels -- 2 -> 7,
    # say -- is a 3.5x "jump" that means nothing anatomically. Comparing only
    # occupied slices does not avoid this, because the tip slices are occupied;
    # the ratio has to be restricted by area, not by occupancy. Measured on six
    # P82 scans every spurious flag sat on the first or last slice of the span,
    # while the interior stayed within 1.2-1.5x.
    substantial = occupied[counts[occupied] >= min_slice_voxels]
    present = counts[substantial]
    if len(present) > 2:
        ratios = present[1:] / np.maximum(present[:-1], 1)
        worst = float(np.maximum(ratios, 1.0 / np.maximum(ratios, 1e-9)).max())
        worst_at = int(substantial[1:][np.argmax(np.maximum(ratios, 1.0 / np.maximum(ratios, 1e-9)))])
        report.add_if(
            "area stability",
            value=worst,
            fail_when=worst > max_area_jump * 2,
            warn_when=worst > max_area_jump,
            message=(
                f"largest slice-to-slice area change is {worst:.1f}x at slice {worst_at}"
                + (" — a step change in cross-section means the mask jumped onto "
                   "an adjacent structure" if worst > max_area_jump else "")
            ),
        )

        # Same restriction, for the same reason: a centroid computed from two
        # voxels moves with any one of them, so the tip slices would dominate
        # this measurement with noise.
        centroids = np.array([
            ndi.center_of_mass(mask[z]) for z in substantial
        ])
        steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        # Only consecutive slices: a step across a gap is already reported by
        # the continuity check and would double-count here.
        consecutive = np.diff(substantial) == 1
        steps = steps[consecutive] if consecutive.any() else steps
        if steps.size:
            worst_step = float(steps.max())

            # Judge the jump in millimetres when the voxel size is known.
            # The tooth's curvature is physical, so the same bend spans more
            # voxels on a finer scan -- a 0.167 mm move is 7 voxels at 24 um
            # but 20.9 at 8 um -- and a limit in voxels therefore tightens as
            # sampling improves, flagging a healthy full-resolution mask that
            # its own downsampled copy passes. Converting the measurement
            # rather than the limit keeps a real jump caught at every pitch,
            # which rescaling the limit alone does not: a 34.7-voxel teleport
            # failed at 24 um and passed at 8 um.
            in_plane = None
            if spacing is not None:
                in_plane = float(np.mean(np.asarray(spacing, dtype=float)[1:]))

            if in_plane and in_plane > 0:
                measured = worst_step * in_plane
                limit = max_centroid_jump_voxels * _CENTROID_REFERENCE_PITCH_MM
                units = "mm"
            else:
                measured = worst_step
                limit = max_centroid_jump_voxels
                units = "voxels"

            report.add_if(
                "trajectory smoothness",
                value=worst_step,
                fail_when=measured > limit * 2,
                warn_when=measured > limit,
                message=(
                    f"largest centroid move between adjacent slices is "
                    f"{measured:.3g} {units} (limit {limit:.3g} {units})"
                    + (" — the incisor curves gently; a jump this large means "
                       "the mask switched structures"
                       if measured > limit else "")
                ),
            )

    # --- 6. Denser than what surrounds it ---
    # Enamel and dentine are the most mineralised tissue in the scan, so the
    # incisor should be brighter than the bone left behind. If it is not, the
    # mask is sitting on bone.
    outside = (preprocessed > 0) & ~mask
    if outside.any():
        inside_mean = float(preprocessed[mask].mean())
        outside_mean = float(preprocessed[outside].mean())
        ratio = inside_mean / outside_mean if outside_mean else 0.0
        report.add_if(
            "intensity contrast",
            value=ratio,
            fail_when=ratio < 0.85,
            warn_when=ratio < 1.0,
            message=(
                f"mean intensity inside the mask is {inside_mean:.4g} vs "
                f"{outside_mean:.4g} outside (ratio {ratio:.2f})"
                + (" — the incisor is the densest tissue present, so a mask no "
                   "brighter than its surroundings is probably bone"
                   if ratio < 1.0 else "")
            ),
        )

    # --- 7. Physical size, when the voxel size is known ---
    if spacing is not None:
        voxel_mm3 = float(np.prod(spacing))
        volume_mm3 = total * voxel_mm3
        # A mouse hemimandible incisor is on the order of 5-15 mm long and
        # under a millimetre across, so a few mm^3. The bounds are an order of
        # magnitude either side, to flag a scale error (wrong voxel size, wrong
        # units) rather than to police biological variation.
        report.add_if(
            "physical volume",
            value=volume_mm3,
            fail_when=False,
            warn_when=not (0.3 <= volume_mm3 <= 30.0),
            message=(
                f"incisor volume is {volume_mm3:.3g} mm^3"
                + (" — outside the 0.3-30 mm^3 range expected for a mouse "
                   "incisor; check the voxel size is right"
                   if not (0.3 <= volume_mm3 <= 30.0) else "")
            ),
        )

    return report


def check_shrink(
    before: np.ndarray,
    after: np.ndarray,
    margin_mm: float,
    *,
    min_retained: float = 0.20,
) -> CheckReport:
    """
    Validate the margin shrink.

    The shrink is meant to shave a partial-volume rim, so it should cost a
    modest share of the mask. Losing most of it means the margin is far wider
    than the rim it was meant to trim -- usually a units mistake (mm read as
    voxels, or a 0.5 that was meant to be 0.05).

    Parameters:
        min_retained: Fraction of the mask that must survive the shrink.
    """
    report = CheckReport("Incisor margin shrink")

    if margin_mm <= 0:
        report.add("shrink applied", Severity.PASS, "no margin shrink requested", 0.0)
        return report

    n_before = int(np.count_nonzero(before))
    n_after = int(np.count_nonzero(after))
    retained = n_after / n_before if n_before else 0.0

    report.add_if(
        "retained fraction",
        value=retained,
        fail_when=retained < min_retained,
        warn_when=retained < 0.50,
        message=(
            f"{retained:.1%} of the mask survived a {margin_mm:.4g} mm shrink "
            f"({n_after:,} of {n_before:,} voxels)"
            + (" — a shrink is meant to shave the partial-volume rim, not most "
               "of the tooth; check the margin's units"
               if retained < 0.50 else "")
        ),
    )

    if n_after:
        largest, n_components = _largest_component_fraction(after.astype(bool, copy=False))
        report.add_if(
            "single component",
            value=largest,
            fail_when=False,
            warn_when=largest < 0.95,
            message=(
                f"largest component holds {largest:.1%} of the shrunk mask "
                f"({n_components} component(s))"
                + (" — the shrink pinched the tooth apart at its thinnest point"
                   if largest < 0.95 else "")
            ),
        )
    return report
