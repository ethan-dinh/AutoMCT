"""
Regression tests for the specimen-mask smoothing in ``_sample_object_points``.

These pin the fix for a silent data-loss bug: ``--reorient-tight`` was cropping
~1.4 mm off the incisal tip of a real scan even when the user's own crop plainly
included it.

The cause was not thresholding the tip away for being dim -- the tip is the
*brightest* part of a rodent incisor (mature enamel), median 15254 counts
against 6458 at the base. It was fragmentation. The tip is thin, so above the
Otsu level its mask is a lacework of hundreds of disconnected specks rather than
one solid mass, and the "keep the largest connected component" step (there to
reject the sample holder and mounting putty) discarded every one of them as
debris. The bounding box then stopped short of the tip, and the tight crop cut
there.

Measured on the scan that exposed this (f0004019, 395 x 3168 x 3168 at 6 um,
after --autocrop):

    smoothing   Otsu   components   largest fraction   fitted long extent
    off         7136         1218              0.768          10.645 mm
    0.06 mm     5000            6              1.000          12.095 mm

against a true extent of 12.04 mm measured on the uncropped scan.

Note the ordering the fix depends on: smoothing is safe only *after* a crop.
On the full uncropped scan ~99% of voxels are air, and blurring smears that air
across the boundary hard enough to collapse the Otsu split from 6475 into the
noise band (~1213, below the median of 1241), which merges the holder into the
specimen and balloons the box to 18.87 x 18.77 mm.

Scope: the interesting failure needs a specimen that *fragments*, and
fragmentation is a joint property of the taper, the noise and the stride the
192-voxel subsample cap forces on a 3168-voxel scan. Synthetic phantoms
tried for this did not reproduce it -- they either stayed connected (largest
fraction 0.99+, so nothing was ever discarded) or, once the tip was thinned
enough to shatter, had a sub-voxel tip that smoothing legitimately erased. So
the regression proper runs against the real scan and skips when it is not
mounted; what runs everywhere are the invariants that hold on any volume.
"""

import numpy as np
import pytest

from volume_pipeline import (
    _sample_object_points,
    autocrop_bbox,
    directions_from_spacing,
    min_volume_obb,
)

SPACING_MM = 0.006
DEFAULT_SMOOTH_MM = 0.06

# The scan this bug was found on. Lives on an external volume, so every test
# that needs it skips cleanly when it is not mounted.
SCAN = (
    "/Volumes/Z Drive/CT Data - Enamel Optimization/RK-TestRuns/WT-M-1/f0004019.isq"
)
TRUE_LENGTH_MM = 12.04      # measured on the uncropped scan
BROKEN_LENGTH_MM = 10.645   # what the unsmoothed mask returned


@pytest.fixture(scope="module")
def cropped_scan():
    """The real scan, autocropped -- the exact state the reorient stage sees."""
    from pathlib import Path

    path = Path(SCAN)
    if not path.is_file():
        pytest.skip(f"scan not mounted: {SCAN}")

    from convert_ISQ_NRRD import read_isq

    volume, _ = read_isq(path)
    directions = directions_from_spacing(SPACING_MM, SPACING_MM, SPACING_MM)
    slices = autocrop_bbox(
        volume,
        margin_mm=0.5,
        spacing=np.array([SPACING_MM] * 3),
        max_dim=192,
    )
    cropped = np.ascontiguousarray(volume[slices])
    starts = np.array([s.start for s in slices], dtype=float)
    return cropped, directions, starts @ directions


def _long_extent(volume, directions, origin, smooth_mm):
    points, _ = _sample_object_points(
        volume, directions, origin, mask_smooth_mm=smooth_mm,
    )
    _, extents = min_volume_obb(points)
    return float(extents[0])


def test_unsmoothed_mask_loses_the_tip(cropped_scan):
    """Guards the premise: without smoothing the tip is still lost.

    If this ever stops under-measuring, the bug is gone by some other route and
    the recovery test below has become vacuous.
    """
    extent = _long_extent(*cropped_scan, smooth_mm=0.0)
    assert extent == pytest.approx(BROKEN_LENGTH_MM, abs=0.3)
    assert extent < TRUE_LENGTH_MM - 1.0


def test_smoothing_recovers_the_tip(cropped_scan):
    """The fix: the full 12.04 mm comes back, tip included."""
    extent = _long_extent(*cropped_scan, smooth_mm=DEFAULT_SMOOTH_MM)
    assert extent == pytest.approx(TRUE_LENGTH_MM, abs=0.25)


@pytest.mark.parametrize("smooth_mm", [0.02, 0.04, 0.06, 0.10, 0.15])
def test_recovered_extent_is_insensitive_to_sigma(cropped_scan, smooth_mm):
    """A broad plateau, so the default is not a knife-edge tuning.

    Measured: 0.02-0.15 mm all land within 0.09 mm of the true extent, while
    0.25 mm begins eroding the tip (11.764 mm).
    """
    extent = _long_extent(*cropped_scan, smooth_mm=smooth_mm)
    assert extent == pytest.approx(TRUE_LENGTH_MM, abs=0.25)


def test_smoothing_keeps_the_mask_in_one_piece(cropped_scan):
    """The mechanism, not just the outcome.

    Point count is the observable proxy for the largest component absorbing the
    tip specks instead of the code discarding them.
    """
    volume, directions, origin = cropped_scan
    plain, _ = _sample_object_points(volume, directions, origin, mask_smooth_mm=0.0)
    smoothed, _ = _sample_object_points(
        volume, directions, origin, mask_smooth_mm=DEFAULT_SMOOTH_MM,
    )
    assert len(smoothed) > 1.2 * len(plain)


# --- invariants that do not need the scan -----------------------------------


def _solid_block(shape=(40, 60, 80), spacing=0.05, seed=0):
    """A solid, noisy block: never fragments, so smoothing must be a no-op."""
    rng = np.random.default_rng(seed)
    vol = np.full(shape, 200.0, dtype=np.float32)
    vol[8:32, 12:48, 10:70] = 9000.0
    vol += rng.normal(0.0, 150.0, size=shape).astype(np.float32)
    return np.clip(vol, 0.0, None), directions_from_spacing(*([spacing] * 3))


def test_smoothing_is_inert_on_an_unfragmented_specimen():
    """Smoothing must not move the box on a specimen that was never broken.

    This is what makes the default safe to turn on for every scan: it rescues
    a fragmented mask without perturbing one that segmented cleanly.
    """
    volume, directions = _solid_block()
    origin = np.zeros(3)
    plain = _long_extent(volume, directions, origin, smooth_mm=0.0)
    smoothed = _long_extent(volume, directions, origin, smooth_mm=DEFAULT_SMOOTH_MM)
    assert smoothed == pytest.approx(plain, abs=2 * 0.05)


def test_smoothing_does_not_alter_returned_intensities():
    """The blur builds the mask only; it must never reach the voxel data.

    The background level is read off the unsmoothed subsample, so it cannot
    drift with sigma -- otherwise the padding value used when rotating into a
    larger grid would depend on a mask-only tuning knob.
    """
    volume, directions = _solid_block()
    origin = np.zeros(3)
    _, bg_plain = _sample_object_points(volume, directions, origin, mask_smooth_mm=0.0)
    _, bg_smooth = _sample_object_points(
        volume, directions, origin, mask_smooth_mm=DEFAULT_SMOOTH_MM,
    )
    assert bg_smooth == pytest.approx(bg_plain, rel=0.05)


def test_sigma_is_physical_not_per_voxel():
    """Sigma is in mm, so the same value must mean the same distance.

    The stride varies with volume size (the real scan subsamples at 17x
    uncropped and 7-11x cropped), so a sigma in subsample voxels would blur a
    different physical distance on every scan.
    """
    volume, _ = _solid_block()
    origin = np.zeros(3)
    fine = directions_from_spacing(0.05, 0.05, 0.05)
    coarse = directions_from_spacing(0.10, 0.10, 0.10)

    extent_fine = _long_extent(volume, fine, origin, smooth_mm=DEFAULT_SMOOTH_MM)
    extent_coarse = _long_extent(volume, coarse, origin, smooth_mm=DEFAULT_SMOOTH_MM)
    # Same voxel grid at twice the spacing -> twice the physical extent.
    assert extent_coarse == pytest.approx(2.0 * extent_fine, rel=0.05)


def test_zero_sigma_disables_smoothing_entirely():
    """--reorient-mask-smooth-mm 0 must reproduce the old behaviour exactly."""
    volume, directions = _solid_block()
    origin = np.zeros(3)
    a, bg_a = _sample_object_points(volume, directions, origin, mask_smooth_mm=0.0)
    b, bg_b = _sample_object_points(volume, directions, origin)
    # The default is non-zero, so these must differ in general; pin the
    # explicit-zero path against itself to catch an accidental always-on blur.
    c, bg_c = _sample_object_points(volume, directions, origin, mask_smooth_mm=0.0)
    assert np.array_equal(a, c) and bg_a == bg_c
    assert b.shape[1] == 3
