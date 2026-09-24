"""
Tests for the segmentation validation checks.

The checks exist to tell a good incisor mask from a bad one, so the tests are
built around synthetic volumes with a known verdict: a tube that behaves like
an incisor must pass, and each characteristic failure mode must be caught by
the check that is supposed to catch it. That is the property worth testing --
not that a particular threshold has a particular value, which would just
restate the source.

Run with:  python -m pytest segment_mandible/tests/test_validation.py
   or:     python segment_mandible/tests/test_validation.py
"""

# The package imports below have to follow the sys.path setup that makes
# segment_mandible/ importable when this file is run directly.
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import (  # noqa: E402
    Severity,
    check_incisor_mask,
    check_incisor_seed,
    check_isolation_volume,
    check_loaded_volume,
    check_preprocessed_volume,
    check_reorientation,
    check_shrink,
)
from validation.summary import ValidationSummary  # noqa: E402


SHAPE = (200, 60, 80)


def _synthetic_mandible(shape=SHAPE):
    """
    A stand-in hemimandible: a bright incisor tube inside a dimmer bone slab.

    Deliberately crude -- the checks measure gross geometry and relative
    intensity, so a tube in a slab exercises them the same way a real scan
    does, without needing one.
    """
    volume = np.zeros(shape, dtype=np.float32)
    # Bone: a slab filling much of the cross-section along the whole length.
    volume[:, 10:50, 10:70] = 0.4
    # Incisor: a small bright tube that drifts gently across the slices, the
    # way a real incisor curves.
    incisor = np.zeros(shape, dtype=bool)
    for z in range(shape[0]):
        y = 18 + int(6 * z / shape[0])
        x = 30 + int(10 * z / shape[0])
        incisor[z, y - 4:y + 4, x - 4:x + 4] = True
    volume[incisor] = 0.9
    return volume, incisor


def _severity(report, name):
    for result in report.results:
        if result.name == name:
            return result.severity
    raise AssertionError(f"no check named {name!r} in {[r.name for r in report.results]}")


# ---------------------------------------------------------------------------
# The good case
# ---------------------------------------------------------------------------

def test_good_incisor_passes():
    volume, incisor = _synthetic_mandible()
    report = check_incisor_mask(incisor, volume)
    assert not report.failed, [r.message for r in report.problems]


def test_good_volume_passes_load_and_preprocess_checks():
    volume, _ = _synthetic_mandible()
    assert not check_loaded_volume(volume).failed
    assert not check_preprocessed_volume(volume).failed


# ---------------------------------------------------------------------------
# "It picked up everything" -- the failure this work is really aimed at
# ---------------------------------------------------------------------------

def test_mask_swallowing_the_whole_jaw_fails():
    volume, _ = _synthetic_mandible()
    everything = volume > 0
    report = check_incisor_mask(everything, volume)
    assert report.failed
    assert _severity(report, "share of foreground") is Severity.FAIL


def test_mask_taking_half_the_bone_fails():
    volume, incisor = _synthetic_mandible()
    leaked = incisor.copy()
    leaked[:, 10:35, 10:70] = True  # incisor plus a big slice of bone
    report = check_incisor_mask(leaked, volume)
    assert report.failed


def test_empty_mask_fails():
    volume, _ = _synthetic_mandible()
    report = check_incisor_mask(np.zeros(volume.shape, dtype=bool), volume)
    assert report.failed
    assert _severity(report, "non-empty") is Severity.FAIL


# ---------------------------------------------------------------------------
# Tracking failures
# ---------------------------------------------------------------------------

def test_mask_covering_few_slices_is_flagged():
    volume, incisor = _synthetic_mandible()
    truncated = incisor.copy()
    truncated[60:] = False  # tracking stopped less than a third of the way
    report = check_incisor_mask(truncated, volume)
    assert report.failed or report.warned
    assert _severity(report, "slice coverage") in (Severity.WARN, Severity.FAIL)


def test_fragmented_mask_is_flagged():
    volume, incisor = _synthetic_mandible()
    broken = incisor.copy()
    broken[80:120] = False  # a hole in the middle of the span
    report = check_incisor_mask(broken, volume)
    assert _severity(report, "continuity") in (Severity.WARN, Severity.FAIL)
    assert _severity(report, "single component") in (Severity.WARN, Severity.FAIL)


def test_centroid_jump_is_flagged():
    volume, incisor = _synthetic_mandible()
    jumped = incisor.copy()
    # From slice 100 on, the mask teleports to the far side of the jaw --
    # what a switch onto a different structure looks like.
    jumped[100:] = False
    for z in range(100, SHAPE[0]):
        jumped[z, 40:48, 55:63] = True
    report = check_incisor_mask(jumped, volume)
    assert _severity(report, "trajectory smoothness") in (Severity.WARN, Severity.FAIL)


def test_area_jump_is_flagged():
    volume, incisor = _synthetic_mandible()
    ballooned = incisor.copy()
    ballooned[100, 12:48, 12:68] = True  # one slice balloons onto bone
    report = check_incisor_mask(ballooned, volume)
    assert _severity(report, "area stability") in (Severity.WARN, Severity.FAIL)


def test_tapering_tips_are_not_read_as_area_jumps():
    """
    A tooth that thins to a few voxels at its ends must not be flagged for it.

    This is a regression test with a specific origin: measured on six real P82
    scans, every area-stability flag sat on the first or last slice of the
    mask, where a change of four or five voxels is arithmetically a multi-fold
    ratio while the interior held steady within 1.5x. One scan was pushed to a
    spurious FAIL that way, which in the pipeline would discard a good mask and
    fall back unnecessarily. Restricting the measurement to slices with enough
    area is what prevents that, so the taper here is built to be gradual and
    entirely legitimate.
    """
    volume, incisor = _synthetic_mandible()
    tapered = incisor.copy()

    # Thin the last few slices down to a handful of voxels, as a real tooth
    # does at its apical end -- a smooth taper, not a jump.
    for offset, width in enumerate([3, 2, 1, 1]):
        z = SHAPE[0] - 1 - offset
        y = 18 + int(6 * z / SHAPE[0])
        x = 30 + int(10 * z / SHAPE[0])
        tapered[z] = False
        tapered[z, y - width:y + width, x - width:x + width] = True

    report = check_incisor_mask(tapered, volume)
    assert _severity(report, "area stability") is Severity.PASS, (
        "a legitimate end taper was mistaken for the mask jumping onto bone"
    )


def test_area_jump_is_still_caught_despite_the_taper_rule():
    """
    The taper rule must not blind the check to a real jump.

    Paired with the test above: restricting the measurement to substantial
    slices is only safe if a genuine balloon on a full-size slice is still
    flagged, so both directions are asserted rather than just the quiet one.
    """
    volume, incisor = _synthetic_mandible()
    ballooned = incisor.copy()
    ballooned[100, 12:48, 12:68] = True
    # Taper the ends too, so the mask has both features at once.
    for offset, width in enumerate([3, 2, 1, 1]):
        z = SHAPE[0] - 1 - offset
        y = 18 + int(6 * z / SHAPE[0])
        x = 30 + int(10 * z / SHAPE[0])
        ballooned[z] = False
        ballooned[z, y - width:y + width, x - width:x + width] = True

    report = check_incisor_mask(ballooned, volume)
    assert _severity(report, "area stability") in (Severity.WARN, Severity.FAIL)


def test_trajectory_is_judged_on_physical_distance():
    """
    The verdict must follow how far the mask actually moved, in millimetres.

    A slice-to-slice centroid move of a fixed number of voxels is a different
    physical distance at different sampling pitches, and it is the physical
    one that says whether the mask jumped structures. Measuring in voxels made
    the check tighten as scans got finer: on a real 8 um P82 scan a healthy
    0.167 mm curve measured 20.9 voxels and warned, where the same anatomy
    downsampled to 24 um measured 7 and passed.

    Here one mask is scored at two pitches. The voxel measurement is identical
    by construction; the physical one is 3x larger at the coarser pitch, and
    the verdicts must follow the physical one.
    """
    volume, incisor = _synthetic_mandible()

    # A localised ~26 voxel step: gentle in mm on a fine scan, a real jump on
    # a coarse one.
    kinked = incisor.copy()
    for z in range(100, SHAPE[0]):
        kinked[z] = False
        kinked[z, 20:28, 56:64] = True

    fine = check_incisor_mask(kinked, volume, spacing=(0.008, 0.008, 0.008))
    coarse = check_incisor_mask(kinked, volume, spacing=(0.024, 0.024, 0.024))

    assert _severity(fine, "trajectory smoothness") is Severity.PASS, (
        "a 0.21 mm move was flagged; the check is still judging voxels"
    )
    assert _severity(coarse, "trajectory smoothness") in (
        Severity.WARN, Severity.FAIL
    ), "the same move at 0.63 mm went unflagged"


def test_blobby_mask_fails_elongation():
    volume, _ = _synthetic_mandible()
    blob = np.zeros(volume.shape, dtype=bool)
    blob[90:110, 20:40, 30:50] = True  # a cube, not a tube
    report = check_incisor_mask(blob, volume)
    assert _severity(report, "elongation") in (Severity.WARN, Severity.FAIL)


def test_mask_on_bone_fails_intensity_contrast():
    volume, incisor = _synthetic_mandible()
    # Same tube geometry, but shifted off the incisor onto plain bone. The real
    # incisor stays bright in the volume, so the mask is measurably dimmer than
    # its surroundings -- which is what tracking onto bone actually looks like.
    on_bone = np.roll(incisor, shift=16, axis=1)
    assert not (on_bone & incisor).any(), "the shifted mask must miss the incisor"
    report = check_incisor_mask(on_bone, volume)
    assert _severity(report, "intensity contrast") in (Severity.WARN, Severity.FAIL)


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------

def test_empty_volume_fails():
    report = check_loaded_volume(np.zeros((100, 60, 80), dtype=np.float32))
    assert report.failed


def test_non_3d_volume_fails():
    report = check_loaded_volume(np.zeros((60, 80), dtype=np.float32))
    assert report.failed
    assert _severity(report, "dimensionality") is Severity.FAIL


def test_partial_load_fails():
    report = check_loaded_volume(np.random.rand(3, 60, 80))
    assert report.failed
    assert _severity(report, "shape") is Severity.FAIL


def test_nan_volume_fails():
    volume, _ = _synthetic_mandible()
    volume = volume.copy()
    volume[0, 0, 0] = np.nan
    assert check_loaded_volume(volume).failed


def test_background_removal_keeping_everything_fails():
    volume = np.random.rand(100, 60, 80).astype(np.float32) + 0.5
    report = check_preprocessed_volume(volume)
    assert report.failed
    assert _severity(report, "foreground fraction") is Severity.FAIL


def test_background_removal_erasing_specimen_fails():
    volume = np.zeros((100, 60, 80), dtype=np.float32)
    volume[50, 30, 40] = 1.0
    report = check_preprocessed_volume(volume)
    assert report.failed


# ---------------------------------------------------------------------------
# Reorientation
# ---------------------------------------------------------------------------

def test_reorientation_with_wrong_long_axis_fails():
    volume = np.zeros((40, 200, 80), dtype=np.float32)
    report = check_reorientation(volume, {"perm": (0, 1, 2)})
    assert report.failed
    assert _severity(report, "long axis") is Severity.FAIL


def test_reorientation_without_tip_detection_warns():
    volume = np.zeros(SHAPE, dtype=np.float32)
    report = check_reorientation(volume, {})
    assert report.warned
    assert _severity(report, "tip detection") is Severity.WARN


def test_good_reorientation_passes():
    volume = np.zeros(SHAPE, dtype=np.float32)
    report = check_reorientation(volume, {"perm": (1, 0, 2), "flip0": True})
    assert not report.failed and not report.warned


# ---------------------------------------------------------------------------
# Seed, isolation and shrink
# ---------------------------------------------------------------------------

def test_missing_seed_fails():
    report = check_incisor_seed(None, 0, depth=200, slice_area=4800)
    assert report.failed


def test_oversized_seed_fails():
    report = check_incisor_seed(180, 3000, depth=200, slice_area=4800)
    assert report.failed
    assert _severity(report, "seed size") is Severity.FAIL


def test_reasonable_seed_passes():
    report = check_incisor_seed(190, 120, depth=200, slice_area=4800)
    assert not report.failed


def test_isolation_removing_everything_fails():
    volume, _ = _synthetic_mandible()
    report = check_isolation_volume(np.zeros_like(volume), volume > 0)
    assert report.failed


def test_isolation_removing_nothing_warns():
    volume, _ = _synthetic_mandible()
    report = check_isolation_volume(volume, volume > 0)
    assert report.warned


def test_isolation_ratio_cannot_exceed_one():
    """
    The numerator is intersected with the region that forms the denominator, so
    a retained fraction above 1 -- the symptom of the old check's mismatched
    baselines -- is unrepresentable however the region is chosen.
    """
    volume, _ = _synthetic_mandible()
    for region in ((volume > 0), (volume > 0.5), np.ones_like(volume, dtype=bool)):
        report = check_isolation_volume(volume, region)
        values = [r.value for r in report.results if r.name == "retained fraction"]
        assert all(v is not None and v <= 1.0 for v in values), values


def test_isolation_ignores_clahe_lifted_air():
    """
    CLAHE lifts background off exact zero across the whole array, so a
    foreground count taken over the CLAHE volume is ~every voxel in the scan.
    Bone removal then reads as a few percent of that denominator however well
    it worked, and the check warns on every run. Restricting both counts to the
    thresholded region has to make the measurement reflect the actual removal.
    """
    # A mandible occupying a few percent of the scan, as on a real microCT
    # volume -- the fixture's slab fills half the array, which is not enough
    # air for the inflated denominator to show its effect.
    shape = (200, 200, 200)
    volume = np.zeros(shape, dtype=np.float32)
    volume[80:120, 80:120, 80:120] = 0.4          # bone
    incisor = np.zeros(shape, dtype=bool)
    incisor[90:110, 90:100, 90:100] = True
    volume[incisor] = 0.9
    region = volume > 0

    # Air, lifted just above zero the way CLAHE leaves it.
    clahe_like = np.where(region, volume, 0.02).astype(np.float32)

    # Bone removed, incisor kept -- a healthy isolation.
    bone = region & ~incisor
    isolation_volume = np.where(bone, 0, clahe_like)

    report = check_isolation_volume(isolation_volume, region)
    assert not report.warned, [r.message for r in report.problems]
    assert not report.failed

    # The old baseline: every lifted-air voxel inflates the denominator, and
    # the same healthy isolation reads as "barely removed anything".
    inflated = check_isolation_volume(isolation_volume, clahe_like > 0)
    assert inflated.warned


def test_isolation_counts_only_inside_the_region():
    """
    Voxels outside the thresholded region are background to this step and must
    not be counted as survivors -- otherwise lifted air outside the mandible
    pads the numerator and hides a bone removal that took the incisor with it.
    """
    volume, _ = _synthetic_mandible()
    region = volume > 0

    # Everything inside the region removed, but air outside it still nonzero.
    isolation_volume = np.where(region, 0, 0.02).astype(np.float32)

    report = check_isolation_volume(isolation_volume, region)
    assert report.failed


def test_shrink_removing_most_of_the_mask_fails():
    _, incisor = _synthetic_mandible()
    shrunk = incisor.copy()
    shrunk[:, :, :] = False
    shrunk[100, 20, 35] = True
    report = check_shrink(incisor, shrunk, margin_mm=0.5)
    assert report.failed


def test_no_shrink_requested_passes():
    _, incisor = _synthetic_mandible()
    assert not check_shrink(incisor, incisor, margin_mm=0.0).failed


# ---------------------------------------------------------------------------
# Summary plumbing
# ---------------------------------------------------------------------------

def test_summary_tracks_status_per_sample():
    volume, incisor = _synthetic_mandible()
    summary = ValidationSummary()
    summary.sample("good").add(check_incisor_mask(incisor, volume))
    summary.sample("bad").add(check_incisor_mask(volume > 0, volume))

    assert summary.sample("good").status == "OK"
    assert summary.sample("bad").status == "FAIL"
    assert [s.sample for s in summary.failed] == ["bad"]
    # sample() must return the same entry rather than appending a duplicate.
    assert len(summary.samples) == 2


def test_summary_serializes_to_json():
    volume, incisor = _synthetic_mandible()
    summary = ValidationSummary()
    summary.sample("s1").add(check_incisor_mask(incisor, volume))
    payload = summary.samples[0].to_dict()
    assert payload["sample"] == "s1"
    assert payload["stages"][0]["results"]


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
