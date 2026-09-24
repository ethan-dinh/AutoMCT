"""
Tests for the fusion-aware threshold-descent incisor segmentation.

The method's whole claim is that intensity *plus connectivity* separates the
incisor from bone where intensity alone does not: at a high threshold the tooth
is its own connected component, and below some lower threshold it fuses to
bone through a bridge. So the volumes here are built to have exactly that
structure -- a bright tube, a dimmer slab, and a bridge of intermediate
intensity joining them -- with the fusion point known by construction. The
property under test is that the descent stops before the bridge conducts, not
that it returns any particular voxel count.

Run with:  python -m pytest segment_mandible/tests/test_incisor_threshold.py
"""

# The package imports below have to follow the sys.path setup that makes
# segment_mandible/ importable when this file is run directly.
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.incisor_threshold import (  # noqa: E402
    find_incisor_seed,
    polish_incisor_mask,
    recover_cervical_loop,
    recover_low_density_margin,
    segment_incisor_by_threshold_descent,
    track_cervical_loop,
)

SHAPE = (120, 64, 96)

# Intensities chosen so the three tissues are unambiguously ordered.
#
# The bridge is only slightly *brighter* than bone, rather than sitting midway
# between bone and tooth. That is deliberate, and it is what makes the fusion
# point well defined: the component can only swallow the slab once the
# threshold is low enough for the slab itself to light up, so a bridge much
# brighter than bone would conduct while bone was still dark and add only its
# own few voxels -- a rise far too small to read as fusion. Fusion in a real
# scan is likewise gated by the bone's own intensity, not the bridge's.
TOOTH = 1.0
BRIDGE = 0.44
BONE = 0.40
AIR = 0.0

# The threshold at which the slab becomes visible, and so the level the descent
# must stop above. Fusion cannot happen before this regardless of the bridge.
FUSION_LEVEL = BONE

# Geometry, named once so the phantom and the assertions cannot drift apart.
# The proportions matter: a mouse incisor is a small share of the
# hemimandible (~1-2% of foreground here), and the seed is taken from a high
# percentile of tissue. Make the tooth a large share instead and that
# percentile cuts *inside* the tooth rather than below it, leaving only
# speckle -- a phantom artifact, not a property of the method.
BONE_REGION = np.s_[:, 30:60, 10:88]
TOOTH_REGION = np.s_[4:116, 14:20, 40:46]
BRIDGE_REGION = np.s_[50:60, 20:30, 41:45]
MOLAR_REGION = np.s_[92:112, 34:52, 16:34]


def _fusing_mandible(shape=SHAPE, bridge_intensity=BRIDGE, seed=0):
    """
    A bright incisor tube running the length of a dimmer bone slab, joined to
    it by a thin bridge of intermediate intensity.

    The bridge is the whole point: without it the tooth never fuses and the
    descent runs to the bottom of its sweep, which is a different code path.

    Each tissue is given a little intensity spread rather than a single flat
    value. This is not decoration: the seed is chosen from voxels *strictly
    above* a high percentile of tissue, and in a volume where the tooth is one
    exact value that percentile lands on the value itself and selects nothing.
    Real tissue always varies, so a flat phantom would be testing a degenerate
    case the method never meets. The spread is kept far smaller than the gaps
    between tissues, so the ordering tooth > bridge > bone still holds
    everywhere.
    """
    rng = np.random.default_rng(seed)
    volume = np.full(shape, AIR, dtype=np.float32)

    def _fill(region, level, spread):
        volume[region] = level + rng.normal(0.0, spread, size=volume[region].shape)

    # Bone slab: a wide, dim block occupying one side of every slice.
    _fill(BONE_REGION, BONE, 0.01)

    # Incisor: a long bright tube spanning nearly the whole Z axis. Its extent
    # along axis 0 is what find_incisor_seed keys on.
    _fill(TOOTH_REGION, TOOTH, 0.01)

    # The bridge: a thin column of intermediate intensity connecting tube to
    # slab over a short stretch of Z. Narrow enough that it only conducts once
    # the threshold drops below its own value.
    _fill(BRIDGE_REGION, bridge_intensity, 0.005)

    return volume


def _tooth_mask(shape=SHAPE):
    """The incisor's true extent, for scoring a returned mask against."""
    mask = np.zeros(shape, dtype=bool)
    mask[TOOTH_REGION] = True
    return mask


def _bone_mask(shape=SHAPE):
    """The bone slab's true extent, i.e. everything the mask must not contain."""
    mask = np.zeros(shape, dtype=bool)
    mask[BONE_REGION] = True
    return mask


def _molar_cluster(volume, intensity=TOOTH):
    """
    Add a compact cluster as dense as the tooth and substantially bulkier than
    it (6480 voxels against the tooth's 4032), standing in for molar cusps.

    This is the case that defeats picking the *largest* bright component: on
    size the cluster wins, on Z extent the tooth wins by 112 slices to 20. That
    gap is the property the seed rule exists to exploit, so it is what these
    tests measure.

    The cluster is given the tooth's own intensity rather than a higher one.
    Making it brighter would put a high percentile of tissue *above* the tooth
    entirely, leaving the tooth out of the candidate set -- the seed would then
    pick the cluster because nothing else was offered, and the test would be
    reporting on the percentile rather than on the selection rule. Equal
    intensity keeps both structures candidates so the extent comparison is
    what decides.
    """
    rng = np.random.default_rng(1)
    out = volume.copy()
    region = MOLAR_REGION
    out[region] = intensity + rng.normal(0.0, 0.01, size=out[region].shape)
    return out


def _max_step_ratio(volume, warmup=5, steps=60):
    """
    Largest single-step growth of the seeded component over the sweep.

    Used by the gradual-fusion tests to assert that their phantom really is
    gradual -- that no step trips ``fusion_ratio`` -- so those tests cannot
    quietly start passing because the ordinary detector caught it instead.
    """
    from scipy import ndimage as ndi

    from segmentation.incisor_threshold import _tissue_floor

    seed = find_incisor_seed(volume)
    if not seed.any():
        return 0.0
    floor = _tissue_floor(volume)
    tissue = volume[volume > floor]
    hi = float(np.percentile(tissue, 99.5))
    lo = max(0.35 * float(np.median(volume[seed])),
             float(np.percentile(tissue, 50.0)) * 0.5)

    previous, worst = None, 0.0
    for i, t in enumerate(np.linspace(hi, lo, steps)):
        labels, n = ndi.label(volume > t)  # type: ignore
        if n == 0:
            continue
        ids, counts = np.unique(labels[seed], return_counts=True)
        keep = ids > 0
        ids, counts = ids[keep], counts[keep]
        if ids.size == 0:
            continue
        size = int((labels == int(ids[np.argmax(counts)])).sum())
        if previous and i >= warmup:
            worst = max(worst, size / previous)
        previous = size
    return worst



# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def test_seed_lands_inside_the_incisor():
    """The seed must sit in the tooth, which is what every later step assumes."""
    volume = _fusing_mandible()
    seed = find_incisor_seed(volume)

    assert seed.any(), "no seed was found in a volume that plainly contains a tooth"

    tooth = _tooth_mask()
    assert seed[~tooth].sum() == 0, "the seed spilled outside the incisor"


def test_seed_prefers_extent_over_size():
    """
    A denser, larger molar cluster must not capture the seed.

    Selecting the biggest bright component would pick the cluster here; picking
    the one with the greatest Z extent picks the tooth. That difference is the
    reason the function is written the way it is.
    """
    volume = _molar_cluster(_fusing_mandible())
    seed = find_incisor_seed(volume, percentile=97.5)

    tooth = _tooth_mask()
    assert seed.any()
    assert seed[tooth].sum() > 0, "the seed missed the incisor entirely"
    assert seed[~tooth].sum() == 0, "the seed latched onto the molar cluster"


def test_seed_is_stable_across_percentiles():
    """
    The seed must not move between structures as the percentile is nudged.

    A seed that flips from tooth to molar over a small parameter change is the
    specific fragility this selection rule was written to remove.
    """
    volume = _molar_cluster(_fusing_mandible())
    tooth = _tooth_mask()

    # Swept over the range where both structures are above the cut and so both
    # are genuine candidates. Above roughly the 98th percentile the cut rises
    # through the enamel level itself and the candidate set empties out, which
    # says nothing about the selection rule.
    for percentile in (97.0, 97.25, 97.5, 97.75, 98.0):
        seed = find_incisor_seed(volume, percentile=percentile)
        assert seed.any(), f"no seed at percentile {percentile}"
        assert seed[~tooth].sum() == 0, f"seed left the incisor at percentile {percentile}"


def test_seed_empty_volume_returns_empty_mask():
    """An all-background volume yields an empty seed rather than raising."""
    seed = find_incisor_seed(np.zeros(SHAPE, dtype=np.float32))
    assert not seed.any()


# ---------------------------------------------------------------------------
# Threshold descent
# ---------------------------------------------------------------------------

def test_descent_stops_before_fusing_with_bone():
    """
    The returned mask must be the tooth, not the tooth plus the slab.

    This is the method's central claim, so it is asserted on both sides: the
    tooth is recovered, and the bone is not.
    """
    volume = _fusing_mandible()
    mask, threshold = segment_incisor_by_threshold_descent(volume)

    tooth = _tooth_mask()
    bone = _bone_mask()

    assert mask.any(), "the descent returned an empty mask"
    assert np.isfinite(threshold)

    recall = mask[tooth].sum() / tooth.sum()
    assert recall > 0.9, f"most of the incisor was missed (recall {recall:.2f})"

    leaked = mask[bone].sum() / bone.sum()
    assert leaked < 0.01, f"the mask fused into bone ({leaked:.1%} of the slab)"


def test_descent_stops_above_the_bridge_intensity():
    """
    The stopping threshold must sit above the level at which the bone slab
    appears, since crossing that is what fusing means here. Anchoring the
    assertion to the constructed intensity tests the mechanism rather than a
    remembered number.
    """
    _, threshold = segment_incisor_by_threshold_descent(_fusing_mandible())
    assert threshold > FUSION_LEVEL, (
        f"descent continued past the fusion point (stopped at {threshold:.3g}, "
        f"bone lights up at {FUSION_LEVEL})"
    )


def test_descent_survives_a_volume_with_no_fusion():
    """
    With no bridge the tooth never fuses, so the sweep runs to its floor and
    must still return the tooth rather than failing.
    """
    volume = _fusing_mandible()
    volume[BRIDGE_REGION] = AIR  # remove the bridge

    mask, threshold = segment_incisor_by_threshold_descent(volume)

    bone = _bone_mask()

    assert mask.any()
    assert np.isfinite(threshold)
    assert mask[bone].sum() == 0, "picked up bone in a volume where nothing connects"


def test_descent_with_empty_seed_returns_empty_mask_and_nan():
    """
    The no-seed path must report NaN, not a number.

    The pipeline branches on ``np.isfinite`` to avoid emptying the conservative
    volume through a NaN comparison, so this contract is load-bearing.
    """
    volume = _fusing_mandible()
    mask, threshold = segment_incisor_by_threshold_descent(
        volume, seed=np.zeros(SHAPE, dtype=bool)
    )
    assert not mask.any()
    assert np.isnan(threshold)


def test_warmup_prevents_aborting_during_the_forming_phase():
    """
    Without a warmup the component's early growth reads as fusion and the
    sweep aborts almost immediately, returning a fragment. The guard's whole
    job is to prevent that, so disabling it should measurably hurt.
    """
    volume = _fusing_mandible()
    guarded, _ = segment_incisor_by_threshold_descent(volume)
    unguarded, _ = segment_incisor_by_threshold_descent(volume, warmup_steps=0)

    assert guarded.sum() > unguarded.sum(), (
        "the warmup guard made no difference; the test volume may no longer "
        "exercise the forming phase"
    )


# ---------------------------------------------------------------------------
# Margin recovery and polish
# ---------------------------------------------------------------------------

def test_margin_recovery_is_a_superset_that_does_not_reach_bone():
    """
    Recovery may only add voxels, and only ones touching the confident mask.

    The corridor is what keeps the relaxed threshold safe, so the test checks
    that the relaxation stayed local rather than flooding.
    """
    volume = _fusing_mandible()
    mask, threshold = segment_incisor_by_threshold_descent(volume)
    grown = recover_low_density_margin(volume, mask, threshold)

    bone = _bone_mask()

    assert grown[mask].all(), "recovery dropped voxels from the confident mask"
    assert grown.sum() >= mask.sum()
    assert grown[bone].sum() / bone.sum() < 0.01, "recovery leaked into bone"


def test_margin_recovery_passes_through_a_nan_threshold():
    """A failed descent must not be amplified into an exception here."""
    mask = np.zeros(SHAPE, dtype=bool)
    out = recover_low_density_margin(_fusing_mandible(), mask, float("nan"))
    assert not out.any()


def test_polish_keeps_one_component_and_leaves_an_open_cavity_open():
    """
    Polish must reduce the mask to a single component, and its 3D fill must not
    close a cavity that is open to the exterior -- filling those per-slice is
    the specific mistake the function's docstring warns against.
    """
    mask = np.zeros(SHAPE, dtype=bool)
    mask[10:100, 12:22, 24:36] = True
    # A channel running the tooth's length and open at both Z ends, standing in
    # for the pulp cavity.
    mask[10:100, 15:19, 28:32] = False
    # A detached fragment that polish should discard.
    mask[5:8, 40:44, 50:54] = True

    out = polish_incisor_mask(mask)

    assert not out[5:8, 40:44, 50:54].any(), "the stray fragment survived"
    assert not out[50, 15:19, 28:32].any(), (
        "the open cavity was filled; a 3D fill should leave it alone"
    )


# ---------------------------------------------------------------------------
# Cervical loop recovery
# ---------------------------------------------------------------------------

def _mandible_with_cervical_loop():
    """
    The fusing phantom plus a faint extension at the tooth's apical (low-Z) end.

    The extension is deliberately below the intensity at which the tooth stays
    separate from bone -- that is what makes it invisible to the descent and so
    the thing this recovery exists to find. It is built as a gradient rather
    than a step because the loop is reached by following mineral density down,
    and a single-valued block would be recoverable in one pass, testing
    something easier than the real case.
    """
    volume = _fusing_mandible()
    rng = np.random.default_rng(7)

    # The loop has to sit in a specific window to test anything: dim enough
    # that the descent does not already have it, bright enough that the
    # relaxed threshold can still find it. Those bounds are the descent's own
    # stopping threshold and 60% of it, so the gradient is placed between them
    # rather than at arbitrary values.
    #
    # It is also placed above the bone level, matching the real scans: measured
    # on P82-Odam-CTR-1 the apical zone's bulk tissue sits at 0.076 while the
    # tooth there averages 0.454, so the relaxed threshold lands between them.
    # A phantom whose loop is dimmer than its bone inverts that relationship
    # and makes the recovery look unsafe for a reason no scan exhibits.
    hi, lo = TOOTH * 0.38, TOOTH * 0.28
    for i, z in enumerate(range(4, 34)):
        level = hi - (i / 29) * (hi - lo)
        region = np.s_[z, 14:20, 40:46]
        volume[region] = level + rng.normal(0.0, 0.004, size=volume[region].shape)

    # The bone slab is cleared from the apical end. In a real mandible the
    # tooth's forming end sits in a marrow space rather than against cortical
    # bone, which is exactly why a threshold low enough to follow the loop does
    # not immediately find bone beside it.
    volume[0:34, 30:60, 10:88] = AIR
    return volume


def test_cervical_loop_recovery_extends_the_apical_end():
    """
    The recovered mask must reach further apically and hold more there.

    Both are asserted: reaching further without filling out would just be a
    thread of voxels, and filling out without reaching further would not be the
    loop.
    """
    volume = _mandible_with_cervical_loop()
    base, threshold = segment_incisor_by_threshold_descent(volume)
    grown = recover_cervical_loop(volume, base, threshold)

    def _apical(mask, width=20):
        counts = mask.reshape(mask.shape[0], -1).sum(1)
        occupied = np.flatnonzero(counts)
        return occupied.min(), counts[occupied.min():occupied.min() + width].mean()

    base_start, base_area = _apical(base)
    grown_start, grown_area = _apical(grown)

    assert grown[base].all(), "recovery dropped voxels from the confident mask"
    assert grown_start <= base_start, "recovery did not reach further apically"
    assert grown_area > base_area, (
        f"apical coverage did not improve ({base_area:.0f} -> {grown_area:.0f})"
    )


def test_cervical_loop_recovery_stays_out_of_bone():
    """
    The apical restriction is the only thing making so low a threshold safe, so
    the test that matters is that bone still stays out.
    """
    volume = _mandible_with_cervical_loop()
    base, threshold = segment_incisor_by_threshold_descent(volume)
    grown = recover_cervical_loop(volume, base, threshold)

    bone = _bone_mask()
    leaked = grown[bone].sum() / bone.sum()
    assert leaked < 0.02, f"recovery leaked into bone ({leaked:.1%} of the slab)"


def test_cervical_loop_recovery_is_confined_to_the_apical_zone():
    """
    Growth must not happen at the incisal end.

    This is what separates this function from simply relaxing the threshold
    everywhere, which was measured to leak badly, so it is asserted directly
    rather than inferred from the total.
    """
    volume = _mandible_with_cervical_loop()
    base, threshold = segment_incisor_by_threshold_descent(volume)
    grown = recover_cervical_loop(volume, base, threshold)

    counts = base.reshape(base.shape[0], -1).sum(1)
    occupied = np.flatnonzero(counts)
    low, high = occupied.min(), occupied.max()
    incisal_start = low + int(0.5 * (high - low))

    added = grown & ~base
    assert added[incisal_start:].sum() == 0, (
        "recovery added voxels in the incisal half, outside the apical zone"
    )


def test_cervical_loop_recovery_passes_through_a_nan_threshold():
    """A failed descent must not become an exception here."""
    out = recover_cervical_loop(
        _mandible_with_cervical_loop(), np.zeros(SHAPE, dtype=bool), float("nan")
    )
    assert not out.any()


def test_cervical_loop_recovery_handles_an_empty_mask():
    """An empty mask has no Z span to take an apical fraction of."""
    volume = _mandible_with_cervical_loop()
    out = recover_cervical_loop(volume, np.zeros(SHAPE, dtype=bool), 0.5)
    assert not out.any()



# ---------------------------------------------------------------------------
# Gradual fusion (the denoised case)
# ---------------------------------------------------------------------------

def _gradually_fusing_mandible():
    """
    A phantom whose tooth fuses to bone over several threshold steps.

    The bridge is a ramp rather than a single intensity, so as the threshold
    falls the connection opens progressively and no one step shows the sharp
    jump the per-step ratio looks for. This is what denoising does to a real
    scan: measured on P82-Odam-CTR-2 the raw volume fuses in one 3.36x step,
    while after NLM and TV denoising the same fusion arrives as a 1.99x step --
    under the 2.5 ratio, so the descent ran to the bottom of its sweep and
    returned 1.33M voxels instead of 264k.
    """
    rng = np.random.default_rng(3)
    volume = np.full(SHAPE, AIR, dtype=np.float32)

    # The bone slab carries a density gradient across its width instead of one
    # flat value. That is what makes the fusion gradual: a uniform slab appears
    # all at once when the threshold reaches its level, which is the sharp jump
    # the per-step ratio is built to catch, whereas a graded slab is absorbed a
    # layer at a time. Real bone is likewise not uniform, and denoising further
    # smooths the transition.
    for j, y in enumerate(range(30, 60)):
        level = BONE * 1.55 - (j / 29) * (BONE * 1.55 - BONE * 0.55)
        volume[:, y, 10:88] = level + rng.normal(0.0, 0.008, size=(SHAPE[0], 78))

    volume[TOOTH_REGION] = TOOTH + rng.normal(0.0, 0.01, size=(112, 6, 6))

    # The bridge sits just above the brightest bone, so contact is made early
    # and the slab is then taken in progressively rather than in one step.
    volume[BRIDGE_REGION] = BONE * 1.60 + rng.normal(0.0, 0.004, size=(10, 10, 4))
    return volume


def test_gradual_fusion_is_caught_by_the_cumulative_bound():
    """
    A fusion too gradual for the per-step ratio must still stop the descent.

    The assertion is on bone content rather than on which guard fired: what
    matters is that the returned mask is the tooth, however the descent
    worked that out.
    """
    volume = _gradually_fusing_mandible()

    # Confirm the phantom really is the gradual case: if a single step ever
    # exceeded fusion_ratio, the ordinary detector would catch it and this
    # test would pass without exercising the cumulative bound at all.
    assert _max_step_ratio(volume) < 2.5, (
        "phantom fuses sharply; it no longer tests gradual fusion"
    )

    mask, threshold = segment_incisor_by_threshold_descent(volume)

    bone = _bone_mask()
    tooth = _tooth_mask()

    assert mask.any() and np.isfinite(threshold)
    leaked = mask[bone].sum() / bone.sum()
    assert leaked < 0.05, (
        f"the descent ran through a gradual fusion and took in bone "
        f"({leaked:.1%} of the slab)"
    )
    assert mask[tooth].sum() / tooth.sum() > 0.5, "the tooth itself was lost"


def test_cumulative_bound_does_not_fire_on_a_clean_descent():
    """
    The bound must not cut short a descent that is merely recovering surface.

    Paired with the test above: a guard that stops early on healthy volumes
    would trade one failure mode for another, so the clean phantom is checked
    to give the same answer with the bound effectively disabled.
    """
    volume = _fusing_mandible()
    bounded, _ = segment_incisor_by_threshold_descent(volume)
    unbounded, _ = segment_incisor_by_threshold_descent(
        volume, max_total_growth=1e9
    )
    assert int(bounded.sum()) == int(unbounded.sum()), (
        "the cumulative bound changed the result on a cleanly-fusing volume"
    )


def test_refinement_cannot_return_a_fused_mask():
    """
    The refine pass must not undo the coarse pass's decision.

    Its sweep covers one coarse step, too narrow for either fusion test to
    fire, so without a bound it returns the largest component in the window --
    the fused one the coarse pass had just stopped before. On P82-Odam-CTR-2
    that turned a 261k mask into 635k.
    """
    volume = _gradually_fusing_mandible()
    refined, _ = segment_incisor_by_threshold_descent(volume, refine=True)
    coarse, _ = segment_incisor_by_threshold_descent(volume, refine=False)

    assert refined.sum() <= 1.5 * coarse.sum(), (
        f"refinement inflated the mask {refined.sum() / max(coarse.sum(), 1):.1f}x "
        "over the unrefined result, so it stepped into the fusion"
    )



# ---------------------------------------------------------------------------
# Cervical loop tracking
# ---------------------------------------------------------------------------

def _mandible_with_dim_loop(loop_level=None, loop_slices=30, tail_slices=0,
                            tooth_start=70):
    """
    A phantom whose cervical loop is *dimmer than the bone beside it*.

    This is the case that defeats every threshold-plus-connectivity method,
    and it is what the real scans show: on P82-OdamcKo-CTR-2, relaxing the
    descent's threshold to 0.9x leaves a sane component that stops short of the
    loop, while 0.7x fuses the tooth to the whole mandible. There is no level
    in between. So the loop here is placed below BONE deliberately -- a
    phantom whose loop is brighter than its bone would be solvable by
    thresholding and would not test tracking at all.

    ``tail_slices`` optionally continues a *different* structure past the loop
    along the same axis, for the stopping-criterion tests.
    """
    rng = np.random.default_rng(23)
    if loop_level is None:
        loop_level = BONE * 0.72

    # Built from scratch rather than from _fusing_mandible(): that phantom's
    # tooth starts at z=4, leaving no room below it for a loop to occupy.
    volume = np.full(SHAPE, AIR, dtype=np.float32)
    volume[BONE_REGION] = BONE + rng.normal(0.0, 0.01, size=volume[BONE_REGION].shape)
    tooth = np.s_[tooth_start:116, 14:20, 40:46]
    volume[tooth] = TOOTH + rng.normal(0.0, 0.01, size=volume[tooth].shape)

    # The loop: continues the tooth's line, apical to it, and tapers.
    start = tooth_start
    for i in range(loop_slices):
        z = start - 1 - i
        if z < 0:
            break
        # Taper, but never below the tracker's minimum slice size, so a track
        # that should stop for another reason is not cut short by this one.
        half = 3 if i < loop_slices // 2 else 2
        region = np.s_[z, 17 - half:17 + half, 43 - half:43 + half]
        volume[region] = loop_level + rng.normal(0.0, 0.004, size=volume[region].shape)

    # An unrelated structure further on, at constant width: what a track that
    # has left the loop latches onto.
    for i in range(tail_slices):
        z = start - 1 - loop_slices - i
        if z < 0:
            break
        # Wider than the tapered loop, so picking it up is a clear re-growth.
        region = np.s_[z, 13:21, 39:47]
        volume[region] = loop_level + rng.normal(0.0, 0.004, size=volume[region].shape)
    return volume


def test_tracking_reaches_a_loop_that_thresholding_cannot():
    """
    The tracker must extend the mask into a loop dimmer than nearby bone.

    The premise is asserted first: if the loop were bright enough for the
    descent to have taken it already, the test would pass without exercising
    the tracker.
    """
    volume = _mandible_with_dim_loop()
    base, threshold = segment_incisor_by_threshold_descent(volume)

    base_start = np.flatnonzero(base.reshape(base.shape[0], -1).sum(1)).min()
    assert base_start >= 70, (
        "the descent already reached the loop; the phantom is not testing tracking"
    )

    tracked = track_cervical_loop(volume, base, threshold)
    tracked_start = np.flatnonzero(tracked.reshape(tracked.shape[0], -1).sum(1)).min()

    assert tracked[base].all(), "tracking dropped voxels from the mask"
    assert tracked_start < base_start, (
        f"tracking did not extend the mask apically ({base_start} -> {tracked_start})"
    )


def test_tracking_stays_out_of_bone():
    """
    The geometric restriction is the only thing making so low a threshold
    safe, so bone exclusion is what has to be asserted.
    """
    volume = _mandible_with_dim_loop()
    base, threshold = segment_incisor_by_threshold_descent(volume)
    tracked = track_cervical_loop(volume, base, threshold)

    bone = _bone_mask()
    leaked = tracked[bone].sum() / bone.sum()
    assert leaked < 0.02, f"tracking leaked into bone ({leaked:.1%} of the slab)"


def test_tracking_stops_when_the_cross_section_grows_back():
    """
    A track that has left the loop and found something else must stop.

    The tail here is a constant-width structure past a tapering loop, which is
    what P82-OdamcKo-CTR-2 showed: the cross-section fell to 0.54 of its peak
    and then climbed back to 0.84 as the track followed a different structure
    to the edge of the volume.
    """
    volume = _mandible_with_dim_loop(loop_slices=24, tail_slices=60)
    base, threshold = segment_incisor_by_threshold_descent(volume)
    tracked = track_cervical_loop(volume, base, threshold)

    start = np.flatnonzero(tracked.reshape(tracked.shape[0], -1).sum(1)).min()
    base_start = np.flatnonzero(base.reshape(base.shape[0], -1).sum(1)).min()
    extended = base_start - start

    # It should follow the loop but not run the whole length of the tail.
    assert extended > 0, "tracking did not start"
    assert extended <= 24 + 8, (
        f"tracking ran {extended} slices past a {24}-slice loop — it followed "
        "the wider tail structure instead of stopping at the turnaround"
    )


def test_tracking_rejects_a_blob_that_sits_off_the_axis():
    """
    A bright structure inside the search disk but away from the predicted
    centre must not be accepted.

    The disk has to be wide enough to hold the loop wherever it curves to, so
    it inevitably admits some neighbouring tissue too. The drift check is what
    distinguishes the two, and without it the track can step sideways onto
    bone and then keep following it.
    """
    volume = _mandible_with_dim_loop(loop_slices=0)
    rng = np.random.default_rng(31)

    # A distractor just inside the disk's edge, offset from the tooth's axis,
    # and brighter than the loop would be -- so only its position disqualifies
    # it.
    for z in range(58, 70):
        region = np.s_[z, 22:28, 48:54]
        volume[region] = TOOTH * 0.5 + rng.normal(0.0, 0.004, size=volume[region].shape)

    base, threshold = segment_incisor_by_threshold_descent(volume)
    tracked = track_cervical_loop(volume, base, threshold)

    distractor = np.zeros(SHAPE, dtype=bool)
    distractor[58:70, 22:28, 48:54] = True
    assert tracked[distractor].sum() == 0, (
        "tracking stepped onto an off-axis structure inside the search disk"
    )


def test_tracking_handles_a_mask_too_short_to_fit_an_axis():
    """A mask of one slice has no trajectory to extrapolate."""
    volume = _mandible_with_dim_loop()
    tiny = np.zeros(SHAPE, dtype=bool)
    tiny[60, 14:20, 40:46] = True
    out = track_cervical_loop(volume, tiny, 0.5)
    assert int(out.sum()) == int(tiny.sum())


def test_tracking_passes_through_a_nan_threshold():
    """A failed descent must not become an exception here."""
    out = track_cervical_loop(
        _mandible_with_dim_loop(), np.zeros(SHAPE, dtype=bool), float("nan")
    )
    assert not out.any()



if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
