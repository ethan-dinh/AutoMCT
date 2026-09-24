"""
Tests for the int16 saturation trim in the ISQ reader.

Scanco reconstructions carry a few voxels pinned to the int16 rails
(-32768 / 32767). Neither rail is a measurement -- at mu_scaling 4096 they land
on exactly -8.000 and +8.000 1/cm, the signature of a fixed-point limit.

On f0004019 both clusters were reconstruction artifacts. The 13 low-rail voxels
sat 3-4 voxels from the rotation centre, where 11.9% of voxels ran below -20000
against 0.000% elsewhere in the same slice; a negative attenuation is
impossible regardless. The 47 high-rail voxels sat 1.76 mm *outside* the
specimen in the mounting medium, in a blob 6 slices thick flanked by values of
-4086 -- streak artifact, not dense material.

Sixty voxels in four billion is ignorable for densitometry and not at all
ignorable for anything keyed to the volume's *extremes* -- a single voxel at
-32768 sets ``volume.min()``, and every min/max-derived quantity downstream
inherits it.

What these tests deliberately do NOT claim: that trimming prevents int16
overflow in downstream arithmetic. Of four scans checked, all four spanned more
than 32767 counts and only one had any saturation at all, so a normalizer that
subtracts int16 extremes in the input dtype overflows on clean scans too. That
fix belongs at the arithmetic (see segment_mandible/preprocessing/filters.py),
and ``test_trim_does_not_guarantee_int16_safe_span`` pins the distinction so no
one later mistakes this for a fix it is not.
"""

import numpy as np
import pytest

from convert_ISQ_NRRD import trim_saturated

INT16 = np.iinfo(np.int16)


def _scan(shape=(40, 60, 60), seed=0, low=-500, high=500):
    """A clean int16 volume with no saturated voxels."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=shape).astype(np.int16)


def test_rails_are_clamped_into_the_real_signal_range():
    volume = _scan()
    volume[5, 5, 5] = INT16.min
    volume[9, 9, 9] = INT16.max

    trimmed, info = trim_saturated(volume)

    assert info["applied"] is True
    assert (info["n_low"], info["n_high"]) == (1, 1)
    assert trimmed.min() > INT16.min
    assert trimmed.max() < INT16.max


def test_non_rail_voxels_are_bit_identical():
    """Densitometry must be unaffected: only rail voxels may change.

    This is what makes the trim safe to leave on by default -- it is not a
    filter, and it does not touch a single measured value.
    """
    volume = _scan()
    volume[1, 2, 3] = INT16.min
    volume[4, 5, 6] = INT16.max
    interior = (volume != INT16.min) & (volume != INT16.max)

    trimmed, _ = trim_saturated(volume)

    assert np.array_equal(trimmed[interior], volume[interior])


def test_clean_volume_is_returned_untouched():
    """No rails means no copy and no change -- the common case stays free."""
    volume = _scan()
    trimmed, info = trim_saturated(volume)

    assert info["applied"] is False
    assert trimmed is volume


def test_float_volume_is_left_alone():
    """--apply-scaling produces floats, which have no int16 rails to clamp."""
    volume = np.array([1.0, 2.0], dtype=np.float32)
    trimmed, info = trim_saturated(volume)

    assert info["applied"] is False
    assert trimmed is volume


def test_fill_comes_from_the_local_neighbourhood():
    """The whole point of neighbour-median over a global constant.

    Two rail voxels sitting in regions of very different intensity must be
    filled from their own surroundings, not from one volume-wide value. On the
    real scan one cluster sat in tissue and the other 1.76 mm outside the
    specimen in mounting medium, where a single global fill would be wrong in
    at least one of the two places.
    """
    volume = np.zeros((20, 20, 20), dtype=np.int16)
    volume[:10] = 500        # "tissue" half
    volume[10:] = 5000       # "dense" half
    volume[5, 10, 10] = INT16.max
    volume[15, 10, 10] = INT16.min

    trimmed, info = trim_saturated(volume)

    assert info["applied"] is True
    assert trimmed[5, 10, 10] == 500
    assert trimmed[15, 10, 10] == 5000


def test_heavily_clipped_volume_still_trims():
    """Rails common enough to dominate a neighbourhood must still resolve.

    Here a sixth of the volume sits on each rail, so many windows are mostly
    rail; the fill must come from whatever clean voxels remain.
    """
    tile = np.array([[INT16.min, 100, 200], [300, INT16.max, 400]], dtype=np.int16)
    volume = np.tile(tile, (50, 50)).astype(np.int16)

    trimmed, info = trim_saturated(volume)

    assert info["applied"] is True
    assert INT16.min < trimmed.min() and trimmed.max() < INT16.max
    # Whatever the route -- neighbourhood or the global-median fallback for
    # windows that were entirely rail -- no rail value may survive.
    assert not ((trimmed == INT16.min) | (trimmed == INT16.max)).any()


def test_all_rail_volume_is_filled_not_crashed():
    """Degenerate input must not crash on an empty neighbourhood.

    Nothing here can be recovered -- there is no clean voxel anywhere -- so the
    contract is only that it terminates and leaves no rail value behind.
    """
    volume = np.full((20, 20), INT16.min, dtype=np.int16)
    trimmed, info = trim_saturated(volume)

    assert info["applied"] is True
    assert info["unresolved"] == volume.size
    assert not (trimmed == INT16.min).any()


@pytest.mark.parametrize("shape", [(64,), (32, 32), (16, 24, 24), (4, 8, 8, 8)])
def test_works_at_any_dimensionality(shape):
    """The subsample stride is built per-axis, so rank must not matter."""
    volume = _scan(shape=shape)
    flat = volume.reshape(-1)
    flat[0] = INT16.min
    flat[-1] = INT16.max

    trimmed, info = trim_saturated(volume)

    assert info["applied"] is True
    assert trimmed.shape == volume.shape
    assert trimmed.min() > INT16.min and trimmed.max() < INT16.max


def test_trim_does_not_guarantee_int16_safe_span():
    """Trimming is not an overflow fix, and must not be mistaken for one.

    A scan can span more than 32767 counts with zero saturated voxels -- three
    of the four real scans checked did exactly that. Downstream code must do
    min/max arithmetic in float regardless of this trim.
    """
    volume = _scan(low=-22266, high=31415)
    volume.flat[0] = -22266
    volume.flat[1] = 31415

    trimmed, info = trim_saturated(volume)

    assert info["applied"] is False           # nothing on the rails
    span = int(trimmed.max()) - int(trimmed.min())
    assert span > INT16.max                   # yet still overflows int16
