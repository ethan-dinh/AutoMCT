"""
Equivalence tests for the margin shrink's fast paths.

The shrink trims the surface that downstream intensity measurements
(mineral density, enamel thickness) are taken inside of, so a fast path that
trimmed a slightly different shape would bias those numbers silently -- there
is no output that would look wrong. Every test here therefore pins the fast
paths to be *bit-identical* to the distance transform they replace, not merely
close to it.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from segmentation.incisor import (
    _ball_footprint,
    _mask_bounding_box,
    _shrink_by_distance,
    shrink_incisor_margin,
)


def _reference(mask, margin_mm, spacing):
    """The unoptimised definition: threshold a full-volume distance transform."""
    return ndi.distance_transform_edt(mask, sampling=spacing) > margin_mm


def _blob(shape=(60, 50, 50), seed=0):
    """A solid, offset blob -- a stand-in for a tooth inside a larger volume."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    mask[10:45, 15:35, 12:38] = True
    # Roughen the surface so the trimmed boundary is not a trivial box face.
    noise = rng.random(shape) > 0.02
    return mask & noise


ISOTROPIC = (0.008, 0.008, 0.008)
ANISOTROPIC = (0.020, 0.008, 0.008)


@pytest.mark.parametrize("spacing", [ISOTROPIC, ANISOTROPIC, None])
@pytest.mark.parametrize("margin", [0.008, 0.016, 0.024])
def test_fast_path_matches_distance_transform(spacing, margin):
    """The ball erosion and the EDT threshold keep exactly the same voxels."""
    mask = _blob()
    margin_mm = margin if spacing is not None else 2.0
    got, _ = _shrink_by_distance(mask, margin_mm, spacing, workers=1)
    assert np.array_equal(got, _reference(mask, margin_mm, spacing))


def test_anisotropic_margin_is_a_ball_not_a_box():
    """
    On anisotropic spacing the trim depth must follow physical distance, so a
    coarse axis is trimmed fewer voxels than a fine one. A box footprint would
    trim the same voxel count on every axis, which is the specific error this
    footprint exists to avoid.
    """
    footprint = _ball_footprint(0.024, ANISOTROPIC)
    assert footprint is not None
    # 0.024mm is 1.2 voxels on the 0.020mm axis but 3 on the 0.008mm axes.
    assert footprint.shape == (5, 7, 7)
    centre = tuple(s // 2 for s in footprint.shape)
    # The corner of the bounding box is outside the ball; the axis ends are in.
    assert not footprint[0, 0, 0]
    assert footprint[centre[0], centre[1], 0]


def test_wide_margin_falls_through_to_the_transform():
    """A margin too wide for a cheap footprint still resolves via the EDT."""
    spacing = ISOTROPIC
    margin_mm = 0.080  # 10 voxels -- past the ball path's ceiling
    assert _ball_footprint(margin_mm, spacing) is None
    mask = _blob()
    got, _ = _shrink_by_distance(mask, margin_mm, spacing, workers=1)
    assert np.array_equal(got, _reference(mask, margin_mm, spacing))


def test_parallel_path_matches_the_serial_one():
    """Slab chunking must not change the result at the seams."""
    mask = _blob(shape=(240, 40, 40))
    margin_mm = 0.080
    serial, _ = _shrink_by_distance(mask, margin_mm, ISOTROPIC, workers=1)
    parallel, _ = _shrink_by_distance(mask, margin_mm, ISOTROPIC, workers=8)
    assert np.array_equal(serial, parallel)
    assert np.array_equal(serial, _reference(mask, margin_mm, ISOTROPIC))


def test_result_is_returned_in_the_original_coordinate_space():
    """
    Cropping to the bounding box is an internal optimisation: the caller
    indexes the returned mask against the full volume, so it must come back at
    the input's shape with the foreground in its original position.
    """
    mask = np.zeros((80, 60, 60), dtype=bool)
    mask[50:70, 40:55, 5:20] = True  # deliberately off-centre
    got, _ = _shrink_by_distance(mask, 0.024, ISOTROPIC, workers=1)

    assert got.shape == mask.shape
    assert np.array_equal(got, _reference(mask, 0.024, ISOTROPIC))
    # The surviving voxels sit inside the original blob, not shifted to the origin.
    assert got[50:70, 40:55, 5:20].any()
    assert not got[:50].any()


def test_bounding_box_pads_by_the_margin():
    """
    The box must include the background just outside the foreground, or a
    surface voxel would measure its distance to the crop face instead.
    """
    mask = np.zeros((60, 60, 60), dtype=bool)
    mask[20:40, 20:40, 20:40] = True
    box = _mask_bounding_box(mask, 0.024, ISOTROPIC)
    assert box is not None
    for axis_slice in box:
        assert axis_slice.start <= 20 - 3
        assert axis_slice.stop >= 40 + 3


def test_mask_touching_the_array_edge_is_not_trimmed_from_outside():
    """
    Outside-the-array is not background: the EDT measures only to background
    voxels that exist, so a mask running off the edge keeps its edge voxels.
    Both the crop and the erosion's border_value have to preserve that.
    """
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[:, 10:30, 10:30] = True  # spans the full Z extent
    got, _ = _shrink_by_distance(mask, 0.024, ISOTROPIC, workers=1)
    assert np.array_equal(got, _reference(mask, 0.024, ISOTROPIC))
    assert got[0].any()


def test_empty_mask_survives_the_crop():
    """An empty mask has no bounding box; it must not raise."""
    mask = np.zeros((30, 30, 30), dtype=bool)
    got, _ = _shrink_by_distance(mask, 0.024, ISOTROPIC, workers=1)
    assert not got.any()
    assert got.shape == mask.shape


def test_shrink_preserves_shape_end_to_end():
    """The public entry point keeps the caller's shape."""
    mask = _blob(shape=(70, 55, 55))
    got = shrink_incisor_margin(mask, 0.024, ISOTROPIC, workers=1)
    assert got.shape == mask.shape
    assert np.array_equal(got, _reference(mask, 0.024, ISOTROPIC))


def test_margin_wide_enough_to_erase_the_mask_raises():
    """A margin past the mask's half-thickness is a units error, not a shrink."""
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[18:22, 18:22, 18:22] = True
    with pytest.raises(ValueError, match="would remove the entire mask"):
        shrink_incisor_margin(mask, 0.5, ISOTROPIC, workers=1)
