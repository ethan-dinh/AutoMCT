"""
Tests for trimming the background that comes away with the molars.

The trim removes two kinds of material. A faint skin, below the surface cut,
goes wherever it is. Dim material above the cut is a mix: the tooth has dim
parts of its own -- its blurred surface, the lining of the pulp chamber, the
tapering root tips -- and the claim is that *where* the dim material sits
tells them apart: background rests on the enamel-covered crown, the rest does
not. So the tooth here is built with one of each, placed by construction on
the right side of each rule, and the tests check which ones survive.

The tissue floor is passed in rather than found, so each test pins the
geometry rather than the histogram; the histogram step has its own test.

Run with:  python -m pytest segment_mandible/tests/test_molar_background.py
"""

# The package imports below have to follow the sys.path setup that makes
# segment_mandible/ importable when this file is run directly.
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.molar_bone import (  # noqa: E402
    _BACKGROUND_ENAMEL_REACH_VOXELS,
    _BACKGROUND_SURFACE_CUT_FRACTION,
    _find_tissue_floor,
    _trim_attached_background,
)

SHAPE = (64, 64, 128)

# The separation and enamel thresholds as segment_molar_bone would pass them,
# and a tissue floor between them.
LOW, HIGH, FLOOR = 0.18, 0.625, 0.35
SURFACE_CUT = LOW + _BACKGROUND_SURFACE_CUT_FRACTION * (FLOOR - LOW)

# FAINT sits under the surface cut, DIM between it and the tissue floor.
FAINT, DIM, DENTIN, ENAMEL = 0.21, 0.30, 0.5, 0.8
NOISE = 0.005

# Regions, as (z, y, x) slices.
DENTIN_BODY = np.s_[20:40, 20:40, 20:100]
ENAMEL_CAP = np.s_[40:44, 20:40, 20:60]          # crown half only
CROWN_BACKGROUND = np.s_[44:52, 24:36, 30:50]    # soft tissue resting on the cap
SPECK = np.s_[50:53, 28:31, 38:41]               # bright, held only by that tissue
PULP = np.s_[24:32, 25:35, 30:50]                # dim, walled off by dentin
ROOT_TIP = np.s_[28:32, 28:32, 100:112]          # dim, far from the enamel
SURFACE = np.s_[20:40, 40:41, 70:100]            # one-voxel blurred surface
FAINT_SKIN = np.s_[20:40, 19:20, 70:100]         # faint, far from the enamel


def _tooth(seed=0):
    """A dentin bar with an enamel cap, and one of each kind of dim material."""
    assert FAINT < SURFACE_CUT < DIM < FLOOR < DENTIN
    volume = np.zeros(SHAPE, dtype=np.float32)
    volume[DENTIN_BODY] = DENTIN
    volume[PULP] = DIM
    volume[ENAMEL_CAP] = ENAMEL
    volume[CROWN_BACKGROUND] = DIM
    volume[SPECK] = ENAMEL
    volume[ROOT_TIP] = DIM
    volume[SURFACE] = DIM
    volume[FAINT_SKIN] = FAINT
    rng = np.random.default_rng(seed)
    tissue = volume > 0
    volume[tissue] += rng.normal(0.0, NOISE, int(tissue.sum())).astype(np.float32)
    return volume, tissue, volume >= HIGH


def _trim(volume, molar, enamel):
    return _trim_attached_background(volume, molar, enamel, LOW, HIGH, tissue_floor=FLOOR)


def _region(sl):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[sl] = True
    return mask


def test_tissue_floor_falls_between_dim_material_and_dentin():
    rng = np.random.default_rng(0)
    vals = np.concatenate([
        rng.normal(DIM, 0.01, 5_000),
        rng.normal(DENTIN, 0.01, 30_000),
    ])
    floor = _find_tissue_floor(vals, LOW, HIGH)
    assert floor is not None
    assert DIM < floor < DENTIN


def test_faint_skin_is_removed_wherever_it_is():
    volume, molar, enamel = _tooth()
    out = _trim(volume, molar, enamel)
    assert not out[FAINT_SKIN].any()


def test_background_resting_on_the_crown_is_removed():
    volume, molar, enamel = _tooth()
    out = _trim(volume, molar, enamel)
    # Two voxels next to the cap are the tooth's blurred surface and stay;
    # everything beyond them is background.
    beyond_rim = _region(CROWN_BACKGROUND) & ~_region(np.s_[44:46, :, :]) & ~_region(SPECK)
    assert not (out & beyond_rim).any()


def test_speck_held_only_by_background_goes_with_it():
    """A bright fragment is solid, but once the tissue holding it is gone it is detached."""
    volume, molar, enamel = _tooth()
    out = _trim(volume, molar, enamel)
    assert not (out & _region(SPECK)).any()


def test_tooth_and_its_own_dim_parts_are_kept():
    volume, molar, enamel = _tooth()
    out = _trim(volume, molar, enamel)
    assert out[DENTIN_BODY].all()
    assert out[ENAMEL_CAP].all()
    assert out[SURFACE].all(), "blurred surface within the rim was trimmed"
    assert out[PULP].all(), "pulp lining walled off from the enamel was trimmed"
    assert out[ROOT_TIP].all(), "root tip far from the enamel was trimmed"


def test_reach_cap_stops_a_sheet_running_down_the_root():
    """
    A dim sheet from the crown down the side of the root is one connected
    piece resting on the enamel, but only its part near the enamel may go.
    """
    volume, molar, enamel = _tooth()
    sheet = _region(np.s_[30:44, 40:48, 20:100])
    volume[sheet] = DIM
    molar |= sheet
    out = _trim(volume, molar, enamel)

    enamel_dist = ndi.distance_transform_edt(~enamel)
    beyond_rim = sheet & (ndi.distance_transform_edt(~(molar & (volume > FLOOR))) > 2)
    assert not (out & beyond_rim & (enamel_dist <= _BACKGROUND_ENAMEL_REACH_VOXELS)).any()
    far = sheet & (enamel_dist > _BACKGROUND_ENAMEL_REACH_VOXELS)
    assert far.any() and out[far].all()


def test_no_enamel_leaves_the_mask_unchanged():
    volume, molar, _ = _tooth()
    out = _trim(volume, molar, np.zeros(SHAPE, dtype=bool))
    assert np.array_equal(out, molar)
