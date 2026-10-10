"""
Tests for cleaning the bone mask: no soft tissue, no islands.

The bone mask is whatever the background threshold let in that is neither
incisor nor molar, and on a real scan that includes the soft tissue round the
teeth (the pulp, the periodontal space, the gingiva) and specks of noise. The
phantom here is a block of bone with a socket through it: a tooth down the
middle, cut out of the mask as the incisor is, a sleeve of soft tissue between
the tooth and the bone, and soft tissue over one face of the block. A few
bright specks float in the background. The densities are on the scale the
pipeline sees them (P22_TGM2_KO_F_1, normalized: background/bone knee 0.13,
soft tissue 0.15, bone 0.34), with blur and noise for the partial-volume
spread of a scan.

Run with:  python -m pytest segment_mandible/tests/test_bone_cleanup.py
"""

# The package imports below have to follow the sys.path setup that makes
# segment_mandible/ importable when this file is run directly.
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.molar_bone import clean_bone_mask, find_bone_floor  # noqa: E402

SHAPE = (48, 72, 72)
BACKGROUND, SOFT, BONE, TOOTH = 0.05, 0.16, 0.35, 0.7
KNEE = 0.13                                   # the background threshold
SOCKET, TOOTH_RADIUS = 15, 7                  # radii round the socket's axis
BLOCK = np.s_[6:40, 8:64, 8:64]
COAT = np.s_[40:44, 8:64, 8:64]               # soft tissue over the block's top face
SPECKS = [np.s_[2:5, 2:5, 2:5], np.s_[44:47, 66:69, 3:6], np.s_[20:23, 2:5, 66:69]]


def _phantom(seed=0, soft=SOFT, bone=BONE):
    """(volume, candidate, {region: mask}) -- the candidate as the pipeline hands it over."""
    z, y, x = np.indices(SHAPE)
    radius = np.hypot(y - 36, x - 36)
    block = np.zeros(SHAPE, bool)
    block[BLOCK] = True
    tooth = block & (radius <= TOOTH_RADIUS)
    sleeve = block & (radius > TOOTH_RADIUS) & (radius <= SOCKET)
    body = block & (radius > SOCKET)
    coat = np.zeros(SHAPE, bool)
    coat[COAT] = True
    specks = np.zeros(SHAPE, bool)
    for sl in SPECKS:
        specks[sl] = True

    volume = np.full(SHAPE, BACKGROUND)
    volume[body] = bone
    volume[sleeve] = soft
    volume[coat] = SOFT
    volume[tooth] = TOOTH
    volume[specks] = bone
    volume = ndi.gaussian_filter(volume, 0.8)
    volume += np.random.default_rng(seed).normal(0.0, 0.01, SHAPE)

    # As the pipeline does it: above the knee, less the tooth dilated by two.
    candidate = (volume > KNEE) & ~ndi.binary_dilation(tooth, iterations=2)
    # Clear of the dilated tooth and of the blurred edges, where the sleeve is itself.
    inner = sleeve & (radius >= TOOTH_RADIUS + 4) & (radius <= SOCKET - 2)
    return volume, candidate, {"body": body, "sleeve": sleeve, "inner": inner, "coat": coat, "specks": specks}


def _share(mask, region):
    return float((mask & region).sum()) / float(region.sum())


def test_the_soft_tissue_round_a_tooth_is_not_bone():
    volume, candidate, parts = _phantom()
    coat = ndi.binary_erosion(parts["coat"], iterations=1)
    assert _share(candidate, parts["inner"]) > 0.9            # the threshold let it in
    assert _share(candidate, coat) > 0.9
    bone, info = clean_bone_mask(volume, candidate)
    assert SOFT < info["floor"] < BONE
    assert _share(bone, parts["inner"]) < 0.02
    assert _share(bone, coat) < 0.02
    assert _share(bone, ndi.binary_erosion(parts["body"], iterations=1)) > 0.99


def test_islands_are_dropped():
    volume, candidate, parts = _phantom()
    assert _share(candidate, parts["specks"]) > 0.5
    bone, info = clean_bone_mask(volume, candidate)
    assert not (bone & ndi.binary_dilation(parts["specks"], iterations=2)).any()
    assert info["islands"] > 0
    labels, n = ndi.label(bone, structure=ndi.generate_binary_structure(3, 3))
    assert n == 1


def test_cleaning_twice_keeps_the_bone():
    # On the P22 scans a second pass finds no floor at all. The phantom's flat,
    # evenly blurred faces leave more partial volume than a scan does, so here
    # it may trim that rim again -- but never into the bone.
    volume, candidate, parts = _phantom()
    bone, _ = clean_bone_mask(volume, candidate)
    again, _ = clean_bone_mask(volume, bone)
    assert not (again & ~bone).any()
    assert _share(again, ndi.binary_erosion(parts["body"], iterations=1)) > 0.99


def test_bone_right_up_to_the_tooth_is_kept():
    # The socket holds bone right up to the tooth: all of it stays.
    volume, candidate, parts = _phantom(soft=BONE)
    bone, info = clean_bone_mask(volume, candidate)
    assert info["floor"] is None or info["floor"] < BONE
    assert _share(bone, parts["inner"] | ndi.binary_erosion(parts["body"], iterations=1)) > 0.99


def test_a_mode_just_under_bone_is_more_bone():
    # Two densities of bone, not soft tissue and bone: the lower one is kept.
    volume, candidate, parts = _phantom(soft=0.30)
    bone, info = clean_bone_mask(volume, candidate)
    assert info["floor"] is None or info["floor"] < 0.30
    assert _share(bone, parts["inner"]) > 0.99


def test_no_valley_no_floor():
    values = np.random.default_rng(1).normal(0.5, 0.05, 20_000)
    assert find_bone_floor(values, 0.2, 0.8) is None
    assert find_bone_floor(np.array([]), 0.2, 0.8) is None
