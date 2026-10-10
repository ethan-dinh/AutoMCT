"""
Tests for finding the threshold at which the molars part from bone.

The phantom is P42_WT_WT_1's jaw in miniature: bone denser than the molar's
dentin (as it is by P42), a molar touching the bone through a dim contact (the
blurred periodontal ligament), and a small dense patch inside the bone that
reaches the enamel threshold. On the real scan there was no enamel peak, the
enamel threshold fell back to a knee at 0.58, and a 4,586-voxel patch of
cortical bone came in with the molar enamel as a second "enamel" cluster.

That patch is part of the bone at every threshold. The search waits for every
enamel-bearing component to leave the bone, so it climbed until the bone was
thresholded away around the patch -- by which point the molar's dentin, being
dimmer than the bone, had gone too, and the molars were their enamel caps.

Run with:  python -m pytest segment_mandible/tests/test_molar_separation.py
"""

# The package imports below have to follow the sys.path setup that makes
# segment_mandible/ importable when this file is run directly.
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.molar_bone import (  # noqa: E402
    _find_molar_separation_threshold,
    _fused_with_bone,
)

SHAPE = (40, 48, 96)
BONE, DENTIN, CONTACT, ENAMEL = 0.5, 0.4, 0.3, 0.9
ENAMEL_THRESHOLD = 0.85

BONE_BLOCK = np.s_[4:36, 4:44, 4:44]
PATCH = np.s_[18:21, 22:25, 20:23]           # 27 voxels, ~1% of the enamel
DENTIN_BODY = np.s_[10:30, 12:36, 52:80]
ENAMEL_CAP = np.s_[30:34, 12:36, 52:80]
CONTACT_BAR = np.s_[14:18, 20:28, 44:52]     # joins the bone to the dentin
ANCHOR = np.s_[:, :, :16]                    # bone well away from the molar


def _phantom(patch=True):
    volume = np.zeros(SHAPE, dtype=np.float32)
    volume[BONE_BLOCK] = BONE
    volume[DENTIN_BODY] = DENTIN
    volume[ENAMEL_CAP] = ENAMEL
    volume[CONTACT_BAR] = CONTACT
    if patch:
        volume[PATCH] = ENAMEL
    foreground = volume > 0
    anchor = np.zeros(SHAPE, dtype=bool)
    anchor[ANCHOR] = True
    return volume, volume >= ENAMEL_THRESHOLD, foreground & anchor


def _region(sl):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[sl] = True
    return mask


def _search(volume, enamel, anchor):
    return _find_molar_separation_threshold(
        volume, enamel, anchor, floor=float(volume[volume > 0].min()), ceiling=ENAMEL_THRESHOLD,
    )


def test_molar_parts_from_bone_below_its_dentin():
    """The molar comes away at the contact, with its dentin, despite the patch."""
    volume, enamel, anchor = _phantom()
    labels, threshold, _ = _search(volume, enamel, anchor)
    assert labels is not None
    assert CONTACT <= threshold < DENTIN, threshold
    tooth = np.unique(labels[_region(ENAMEL_CAP)])
    assert tooth.size == 1 and tooth[0] > 0
    assert (labels[_region(DENTIN_BODY)] == tooth[0]).all(), "the dentin was cut off the molar"


def test_patch_stays_with_the_bone():
    """At that threshold the patch's component is the bone, and is fused, not a molar."""
    volume, enamel, anchor = _phantom()
    labels, _, _ = _search(volume, enamel, anchor)
    ids = np.unique(labels[enamel])
    ids = ids[ids > 0]
    fused = _fused_with_bone(labels, ids, anchor)
    patch_label = np.unique(labels[_region(PATCH)])
    assert list(fused) == list(patch_label)
    assert labels[_region(ENAMEL_CAP)][0] not in fused


def test_same_threshold_without_the_patch():
    """The patch must not move the threshold at all."""
    with_patch = _search(*_phantom())[1]
    without = _search(*_phantom(patch=False))[1]
    assert abs(with_patch - without) < 0.01


def test_molar_still_held_while_fused():
    """
    The rule must not let a real tooth go early: with the contact as bright as
    the dentin, the molar is part of the bone until the dentin itself is cut,
    and the search has to climb past it.
    """
    volume, enamel, anchor = _phantom()
    volume[CONTACT_BAR] = DENTIN
    _, threshold, _ = _search(volume, enamel, anchor)
    assert threshold >= DENTIN


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
