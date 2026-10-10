"""
Tests for keeping the slice tracker off bone that touches the incisor.

The phantom is the P42_WT_WT_1 failure in miniature: a tooth running the
length of the volume inside a thin sleeve of bone, with the dark periodontal
gap between them, and a stretch of slices where a bar bridges the gap. On those
slices the per-slice threshold returns tooth, bridge and sleeve as one region.
The tracker's own guard erodes a merged region until it splits, and accepts
the first split it sees. On the real scan that split was a chip: the first
erosion of a 12,525-voxel merge at slice 854 broke off slivers of 93, 21 and 1
voxels, and the guard kept the rest, tooth and bone ring still joined. The
sleeve here carries the same kind of spur, which the first erosion parts from
it.

The veto is the incisor's watershed basin, flooded from a core inside the
tooth against markers on the densest part of the sleeve. The bridge is dimmer
than the sleeve it leads to, as a blurred ligament is, so the sleeve is
claimed from its own markers before the tooth's flood crosses.

Run with:  python -m pytest segment_mandible/tests/test_incisor_veto.py
"""

# The package imports below have to follow the sys.path setup that makes
# segment_mandible/ importable when this file is run directly.
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.incisor import incisor_basin, segment_incisor  # noqa: E402

SHAPE = (100, 64, 64)
CENTRE = 32
TOOTH_RADIUS = 7
SLEEVE = (11, 14)                  # inner and outer radius of the bone sleeve
BRIDGED = slice(40, 71)            # slices where a bar crosses the gap
TOOTH, SLEEVE_BONE, DENSE_BONE, BRIDGE = 0.85, 0.6, 0.75, 0.55
MIN_TIP_INTENSITY = 0.7            # the seed must be tooth, not sleeve


def _phantom(bridged=True):
    """(volume, tooth mask, sleeve mask)."""
    _, y, x = np.indices(SHAPE)
    radius = np.hypot(y - CENTRE, x - CENTRE)
    tooth = radius <= TOOTH_RADIUS
    sleeve = (radius > SLEEVE[0]) & (radius <= SLEEVE[1])

    volume = np.zeros(SHAPE, dtype=np.float32)
    volume[tooth] = TOOTH
    volume[sleeve] = SLEEVE_BONE
    # The densest bone, on the far side from the bridge: what the threshold
    # descent leaves above its threshold and the basin marks as not-tooth.
    volume[sleeve & (x > CENTRE + SLEEVE[0])] = DENSE_BONE
    # A spur on the outside of the sleeve, joined by a one-voxel neck: the
    # first erosion parts it from the sleeve.
    volume[:, CENTRE - 1:CENTRE + 2, CENTRE + SLEEVE[1] + 2:CENTRE + SLEEVE[1] + 5] = SLEEVE_BONE
    volume[:, CENTRE, CENTRE + SLEEVE[1] + 1] = SLEEVE_BONE
    if bridged:
        bar = np.zeros(SHAPE, dtype=bool)
        bar[BRIDGED, CENTRE - 3:CENTRE + 4, CENTRE - SLEEVE[0]:CENTRE - TOOTH_RADIUS] = True
        volume[bar] = BRIDGE
    return volume, tooth, sleeve


def _basin(volume, tooth):
    """The basin as the pipeline builds it: descent core vs bone above the descent threshold."""
    core = volume >= TOOTH
    other = (volume > MIN_TIP_INTENSITY) & ~core
    return incisor_basin(volume, core, other, volume > 0)


def test_phantom_reproduces_the_leak():
    """Without the veto the tracker takes the sleeve on the bridged slices."""
    volume, tooth, sleeve = _phantom()
    mask = segment_incisor(volume, MIN_TIP_INTENSITY)
    assert (mask & sleeve)[BRIDGED].sum() > 0.5 * sleeve[BRIDGED].sum()


def test_basin_holds_the_tooth_and_not_the_sleeve():
    volume, tooth, sleeve = _phantom()
    basin = _basin(volume, tooth)
    assert basin[tooth].all()
    assert not basin[sleeve].any()


def test_veto_keeps_the_sleeve_out():
    volume, tooth, sleeve = _phantom()
    mask = segment_incisor(volume, MIN_TIP_INTENSITY, veto=_basin(volume, tooth))
    assert not (mask & sleeve).any(), "the tracker still took in bone"
    assert (mask & tooth).sum() >= 0.99 * tooth.sum(), "the veto cost the tooth"


def test_veto_changes_nothing_where_nothing_merges():
    """With no bridge there is nothing to veto, and the mask is the tracker's own."""
    volume, tooth, _ = _phantom(bridged=False)
    plain = segment_incisor(volume, MIN_TIP_INTENSITY)
    vetoed = segment_incisor(volume, MIN_TIP_INTENSITY, veto=_basin(volume, tooth))
    assert np.array_equal(plain, vetoed)


def test_veto_spares_tooth_the_basin_gets_wrong():
    """
    Where the dentin is dimmer than the bone it touches, the basin can hand
    part of the tooth to the bone (on P42, a wedge at a trabecular contact).
    The tracker already held that part on the slice before, so the veto must
    leave it alone.
    """
    volume, tooth, _ = _phantom(bridged=False)
    basin = _basin(volume, tooth)
    wrong = basin.copy()
    wrong[30:60, CENTRE:CENTRE + 8, CENTRE:CENTRE + 8] = False   # basin loses a quadrant
    plain = segment_incisor(volume, MIN_TIP_INTENSITY)
    vetoed = segment_incisor(volume, MIN_TIP_INTENSITY, veto=wrong)
    assert np.array_equal(plain, vetoed)


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
