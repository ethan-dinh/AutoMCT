"""
Tests for finding eruption, and for keeping the landmarks set by hand.

The phantom is a straight incisor -- a hollow tube of dentin along z, tip at
high z, with a band of bright enamel over 120 degrees of its surface, as the
labial band is -- in a sleeve of bone with a gap round it, as a socket has.
Toward the tip the bone over the enamel opens, the open arc widening steadily
from nothing to half the circle over 20 slices, while a strip of bone under
the lingual side runs on for a while. Eruption is where no ray from the enamel
touches bone: with the band's edge rays left out (3 degrees each side), where
the opening passes 114 degrees, 12.7 slices into it.

Run with:  python -m pytest longitudinal_incisor_analysis/test_detect_landmarks.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_landmarks import (  # noqa: E402
    enamel_rays,
    enamel_threshold,
    find_eruption,
    merge_marks,
    ray_survey,
)

SHAPE = (160, 64, 64)
CENTRE = (32.0, 32.0)
TOOTH_Z = (5, 150)               # the incisor, apex to tip
OPENING = (100, 120)             # where the bone over the enamel opens, 0 to 180 degrees
STRIP_END = 130                  # the strip of bone under the lingual side runs to here
DENTIN, ENAMEL = 14000.0, 25000.0
ENAMEL_HALF_DEG = 60.0
FREE_Z = OPENING[0] + (OPENING[1] - OPENING[0]) * 2 * (ENAMEL_HALF_DEG - 3.0) / 180.0


def _phantom(enamel=ENAMEL, over_enamel_until=None, gap=False, bone=True, seed=0):
    """(incisor volume, bone, centerline, frames) -- the centerline one point per slice, apex first."""
    z, y, x = np.indices(SHAPE, dtype=np.float32)
    r = np.hypot(y - CENTRE[0], x - CENTRE[1])
    angle = np.degrees(np.arctan2(y - CENTRE[0], x - CENTRE[1])) % 360.0     # 0 along +x
    off_labial = np.minimum(angle, 360.0 - angle)                           # degrees from +x
    tube = (r >= 3) & (r <= 8) & (z >= TOOTH_Z[0]) & (z < TOOTH_Z[1])      # dentin round the pulp
    volume = np.where(tube, DENTIN, 0.0)
    volume[tube & (r >= 6) & (off_labial <= ENAMEL_HALF_DEG)] = enamel
    volume[tube] += np.random.default_rng(seed).normal(0.0, 400.0, int(tube.sum()))
    wall = (r >= 12) & (r <= 18)
    opened = np.clip((z - OPENING[0]) / (OPENING[1] - OPENING[0]), 0.0, 1.0) * 180.0
    bones = wall & (z >= 2) & (z < OPENING[1]) & (off_labial >= opened / 2.0)
    bones |= wall & (z >= OPENING[1]) & (z < STRIP_END) & (np.abs(angle - 180.0) <= 45.0)   # lingual strip
    if over_enamel_until is not None:       # a strip of bone over part of the enamel, running on
        bones |= wall & (z >= 2) & (z < over_enamel_until) & (np.abs(angle - 30.0) <= 8.0)
    if gap:                                  # a hole through the bone over all of the enamel, inside the jaw
        bones &= ~((z >= 60) & (z < 66) & (off_labial <= 75.0))
    if not bone:
        bones[:] = False
    centerline = np.array([(k, *CENTRE) for k in range(TOOTH_Z[0], TOOTH_Z[1])], dtype=float)
    frames = np.tile(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), (len(centerline), 1, 1))
    return volume.astype(np.float32), bones, centerline, frames


def _eruption(flip=False, threshold=None, **kw):
    """(z of the eruption or None, find_eruption's answer)."""
    volume, bones, centerline, frames = _phantom(**kw)
    if flip:                                   # traced from the tip instead
        centerline, frames = centerline[::-1], frames[::-1]
    hits, surface = ray_survey(centerline, frames, volume, bones, tooth_reach_vox=12, reach_vox=12,
                               depth_vox=2, step=0.5)
    level = enamel_threshold(volume[volume > 0]) if threshold is None else threshold
    found = find_eruption(hits, enamel_rays(surface, level, edge=1), tip_at_index_0=flip, run_sections=5)
    return (None if found["index"] is None else float(centerline[found["index"], 0])), found


def test_eruption_is_where_no_ray_from_the_enamel_touches_bone():
    z, found = _eruption()
    assert abs(z - FREE_Z) <= 2, z
    assert found["reason"] is None
    assert OPENING[0] - 6 <= found["covered_until"] + TOOTH_Z[0] <= OPENING[0] + 6


def test_bone_still_over_the_enamel_holds_it_back():
    z, _ = _eruption(over_enamel_until=128)
    assert abs(z - 128) <= 2, z


def test_bone_under_the_lingual_side_does_not():
    # The lingual strip runs to z 130, well past eruption: none of its rays start on enamel.
    assert _eruption()[0] < STRIP_END - 10


def test_which_end_the_trace_starts_from_does_not_move_it():
    assert _eruption(flip=True)[0] == _eruption()[0]


def test_a_gap_in_the_bone_inside_the_jaw_does_not_free_it_early():
    assert _eruption(gap=True)[0] == _eruption()[0]


def test_without_bone_or_enamel_there_is_no_eruption_and_it_says_why():
    z, found = _eruption(bone=False)
    assert z is None and "never covers" in found["reason"]
    z, found = _eruption(enamel=DENTIN, threshold=20000.0)
    assert z is None and "No enamel" in found["reason"]


def test_the_enamel_threshold_falls_between_dentin_and_enamel():
    volume = _phantom()[0]
    assert DENTIN + 1500 < enamel_threshold(volume[volume > 0]) < ENAMEL - 1500
    # A flatter enamel peak than a molar's, as an incisor has, is still found.
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(DENTIN, 1500, 400_000), rng.normal(ENAMEL, 1500, 3_000)])
    assert DENTIN + 3000 < enamel_threshold(values) < ENAMEL - 1500


def test_the_enamel_bands_edges_are_left_out_round_the_circle():
    surface = np.zeros((1, 12))
    surface[0, [10, 11, 0, 1, 2]] = 1.0            # one band, across 0 degrees
    assert np.flatnonzero(enamel_rays(surface, 0.5, edge=1)[0]).tolist() == [0, 1, 11]
    assert enamel_rays(surface, 0.5, edge=0)[0].sum() == 5


def test_a_landmark_set_by_hand_is_kept():
    earlier = [{"landmark": "eruption", "z_start": 40.0, "z_end": None, "source": "manual"},
               {"landmark": "M1", "z_start": 9.0, "z_end": 14.0, "source": "teeth"},
               {"landmark": "M2", "z_start": 19.0, "z_end": 24.0, "source": "auto"},
               {"landmark": "M3", "z_start": 31.0, "z_end": 34.0}]
    found = [{"landmark": "eruption", "z_start": 44.0, "z_end": None, "source": "auto"},
             {"landmark": "M1", "z_start": 10.0, "z_end": 15.0, "source": "auto"},
             {"landmark": "M2", "z_start": 20.0, "z_end": 25.0, "source": "auto"},
             {"landmark": "M3", "z_start": 30.0, "z_end": 35.0, "source": "auto"}]
    merged = {m["landmark"]: (m["z_start"], m.get("source")) for m in merge_marks(earlier, found)}
    # Hand marks (M3's, from before marks had a source, counts as one) win; found ones are replaced.
    assert merged == {"eruption": (40.0, "manual"), "M1": (10.0, "auto"), "M2": (20.0, "auto"), "M3": (31.0, None)}
