"""
Tests for the patches of thinner and less dense enamel.

Each builds a sheet of junction points -- 900 um square, 6 um apart, jittered
and tilted off the axes -- and paints on it a thickness or core density whose
patches are known: disks thinner than the rest, steady slopes that are no
patch at all, a taper toward an edge, a contact, a ring around a hole. The
claims under test are the ones a reader of the numbers relies on: a planted
patch is found once, at its size and depth; anatomy that changes steadily is
not a patch; nothing is judged where the enamel tapers to the margin or meets
the next tooth.

Run with:  python -m pytest quantify_molar/tests/test_enamel_patches.py
"""

# pylint: disable=wrong-import-position

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import enamel_patches as P  # noqa: E402

STEP, SIZE, VOXEL = 6.0, 900.0, 6.0
THIN = dict(by=0.25, by_abs=1.5 * VOXEL)


def _sheet(size=SIZE, step=STEP, seed=0, offset=(0.0, 0.0)):
    """Points on a square sheet, jittered, tilted out of every axis. Returns (points um, x, y, area each)."""
    g = np.arange(0.0, size, step)
    x, y = np.meshgrid(g + offset[0], g + offset[1], indexing="ij")
    rng = np.random.default_rng(seed)
    x = (x + rng.uniform(-0.2, 0.2, x.shape) * step).ravel()
    y = (y + rng.uniform(-0.2, 0.2, y.shape) * step).ravel()
    return _tilt(np.stack([x, y, np.zeros(x.size)], axis=1)), x, y, np.full(x.size, step * step)


def _tilt(points):
    a, b = np.deg2rad(25.0), np.deg2rad(40.0)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return points @ (ry @ rx).T


def _run(points, areas, values, usable=None, eligible=None, kind=THIN, params=P.Params()):
    usable = np.ones(len(points), bool) if usable is None else usable
    eligible = usable if eligible is None else eligible
    return P.tooth_patches(points, areas, values, usable, eligible, params=params, voxel_um=VOXEL, **kind)


def _disk(x, y, radius, depth, base=60.0, at=(450.0, 450.0)):
    t = np.full(x.size, base)
    t[np.hypot(x - at[0], y - at[1]) < radius] *= 1.0 - depth
    return t


# ---------------------------------------------------------------------------
# Nothing where nothing was put
# ---------------------------------------------------------------------------

def test_uniform_enamel_has_no_patches():
    points, x, y, areas = _sheet()
    t = 60.0 + np.random.default_rng(1).normal(0.0, 3.0, x.size)
    found = _run(points, areas, t)
    numbers = P.numbers(found.sums)
    assert numbers["patches"] == 0 and numbers["share"] == 0.0
    assert numbers["shortfall"] < 0.02
    assert numbers["share_judged"] > 0.95


def test_coverage_is_a_flat_sheets_inside_and_half_at_an_edge():
    points, x, y, areas = _sheet()
    _, coverage = P.around(points, np.full(x.size, 60.0), areas, 60.0, fits=1)
    inside = (x > 300) & (x < 600) & (y > 300) & (y < 600)
    edge = (x < 3) & (y > 300) & (y < 600)
    assert 0.95 < coverage[inside].min() and coverage[inside].max() < 1.05
    assert 0.45 < coverage[edge].mean() < 0.55


@pytest.mark.parametrize("gradient", [0.1, 0.3])
def test_a_steady_slope_is_not_a_patch(gradient):
    """Thickness climbing steadily, as up a cusp's flank: the fit follows it, so nothing falls short."""
    points, x, y, areas = _sheet()
    t = 30.0 + gradient * x
    found = _run(points, areas, t)
    assert found.sums["count"] == 0
    away = (x > 120) & (x < 780) & (y > 120) & (y < 780)
    assert np.nanmax(np.abs(found.deficit[away])) <= 0.06


# ---------------------------------------------------------------------------
# Planted patches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [40.0, 60.0, 90.0])
def test_a_thin_disk_is_one_patch_of_its_size_and_depth(radius):
    points, x, y, areas = _sheet()
    found = _run(points, areas, _disk(x, y, radius, 0.4))
    numbers = P.numbers(found.sums)
    assert numbers["patches"] == 1
    truth = np.pi * radius ** 2 / 1e6
    assert numbers["largest_mm2"] == pytest.approx(truth, rel=0.2)
    assert numbers["depth"] == pytest.approx(0.4, abs=0.05)
    centre = found.records[0]["centre_um"]
    assert np.linalg.norm(centre - _tilt(np.array([[450.0, 450.0, 0.0]]))[0]) < 6.0


@pytest.mark.parametrize("radius", [60.0, 90.0])
def test_a_disk_thirty_percent_thinner_is_found(radius):
    points, x, y, areas = _sheet()
    found = _run(points, areas, _disk(x, y, radius, 0.3))
    assert found.sums["count"] == 1


def test_wide_thinning_is_found_only_in_part():
    """The enamel around follows thinning wider than itself: 150 um disks are found, 300 um ones in part."""
    points, x, y, areas = _sheet()
    mid = P.numbers(_run(points, areas, _disk(x, y, 150.0, 0.4)).sums)
    wide = P.numbers(_run(points, areas, _disk(x, y, 300.0, 0.4)).sums)
    assert mid["largest_mm2"] >= 0.5 * np.pi * 150.0 ** 2 / 1e6
    assert (wide["largest_mm2"] or 0.0) < 0.7 * np.pi * 300.0 ** 2 / 1e6


def test_below_the_bars_nothing_is_called():
    points, x, y, areas = _sheet()
    # Too shallow: 18% thinner.
    assert _run(points, areas, _disk(x, y, 80.0, 0.18)).sums["count"] == 0
    # Too small: 40% thinner over 0.0007 mm2.
    assert _run(points, areas, _disk(x, y, 15.0, 0.4)).sums["count"] == 0
    # A third thinner, but only 8 um: under the voxel and a half a call needs.
    assert _run(points, areas, _disk(x, y, 60.0, 0.3, base=26.7)).sums["count"] == 0


def test_points_twice_as_dense_count_the_same_area():
    """Half the sheet at 4 um, half at 8 um, each point standing for its own area."""
    left, xl, yl, _ = _sheet(size=450.0, step=4.0, seed=2)
    right, xr, yr, _ = _sheet(size=450.0, step=8.0, seed=3, offset=(450.0, 0.0))
    points = np.concatenate([left, right])
    x, y = np.concatenate([xl, xr]), np.concatenate([yl, yr])
    areas = np.concatenate([np.full(xl.size, 16.0), np.full(xr.size, 64.0)])
    t = np.full(x.size, 60.0)
    for cx in (225.0, 675.0):
        t[np.hypot(x - cx, y - 225.0) < 60.0] = 30.0
    found = _run(points, areas, t)
    assert found.sums["count"] == 2
    a, b = (r["area_um2"] for r in found.records)
    assert a == pytest.approx(b, rel=0.1)


def test_a_ring_around_a_hole_counts_the_hole():
    """Enamel too thin to place the junction leaves no points: the ring around it carries the hole."""
    points, x, y, areas = _sheet()
    r = np.hypot(x - 450.0, y - 450.0)
    keep = r >= 30.0
    t = np.where(r < 60.0, 12.0, 60.0)
    found = _run(points[keep], areas[keep], t[keep])
    assert found.sums["count"] == 1
    record = found.records[0]
    assert record["area_um2"] == pytest.approx(np.pi * 60.0 ** 2, rel=0.2)
    assert record["hole_um2"] == pytest.approx(np.pi * 30.0 ** 2, rel=0.35)


# ---------------------------------------------------------------------------
# Where nothing is judged
# ---------------------------------------------------------------------------

def test_the_taper_to_the_margin_is_not_a_patch():
    """Enamel thinning from 60 um to 5 um over the last 150 um to its edge, the margin band left unjudged."""
    points, x, y, areas = _sheet()
    t = np.where(x < 150.0, 5.0 + 55.0 * x / 150.0, 60.0)
    cervical = x                                   # the edge runs along x = 0
    usable = np.ones(x.size, bool)
    eligible = cervical >= P.Params().margin_um
    found = _run(points, areas, t, usable, eligible)
    assert found.sums["count"] == 0
    judged = found.judged.mean()
    assert judged == pytest.approx(eligible.mean(), abs=0.03)


def test_a_contact_is_neither_judged_nor_lowers_its_neighbours():
    points, x, y, areas = _sheet()
    t = np.full(x.size, 60.0)
    square = (np.abs(x - 450.0) < 75.0) & (np.abs(y - 450.0) < 75.0)
    t[square] = 20.0
    found = _run(points, areas, t, usable=~square)
    assert found.sums["count"] == 0
    beside = (np.abs(x - 450.0) < 75.0) & (y > 525.0 + 25.0) & (y < 525.0 + 35.0)
    assert np.nanmax(np.abs(found.deficit[beside])) < 0.03


def test_a_fragment_on_its_own_is_not_judged():
    points, x, y, areas = _sheet()
    g = np.arange(0.0, 30.0, STEP)
    fx, fy = np.meshgrid(g, g, indexing="ij")
    fragment = _tilt(np.stack([fx.ravel(), fy.ravel(), np.full(fx.size, 300.0)], axis=1))
    every = np.concatenate([points, fragment])
    t = np.concatenate([np.full(x.size, 60.0), np.full(fx.size, 20.0)])
    found = _run(every, np.full(len(every), STEP * STEP), t)
    assert not found.judged[x.size:].any()
    assert found.sums["count"] == 0


# ---------------------------------------------------------------------------
# The grid against the sums it stands for
# ---------------------------------------------------------------------------

def test_the_grid_matches_the_gaussian_sums_and_keeps_a_plane():
    points, x, y, areas = _sheet(size=600.0)
    sigma = 60.0
    _, coverage = P.around(points, np.full(x.size, 1.0), areas, sigma, fits=1)
    pick = np.random.default_rng(4).choice(x.size, 60, replace=False)
    d2 = ((points[pick, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    brute = (areas[None, :] * np.exp(-0.5 * d2 / sigma ** 2)).sum(axis=1) / (2.0 * np.pi * sigma ** 2)
    assert np.allclose(coverage[pick], brute, rtol=0.03)

    plane = 40.0 + 0.05 * x + 0.02 * y
    reference, _ = P.around(points, plane, areas, sigma, fits=1)
    inside = (x > 150) & (x < 450) & (y > 150) & (y < 450)
    assert np.allclose(reference[inside], plane[inside], rtol=0.005)


# ---------------------------------------------------------------------------
# The core density, judged the same way
# ---------------------------------------------------------------------------

DENSE = dict(by=0.05, by_abs=0.0, steps=P.DENSITY_STEPS)


def test_less_dense_disks_are_found_at_their_depth():
    points, x, y, areas = _sheet()
    rng = np.random.default_rng(6)
    for depth, expected in ((0.08, 1), (0.03, 0)):
        value = 1000.0 + rng.normal(0.0, 10.0, x.size)
        value[np.hypot(x - 450.0, y - 450.0) < 60.0] *= 1.0 - depth
        found = _run(points, areas, value, kind=DENSE)
        assert found.sums["count"] == expected
        if expected:
            assert P.numbers(found.sums)["depth"] == pytest.approx(depth, abs=0.015)


# ---------------------------------------------------------------------------
# Areas, the row, and the JSON
# ---------------------------------------------------------------------------

def test_point_areas_add_up_to_the_surface_they_stand_for():
    points, x, y, _ = _sheet(size=120.0)
    centres = points + np.random.default_rng(7).normal(0.0, 1.0, points.shape)
    tri = np.full(len(centres), 30.0)
    far = np.array([[0.0, 0.0, 500.0]])
    areas = P.point_areas(np.concatenate([points, far]), np.concatenate([centres, far + 50.0]),
                          np.concatenate([tri, [99.0]]), reach_um=18.0)
    assert areas[:-1].sum() == pytest.approx(tri.sum())
    assert areas[-1] == 0.0
    assert (areas[:-1] > 0).mean() > 0.95


def test_the_row_and_its_teeth_are_written_as_strict_json():
    points, x, y, areas = _sheet()
    voxel = np.array([VOXEL] * 3)
    t = _disk(x, y, 60.0, 0.6)                           # past the default bar (45%)
    tooth = np.where(x < 450.0, 1, 2).astype(np.uint8)
    tooth[(x > 850.0)] = 3                               # a sliver of M3: too little to judge
    core = 3000.0 + 0.0 * x
    block, shades = P.measure_patches(points / voxel, voxel, areas_um2=areas, tooth=tooth,
                                      names={1: "M1", 2: "M2", 3: "M3"}, thickness_um=t,
                                      to_contact=np.zeros(x.size, bool), edge_um=np.full(x.size, np.inf),
                                      core_raw=core, background=500.0, edge_blur_um=7.5, clear_um=18.0)
    text = json.dumps(block, allow_nan=False)
    assert "NaN" not in text
    thin = block["thin"]
    assert [t["name"] for t in thin["teeth"]] == ["M1", "M2", "M3"]
    # The disk straddles M1 and M2: one patch in each, and the row's share weighs both.
    assert thin["patches"] == 2
    assert sum(t["patches"] for t in thin["teeth"]) == 2
    by = {t["name"]: t for t in thin["teeth"]}
    row = thin["share"]
    assert min(by["M1"]["share"], by["M2"]["share"]) <= row <= max(by["M1"]["share"], by["M2"]["share"])
    assert block["low_density"]["patches"] == 0
    assert block["low_density"]["core_density_raw"] == pytest.approx(3000.0)
    assert shades["thin"].shape == (x.size,) and shades["thin_patch"].sum() > 0
