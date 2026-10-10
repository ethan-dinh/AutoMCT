"""
Tests for the molar enamel quantification.

Every test builds a phantom whose answer is known by construction: dentin
blocks with an enamel cap of a stated thickness, laid out along axis 0 the way
the pipeline's reorientation lays out M1-M3. The claims under test are the
ones a reader of the numbers relies on -- the enamel fraction is the enamel
that was put there, teeth come out named by size rather than by array order,
touching teeth are still counted as three, and nothing in the JSON trips up a
strict parser.

Run with:  python -m pytest quantify_molar/test_molar_enamel.py
"""

# pylint: disable=wrong-import-position

import json
import os
import re
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import enamel_surfaces as S  # noqa: E402
import molar_enamel as M  # noqa: E402
from data_io import save_volume  # noqa: E402  (importable once molar_enamel set the path)

SHAPE = (120, 64, 64)
DENTIN, ENAMEL, NOISE = 500.0, 900.0, 15.0
SPACING = (0.01, 0.01, 0.01)

# (z range, y/x half-width, enamel cap thickness), largest tooth first: M1 > M2 > M3.
TEETH = (
    ((10, 50), 15, 6),
    ((56, 84), 12, 5),
    ((90, 108), 10, 4),
)


def _phantom(seed=0, bridge=False):
    """Three dentin blocks, each capped with enamel on its low-y face."""
    volume = np.zeros(SHAPE, dtype=np.float32)
    enamel_truth = np.zeros(SHAPE, dtype=bool)
    centre = SHAPE[1] // 2
    for (z0, z1), half, cap in TEETH:
        box = np.s_[z0:z1, centre - half:centre + half, centre - half:centre + half]
        volume[box] = DENTIN
        cap_box = np.s_[z0:z1, centre - half:centre - half + cap, centre - half:centre + half]
        volume[cap_box] = ENAMEL
        enamel_truth[cap_box] = True
    if bridge:
        # A 2x2 dentin contact point joining M1 to M2 across the gap.
        volume[50:56, centre:centre + 2, centre:centre + 2] = DENTIN
    tissue = volume > 0
    rng = np.random.default_rng(seed)
    volume[tissue] += rng.normal(0.0, NOISE, int(tissue.sum())).astype(np.float32)
    return volume, enamel_truth


def test_threshold_sits_between_dentin_and_enamel():
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, thickness=False)
    assert DENTIN + 4 * NOISE < summary["threshold"]["value"] < ENAMEL - 4 * NOISE
    assert summary["threshold"]["method"] == "log-valley"
    assert summary["threshold"]["dentin_mode"] == pytest.approx(DENTIN, abs=3 * NOISE)


def test_enamel_fraction_matches_the_cap_that_was_built():
    volume, truth = _phantom()
    summary, arrays = M.quantify(volume, SPACING, thickness=False)
    expected = truth.sum() / (volume > 0).sum()
    assert summary["enamel"]["fraction"] == pytest.approx(expected, abs=0.002)
    voxel = np.prod(SPACING)
    assert summary["total"]["volume_mm3"] == pytest.approx((volume > 0).sum() * voxel)
    assert summary["enamel"]["volume_mm3"] + summary["dentin"]["volume_mm3"] == pytest.approx(
        summary["total"]["volume_mm3"])
    # The mask is returned cropped to the molar's box; it must hold the cap.
    full = np.zeros(SHAPE, dtype=bool)
    full[arrays["box"]] = arrays["enamel"]
    assert (full & truth).sum() / truth.sum() > 0.99


def test_teeth_are_named_by_size_not_array_order():
    volume, truth = _phantom()
    for flipped in (False, True):
        vol = volume[::-1] if flipped else volume
        summary, _ = M.quantify(vol, SPACING, thickness=False)
        names = [t["name"] for t in summary["teeth"]]
        assert names == ["M1", "M2", "M3"]
        sizes = [t["voxels"] for t in summary["teeth"]]
        assert sizes == sorted(sizes, reverse=True), "M1 must be the largest tooth"
        assert summary["teeth_split"] == "dentin-0"   # apart already: no erosion needed
    fractions = {t["name"]: t["enamel_fraction"] for t in summary["teeth"]}
    for name, ((z0, z1), half, cap) in zip(("M1", "M2", "M3"), TEETH):
        assert fractions[name] == pytest.approx(cap / (2 * half), abs=0.01)


def test_each_crown_runs_where_its_enamel_does():
    """Each crown's extent along the jaw -- the landmarks' M1-M3 start and end -- is its block's z range."""
    volume, _ = _phantom()
    for flipped in (False, True):
        summary, _ = M.quantify(volume[::-1] if flipped else volume, SPACING, thickness=False)
        for tooth, ((z0, z1), _, _) in zip(summary["teeth"], TEETH):
            built = (SHAPE[0] - z1, SHAPE[0] - 1 - z0) if flipped else (z0, z1 - 1)
            assert (tooth["crown_z_start"], tooth["crown_z_end"]) == built
            # Apart, a molar's landmark span is its crown, started at the end toward M1.
            assert (tooth["landmark_z_start"], tooth["landmark_z_end"]) == (built[::-1] if flipped else built)


def test_the_crowns_alone_are_the_molar_steps_own():
    """``molar_crowns`` -- what the Landmarks step places M1-M3 from -- is the split quantify makes, touching teeth too."""
    for volume in (_phantom()[0], _phantom(bridge=True)[0], _phantom()[0][::-1]):
        summary, _ = M.quantify(volume, SPACING, thickness=False)
        crowns, how = M.molar_crowns(volume)
        assert how == summary["teeth_split"]
        assert [(c.name, c.crown_z_start, c.crown_z_end, c.landmark_z_start, c.landmark_z_end) for c in crowns] == [
            (t["name"], t["crown_z_start"], t["crown_z_end"], t["landmark_z_start"], t["landmark_z_end"])
            for t in summary["teeth"]]
    with pytest.raises(ValueError):
        M.molar_crowns(np.zeros((8, 8, 8), dtype=np.float32))


def test_where_crowns_overlap_the_tooth_in_front_keeps_them():
    """P22_TGM2_KO_F_1's crowns, M2's running on beside M1's: marked by eye at 973-1219, 832-970, 715-829."""
    for flip in (False, True):
        crowns = ((975, 1216), (835, 1000), (722, 828))
        if flip:
            crowns = tuple((2000 - hi, 2000 - lo) for lo, hi in crowns)
        teeth = [M.Tooth(name=f"M{k}", voxels=1, volume_mm3=0.0, enamel_voxels=1, enamel_volume_mm3=0.0,
                         enamel_fraction=0.0, enamel_median=None, dentin_median=None, z_start=0, z_end=0,
                         crown_z_start=lo, crown_z_end=hi) for k, (lo, hi) in enumerate(crowns, start=1)]
        M.landmark_spans(teeth)
        spans = [(t.landmark_z_start, t.landmark_z_end) for t in teeth]
        expected = [(1216, 975), (974, 835), (828, 722)]
        assert spans == ([(2000 - a, 2000 - b) for a, b in expected] if flip else expected)


def test_touching_teeth_are_split_back_into_three():
    volume, _ = _phantom(bridge=True)
    summary, _ = M.quantify(volume, SPACING, thickness=False)
    assert summary["n_teeth"] == 3
    assert summary["teeth_split"] == "dentin-1"   # one voxel of erosion cuts the 2x2 bridge
    m1 = next(t for t in summary["teeth"] if t["name"] == "M1")
    (z0, z1), half, _ = TEETH[0]
    # The bridge's six slices go to one tooth or the other, never lost.
    assert abs(m1["voxels"] - (z1 - z0) * (2 * half) ** 2) <= 6 * 4


def test_fixed_threshold_overrides_the_histogram():
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, fixed_threshold=ENAMEL + 5 * NOISE, thickness=False)
    assert summary["threshold"]["method"] == "fixed"
    assert summary["threshold"]["value"] == ENAMEL + 5 * NOISE
    assert summary["enamel"]["fraction"] < 0.01


def test_hypomineralized_enamel_still_reports():
    """Enamel barely brighter than dentin: no clean second peak, but a number all the same."""
    volume = np.zeros((60, 40, 40), dtype=np.float32)
    volume[10:50, 10:30, 10:30] = DENTIN
    volume[10:50, 10:14, 10:30] = DENTIN + NOISE
    rng = np.random.default_rng(2)
    tissue = volume > 0
    volume[tissue] += rng.normal(0.0, NOISE, int(tissue.sum())).astype(np.float32)
    summary, _ = M.quantify(volume, SPACING, split=False, thickness=False)
    assert 0.0 <= summary["enamel"]["fraction"] <= 1.0


# The scanner calibration of f0004028, with a divisor that puts the phantom's
# 900-unit enamel at a real enamel's density: 900 / 160 * 375.507 - 197.288.
CALIBRATION = {"slope": 375.507, "intercept": -197.288, "divisor": 160.0, "offset": 0.0, "source": "test"}
Z95 = 1.6449  # the normal distribution's 95th percentile, in standard deviations


def test_enamel_density_is_the_enamel_that_was_built():
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, thickness=False)
    density = summary["enamel"]["density"]
    assert density["percentiles"] == [5.0, 95.0]
    assert density["raw"]["mean"] == pytest.approx(ENAMEL, abs=2)
    assert density["raw"]["low"] == pytest.approx(ENAMEL - Z95 * NOISE, abs=2)
    assert density["raw"]["high"] == pytest.approx(ENAMEL + Z95 * NOISE, abs=2)
    assert density["mg_ha_cm3"] is None and summary["calibration"] is None


def test_density_converts_with_the_calibration():
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, thickness=False, calibration=CALIBRATION)
    density = summary["enamel"]["density"]
    for stat in ("mean", "low", "high"):
        assert density["mg_ha_cm3"][stat] == pytest.approx(density["raw"][stat] / 160.0 * 375.507 - 197.288)
    assert density["mg_ha_cm3"]["mean"] == pytest.approx(900 / 160.0 * 375.507 - 197.288, abs=5)
    assert summary["calibration"]["source"] == "test" and not summary["warnings"]


def test_a_conversion_no_enamel_could_read_is_flagged():
    """Dividing by 1 what should be divided by mu_scaling: hundreds of thousands of mg HA/cm³."""
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, thickness=False, calibration={**CALIBRATION, "divisor": 1.0})
    assert any("which no enamel reads" in w for w in summary["warnings"])


def test_the_percentiles_are_the_ones_asked_for():
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, thickness=False, percentiles=(1.0, 99.0))
    density = summary["enamel"]["density"]
    assert density["percentiles"] == [1.0, 99.0]
    assert density["raw"]["low"] == pytest.approx(ENAMEL - 2.3263 * NOISE, abs=3)


def test_empty_volume_is_an_error():
    with pytest.raises(ValueError):
        M.quantify(np.zeros((10, 10, 10), dtype=np.float32), SPACING)


def test_sample_names_follow_the_pipeline_layout():
    assert M.derive_sample_name("/x/P52_Odam-Cre_WT_1/volumes/P52_Odam-Cre_WT_1_molar_volume.nrrd") \
        == "P52_Odam-Cre_WT_1"
    assert M.derive_sample_name("/x/Odam-CRE-A/volumes/molar_volume.tif") == "Odam-CRE-A"


def test_cli_batch_writes_every_output(tmp_path):
    root = tmp_path / "Segments"
    for name, seed in (("P52_WT_1", 0), ("P52_KO_1", 1)):
        volume, _ = _phantom(seed=seed)
        save_volume(volume, os.path.join(name, "volumes"), f"{name}_molar_volume",
                    base_dir=str(root), output_format="nrrd", spacing=SPACING)
    out = tmp_path / "out"
    assert M.main(["-i", str(root), "-o", str(out), "--no-plot"]) == 0

    summary_csv = out / "molar_enamel_summary.csv"
    assert summary_csv.exists()
    assert len(summary_csv.read_text().strip().splitlines()) == 3

    for name in ("P52_WT_1", "P52_KO_1"):
        sample_dir = out / name
        text = (sample_dir / f"{name}_molar_enamel.json").read_text()

        def _reject(constant):
            raise ValueError(f"non-standard JSON constant {constant}")

        record = json.loads(text, parse_constant=_reject)
        assert record["sample"] == name
        assert record["spacing_mm_zyx"] == list(SPACING)
        for suffix in ("_molar_enamel_teeth.csv", "_molar_enamel_profile.csv", "_enamel_mask.nrrd"):
            assert (sample_dir / f"{name}{suffix}").exists()


def test_cli_converts_density_with_a_calibration_file(tmp_path):
    volume, _ = _phantom()
    root = tmp_path / "Segments"
    save_volume(volume, os.path.join("P52_WT_1", "volumes"), "P52_WT_1_molar_volume",
                base_dir=str(root), output_format="nrrd", spacing=SPACING)
    calibration = tmp_path / "phantom_calibration.json"
    calibration.write_text(json.dumps({"slope": 375.507, "intercept": -197.288, "mu_scaling": 160}))
    out = tmp_path / "out"
    assert M.main(["-i", str(root), "-o", str(out), "--no-plot", "--no-thickness",
                   "--calibration", str(calibration), "--percentiles", "10", "90"]) == 0

    record = json.loads((out / "P52_WT_1_molar_enamel.json").read_text())
    density = record["enamel"]["density"]
    assert density["percentiles"] == [10.0, 90.0]
    assert density["mg_ha_cm3"]["mean"] == pytest.approx(density["raw"]["mean"] / 160 * 375.507 - 197.288)
    assert record["calibration"]["divisor"] == 160 and record["version"] == M.VERSION

    row = M.summary_row("P52_WT_1", record)
    assert row["enamel_mean_mgHA"] == density["mg_ha_cm3"]["mean"]
    assert row["enamel_low_raw"] == density["raw"]["low"]
    assert row["density_percentiles"] == "10/90" and row["calibration"] == "phantom_calibration.json"


def test_cli_says_when_the_calibration_is_missing(tmp_path):
    volume, _ = _phantom()
    path = tmp_path / "P52_WT_1_molar_volume.nrrd"
    save_volume(volume, "", "P52_WT_1_molar_volume", base_dir=str(tmp_path), output_format="nrrd", spacing=SPACING)
    out = tmp_path / "out"
    assert M.main(["-i", str(path), "-o", str(out), "--no-plot", "--no-thickness",
                   "--calibration", str(tmp_path / "nowhere_meta.json")]) == 0
    record = json.loads((out / "P52_WT_1_molar_enamel.json").read_text())
    assert record["enamel"]["density"]["mg_ha_cm3"] is None and record["enamel"]["density"]["raw"]
    assert any("raw intensity only" in w for w in record["warnings"])


def test_a_hand_calibration_needs_both_numbers():
    with pytest.raises(SystemExit):
        M.parse_arguments(["-i", "x", "--ha-slope", "375"])
    with pytest.raises(SystemExit):
        M.parse_arguments(["-i", "x", "--percentiles", "95", "5"])


# ---------------------------------------------------------------------------
# Enamel thickness and the junction's area
# ---------------------------------------------------------------------------

AIR = 120.0   # the scan's background: not zero, so it has to be read, not assumed


def _shell(thick=6, radius=12, sigma=1.2, seed=3):
    """A dentin ball in a closed enamel shell ``thick`` voxels deep, blurred as a scan blurs.

    Closed, so every point of the junction has the full thickness above it: no
    edge for the nearest outer point to escape to. Returns (scan, molar
    volume): the molar volume is zero outside its mask, as the segmentation
    writes it, and keeps only the upper part of the blurred outer edge.
    """
    n = 48
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    r = np.sqrt((z - 23.7) ** 2 + (y - 24.2) ** 2 + (x - 23.9) ** 2)   # off the voxel grid
    dentin = np.clip(radius - r + 0.5, 0, 1)
    outer = np.clip(radius + thick - r + 0.5, 0, 1)
    scan = ndi.gaussian_filter(DENTIN * dentin + ENAMEL * (outer - dentin), sigma) + AIR
    scan += np.random.default_rng(seed).normal(0.0, NOISE, scan.shape)
    scan = scan.astype(np.float32)
    return scan, np.where(scan > AIR + 0.3 * DENTIN, scan, 0).astype(np.float32)


# The shells are mostly enamel, which the log-histogram valley is not built for:
# a fixed threshold halfway between their dentin and enamel, background included.
MIDWAY = AIR + (DENTIN + ENAMEL) / 2


def test_thickness_is_the_shell_thickness():
    scan, volume = _shell()
    summary, _ = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, split=False)
    thickness = summary["enamel"]["thickness_um"]
    assert thickness["median"] == pytest.approx(60.0, abs=3.0)
    assert thickness["share_measured"] > 0.95
    assert thickness["levels"]["background_from"] == "scan"
    assert thickness["levels"]["background"] == pytest.approx(AIR, abs=3 * NOISE)


def test_junction_area_is_the_inner_sphere():
    """The junction is a sphere of radius 12 voxels: 4 pi r^2.

    Read on the scan smoothed by a voxel, a tight curve like this one comes out a few percent small.
    """
    scan, volume = _shell()
    summary, _ = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, split=False)
    truth = 4 * np.pi * (12 * SPACING[0]) ** 2
    assert 0.93 * truth < summary["enamel"]["thickness_um"]["junction_area_mm2"] <= truth


def test_thickness_without_the_scan_says_so():
    _, volume = _shell()
    summary, _ = M.quantify(volume, SPACING, fixed_threshold=MIDWAY, split=False)
    thickness = summary["enamel"]["thickness_um"]
    assert thickness["levels"]["background_from"] == "assumed zero (no scan)"
    assert thickness["median"] is not None
    assert any("--scan" in w for w in summary["warnings"])


def test_thickness_per_tooth():
    """Caps 6, 5 and 4 voxels deep: the teeth come out in that order.

    The caps are slabs whose sides are open to the air, so near their edges the
    nearest way out is sideways and the medians sit below the cap depths.
    """
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, scan=volume)
    by_name = {t["name"]: t["thickness_median_um"] for t in summary["teeth"]}
    assert by_name["M1"] > by_name["M2"] > by_name["M3"]
    for (_, _, cap), name in zip(TEETH, ("M1", "M2", "M3")):
        assert by_name[name] <= cap * 10.0 + 3.0
    # Each cap's underside is a flat junction, 2 half-widths across by the tooth's length. Where it
    # meets the air at the cap's open sides it is no longer enamel over dentin, so it stops short.
    areas = {t["name"]: t["junction_area_mm2"] for t in summary["teeth"]}
    assert areas["M1"] > areas["M2"] > areas["M3"]
    for ((z0, z1), half, _), name in zip(TEETH, ("M1", "M2", "M3")):
        flat = (z1 - z0) * 2 * half * SPACING[0] ** 2
        assert 0.5 * flat < areas[name] <= flat


def test_the_thickness_map_is_written_and_readable(tmp_path):
    scan, volume = _shell()
    sample = tmp_path / "P22_WT_1"
    save_volume(volume, "volumes", "P22_WT_1_molar_volume", base_dir=str(sample), output_format="nrrd", spacing=SPACING)
    save_volume(scan, "", "P22_WT_1_reoriented_volume", base_dir=str(sample), output_format="nrrd", spacing=SPACING)
    out = tmp_path / "out"
    assert M.main(["-i", str(sample), "-o", str(out), "--threshold", str(MIDWAY), "--thickness-max", "90"]) == 0

    def _reject(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    record = json.loads((out / "P22_WT_1_molar_enamel.json").read_text(), parse_constant=_reject)
    assert record["scan"].endswith("P22_WT_1_reoriented_volume.nrrd")   # found beside the volumes
    assert record["enamel"]["thickness_um"]["levels"]["background_from"] == "scan"
    assert (out / "P22_WT_1_thickness_map.png").exists()

    page = (out / "P22_WT_1_thickness_map.html").read_text(encoding="utf-8")
    assert "__DATA__" not in page and "__META__" not in page and "__SAMPLE__" not in page
    meta = json.loads(re.search(r'<script id="meta" type="application/json">(.*?)</script>', page, re.S).group(1))
    assert meta["vmax"] == 90.0 and meta["reported"]["median"] == pytest.approx(60.0, abs=3.0)
    assert meta["layout"]["layers"]["enamel"]["n_faces"] > 0 and meta["layout"]["layers"]["dentin"]["n_faces"] > 0
    assert meta["reported"]["junction_area_mm2"] > 0

    row = M.summary_row("P22_WT_1", record)
    assert row["thickness_median_um"] == pytest.approx(60.0, abs=3.0) and row["junction_area_mm2"] > 0
    assert row["thickness_background_from"] == "scan"


def test_the_map_data_decodes_as_the_viewer_reads_it():
    """Offsets aligned for the typed arrays, positions to 16 bits, thickness in tenths, 65535 for none."""
    pos = np.array([[0, 0, 0], [100.0, 0, 0], [0, 50.0, 25.0]], dtype=np.float32)
    layers = {"outer": {"pos": pos, "faces": np.array([[0, 1, 2]], dtype=np.uint32),
                        "value": np.array([61.24, np.nan, 3.0], dtype=np.float32), "tooth": np.array([1, 1, 2], np.uint8)},
              "context": {"pos": pos + 1.0, "faces": np.array([[2, 1, 0]], dtype=np.uint32)}}
    blob, layout = S._pack(layers)
    lo, scale = np.array(layout["bbox_min"]), np.array(layout["scale"])
    for name, layer in layers.items():
        spec = layout["layers"][name]
        for key in ("pos", "faces", "value", "tooth"):
            if key in spec:
                assert spec[key] % 4 == 0
        q = np.frombuffer(blob, "<u2", spec["n_vertices"] * 3, spec["pos"]).reshape(-1, 3)
        assert np.allclose(lo + q * scale, layer["pos"], atol=0.01)
        faces = np.frombuffer(blob, "<u4", spec["n_faces"] * 3, spec["faces"]).reshape(-1, 3)
        assert (faces == layer["faces"]).all()
    value = np.frombuffer(blob, "<u2", 3, layout["layers"]["outer"]["value"])
    assert list(value) == [612, 65535, 30]
    assert list(np.frombuffer(blob, "u1", 3, layout["layers"]["outer"]["tooth"])) == [1, 1, 2]


# ---------------------------------------------------------------------------
# Teeth that touch, enamel to enamel
# ---------------------------------------------------------------------------

# A molar row whose neighbours are pressed together: (z range, x half-width,
# enamel depth) per tooth along axis 0, M1 > M2 > M3, each z range starting
# where the last one stops.
ROW = (((6, 58), 14, 6), ((58, 94), 12, 5), ((94, 118), 10, 4))
ROW_SHAPE = (124, 52, 44)
ROW_TOP, ROW_CERVIX, ROW_BOTTOM, ROW_X = 6, 28, 46, 22   # crown face (low y), enamel's lower edge, root end, x centre


def _molar_row(teeth=ROW, sigma=1.2, seed=5):
    """Teeth along axis 0 whose enamel meets enamel, as M1 and M2 do by P22.

    Each tooth is a dentin block under enamel on its crown face and on both
    faces along the jaw, down to the cervix; below it the root dentin
    narrows, leaving a gap between neighbours that a 3x3 dentin-bright rod
    bridges, like the dim link the real scans have under their contacts. The
    blurred scan keeps its background; the molar volume keeps what the
    segmentation would, blurred rim and all. Returns (scan, molar volume, the
    tooth each built voxel belongs to).
    """
    dentin = np.zeros(ROW_SHAPE, dtype=np.float32)
    enamel = np.zeros(ROW_SHAPE, dtype=np.float32)
    truth = np.zeros(ROW_SHAPE, dtype=np.uint8)
    for number, ((z0, z1), half, depth) in enumerate(teeth, start=1):
        x = slice(ROW_X - half, ROW_X + half)
        enamel[z0:z1, ROW_TOP:ROW_CERVIX, x] = 1
        enamel[z0 + depth:z1 - depth, ROW_TOP + depth:ROW_CERVIX, x] = 0
        dentin[z0 + depth:z1 - depth, ROW_TOP + depth:ROW_BOTTOM, x] = 1
        truth[z0:z1, ROW_TOP:ROW_CERVIX, x] = number
        truth[z0 + depth:z1 - depth, ROW_TOP + depth:ROW_BOTTOM, x] = number
    for number, (((_, end), _, before), ((start, _), _, after)) in enumerate(zip(teeth[:-1], teeth[1:]), start=1):
        if end == start:
            rod = (slice(38, 41), slice(ROW_X - 1, ROW_X + 2))
            dentin[(slice(end - before, start + after), *rod)] = 1
            truth[(slice(end - before, end), *rod)] = number
            truth[(slice(start, start + after), *rod)] = number + 1
    scan = ndi.gaussian_filter(DENTIN * dentin + ENAMEL * enamel, sigma) + AIR
    scan += np.random.default_rng(seed).normal(0.0, NOISE, scan.shape)
    scan = scan.astype(np.float32)
    return scan, np.where(scan > AIR + 0.3 * DENTIN, scan, 0).astype(np.float32), truth


def _contact_pair(enamel=(7, 3), seam=0.0, size=40, sigma=1.2, seed=7):
    """Two teeth pressed together along axis 0, enamel only on the faces that meet.

    Dentin to z 40, then ``enamel[0]`` voxels of the first tooth's enamel and
    ``enamel[1]`` of the second's, then 30 slices of dentin: the first
    junction lies at z 39.5, the contact ``enamel[0]`` past it, the second
    junction ``enamel[1]`` further on. ``seam`` dims the second tooth's first
    voxel plane of enamel by that much before the blur. Cross-section ``size``
    voxels square. Returns (scan, molar volume).
    """
    first, second = enamel
    shape = (80 + first + second, size + 12, size + 12)
    face = (slice(6, 6 + size), slice(6, 6 + size))
    contact = 40 + first
    value = np.zeros(shape, dtype=np.float32)
    value[(slice(10, 40), *face)] = DENTIN
    value[(slice(40, contact + second), *face)] = ENAMEL
    value[(slice(contact, contact + 1), *face)] = ENAMEL - seam
    value[(slice(contact + second, contact + second + 30), *face)] = DENTIN
    scan = ndi.gaussian_filter(value, sigma) + AIR
    scan += np.random.default_rng(seed).normal(0.0, NOISE, shape)
    scan = scan.astype(np.float32)
    return scan, np.where(scan > AIR + 0.3 * DENTIN, scan, 0).astype(np.float32)


def _contact_with_a_gap(size=40, sigma=1.2, seed=11):
    """Two teeth whose enamel is parted by a dim gap 5 voxels across, reaching the first tooth's dentin at one edge.

    As on P22_TGM2_KO_F_1: the gap reads below the enamel threshold, like
    dentin, and is thick enough to outlast the erosion; taken for the first
    tooth's dentin, it would flood the second tooth's enamel from inside the
    contact. Enamel 7 voxels deep on each tooth, the gap between z 47 and 52;
    along a 4-voxel strip at one edge there is no enamel, and the first
    tooth's dentin runs up to the gap. Returns (scan, molar volume).
    """
    shape = (100, size + 12, size + 12)
    across = slice(6, 6 + size)
    face = (slice(10, 6 + size), across)          # y 6-10 is the strip without enamel
    value = np.zeros(shape, dtype=np.float32)
    value[(slice(10, 40), *face)] = DENTIN
    value[(slice(40, 47), *face)] = ENAMEL
    value[(slice(47, 52), across, across)] = 0.5 * DENTIN
    value[(slice(52, 59), *face)] = ENAMEL
    value[(slice(59, 89), *face)] = DENTIN
    value[(slice(10, 47), slice(6, 10), across)] = DENTIN
    scan = ndi.gaussian_filter(value, sigma) + AIR
    scan += np.random.default_rng(seed).normal(0.0, NOISE, shape)
    scan = scan.astype(np.float32)
    return scan, np.where(scan > AIR + 0.3 * DENTIN, scan, 0).astype(np.float32)


def _in_frame(cropped, box, shape):
    out = np.zeros(shape, dtype=cropped.dtype)
    out[box] = cropped
    return out


def _built_tooth(truth, mask):
    """Each mask voxel given the nearest tooth that was built: the blurred rim belongs to the tooth it rims."""
    _, index = ndi.distance_transform_edt(truth == 0, return_indices=True)
    return np.where(mask, truth[tuple(index)], 0)


def test_teeth_fused_enamel_to_enamel_are_split_into_three():
    """One piece to start with -- enamel fused at both contacts, a dim rod beneath each -- and three teeth out."""
    scan, volume, truth = _molar_row()
    six = ndi.generate_binary_structure(3, 1)
    assert ndi.label(volume > 0, structure=six)[1] == 1
    summary, arrays = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, thickness=False)
    assert summary["n_teeth"] == 3 and summary["teeth_split"] == "dentin-2"
    assert [t["name"] for t in summary["teeth"]] == ["M1", "M2", "M3"]
    sizes = [t["voxels"] for t in summary["teeth"]]
    assert sizes == sorted(sizes, reverse=True)
    labels = _in_frame(arrays["teeth"], arrays["box"], volume.shape)
    built = _built_tooth(truth, volume > 0)
    for number in (1, 2, 3):
        got, want = labels == number, built == number
        assert ndi.label(got, structure=six)[1] == 1, f"M{number} is in pieces"
        assert 2 * (got & want).sum() / (got.sum() + want.sum()) >= 0.97, f"M{number} is not the tooth built"


def test_the_crowns_of_touching_teeth_part_at_their_contacts():
    """Fused enamel to enamel, each crown still starts and ends within a voxel of where it was built."""
    scan, volume, _ = _molar_row()
    summary, _ = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, thickness=False)
    for tooth, ((z0, z1), _, _) in zip(summary["teeth"], ROW):
        assert abs(tooth["crown_z_start"] - z0) <= 1 and abs(tooth["crown_z_end"] - (z1 - 1)) <= 1, tooth["name"]


def _cuts(enamel, seam):
    """Where along axis 0 the two teeth part, at each point of the slab away from its edges."""
    scan, volume = _contact_pair(enamel=enamel, seam=seam)
    summary, arrays = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, thickness=False)
    assert summary["n_teeth"] == 2
    labels = _in_frame(arrays["teeth"], arrays["box"], volume.shape)[:, 12:-12, 12:-12]
    lo, hi = labels[:-1], labels[1:]
    return np.nonzero((lo != hi) & (lo > 0) & (hi > 0))[0] + 0.5


def test_without_a_seam_the_cut_is_midway_between_the_junctions():
    """Enamel 10 and 6 voxels deep, fused: cut flat, midway (47.5), 2 voxels short of the contact (49.5)."""
    cuts = _cuts((10, 6), 0.0)
    assert (cuts == 47.5).all()
    assert (_cuts((8, 8), 0.0) == 47.5).all(), "equal enamel: midway is the contact"


def test_a_seam_pulls_the_cut_to_the_contact():
    """The same contact with a seam: the cut leaves the midline for the contact, within a voxel of it."""
    cuts = _cuts((10, 6), 450.0)
    assert 48.5 <= np.median(cuts) <= 49.5
    assert cuts.min() >= 48.5 and cuts.max() <= 49.5


def test_a_gap_between_the_enamel_layers_is_not_taken_for_dentin():
    """The cut falls in the gap (z 47-52, within a voxel of its blurred faces), not through
    the second tooth's enamel (z 52-59), where the gap taken for dentin would put it (55.5)."""
    scan, volume = _contact_with_a_gap()
    summary, arrays = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, thickness=False)
    assert summary["n_teeth"] == 2
    labels = _in_frame(arrays["teeth"], arrays["box"], volume.shape)[:, 16:-12, 12:-12]
    lo, hi = labels[:-1], labels[1:]
    cuts = np.nonzero((lo != hi) & (lo > 0) & (hi > 0))[0] + 0.5
    assert cuts.size and 47 <= np.median(cuts) <= 52
    assert 46.5 <= cuts.min() and cuts.max() <= 52.5


def test_contacts_report_the_area_shared_or_the_gap():
    """M1 and M2 share their crowns' facing enamel (24 x 22 voxels) and a 3x3 rod; M3 stands 4 voxels off M2.

    Equal enamel either side of the contact, so the cut (midway between the
    junctions, with no seam) is the contact itself, not a plane half a voxel
    into the wider M1.
    """
    teeth = (((6, 58), 14, 5), ((58, 90), 12, 5), ROW[2])
    scan, volume, _ = _molar_row(teeth=teeth, sigma=0.0)
    summary, _ = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, thickness=False)
    first, second = summary["contacts"]
    assert first["teeth"] == ["M1", "M2"] and first["touching"] and first["gap_um"] == 0.0
    shared = (24 * 22 + 9) * SPACING[1] * SPACING[2]
    assert first["contact_area_mm2"] == pytest.approx(shared, rel=0.15)
    assert second["teeth"] == ["M2", "M3"] and not second["touching"] and second["contact_area_mm2"] == 0.0
    assert second["gap_um"] == pytest.approx(40.0, abs=10.0)
    row = M.summary_row("P60_WT_1", summary)
    assert row["M1_M2_contact_mm2"] == first["contact_area_mm2"] and row["M2_M3_gap_um"] == second["gap_um"]


def test_thickness_under_a_contact_is_measured_to_the_contact():
    """Enamel 7 voxels deep on each tooth, fused: each reads its own 70 um.

    Measured to the nearest air instead, the enamel under the contact would
    read out to the edge of the fused slab, up to 48 voxels away.
    """
    scan, volume = _contact_pair(enamel=(7, 7), size=96)
    summary, _ = M.quantify(volume, SPACING, scan=scan, fixed_threshold=MIDWAY, thickness_map=False)
    thickness = summary["enamel"]["thickness_um"]
    assert thickness["contact_points"] > 0
    assert [t["name"] for t in summary["teeth"]] == ["M1", "M2"]
    for tooth in summary["teeth"]:
        assert tooth["thickness_median_um"] == pytest.approx(70.0, abs=5.0)


def test_the_tooth_labels_are_written(tmp_path):
    scan, volume, _ = _molar_row()
    sample = tmp_path / "P60_WT_1"
    save_volume(volume, "volumes", "P60_WT_1_molar_volume", base_dir=str(sample), output_format="nrrd", spacing=SPACING)
    out = tmp_path / "out"
    assert M.main(["-i", str(sample), "-o", str(out), "--threshold", str(MIDWAY), "--no-thickness",
                   "--no-plot"]) == 0
    labels, spacing = M.load_volume(str(out / "P60_WT_1_tooth_labels.nrrd"))
    assert labels.shape == volume.shape and spacing == pytest.approx(SPACING)
    assert ((labels > 0) == (volume > 0)).all(), "every molar voxel belongs to a tooth"
    sizes = np.bincount(labels.ravel().astype(np.int64), minlength=4)[1:]
    assert sizes[0] > sizes[1] > sizes[2] > 0, "1 is M1, the largest"
    record = json.loads((out / "P60_WT_1_molar_enamel.json").read_text())
    assert [c["teeth"] for c in record["contacts"]] == [["M1", "M2"], ["M2", "M3"]]
    assert all(c["touching"] for c in record["contacts"])


# ---------------------------------------------------------------------------
# The enamel between its halfway faces
# ---------------------------------------------------------------------------

# A molar row as P82_Odam-Cre_WT_2 was scanned: SkyScan, 8 um voxels, whole
# 8-bit units, edges blurred over 1.75 voxels, and dimming toward M1 -- 85% and
# 93% of M3's contrast over the background, the trend that scan has from M3 to
# M1. M1's enamel is the thinnest and one of its cusps is worn to the dentin.
AIR8, DENTIN8, ENAMEL8, PULP8 = 14.0, 128.0, 222.0, 40.0
SPACING8 = (0.008, 0.008, 0.008)


def _dimming_row(caps=(5, 8, 10), dim=(0.85, 0.93, 1.0), sigma=1.75, tail=0, seed=1):
    """Three teeth along axis 0, enamel to enamel, each with a pulp, a cusp notch and enamel down to its cervix.

    ``tail`` adds that many slices of root dentin with a canal and no enamel
    past the last tooth. Returns (scan, molar volume, the enamel built, the
    tooth each built voxel belongs to: 1-3, 4 for the tail).
    """
    shape = (150 + tail, 80, 60)
    value = np.zeros(shape, dtype=np.float32)
    truth = np.zeros(shape, dtype=bool)
    tooth = np.zeros(shape, dtype=np.uint8)
    top, cervix, bottom, xa, xb = 10, 40, 74, 8, 52
    z0 = 6
    for number, (cap, f) in enumerate(zip(caps, dim), start=1):
        z1 = z0 + 42
        enamel = np.zeros(shape, dtype=bool)
        dentin = np.zeros(shape, dtype=bool)
        enamel[z0:z1, top:cervix, xa:xb] = True
        dentin[z0 + cap:z1 - cap, top + cap:bottom, xa + cap:xb - cap] = True
        enamel &= ~dentin
        notch = (slice(z0 + 14, z1 - 14), slice(top, top + 6), slice(xa + 12, xb - 12))
        enamel[notch] = dentin[notch] = False
        floor = (slice(z0 + 14, z1 - 14), slice(top + 6, top + 6 + cap), slice(xa + 12, xb - 12))
        enamel[floor], dentin[floor] = True, False
        pulp = np.zeros(shape, dtype=bool)
        pulp[z0 + cap + 6:z1 - cap - 6, top + cap + 10:bottom - 6, xa + cap + 6:xb - cap - 6] = True
        dentin &= ~pulp
        if number == 1:   # a cusp worn through its enamel
            worn = (slice(z0, z0 + 12), slice(top, top + cap + 2), slice(xa, xa + 14))
            enamel[worn] = dentin[worn] = False
            dentin[z0 + 2:z0 + 12, top + cap // 2 + 1:top + cap + 2, xa + 2:xa + 14] = True
        value[dentin], value[enamel], value[pulp] = (DENTIN8 - AIR8) * f, (ENAMEL8 - AIR8) * f, (PULP8 - AIR8) * f
        truth |= enamel
        tooth[enamel | dentin | pulp] = number
        z0 = z1
    if tail:
        value[z0:z0 + tail, 20:70, 14:46] = DENTIN8 - AIR8
        value[z0:z0 + tail, 36:56, 24:36] = PULP8 - AIR8
        tooth[z0:z0 + tail, 20:70, 14:46] = 4
    scan = ndi.gaussian_filter(value, sigma) + AIR8
    scan += np.random.default_rng(seed).normal(0.0, 2.0, shape)
    scan = np.round(np.clip(scan, 0, 255)).astype(np.float32)
    molar = ndi.binary_fill_holes(scan > 60)
    return scan, np.where(molar, scan, 0).astype(np.float32), truth, tooth


def _enamel_by_tooth(found, truth, tooth):
    """The enamel found at each tooth built, as a share of its enamel built (1.0: all of it and no more).

    Counted out to 3 voxels past the tooth, where its blurred rim lies.
    """
    shares = []
    for number in (1, 2, 3):
        others = np.isin(tooth, [k for k in (1, 2, 3) if k != number])
        near = ndi.binary_dilation(tooth == number, iterations=3) & ~others
        shares.append((found & near).sum() / (truth & (tooth == number)).sum())
    return shares


def test_dim_thin_enamel_is_found_whole():
    """The bright enamel sets one threshold above the dim tooth's enamel; its halfway faces find all three.

    M1's cap, 5 voxels deep, reads about a tenth thick: blurred over 1.75
    voxels a layer that thin peaks below the enamel's full brightness, and its
    halfway points lie outside it (+13% for a slab).
    """
    scan, volume, truth, tooth = _dimming_row()
    summary, arrays = M.quantify(volume, SPACING8, scan=scan, split=False, thickness=False)
    found = _in_frame(arrays["enamel"], arrays["box"], volume.shape)
    assert summary["enamel_mask"]["method"] == "local-halfway"
    assert found.sum() / truth.sum() == pytest.approx(1.0, abs=0.04)
    assert _enamel_by_tooth(found, truth, tooth) == pytest.approx([1.11, 1.0, 1.0], abs=0.05)

    summary, arrays = M.quantify(volume, SPACING8, scan=scan, split=False, thickness=False, enamel_mask="threshold")
    found = _in_frame(arrays["enamel"], arrays["box"], volume.shape)
    assert summary["enamel_mask"]["method"] == "threshold"
    assert _enamel_by_tooth(found, truth, tooth)[0] < 0.1, "one threshold loses the dim tooth's enamel"


def test_a_stretch_without_enamel_keeps_its_dentin():
    """Root dentin past the last crown, with a canal and no enamel, longer than the stretch the dentin is pooled over.

    Read there, the dentin under the enamel came out at the canal's level on
    P82_Odam-Cre_WT_2, and the stretch's dentin all passed for enamel.
    """
    scan, volume, _, tooth = _dimming_row(tail=60)
    _, arrays = M.quantify(volume, SPACING8, scan=scan, split=False, thickness=False)
    found = _in_frame(arrays["enamel"], arrays["box"], volume.shape)
    tail = tooth == 4
    start = int(np.flatnonzero(tail.any(axis=(1, 2)))[0])
    tail[:start + 3] = False     # its first slices meet the last crown's enamel
    assert not (found & tail).any()


def test_the_blurred_outer_rim_is_enamel():
    """On the shell, the halfway faces hold the enamel built; one threshold between dentin and enamel loses its rim."""
    scan, volume = _shell()
    n = scan.shape[0]
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    r = np.sqrt((z - 23.7) ** 2 + (y - 24.2) ** 2 + (x - 23.9) ** 2)
    built = (np.clip(18 - r + 0.5, 0, 1) - np.clip(12 - r + 0.5, 0, 1)).sum()
    molar = volume != 0
    enamel, _, record = S.enamel_layer(scan, molar, background=AIR, dentin_level=AIR + DENTIN,
                                       enamel_level=AIR + ENAMEL)
    assert enamel.sum() / built == pytest.approx(1.0, abs=0.04)
    assert record["outer_rim_voxels"] > 0
    assert (molar & (scan >= MIDWAY)).sum() / built < 0.85


def _two_shells(dim=0.85, sigma=1.2, seed=3):
    """Two closed shells along axis 0, enamel 6 voxels deep over dentin, the first dimmed to 85%.

    A scan that dims along the jaw. Returns (scan, which shell each voxel
    belongs to, the enamel built).
    """
    n = 48
    shape = (2 * n, n, n)
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float32)
    value = np.zeros(shape, dtype=np.float32)
    which = np.zeros(shape, dtype=np.int32)
    built = np.zeros(shape, dtype=bool)
    for number, (centre, f) in enumerate(((23.7, dim), (71.7, 1.0)), start=1):
        r = np.sqrt((z - centre) ** 2 + (y - 24.2) ** 2 + (x - 23.9) ** 2)
        dentin, outer = np.clip(12 - r + 0.5, 0, 1), np.clip(18 - r + 0.5, 0, 1)
        value += f * (DENTIN * dentin + ENAMEL * (outer - dentin))
        which[r <= 21] = number
        built |= (r > 12) & (r <= 18)
    scan = ndi.gaussian_filter(value, sigma) + AIR
    scan += np.random.default_rng(seed).normal(0.0, NOISE, shape)
    return scan.astype(np.float32), which, built


def test_each_stretch_of_the_jaw_reads_its_own_levels():
    """The dimmed shell's dentin and enamel are read where they are: its enamel is found whole and as deep as the other's.

    One level for both would sit halfway up the dim shell's enamel.
    """
    scan, which, built = _two_shells()
    molar = scan > AIR + 150
    enamel, levels, _ = S.enamel_layer(scan, molar, background=AIR, dentin_level=AIR + DENTIN,
                                       enamel_level=AIR + ENAMEL)
    assert levels.dentin_z[24] == pytest.approx(AIR + 0.85 * DENTIN, abs=10)
    assert levels.dentin_z[72] == pytest.approx(AIR + DENTIN, abs=10)
    for number in (1, 2):
        assert (enamel & (which == number)).sum() / (built & (which == number)).sum() == pytest.approx(1.0, abs=0.05)
    summary, _, _ = S.measure(scan, molar, enamel, SPACING, dentin_level=AIR + DENTIN, enamel_level=AIR + ENAMEL,
                              labels=which, names={1: "M1", 2: "M2"}, display=False, dentin_z=levels.dentin_z)
    for tooth in summary["teeth"]:
        assert tooth["median"] == pytest.approx(60.0, abs=4.0), tooth["name"]
    assert summary["levels"]["placed"]["dentin_under_enamel"]["p5"] < AIR + 0.9 * DENTIN


def test_the_threshold_method_says_which_way_it_went():
    """A clean enamel peak gives the log-histogram valley; a flat tail with none, the knee."""
    rng = np.random.default_rng(0)
    dentin = rng.normal(DENTIN, NOISE, 200_000)
    for tail, way in ((rng.normal(ENAMEL, NOISE, 8_000), "log-valley"),
                      (rng.uniform(DENTIN + 60, ENAMEL, 8_000), "knee")):
        info = {}
        M.find_threshold(np.concatenate([dentin, tail]).astype(np.float32), info=info)
        assert info["method"] == way


def test_the_enamel_mask_says_how_it_was_found():
    volume, _ = _phantom()
    summary, _ = M.quantify(volume, SPACING, thickness=False)
    found = summary["enamel_mask"]
    assert found["method"] == "local-halfway" and found["background_from"].startswith("molar volume")
    assert DENTIN < found["junction_level"]["median"] < ENAMEL
    assert found["dentin_under_enamel"]["median"] == pytest.approx(DENTIN, abs=3 * NOISE)
    row = M.summary_row("P52_WT_1", summary)
    assert row["enamel_mask"] == "local-halfway" and row["junction_level"] == found["junction_level"]["median"]

    fixed, _ = M.quantify(volume, SPACING, fixed_threshold=(DENTIN + ENAMEL) / 2, thickness=False)
    assert fixed["enamel_mask"] == {"method": "threshold", "threshold": (DENTIN + ENAMEL) / 2}
    assert fixed["threshold"]["method"] == "fixed"
    # A summary from before the local mask reads as one threshold.
    del fixed["enamel_mask"]
    assert M.summary_row("P52_WT_1", fixed)["enamel_mask"] == "threshold"


def test_a_molar_that_is_mostly_enamel_falls_back_to_the_threshold():
    """The shell's histogram puts both its peaks in the enamel: no dentin to read the faces from, so one threshold, said so."""
    scan, volume = _shell()
    summary, _ = M.quantify(volume, SPACING, scan=scan, split=False, thickness=False)
    assert summary["enamel_mask"] == {"method": "threshold", "threshold": summary["threshold"]["value"]}
    assert any("too little above its dentin peak" in w for w in summary["warnings"])


def test_no_junction_runs_through_the_dentin():
    """Where no enamel stands, neither surface has a level.

    Read off the dentin's own texture, a junction level would run all through
    the dentin near the enamel, at points no line through enamel leaves from
    (35% of P22_TGM2_KO_F_1's junction points; on this shell, 14%).
    """
    n = 48
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    r = np.sqrt((z - 23.7) ** 2 + (y - 24.2) ** 2 + (x - 23.9) ** 2)
    dentin, outer = np.clip(12 - r + 0.5, 0, 1), np.clip(18 - r + 0.5, 0, 1)
    scan = ndi.gaussian_filter(DENTIN * dentin + ENAMEL * (outer - dentin), 1.2) + AIR
    scan = (scan + np.random.default_rng(3).normal(0.0, 40.0, scan.shape)).astype(np.float32)
    molar = ndi.binary_fill_holes(scan > AIR + 0.3 * DENTIN)
    enamel, levels, _ = S.enamel_layer(scan, molar, background=AIR, dentin_level=AIR + DENTIN,
                                       enamel_level=AIR + ENAMEL)
    summary, _, _ = S.measure(scan, molar, enamel, SPACING, dentin_level=AIR + DENTIN, enamel_level=AIR + ENAMEL,
                              display=False, dentin_z=levels.dentin_z)
    assert summary["share_measured"] > 0.97
    assert summary["median"] == pytest.approx(60.0, abs=3.0)


def test_a_blurred_outer_face_is_found_where_enamel_stands_little_above_dentin():
    """P42_C67B6_WT_1: SkyScan, outer edge blurred over 2.4 voxels, enamel at 175 over dentin at 128.

    Two voxels inside that edge the scan has not reached the junction's level,
    so a test of brightness there refused the outer surface along the crowns'
    walls and the thickness came out wrong or not at all. Enamel inside is the
    enamel mask's to say. A layer 8 voxels deep at that blur reads about a
    voxel thick: its halfway points lie outside it.
    """
    n = 64
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    r = np.sqrt((z - 31.7) ** 2 + (y - 32.2) ** 2 + (x - 31.9) ** 2)
    dentin, outer = np.clip(16 - r + 0.5, 0, 1), np.clip(24 - r + 0.5, 0, 1)
    scan = ndi.gaussian_filter(108.0 * dentin + 155.0 * (outer - dentin), 2.4) + 20.0
    scan = np.round(np.clip(scan + np.random.default_rng(4).normal(0.0, 2.0, scan.shape), 0, 255)).astype(np.float32)
    molar = ndi.binary_fill_holes(scan > 60)
    enamel, levels, _ = S.enamel_layer(scan, molar, background=20.0, dentin_level=128.0, enamel_level=175.0)
    summary, _, _ = S.measure(scan, molar, enamel, SPACING8, dentin_level=128.0, enamel_level=175.0,
                              display=False, dentin_z=levels.dentin_z)
    assert summary is not None and summary["share_measured"] > 0.95
    assert summary["median"] == pytest.approx(64.0, abs=8.0)


# ---------------------------------------------------------------------------
# Patches of enamel thinner, or less dense, than the enamel around them
# ---------------------------------------------------------------------------

import enamel_patches as P  # noqa: E402

# A tooth at 6 um, its enamel 12 voxels deep on its crown and down its sides to
# a cervical edge, the root bare below. In the crown, against the default bars
# (45% thinner, 11% less mineral): two pits (the enamel 9 and 7 voxels
# shallower), one too shallow to call (4, a third), a column of enamel missing a
# fifth of its mineral and one missing too little (5%), and a patch of crown
# left as it is. y runs from the crown to the root.
CAP_SPACING = (0.006, 0.006, 0.006)
CAP_SHAPE = (200, 96, 130)
CAP_TOP, CAP_JUNCTION, CAP_CERVIX, CAP_BOTTOM = 12, 24, 50, 90
CAP_BODY = ((15, 185), (15, 115))                      # the dentin's extent along z and x
CAP_PITS = {(45, 40): 9, (45, 90): 7, (100, 40): 4}    # (z, x): voxels shallower
CAP_DIM = {(155, 40): 0.20, (155, 90): 0.05}            # (z, x): share of the mineral missing
CAP_CONTROL = (100, 90)
CAP_R = 8


def _patchy_cap(sigma=1.3, noise=20.0, seed=11, pits=CAP_PITS, dim=CAP_DIM, bare=None):
    """The tooth above, blurred and noisy as a 6 um scan. Returns (scan, molar volume, enamel, dentin).

    ``bare`` (z, x, radius) leaves a disk of the crown without enamel, as a cusp tip without enamel reads.
    """
    z, y, x = np.mgrid[:CAP_SHAPE[0], :CAP_SHAPE[1], :CAP_SHAPE[2]]
    (z0, z1), (x0, x1) = CAP_BODY
    deep = CAP_JUNCTION - CAP_TOP
    dentin = (z >= z0) & (z < z1) & (x >= x0) & (x < x1) & (y >= CAP_JUNCTION) & (y < CAP_BOTTOM)
    shell = ((z >= z0 - deep) & (z < z1 + deep) & (x >= x0 - deep) & (x < x1 + deep)
             & (y >= CAP_TOP) & (y < CAP_CERVIX))
    top = np.full((CAP_SHAPE[0], CAP_SHAPE[2]), float(CAP_TOP))
    zz, xx = np.mgrid[:CAP_SHAPE[0], :CAP_SHAPE[2]]
    for (cz, cx), shallower in pits.items():
        top[np.hypot(zz - cz, xx - cx) < CAP_R] = CAP_TOP + shallower
    if bare is not None:
        top[np.hypot(zz - bare[0], xx - bare[1]) < bare[2]] = CAP_JUNCTION
    enamel = shell & ~dentin & (y >= top[:, None, :])
    value = np.where(enamel, ENAMEL, 0.0) + np.where(dentin, DENTIN, 0.0)
    for (cz, cx), missing in dim.items():
        value[(np.hypot(z - cz, x - cx) < CAP_R) & enamel] = ENAMEL * (1.0 - missing)
    scan = ndi.gaussian_filter(value.astype(np.float32), sigma) + AIR
    scan += np.random.default_rng(seed).normal(0.0, noise, scan.shape).astype(np.float32)
    scan = scan.astype(np.float32)
    return scan, np.where(scan > AIR + 0.3 * DENTIN, scan, 0).astype(np.float32), enamel, dentin


@pytest.fixture(scope="module")
def patchy_cap():
    scan, volume, enamel, dentin = _patchy_cap()
    summary, arrays = M.quantify(volume, CAP_SPACING, scan=scan, split=False, save_points=True)
    return summary, arrays, enamel, dentin


def _near(record, zx, within=2.0):
    return abs(record["centre_zyx"][0] - zx[0]) <= within and abs(record["centre_zyx"][2] - zx[1]) <= within


def test_the_planted_patches_are_found_and_nothing_else(patchy_cap):
    summary, _, _, _ = patchy_cap
    patches = summary["enamel"]["patches"]
    thin, low = patches["thin"]["largest"], patches["low_density"]["largest"]
    # The two pits deep enough, and only they: not the shallow one, the control, or the dim columns.
    assert len(thin) == 2
    assert {next(zx for zx in ((45, 40), (45, 90)) if _near(r, zx)) for r in thin} == {(45, 40), (45, 90)}
    # The column missing a fifth of its mineral, and only it: not the pits, whose blurred faces the core keeps clear of.
    assert len(low) == 1 and _near(low[0], (155, 40))
    assert low[0]["depth"] == pytest.approx(0.20, abs=0.04)
    assert low[0]["area_mm2"] == pytest.approx(np.pi * (CAP_R * 6.0) ** 2 / 1e6, rel=0.35)
    assert 0.4 < patches["thin"]["share_judged"] < 0.9 and patches["low_density"]["share_judged"] > 0.3


def test_a_pit_is_as_thin_as_the_nearest_point_measure_says(patchy_cap):
    """A pit's patch reaches past its rim: beside it the enamel's nearest way out is the pit's wall.

    The truth is the same nearest-point thickness, read off the built enamel
    from the junction: the junction under enamel at least a quarter (and a
    voxel and a half) thinner than the crown's 12 voxels.
    """
    summary, _, enamel, dentin = patchy_cap
    air = ~(enamel | dentin)
    thickness = ndi.distance_transform_edt(~air)[:, CAP_JUNCTION, :]           # voxels, from the junction
    (z0, z1), (x0, x1) = CAP_BODY
    deep = CAP_JUNCTION - CAP_TOP
    bar = P.Params().thinner_by
    for (cz, cx), shallower in CAP_PITS.items():
        if shallower / deep < bar:
            continue
        zz, xx = np.mgrid[z0:z1, x0:x1]
        local = np.hypot(zz - cz, xx - cx) < 3 * CAP_R
        t = thickness[z0:z1, x0:x1]
        truth = local & (t <= (1.0 - bar) * deep + 0.5) & (deep - t >= 1.5 - 0.5)
        truth_mm2 = truth.sum() * 36.0 / 1e6
        found = next(r for r in summary["enamel"]["patches"]["thin"]["largest"] if _near(r, (cz, cx)))
        assert 0.7 * truth_mm2 <= found["area_mm2"] <= 1.4 * truth_mm2
        truth_depth = float(np.mean(1.0 - t[truth] / deep))
        assert found["depth"] == pytest.approx(truth_depth, abs=0.08)


def test_the_blur_and_the_cervical_edge_are_read_off_the_scan(patchy_cap):
    summary, arrays, _, _ = patchy_cap
    assert summary["enamel"]["thickness_um"]["edge_blur_um"] == pytest.approx(1.3 * 6.0, rel=0.2)
    points = arrays["junction_points"]
    patches = summary["enamel"]["patches"]
    assert patches["margin"]["pieces"] == 1
    y = points["verts"][:, 1]
    # The lowest junction points lie at the edge: the bare root surface starts a
    # voxel or two outside the dentin's face, in the molar mask's blurred rim.
    lowest = y >= CAP_CERVIX - 2
    assert lowest.any() and np.all(points["cervical_um"][lowest] <= 6 * 6.0)
    crown = y <= CAP_JUNCTION + 1
    assert np.all(points["cervical_um"][crown] >= (CAP_CERVIX - CAP_JUNCTION - 2) * 6.0)
    # The crown has no bare area, so the enamel's only edge is the cervical one.
    assert patches["margin"]["bare_crown_pieces"] == 0
    assert np.allclose(points["edge_um"], points["cervical_um"])
    # Nothing is called within the margin band.
    _, shades = P.patches_from_points(points)
    for kind in ("thin", "low_density"):
        assert not np.any(shades[f"{kind}_patch"] & (points["edge_um"] < P.Params().margin_um))


def test_the_saved_points_give_the_same_patches_again(patchy_cap):
    summary, arrays, _, _ = patchy_cap
    block, _ = P.patches_from_points(arrays["junction_points"])
    stored = summary["enamel"]["patches"]
    for kind in ("thin", "low_density"):
        for key in ("share_judged", "share", "patches", "depth", "shortfall"):
            assert block[kind][key] == pytest.approx(stored[kind][key])
    json.dumps(summary, allow_nan=False)


def test_the_rim_of_a_bare_cusp_is_an_edge_not_a_patch():
    """A disk of crown without enamel: the enamel tapers to nothing around it, as at the cervix."""
    scan, volume, _, _ = _patchy_cap(pits={}, dim={}, bare=(100, 65, 14))
    summary, arrays = M.quantify(volume, CAP_SPACING, scan=scan, split=False, save_points=True, thickness_map=False)
    patches = summary["enamel"]["patches"]
    assert patches["margin"]["bare_crown_pieces"] == 1
    assert patches["margin"]["bare_crown_mm2"] == pytest.approx(np.pi * (14 * 6.0) ** 2 / 1e6, rel=0.5)
    assert patches["thin"]["patches"] == 0
    points = arrays["junction_points"]
    rim = np.hypot(points["verts"][:, 0] - 100, points["verts"][:, 2] - 65) < 14 + 8
    assert rim.any() and np.all(points["edge_um"][rim] < P.Params().margin_um)
    assert np.all(np.isinf(points["cervical_um"]) | (points["cervical_um"] >= points["edge_um"]))


def test_the_core_density_keeps_clear_of_the_blurred_faces():
    """A flat slab of enamel between dentin and the air, blurred: the core reads the enamel, the whole line does not."""
    sigma, size = 1.3, 6.0
    for deep in (8, 10, 14):
        shape = (24, 24 + deep + 24, 24)
        value = np.zeros(shape, dtype=np.float32)
        top, junction = 24, 24 + deep
        value[:, top:junction, :] = ENAMEL
        value[:, junction:, :] = DENTIN
        image = ndi.gaussian_filter(value, sigma) + AIR
        g = np.arange(6.0, 18.0, 1.0)
        zz, xx = np.meshgrid(g, g, indexing="ij")
        start = np.stack([zz.ravel(), np.full(zz.size, junction - 0.5), xx.ravel()], axis=1)
        end = np.stack([zz.ravel(), np.full(zz.size, top - 0.5), xx.ravel()], axis=1)
        faces = np.concatenate([start, end]) * size
        core = S.core_density(image, start, end, np.full(3, size), {"core": 2.5 * sigma, "line": 0.0}, faces)
        miss = (ENAMEL - (core["core"] - AIR)) / ENAMEL
        assert np.all(np.isfinite(core["core"]))
        assert np.max(np.abs(miss)) < (0.015 if deep < 12 else 0.005)
        if deep == 8:
            assert np.min((ENAMEL - (core["line"] - AIR)) / ENAMEL) >= 0.05


@pytest.mark.parametrize("sigma", [1.2, 1.75, 2.4])
def test_the_edge_blur_is_read_within_an_eighth(sigma):
    """The outer face of a flat slab 16 voxels deep, blurred at sigma, 2% noise."""
    rng = np.random.default_rng(int(sigma * 10))
    shape = (40, 60, 40)
    value = np.zeros(shape, dtype=np.float32)
    value[:, 20:36, :] = ENAMEL
    value[:, 36:, :] = DENTIN
    image = ndi.gaussian_filter(value, sigma) + AIR
    image += rng.normal(0.0, 0.02 * ENAMEL, shape).astype(np.float32)
    g = np.arange(8.0, 32.0, 0.5)
    zz, xx = np.meshgrid(g, g, indexing="ij")
    outer = np.stack([zz.ravel(), np.full(zz.size, 19.5), xx.ravel()], axis=1)
    inward = np.tile([0.0, 1.0, 0.0], (len(outer), 1))
    blur = S.edge_blur(image, outer, inward, np.full(len(outer), 16.0), AIR)
    assert blur == pytest.approx(sigma, rel=0.12)


def test_the_patch_shades_pack_as_the_viewer_reads_them():
    """Deficits in tenths of a percent, signed, -32768 where not judged; patches as 0 or 1."""
    pos = np.array([[0, 0, 0], [100.0, 0, 0], [0, 50.0, 25.0]], dtype=np.float32)
    layers = {"enamel": {"pos": pos, "faces": np.array([[0, 1, 2]], dtype=np.uint32),
                         "value": np.array([61.0, 50.0, 3.0], dtype=np.float32),
                         "thin": np.array([0.3124, np.nan, -0.05]), "thin_patch": np.array([1.0, np.nan, 0.0]),
                         "low_density": np.array([0.0, 0.081, np.nan]),
                         "low_density_patch": np.array([0.0, 1.0, np.nan])}}
    blob, layout = S._pack(layers)
    spec = layout["layers"]["enamel"]
    for key in ("thin", "thin_patch", "low_density", "low_density_patch"):
        assert spec[key] % 4 == 0
    assert list(np.frombuffer(blob, "<i2", 3, spec["thin"])) == [312, -32768, -50]
    assert list(np.frombuffer(blob, "u1", 3, spec["thin_patch"])) == [1, 0, 0]
    assert list(np.frombuffer(blob, "<i2", 3, spec["low_density"])) == [0, 81, -32768]
    assert list(np.frombuffer(blob, "u1", 3, spec["low_density_patch"])) == [0, 1, 0]


def test_the_patches_reach_the_csv_the_map_and_the_saved_points(tmp_path):
    scan, volume, _, _ = _patchy_cap()
    sample = tmp_path / "P22_WT_9"
    save_volume(volume, "volumes", "P22_WT_9_molar_volume", base_dir=str(sample), output_format="nrrd",
                spacing=CAP_SPACING)
    save_volume(scan, "", "P22_WT_9_reoriented_volume", base_dir=str(sample), output_format="nrrd",
                spacing=CAP_SPACING)
    out = tmp_path / "out"
    assert M.main(["-i", str(sample), "-o", str(out), "--no-split", "--save-points", "--no-plot"]) == 0
    record = json.loads((out / "P22_WT_9_molar_enamel.json").read_text())
    assert record["version"] == M.VERSION == 6
    row = M.summary_row("P22_WT_9", record)
    assert row["thin_patch_share"] == pytest.approx(record["enamel"]["patches"]["thin"]["share"])
    assert row["low_density_patch_share"] > 0 and row["edge_blur_um"] > 0
    assert row["low_density_patch_depth_mgHA"] == ""        # no calibration: raw only
    page = (out / "P22_WT_9_thickness_map.html").read_text(encoding="utf-8")
    meta = json.loads(re.search(r'<script id="meta" type="application/json">(.*?)</script>', page, re.S).group(1))
    assert meta["modes"] == ["thickness", "thin", "low_density"]
    assert meta["patches"]["thin"]["patches"] == 2 and meta["patches"]["low_density"]["patches"] == 1
    assert "thin" in meta["layout"]["layers"]["enamel"]
    results = P.sweep([str(out / "P22_WT_9_junction_points.npz")], P._parse_sets(["thinner_by=0.25,0.6,0.85"]))
    # At 25% the third pit (a third thinner) joins the two; at 60% only the deepest stands (built three quarters
    # thinner, its 3-voxel floor reads about 64%: a thin layer reads thick under blur); at 85% none.
    assert [r["patches"]["thin"]["patches"] for r in results] == [3, 1, 0]


def test_no_patches_when_asked_for_none():
    scan, volume, _, _ = _patchy_cap()
    summary, arrays = M.quantify(volume, CAP_SPACING, scan=scan, split=False, patches=None, thickness_map=False)
    assert summary["enamel"]["patches"] is None
    assert arrays["junction_points"] is None
    assert summary["enamel"]["thickness_um"]["median"] == pytest.approx(72.0, abs=4.0)
