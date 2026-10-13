"""
What the results page and the exports are given about each specimen.

The incisor profile's densities are in the scan's stored units. These tests
pin which HA calibration converts them -- the molar step's when it has run, so
the incisor and molar read on one scale; else the one entered in the Molar
enamel settings; else the scan's own -- and that the incisor export carries
them in mg HA/cm³ beside the raw values, with the specimen's sex to group by.
The molar results carry the enamel thickness per mm² of enamel–dentin
junction, for the row and each tooth, beside the two it is made from. The
incisor metrics are read off the profile as the R report cleans it, and the
incisor export says what the cleanup made of every slice.

They also pin what a flag does: a flagged specimen is left out of the exports
of the analyses it names, is listed with its note in the metadata and in the
exclusions, and stays in the R specimen list commented out.

Run with:  python -m pytest studio/tests/test_results.py
"""

import csv
import io
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio import results  # noqa: E402
from studio.workspace import Workspace  # noqa: E402

SLOPE, INTERCEPT, MU = 375.507, -197.288, 4096.0
ROWS = [  # z, z_percent, status, enamel_ratio, enamel_density, dentin_reference, dentin_mode
    (0, 0.0, "pre_onset", 0.0, 0.0, 13000, 12800),
    (1, 50.0, "resolved_lo", 0.8, 10400, 13000, 13100),
    (2, 100.0, "resolved_hi", 1.5, 19500, 13000, 13900),
]


def mg(value):
    return value / MU * SLOPE + INTERCEPT


def write_profile(folder: Path, name: str) -> Path:
    path = folder / f"{name}_enamel_density_profile.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z", "z_percent", "status", "enamel_ratio", "enamel_density", "dentin_reference", "dentin_mode"])
        writer.writerows(ROWS)
    return path


def write_meta(folder: Path, name: str) -> Path:
    """A converter's <stem>_meta.json with the scanner's calibration, values left as stored."""
    path = folder / f"{name}_meta.json"
    path.write_text(json.dumps({"mu_scaling": MU, "_filter_params": {"apply_rescale": False},
                                "calibration": {"slope": SLOPE, "intercept": INTERCEPT, "mu_scaling": MU,
                                                "description": "70 kVp, 0.5mm Al"}}))
    return path


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws")


def write_molar(folder: Path, name: str, calibration=None, thickness=None, teeth=(), patches=None,
                tooth_patches=None) -> Path:
    """A molar step summary with the fields the results read; teeth as (name, thickness µm, junction mm²).

    ``patches`` is the ``enamel.patches`` block (version 6 on), ``tooth_patches``
    each tooth's patch fields by name.
    """
    path = folder / f"{name}_molar_enamel.json"
    path.write_text(json.dumps({
        "enamel": {"fraction": 0.2, "volume_mm3": 0.1, "density": {"raw": {"mean": 20000.0}}, "thickness_um": thickness or {},
                   "patches": patches},
        "total": {"volume_mm3": 0.5}, "dentin": {"volume_mm3": 0.4}, "threshold": {"value": 15000, "method": "otsu"},
        "teeth": [{"name": tooth, "enamel_fraction": 0.2, "volume_mm3": 0.2, "enamel_volume_mm3": 0.04,
                   "thickness_median_um": um, "junction_area_mm2": mm2, **(tooth_patches or {}).get(tooth, {})}
                  for tooth, um, mm2 in teeth],
        "calibration": calibration}))
    return path


def specimen(ws, tmp_path, name, *, meta=True, sex="F", molar=False):
    folder = tmp_path / "data" / name
    folder.mkdir(parents=True)
    artifacts = {"incisor_profile": str(write_profile(folder, name))}
    if meta:
        artifacts["convert_meta"] = str(write_meta(folder, name))
    if molar:
        artifacts["molar_quant"] = str(write_molar(folder, name))
    return ws.create({"age": "P22", "line": "TGM2", "genotype": "WT", "sex": sex, "replicate": 1},
                     name=name, artifacts=artifacts)


def flag(ws, name, scopes="all", note="Motion in the scan"):
    ws.update(name, lambda rec: rec.update(excluded={"from": scopes, "note": note, "at": "2026-09-29T10:00:00"}))


def test_the_scans_own_calibration_converts_the_incisor_profile(ws, tmp_path):
    rec = specimen(ws, tmp_path, "P22_TGM2_WT_F_1")
    got = results.gather(ws)
    cal = got["incisor"]["P22_TGM2_WT_F_1"]["calibration"]
    assert (cal["slope"], cal["intercept"], cal["divisor"]) == (SLOPE, INTERCEPT, MU)
    assert cal["description"] == "70 kVp, 0.5mm Al" and Path(cal["source"]).name.endswith("_meta.json")
    # The per-slice dentin the tool's own figure draws comes along with the ratio's reference.
    prof = got["incisor"]["P22_TGM2_WT_F_1"]
    assert prof["dentin_mode"] == [12800, 13100, 13900] and prof["dentin"] == [13000] * 3
    assert results.to_mg_ha([19500, None], cal) == [pytest.approx(mg(19500)), None]
    assert results.ha_calibration(ws, rec) == cal


def test_the_molar_steps_calibration_wins_so_both_teeth_read_on_one_scale(ws, tmp_path):
    rec = specimen(ws, tmp_path, "P22_TGM2_WT_F_2")
    by_hand = {"source": None, "description": "by hand", "slope": 400.0, "intercept": -200.0, "divisor": 1.0,
               "offset": 0.0, "units": "mg HA/ccm"}
    summary = write_molar(tmp_path / "data" / "P22_TGM2_WT_F_2", "P22_TGM2_WT_F_2", calibration=by_hand)
    ws.set_artifacts(rec["name"], {"molar_quant": str(summary)})
    assert results.gather(ws)["incisor"][rec["name"]]["calibration"] == by_hand


def test_a_calibration_entered_in_the_molar_settings_converts_an_uncalibrated_scan(ws, tmp_path):
    rec = specimen(ws, tmp_path, "P22_TGM2_WT_F_3", meta=False)
    assert results.ha_calibration(ws, rec) is None
    ws.update_settings({"step_defaults": {"molar_quant": {"ha_slope": 350.0, "ha_intercept": -180.0, "mu_scaling": 8192}}})
    cal = results.ha_calibration(ws, rec)
    assert (cal["slope"], cal["intercept"], cal["divisor"], cal["offset"]) == (350.0, -180.0, 8192.0, 0.0)


def test_the_settings_mu_scaling_overrides_the_files(ws, tmp_path):
    rec = specimen(ws, tmp_path, "P22_TGM2_WT_F_4")
    ws.update_settings({"step_defaults": {"molar_quant": {"mu_scaling": 1}}})
    assert results.ha_calibration(ws, rec)["divisor"] == 1.0


def test_the_incisor_export_carries_mg_ha_and_the_specimens_sex(ws, tmp_path):
    specimen(ws, tmp_path, "P22_TGM2_WT_M_1", sex="M")
    specimen(ws, tmp_path, "P22_TGM2_WT_F_5", meta=False)
    name, text = results.export(ws, "incisor")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0])[:5] == ["sample", "group", "genotype", "age", "sex"]
    male = [r for r in rows if r["sample"] == "P22_TGM2_WT_M_1"]
    assert [r["sex"] for r in male] == ["M"] * 3
    assert float(male[2]["enamel_density_mgHA"]) == pytest.approx(mg(19500))
    assert float(male[1]["dentin_mode_mgHA"]) == pytest.approx(mg(13100))
    assert float(male[1]["dentin_reference_mgHA"]) == pytest.approx(mg(13000))
    # Before enamel onset there is no enamel: 0 in every unit, never the calibration's intercept.
    assert (male[0]["enamel_density"], male[0]["enamel_density_mgHA"]) == ("0.0", "0.0")
    assert male[0]["calibration"].endswith("_meta.json")
    # A scan with no calibration keeps its raw values and leaves the converted ones empty.
    female = [r for r in rows if r["sample"] == "P22_TGM2_WT_F_5"]
    assert female[2]["enamel_density"] == "19500.0" and female[2]["enamel_density_mgHA"] == ""
    assert female[2]["calibration"] == ""


def test_the_thickness_is_also_given_per_mm2_of_junction(ws, tmp_path):
    name = specimen(ws, tmp_path, "P22_TGM2_WT_F_13")["name"]
    summary = write_molar(tmp_path / "data" / name, name,
                          thickness={"median": 63.0, "p25": 50.0, "p75": 75.0, "junction_area_mm2": 5.25},
                          teeth=[("M1", 66.0, 2.4), ("M2", 60.0, 1.6), ("M3", 55.0, 0.0)])
    ws.set_artifacts(name, {"molar_quant": str(summary)})
    mq = results.gather(ws)["molar"][name]
    assert mq["thickness_um_per_mm2"] == pytest.approx(63.0 / 5.25)
    # Each tooth over its own junction, and none for a tooth whose junction was not found.
    assert [t["thickness_um_per_mm2"] for t in mq["teeth"]] == [pytest.approx(66.0 / 2.4), pytest.approx(60.0 / 1.6), None]
    row = next(csv.DictReader(io.StringIO(results.export(ws, "molar")[1])))
    assert float(row["thickness_um_per_mm2"]) == pytest.approx(12.0)
    assert float(row["M2_thickness_um_per_mm2"]) == pytest.approx(37.5) and row["M3_thickness_um_per_mm2"] == ""
    # A summary from before the junction was measured has the thickness alone.
    old = specimen(ws, tmp_path, "P22_TGM2_WT_F_14")["name"]
    ws.set_artifacts(old, {"molar_quant": str(write_molar(tmp_path / "data" / old, old, thickness={"median": 40.0}))})
    assert results.gather(ws)["molar"][old]["thickness_um_per_mm2"] is None


def test_the_thickness_mean_and_sd_reach_results_and_the_export(ws, tmp_path):
    name = specimen(ws, tmp_path, "P22_TGM2_WT_F_15")["name"]
    summary = write_molar(tmp_path / "data" / name, name,
                          thickness={"median": 63.0, "mean": 65.5, "sd": 11.25, "p25": 50.0, "p75": 75.0},
                          teeth=[("M1", 66.0, 2.4), ("M2", 60.0, 1.6)],
                          tooth_patches={"M1": {"thickness_mean_um": 68.0, "thickness_sd_um": 9.5}})
    ws.set_artifacts(name, {"molar_quant": str(summary)})
    mq = results.gather(ws)["molar"][name]
    assert mq["thickness_mean_um"] == 65.5 and mq["thickness_sd_um"] == 11.25
    assert [(t["thickness_mean_um"], t["thickness_sd_um"]) for t in mq["teeth"]] == [(68.0, 9.5), (None, None)]
    row = next(csv.DictReader(io.StringIO(results.export(ws, "molar")[1])))
    assert float(row["thickness_mean_um"]) == 65.5 and float(row["thickness_sd_um"]) == 11.25
    assert float(row["M1_thickness_mean_um"]) == 68.0 and float(row["M1_thickness_sd_um"]) == 9.5
    assert row["M2_thickness_mean_um"] == "" and row["M3_thickness_sd_um"] == ""
    # A summary from before the mean and SD were kept has neither.
    old = specimen(ws, tmp_path, "P22_TGM2_WT_F_16")["name"]
    ws.set_artifacts(old, {"molar_quant": str(write_molar(tmp_path / "data" / old, old, thickness={"median": 40.0}))})
    assert results.gather(ws)["molar"][old]["thickness_mean_um"] is None


def test_the_mean_over_the_root_of_the_junction_area_has_no_unit(ws, tmp_path):
    """Mean µm over √(area) in µm. Version 6 summaries keep the per-tooth mean only in the thickness block."""
    name = specimen(ws, tmp_path, "P22_TGM2_WT_F_17")["name"]
    summary = write_molar(tmp_path / "data" / name, name,
                          thickness={"median": 63.0, "mean": 60.0, "junction_area_mm2": 4.0,
                                     "teeth": [{"name": "M1", "mean": 50.0}, {"name": "M2", "mean": 45.0}]},
                          teeth=[("M1", 52.0, 2.5), ("M2", 47.0, 0.0), ("M3", 40.0, 1.0)])
    ws.set_artifacts(name, {"molar_quant": str(summary)})
    mq = results.gather(ws)["molar"][name]
    assert mq["thickness_mean_per_sqrt_junction"] == pytest.approx(60.0 / 2000.0)
    # M1 from the thickness block's mean; M2 has no junction; M3 has no mean anywhere.
    assert [t["thickness_mean_um"] for t in mq["teeth"]] == [50.0, 45.0, None]
    assert mq["teeth"][0]["thickness_mean_per_sqrt_junction"] == pytest.approx(50.0 / (2.5 ** 0.5 * 1000))
    assert [t["thickness_mean_per_sqrt_junction"] for t in mq["teeth"][1:]] == [None, None]
    row = next(csv.DictReader(io.StringIO(results.export(ws, "molar")[1])))
    assert float(row["thickness_mean_per_sqrt_junction"]) == pytest.approx(0.03)
    assert float(row["M1_thickness_mean_per_sqrt_junction"]) == pytest.approx(0.0316228, rel=1e-5)
    assert row["M2_thickness_mean_per_sqrt_junction"] == ""
    # A tooth scaled up keeps the ratio: twice the thickness, four times the area.
    assert results.per_sqrt_junction_area(120.0, 16.0) == results.per_sqrt_junction_area(60.0, 4.0)
    assert results.per_sqrt_junction_area(None, 4.0) is None and results.per_sqrt_junction_area(60.0, 0.0) is None


def test_the_patches_reach_results_and_the_export(ws, tmp_path):
    name = specimen(ws, tmp_path, "P22_TGM2_KO_F_15")["name"]
    patches = {"thin": {"share": 0.036, "patches_per_mm2": 2.5, "depth": 0.36, "shortfall": 0.077},
               "low_density": {"share": 0.0088, "patches_per_mm2": 1.3, "depth": 0.084, "shortfall": 0.009,
                               "core_density_raw": 2210.0, "core_density_mg_ha": 1980.0}}
    summary = write_molar(tmp_path / "data" / name, name, patches=patches,
                          thickness={"median": 58.0, "junction_area_mm2": 5.0, "edge_blur_um": 7.6},
                          teeth=[("M1", 57.0, 2.9), ("M2", 60.0, 2.0), ("M3", 56.0, 0.8)],
                          tooth_patches={"M1": {"thin_patch_share": 0.05, "low_density_patch_share": 0.01},
                                         "M2": {"thin_patch_share": 0.02}})
    ws.set_artifacts(name, {"molar_quant": str(summary)})
    mq = results.gather(ws)["molar"][name]
    assert mq["thin_patch_share"] == pytest.approx(0.036) and mq["low_density_patch_share"] == pytest.approx(0.0088)
    assert mq["core_density_mgHA"] == pytest.approx(1980.0) and mq["edge_blur_um"] == pytest.approx(7.6)
    assert [t["thin_patch_share"] for t in mq["teeth"]] == [pytest.approx(0.05), pytest.approx(0.02), None]
    row = next(csv.DictReader(io.StringIO(results.export(ws, "molar")[1])))
    assert float(row["thin_patch_share"]) == pytest.approx(0.036)
    assert float(row["M1_low_density_patch_share"]) == pytest.approx(0.01) and row["M2_low_density_patch_share"] == ""
    assert float(row["density_shortfall"]) == pytest.approx(0.009)
    # A summary from before the patches were looked for has none, and the export leaves them empty.
    old = specimen(ws, tmp_path, "P22_TGM2_WT_F_16")["name"]
    ws.set_artifacts(old, {"molar_quant": str(write_molar(tmp_path / "data" / old, old, thickness={"median": 40.0}))})
    record = results.gather(ws)["molar"][old]
    assert record["thin_patch_share"] is None and record["core_density_raw"] is None


def write_long_profile(folder: Path, name: str, flip: int = 150) -> Path:
    """300 slices of a clean rise through dentin density, the fit at slice `flip` on the wrong mode."""
    path = folder / f"{name}_enamel_density_profile.csv"
    zp = [100 * i / 299 for i in range(300)]
    ratio = [0.6 + 1.2 / (1 + math.exp(-(z - 40) / 6)) for z in zp]
    ratio[flip] += 0.5
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z", "z_percent", "status", "enamel_ratio", "enamel_density", "dentin_reference", "dentin_mode",
                         "z_window", "solidity"])
        for i, (z, r) in enumerate(zip(zp, ratio)):
            status = "resolved_hi" if r > 1.03 else "resolved_lo" if r < 0.97 else "coincident"
            writer.writerow([i, z, status, r, r * 13000, 13000, 13000, 2, 0.3])
    return path


def test_the_incisor_metrics_and_export_read_the_cleaned_profile(ws, tmp_path):
    folder = tmp_path / "data" / "P22_TGM2_WT_F_16"
    folder.mkdir(parents=True)
    name = ws.create({"age": "P22", "line": "TGM2", "genotype": "WT", "sex": "F", "replicate": 16}, name="P22_TGM2_WT_F_16",
                     artifacts={"incisor_profile": str(write_long_profile(folder, "P22_TGM2_WT_F_16")),
                                "convert_meta": str(write_meta(folder, "P22_TGM2_WT_F_16"))})["name"]
    got = results.gather(ws)
    metrics = got["incisor_metrics"][name]
    assert (metrics["kept_slices"], metrics["outlier_slices"], metrics["filtered_slices"]) == (299, 1, 0)
    # The report's crossover: where 0.6 + 1.2 / (1 + e^-(z - 40)/6) reaches 1.
    assert metrics["crossover_z"] == pytest.approx(40 - 6 * math.log(2), abs=0.2)
    assert 1.75 < metrics["plateau_ratio"] < metrics["peak_ratio"] < 1.81
    # The plateau and peak in density, off the enamel's own running median: the
    # profile's density is the ratio times a dentin reference of 13000.
    for key in ("plateau", "peak"):
        assert metrics[f"{key}_raw"] == pytest.approx(13000 * metrics[f"{key}_ratio"])
        assert metrics[f"{key}_mgHA"] == pytest.approx(mg(metrics[f"{key}_raw"]))
    assert (metrics["dentin_raw"], metrics["dentin_mgHA"]) == (13000, pytest.approx(mg(13000)))
    prof = got["incisor"][name]
    uncalibrated = results.incisor_metrics(prof)
    assert uncalibrated["plateau_raw"] == metrics["plateau_raw"] and uncalibrated["plateau_mgHA"] is None
    assert prof["outlier"][150] and not prof["kept"][150] and "residual" not in prof, "the page gets no residuals"
    rows = list(csv.DictReader(io.StringIO(results.export(ws, "incisor")[1])))
    flipped = next(r for r in rows if r["z"] == "150")
    assert (flipped["kept"], flipped["outlier"], flipped["ratio_med"]) == ("FALSE", "TRUE", "")
    assert float(flipped["residual"]) == pytest.approx(0.5, abs=0.02)
    middle = next(r for r in rows if r["z"] == "200")
    assert middle["kept"] == "TRUE" and float(middle["density_med_mgHA"]) == pytest.approx(mg(float(middle["density_med"])))
    assert rows[0]["ratio_med"] == "", "no partial window at the cervical end"


def test_a_profile_from_an_older_incisor_build_is_flagged_for_a_rerun(ws, tmp_path):
    names = {}
    for tag, constants in (("old", {"MIN_ONSET_RISE": 2.0}), ("new", {"MIN_ONSET_RISE": 2.0, "OPENING_RISE": 0.5})):
        rec = specimen(ws, tmp_path, f"P22_TGM2_WT_F_{20 if tag == 'old' else 21}")
        run = tmp_path / "data" / rec["name"] / f"{rec['name']}_enamel_density_run.json"
        run.write_text(json.dumps({"constants": constants}))
        ws.set_artifacts(rec["name"], {"incisor_run": str(run)})
        names[tag] = rec["name"]
    got = results.gather(ws)["incisor"]
    assert got[names["old"]]["stale"], "run before the opening stretch's dentin readings were withdrawn"
    assert not got[names["new"]]["stale"]


def test_every_measurement_export_leads_with_what_to_group_by(ws, tmp_path):
    specimen(ws, tmp_path, "P22_TGM2_WT_F_6")
    for kind in ("molar", "segmentation", "incisor"):
        header = results.export(ws, kind)[1].splitlines()[0].split(",")
        assert header[:5] == ["sample", "group", "genotype", "age", "sex"], kind


def names_in(ws, kind):
    return {row["sample"] for row in csv.DictReader(io.StringIO(results.export(ws, kind)[1]))}


def test_a_flagged_specimen_is_left_out_of_the_analyses_its_flag_names(ws, tmp_path):
    keep = specimen(ws, tmp_path, "P22_TGM2_WT_F_7", molar=True)["name"]
    out = specimen(ws, tmp_path, "P22_TGM2_WT_F_8", molar=True)["name"]
    tip = specimen(ws, tmp_path, "P22_TGM2_WT_F_9", molar=True)["name"]
    flag(ws, out)
    flag(ws, tip, ["incisor"], note="Incisor tip broken in dissection")
    assert names_in(ws, "incisor") == {keep}
    assert names_in(ws, "molar") == {keep, tip}, "a broken incisor leaves the molar measured"
    assert names_in(ws, "segmentation") == {keep, tip}
    # The results page gets every specimen, each with its flag, to leave out and list itself.
    flags = {s["name"]: s["excluded"] for s in results.gather(ws)["samples"]}
    assert flags[keep] is None and flags[out]["from"] == "all"
    assert flags[tip] == {"from": ["incisor"], "note": "Incisor tip broken in dissection", "at": "2026-09-29T10:00:00"}


def test_the_exclusions_are_listed_with_their_notes(ws, tmp_path):
    keep = specimen(ws, tmp_path, "P22_TGM2_WT_F_10")["name"]
    out = specimen(ws, tmp_path, "P22_TGM2_WT_F_11")["name"]
    tip = specimen(ws, tmp_path, "P22_TGM2_WT_F_12")["name"]
    flag(ws, out)
    flag(ws, tip, ["molar", "incisor"], note="Incisor tip broken")
    meta = {r["sample"]: r for r in csv.DictReader(io.StringIO(results.export(ws, "metadata")[1]))}
    assert set(meta) == {keep, out, tip}, "the metadata lists every specimen"
    assert (meta[out]["excluded_from"], meta[out]["exclusion_note"], meta[out]["flagged_on"]) == (
        "all", "Motion in the scan", "2026-09-29")
    assert meta[tip]["excluded_from"] == "molar;incisor" and meta[keep]["excluded_from"] == ""
    assert names_in(ws, "excluded") == {out, tip}
    # The R report reads the incisor: its list keeps the flagged ones, commented out with why.
    lines = results.export(ws, "specimens")[1].splitlines()
    assert lines[0] == "  specimens:"
    by_name = {line.split('"')[1].split()[0]: line for line in lines[1:]}
    assert by_name[keep].startswith('    - "')
    assert by_name[out].startswith('    # - "') and by_name[out].endswith("# excluded: Motion in the scan")
    assert by_name[tip].endswith("# excluded: Incisor tip broken")


def test_the_exports_take_the_specimens_the_results_page_shows(ws, tmp_path):
    a = specimen(ws, tmp_path, "P22_TGM2_WT_F_13", molar=True)["name"]
    b = specimen(ws, tmp_path, "P22_TGM2_WT_F_14", molar=True)["name"]
    specimen(ws, tmp_path, "P22_TGM2_WT_F_15", molar=True)
    flag(ws, b, ["molar"])
    picked = lambda kind: {row["sample"] for row in csv.DictReader(io.StringIO(results.export(ws, kind, [a, b])[1]))}  # noqa: E731
    assert picked("metadata") == {a, b}
    assert picked("molar") == {a}, "a flag still holds within the picked specimens"
    assert picked("incisor") == {a, b}
    assert len(names_in(ws, "metadata")) == 3, "with no names, every specimen"
    spoken = [line for line in results.export(ws, "specimens", [a])[1].splitlines()[1:]]
    assert len(spoken) == 1 and a in spoken[0]
    # The page lists each specimen's library folder, to find and pick a batch by.
    ws.update(a, lambda rec: rec.update(folder="Odam study/Batch 2"))
    assert {s["name"]: s["folder"] for s in results.gather(ws)["samples"]}[a] == "Odam study/Batch 2"


def test_a_flag_that_names_every_analysis_reads_as_all_of_them():
    every = {"excluded": {"from": ["segmentation", "molar", "incisor"], "note": "x"}}
    assert results.exclusion(every)["from"] == "all"
    assert results.exclusion({"excluded": {"from": ["incisor", "molar"]}})["from"] == ["molar", "incisor"]
    assert results.exclusion({}) is None and not results.excluded_from({}, "molar")
    assert results.excluded_from({"excluded": {"from": "all"}}, "segmentation")


def test_each_view_converts_with_the_calibration_its_values_answer_to(ws, tmp_path):
    """Views cut from the segmented scan share the specimen's calibration; the raw scans each have their own."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import struct  # noqa: PLC0415
    import scanco_calibration as sc  # noqa: PLC0415

    folder = tmp_path / "data" / "scans"
    folder.mkdir(parents=True)
    isq = folder / "C0001.ISQ"
    extended = sc.extended_header(SLOPE, INTERCEPT)
    pre = bytearray(sc.BLOCK)
    pre[:16] = b"CTDATA-HEADER_V1"
    struct.pack_into("<i", pre, 88, 4096)
    struct.pack_into("<i", pre, 508, len(extended) // sc.BLOCK)
    isq.write_bytes(bytes(pre) + extended + bytes(128))
    # The conversion was rescaled to 1/cm: its values divide by 1, the ISQ's own by 4096.
    meta = folder / "C0001_meta.json"
    meta.write_text(json.dumps({"_filter_params": {"apply_rescale": True},
                                "calibration": {"slope": SLOPE, "intercept": INTERCEPT, "mu_scaling": 4096}}))
    details = {"age": "P22", "line": "TGM2", "genotype": "KO", "sex": "F"}
    rec = ws.create({**details, "replicate": 1}, name="P22_TGM2_KO_F_1", source={"kind": "isq", "path": str(isq)},
                    artifacts={"convert_meta": str(meta)})
    for view in ("converted", "mandible", "molar", "cmpr"):
        assert results.view_calibration(ws, rec, view) == results.ha_calibration(ws, rec)
    assert results.view_calibration(ws, rec, "molar")["divisor"] == 1.0
    raw = results.view_calibration(ws, rec, "source")
    assert (raw["slope"], raw["divisor"], raw["offset"]) == (pytest.approx(SLOPE), 4096.0, 0.0)
    assert Path(raw["source"]).name == "C0001.ISQ"

    # A raw DICOM series carries no calibration of its own.
    dicom = ws.create({**details, "replicate": 2}, name="P22_TGM2_KO_F_2", source={"kind": "dicom", "path": str(folder)},
                      artifacts={"convert_meta": str(meta)})
    assert results.view_calibration(ws, dicom, "source") is None
    assert results.view_calibration(ws, dicom, "molar")["divisor"] == 1.0

    # An NRRD is segmented as it is: its own values are the ones calibrated.
    nrrd = folder / "scan.nrrd"
    nrrd.write_bytes(b"")
    (folder / "scan_meta.json").write_text(json.dumps({"calibration": {"slope": SLOPE, "intercept": INTERCEPT,
                                                                       "mu_scaling": 1024}}))
    volume = ws.create({**details, "replicate": 3}, name="P22_TGM2_KO_F_3", source={"kind": "volume", "path": str(nrrd)})
    assert results.view_calibration(ws, volume, "source")["divisor"] == 1024.0
    assert results.view_calibration(ws, volume, "source") == results.ha_calibration(ws, volume)


def test_the_landmarks_say_which_straightened_slices_they_map_to(ws, tmp_path):
    """The compare page draws each landmark on the straightened incisor at the slice map_landmarks matched it to."""
    rec = specimen(ws, tmp_path, "P22_TGM2_WT_F_1")
    path = tmp_path / "data" / rec["name"] / f"{rec['name']}_landmarks.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["landmark", "kind", "z_percent_start", "z_percent_end", "z_percent_mid", "z_start", "z_end",
                         "slice_start", "slice_end", "length_mm", "source"])
        writer.writerow(["eruption", "event", 81.5, "", 81.5, 1621, "", 412, "", "", "auto"])
        writer.writerow(["M1", "span", 30.0, 45.0, 37.5, 820, 973, 150, 228, 1.2, "manual"])
    rec = ws.set_artifacts(rec["name"], {"landmarks": path})
    got = {r["landmark"]: r for r in results.landmarks(ws, rec)}
    assert (got["eruption"]["slice_start"], got["eruption"]["slice_end"]) == (412, None)
    assert (got["M1"]["slice_start"], got["M1"]["slice_end"]) == (150, 228)
    assert (got["M1"]["start"], got["M1"]["end"], got["M1"]["source"]) == (30.0, 45.0, "manual")
