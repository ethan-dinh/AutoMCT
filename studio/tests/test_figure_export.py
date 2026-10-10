"""
The results page's exports for Prism, run under Node.

The browser draws the figures (web/js/figure.js), lays out the Prism tables
(web/js/prism.js) and writes the files (web/js/files.js) itself. These tests
run those modules under Node and read what they write back with Pillow,
zipfile and ElementTree -- and with the R pzfx package, built from Prism's own
files, where it is installed. They pin:

- GraphPad's significance stars, and brackets that sit over the data inside
  the figure, ns left off when asked;
- axes that run from one tick to another, tick labels with the step's digits;
- points spread so none overlaps another;
- a figure's SVG sized in inches, and the PNG and TIFF recording their DPI;
- a Column table for an unsplit measurement, a Grouped table (a stratum per
  row, a specimen per subcolumn) for a split one, an XY table for a profile,
  and a HugeTable past Prism's 52 replicates.

Run with:  python -m pytest studio/tests/test_figure_export.py
"""

import csv
import io
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

JS = Path(__file__).resolve().parents[1] / "web" / "js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node to run the page's own code")


def js(body: str, **inputs):
    """Run `body` (which ends by printing JSON) with figure.js as F, prism.js as P and files.js as B imported."""
    bindings = "".join(f"const {k} = {json.dumps(v)};\n" for k, v in inputs.items())
    code = (f"import * as F from {json.dumps((JS / 'figure.js').as_uri())};\n"
            f"import * as P from {json.dumps((JS / 'prism.js').as_uri())};\n"
            f"import * as B from {json.dumps((JS / 'files.js').as_uri())};\n"
            f"import fs from 'node:fs';\n{bindings}{body}")
    done = subprocess.run([NODE, "--input-type=module", "-e", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout) if done.stdout.strip() else None


def value(name, v):
    return {"label": name, "value": v}


def column_spec(splits=(), strata=None, tests=()):
    rng = np.random.default_rng(1)
    levels = [{"level": "WT", "name": "WT", "color": "#2a78d6"}, {"level": "KO", "name": "KO", "color": "#eb6834"}]
    if strata is None:
        strata = [{"label": "", "cells": [
            {**levels[0], "values": [value(f"wt{i}", float(v)) for i, v in enumerate(rng.normal(1900, 30, 6))]},
            {**levels[1], "values": [value(f"ko{i}", float(v)) for i, v in enumerate(rng.normal(1750, 40, 5))]}]}]
    return {"kind": "column", "title": "Enamel density, mean (mg HA/cm³)", "yLabel": "Enamel density, mean (mg HA/cm³)",
            "splits": list(splits), "levels": levels, "strata": strata, "tests": list(tests)}


def split_spec():
    rng = np.random.default_rng(2)
    strata = []
    for label, n_wt, n_ko in (("Female, P22", 4, 3), ("Male, P22", 5, 2), ("Female, P30", 2, 4)):
        strata.append({"label": label, "cells": [
            {"level": "WT", "name": "WT", "color": "#2a78d6", "values": [value(f"{label}wt{i}", float(v)) for i, v in enumerate(rng.normal(0.52, 0.02, n_wt))]},
            {"level": "KO", "name": "KO", "color": "#eb6834", "values": [value(f"{label}ko{i}", float(v)) for i, v in enumerate(rng.normal(0.44, 0.02, n_ko))]}]})
    tests = [{"a": "WT", "b": "KO", "within": "Female, P22", "p": 0.004},
             {"a": "WT", "b": "KO", "within": "Male, P22", "p": 0.2},
             {"a": "WT", "b": "KO", "within": "Female, P30", "p": 0.00001}]
    return column_spec(splits=["Sex", "Age"], strata=strata, tests=tests)


def xy_spec(panels=2):
    x = list(range(101))
    curve = lambda a: [[i, None if i < 8 else a + i / 100] for i in x]  # noqa: E731
    return {
        "kind": "xy", "title": "Incisor enamel/dentin ratio along the tooth", "xLabel": "Position along the incisor (%)",
        "yLabel": "Enamel / dentin", "xDomain": [0, 100], "yDomain": [0.55, 1.83], "rules": [{"y": 1, "label": "dentin density"}],
        "legend": [{"label": "WT", "color": "#2a78d6", "slot": 0}, {"label": "KO", "color": "#eb6834", "slot": 1}],
        "panels": [{"title": f"P{22 + 8 * p}", "spans": [{"x0": 30, "x1": 47, "sd0": 2, "sd1": 3, "label": "M1", "tone": 1}],
                    "series": [{"role": "mean", "slot": 0, "color": "#2a78d6", "points": curve(0.7),
                                "band": [[i, None if y is None else y - 0.05, None if y is None else y + 0.05] for i, y in curve(0.7)]},
                               {"role": "mean", "slot": 1, "color": "#eb6834", "points": curve(0.6)}]} for p in range(panels)],
        "x": x,
        "replicates": [{"title": "WT", "specimens": [{"label": f"wt{k}", "values": [None if i < 8 else 0.7 + i / 100 + k / 50 for i in x]} for k in range(3)]},
                       {"title": "KO", "specimens": [{"label": "ko0", "values": [None if i < 20 else 0.6 + i / 100 for i in x]}]}],
    }


# ---------------------------------------------------------------- significance, scales, points

def test_stars_follow_graphpads_thresholds():
    ps = [0.2, 0.05, 0.049, 0.01, 0.0099, 0.001, 0.00099, 0.0001, 0.00001, None]
    assert js("console.log(JSON.stringify(ps.map(F.stars)))", ps=ps) == [
        "ns", "ns", "*", "*", "**", "**", "***", "***", "****", ""]
    assert js("console.log(JSON.stringify([F.pText(0.0321), F.pText(0.00042), F.pText(0.00001)]))") == [
        "p = 0.032", "p = 0.0004", "p < 0.0001"]


def test_an_axis_runs_from_one_tick_to_another_and_holds_the_data():
    rng = np.random.default_rng(5)
    ranges = [sorted(map(float, rng.normal(0, 10 ** rng.uniform(-3, 4), 2))) for _ in range(60)] + [[0, 0], [1900, 1900.5]]
    got = js("console.log(JSON.stringify(ranges.map(([a, b]) => F.niceScale(a, b, 5))))", ranges=ranges)
    for (lo, hi), s in zip(ranges, got):
        assert s["min"] <= lo and s["max"] >= hi
        assert s["ticks"][0] == pytest.approx(s["min"], abs=s["step"] * 1e-6)
        assert s["ticks"][-1] == pytest.approx(s["max"], abs=s["step"] * 1e-6)
        assert 2 <= len(s["ticks"]) <= 12
        mantissa = s["step"] / 10 ** np.floor(np.log10(s["step"]))
        assert round(mantissa, 6) in (1, 2, 5)


def test_tick_labels_carry_as_many_decimals_as_their_step():
    cases = [[1.25, 0.25], [1.5, 0.5], [2000, 500], [-5, 5], [0.3, 0.1], [0.04, 0.02], [-0.0, 0.1]]
    assert js("console.log(JSON.stringify(cases.map(([v, s]) => F.tickLabel(v, s))))", cases=cases) == [
        "1.25", "1.5", "2000", "−5", "0.3", "0.04", "0.0"]


def test_points_are_spread_so_none_overlaps_another():
    rng = np.random.default_rng(3)
    ys = [float(v) for v in rng.normal(100, 6, 14)]
    xs = js("console.log(JSON.stringify(F.swarm(ys, 5, 1000)))", ys=ys)
    pts = np.array([xs, ys]).T
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    assert d[np.triu_indices(len(ys), 1)].min() >= 5 - 1e-6
    # Points with room to themselves stay on the column's centre line.
    assert js("console.log(JSON.stringify(F.swarm([0, 50, 100], 5, 20)))") == [0, 0, 0]


# ---------------------------------------------------------------- figures

def render(spec, **options):
    return js("const f = F.renderFigure(spec, options); console.log(JSON.stringify({svg: f.svg(), w: f.width, h: f.height, layout: f.layout}))",
              spec=spec, options=options)


def test_a_figure_is_sized_in_inches_and_its_text_is_escaped():
    spec = column_spec()
    spec["yLabel"] = "Ratio <enamel> & dentin"
    fig = render(spec, width=3.5, height=2.6)
    root = ET.fromstring(fig["svg"])
    assert root.get("width") == "3.5in" and root.get("height") == "2.6in"
    assert root.get("viewBox") == "0 0 252 187.2"
    texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "Ratio <enamel> & dentin" in texts and "WT" in texts and "KO" in texts


def test_brackets_carry_the_stars_and_sit_over_the_data_inside_the_figure():
    fig = render(split_spec(), width=5, height=2.6)
    labels = [b["label"] for b in fig["layout"]["brackets"]]
    assert sorted(labels) == sorted(["**", "ns", "****"])
    x0, y0, x1, y1 = fig["layout"]["plot"]
    for b in fig["layout"]["brackets"]:
        assert b["top"] >= 0 and b["yb"] < y1
    assert [b["label"] for b in render(split_spec(), ns=False)["layout"]["brackets"]] == ["**", "****"]
    assert render(split_spec(), significance="off")["layout"]["brackets"] == []
    assert sorted(b["label"] for b in render(split_spec(), significance="p")["layout"]["brackets"]) == [
        "p < 0.0001", "p = 0.004", "p = 0.200"]


def test_the_y_axis_ends_at_ticks_and_bars_start_at_zero():
    fig = render(column_spec())
    scale = fig["layout"]["scale"]
    assert scale["min"] > 0 and scale["ticks"][0] == scale["min"] and scale["ticks"][-1] == scale["max"]
    assert render(column_spec(), style="bars")["layout"]["scale"]["min"] == 0
    assert render(column_spec(), zero=True)["layout"]["scale"]["min"] == 0


def test_a_profile_figure_puts_its_panels_side_by_side_on_one_scale():
    fig = render(xy_spec(2), width=7.2, height=2.8)
    plots = fig["layout"]["plots"]
    assert len(plots) == 2
    assert plots[0][1] == plots[1][1] and plots[0][3] == plots[1][3]           # one height
    assert plots[0][2] < plots[1][0]                                           # side by side, apart
    assert all(0 <= p[0] < p[2] <= fig["w"] and 0 <= p[1] < p[3] <= fig["h"] for p in plots)
    assert fig["layout"]["xscale"]["min"] == 0 and fig["layout"]["xscale"]["max"] == 100
    root = ET.fromstring(fig["svg"])
    assert len(root.findall(".//{http://www.w3.org/2000/svg}linearGradient")) == 2   # the landmark's spread, per panel
    assert len(render(xy_spec(4))["layout"]["plots"]) == 4


def svg_texts(fig):
    return {t.text: t for t in ET.fromstring(fig["svg"]).iter("{http://www.w3.org/2000/svg}text")}


def test_a_figure_carries_its_heading_above_the_plot_when_asked():
    plain = render(column_spec())
    spec = {**column_spec(), "heading": "Molar enamel density"}
    titled = render(spec)
    texts = svg_texts(titled)
    assert texts["Molar enamel density"].get("font-weight") == "bold"
    assert titled["layout"]["heading"] == ["Molar enamel density"]
    # The plot moves down to make room, and the brackets stay under the heading.
    assert titled["layout"]["plot"][1] > plain["layout"]["plot"][1]
    assert "Molar enamel density" not in svg_texts(render(spec, title=False))
    split = render({**split_spec(), "heading": "Molar enamel density"}, width=5, height=2.6)
    heading_bottom = float(svg_texts(split)["Molar enamel density"].get("y"))
    assert all(b["top"] >= heading_bottom for b in split["layout"]["brackets"])
    # A long heading wraps onto a second line rather than running off the figure.
    long = render({**column_spec(), "heading": "Molar enamel density, mean of every enamel voxel in the molar row (mg HA/cm³)"}, width=2.4)
    assert len(long["layout"]["heading"]) == 2
    xy = render({**xy_spec(2), "heading": "Mineralization trajectory"}, width=7.2, height=2.8)
    assert xy["layout"]["heading"] == ["Mineralization trajectory"]
    assert xy["layout"]["plots"][0][1] > render(xy_spec(2), width=7.2, height=2.8)["layout"]["plots"][0][1]


def test_every_mark_takes_the_colour_picked_for_it():
    ink = {"mean": "#aa0000", "error": "#00aa00", "brackets": "#0000aa", "axes": "#555555", "text": "#123456",
           "axisTitle": "#654321", "title": "#abcdef", "background": "#fffbe6"}
    fig = render({**split_spec(), "heading": "Density", "ink": ink}, width=5, height=2.6)
    root = ET.fromstring(fig["svg"])
    svg = "{http://www.w3.org/2000/svg}"
    strokes = {el.get("stroke") for el in root.iter() if el.get("stroke")}
    assert {"#aa0000", "#00aa00", "#0000aa", "#555555"} <= strokes and "#000" not in strokes
    texts = svg_texts(fig)
    assert texts["Density"].get("fill") == "#abcdef"
    axis_titles = [t for t in root.iter(f"{svg}text") if "rotate(-90" in (t.get("transform") or "")]
    assert axis_titles and {t.get("fill") for t in axis_titles} == {"#654321"}
    assert texts["**"].get("fill") == "#0000aa" and texts["WT"].get("fill") == "#123456"
    assert root.find(f"{svg}rect").get("fill") == "#fffbe6"
    assert {t.get("fill") for t in root.iter(f"{svg}text")} <= set(ink.values())
    # Unset, a figure prints black on white.
    plain = ET.fromstring(render(split_spec())["svg"])
    assert plain.find(f"{svg}rect").get("fill") == "#ffffff"
    assert {t.get("fill") for t in plain.iter(f"{svg}text")} == {"#000000"}


def test_a_profiles_bands_landmarks_and_reference_line_take_their_picked_colours():
    spec = xy_spec(1)
    spec["panels"][0]["spans"][0]["color"] = "#ff00ff"
    spec["panels"][0]["series"][0]["bandColor"] = "#00ffff"
    spec["ink"] = {"rule": "#999900"}
    root = ET.fromstring(render(spec, width=4.5, height=2.6)["svg"])
    svg = "{http://www.w3.org/2000/svg}"
    assert {s.get("stop-color") for s in root.iter(f"{svg}stop")} == {"#ff00ff"}
    assert any(p.get("fill") == "#00ffff" for p in root.iter(f"{svg}path"))
    assert any(l.get("stroke") == "#999900" for l in root.iter(f"{svg}line"))
    assert svg_texts({"svg": ET.tostring(root, encoding="unicode")})["dentin density"].get("fill") == "#999900"


# ---------------------------------------------------------------- Prism tables


def tables_of(*specs):
    return js("const t = specs.map(P.tableOf); console.log(JSON.stringify({xml: P.pzfx(t, {notes: ['Test export']}), "
              "csv: t.map(P.csv), tsv: t.map(P.tsv), long: P.longCsv(t), tables: t}))", specs=list(specs))


def subcolumns(column):
    return [[d.text for d in sub.findall("d")] for sub in column.findall("Subcolumn")]


def test_an_unsplit_measurement_is_a_column_table_a_group_per_column():
    spec = column_spec()
    out = tables_of(spec)
    root = ET.fromstring(out["xml"])
    table = root.find("Table")
    assert table.get("TableType") == "OneWay" and table.get("XFormat") == "none"
    assert table.find("Title").text == "Enamel density, mean (mg HA/cm³)"
    cols = table.findall("YColumn")
    assert [c.find("Title").text for c in cols] == ["WT", "KO"]
    for col, cell in zip(cols, spec["strata"][0]["cells"]):
        assert [float(v) for v in subcolumns(col)[0]] == pytest.approx([v["value"] for v in cell["values"]], rel=1e-9)
    rows = list(csv.reader(io.StringIO(out["csv"][0])))
    assert rows[0] == ["WT", "KO"] and len(rows) == 7 and rows[6][1] == ""
    assert out["tsv"][0].splitlines()[0] == "WT\tKO"
    # The info sheet names which specimen sits where.
    notes = "".join(root.find("Info/Notes").itertext())
    assert "WT: wt0, wt1" in notes and "Test export" in notes


def test_a_split_measurement_is_a_grouped_table_a_specimen_per_subcolumn():
    spec = split_spec()
    out = tables_of(spec)
    table = ET.fromstring(out["xml"]).find("Table")
    assert table.get("TableType") == "TwoWay" and table.get("YFormat") == "replicates" and table.get("Replicates") == "5"
    assert subcolumns(table.find("RowTitlesColumn")) == [["Female, P22", "Male, P22", "Female, P30"]]
    wt, ko = table.findall("YColumn")
    assert (wt.find("Title").text, ko.find("Title").text) == ("WT", "KO")
    grid = subcolumns(ko)                     # subcolumn j: each stratum's j-th KO specimen
    assert len(grid) == 5 and [g[1] for g in grid[2:]] == [None, None, None]
    male_ko = [v["value"] for v in spec["strata"][1]["cells"][1]["values"]]
    assert [float(grid[j][1]) for j in range(2)] == pytest.approx(male_ko)
    # CSV has no subcolumns: a column per group in each stratum.
    assert next(csv.reader(io.StringIO(out["csv"][0]))) == [
        "WT, Female, P22", "KO, Female, P22", "WT, Male, P22", "KO, Male, P22", "WT, Female, P30", "KO, Female, P30"]
    long = list(csv.DictReader(io.StringIO(out["long"])))
    assert len(long) == 20 and {r["stratum"] for r in long} == {"Female, P22", "Male, P22", "Female, P30"}


def test_a_profile_is_an_xy_table_a_specimen_per_subcolumn():
    out = tables_of(xy_spec())
    table = ET.fromstring(out["xml"]).find("Table")
    assert table.get("TableType") == "XY" and table.get("XFormat") == "numbers" and table.get("Replicates") == "3"
    assert subcolumns(table.find("XColumn"))[0] == [str(i) for i in range(101)]
    wt, ko = table.findall("YColumn")
    assert len(subcolumns(wt)) == 3 and subcolumns(wt)[0][7] is None and float(subcolumns(wt)[2][50]) == pytest.approx(0.7 + 0.5 + 2 / 50)
    assert subcolumns(ko)[1] == [None] * 101                   # KO has one specimen: its other subcolumns are empty
    header = next(csv.reader(io.StringIO(out["csv"][0])))
    assert header[:2] == ["Position along the incisor (%)", "WT: wt0"] and header[-1] == "KO: ko0"


def test_more_replicates_than_prism_keeps_in_a_table_make_a_huge_table():
    spec = column_spec(splits=["Age"], strata=[{"label": "P22", "cells": [
        {"level": "WT", "name": "WT", "color": "#000", "values": [value(f"s{i}", float(i)) for i in range(53)]}]}])
    root = ET.fromstring(tables_of(spec)["xml"])
    assert root.find("Table") is None and root.find("HugeTable").get("Replicates") == "53"


def test_numbers_are_written_as_plain_decimals():
    got = js("console.log(JSON.stringify([P.num(0.00001234), P.num(1e-7), P.num(1234.5678901234), P.num(12), P.num(null), P.num(NaN)]))")
    assert got == ["0.00001234", "0.0000001", "1234.56789", "12", "", ""]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R")
def test_the_r_pzfx_package_reads_every_table(tmp_path):
    probe = subprocess.run(["Rscript", "-e", "cat(requireNamespace('pzfx', quietly=TRUE))"], capture_output=True, text=True)
    if probe.stdout.strip() != "TRUE":
        pytest.skip("needs the R pzfx package")
    path = tmp_path / "t.pzfx"
    path.write_text(tables_of(column_spec(), split_spec(), xy_spec())["xml"], encoding="utf-8")
    script = (f"library(pzfx); f <- '{path}'; t <- pzfx_tables(f); "
              "cat(length(t), dim(read_pzfx(f, 1)), dim(read_pzfx(f, 2)), dim(read_pzfx(f, 3)))")
    done = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["3", "6", "2", "3", "11", "101", "7"]


# ---------------------------------------------------------------- files

def test_a_tiff_holds_the_pixels_and_its_dpi(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    w, hgt = 97, 41
    rgba = rng.integers(0, 256, (hgt, w, 4), dtype=np.uint8)
    rgba[:, :30] = 255                                  # runs, for PackBits to compress
    for alpha, dpi in ((False, 600), (True, 300)):
        path = tmp_path / f"t{alpha}.tif"
        js(f"fs.writeFileSync({json.dumps(str(path))}, B.encodeTiff(Uint8Array.from(px), w, h, {dpi}, {{alpha: {json.dumps(alpha)}}}))",
           px=rgba.ravel().tolist(), w=w, h=hgt)
        img = Image.open(path)
        assert img.info["dpi"] == pytest.approx((dpi, dpi))
        assert (np.array(img) == (rgba if alpha else rgba[..., :3])).all()


def test_a_png_is_given_its_dpi(tmp_path):
    from PIL import Image
    source = tmp_path / "in.png"
    Image.fromarray(np.full((20, 30, 3), 200, np.uint8)).save(source, dpi=(72, 72))
    out = tmp_path / "out.png"
    js(f"fs.writeFileSync({json.dumps(str(out))}, B.pngWithDpi(fs.readFileSync({json.dumps(str(source))}), 1200))")
    img = Image.open(out)
    img.load()
    assert img.info["dpi"] == pytest.approx((1200, 1200), abs=0.01)
    assert out.read_bytes().count(b"pHYs") == 1


def test_a_zip_holds_its_files_under_their_names(tmp_path):
    path = tmp_path / "t.zip"
    js(f"fs.writeFileSync({json.dumps(str(path))}, B.zip([{{name: 'csv/µm.csv', data: 'a,b\\n1,2\\n'}}, {{name: 'figures/x.bin', data: new Uint8Array([0, 1, 255])}}]))")
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert z.read("csv/µm.csv") == b"a,b\n1,2\n" and z.read("figures/x.bin") == bytes([0, 1, 255])


def test_file_names_come_from_titles():
    assert js("console.log(JSON.stringify(['Enamel density, mean (mg HA/cm³)', 'M1 thickness ÷ area (µm/mm²)', '***'].map(B.slug)))") == [
        "enamel_density_mean_mg_ha_cm3", "m1_thickness_area_um_mm2", "figure"]
