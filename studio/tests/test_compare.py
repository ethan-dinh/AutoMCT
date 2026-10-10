"""
The results page's grouping and statistics, run under Node.

The page groups specimens and runs its Welch tests in the browser, so the
numbers it shows come from web/js/compare.js. These tests run that module
under Node and hold it to SciPy: the same t, degrees of freedom and p-value
for the same data. They also pin which comparisons a grouping makes -- the
main comparison inside each stratum and pooled over the splits, never pooled
over age -- and the order levels are shown in.

Run with:  python -m pytest studio/tests/test_compare.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

MODULE = (Path(__file__).resolve().parents[1] / "web" / "js" / "compare.js").as_uri()
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node to run the page's own code")


def js(expression: str, **inputs):
    """The JSON value of `expression`, evaluated with compare.js imported as C and each input bound."""
    bindings = "".join(f"const {k} = {json.dumps(v)};\n" for k, v in inputs.items())
    code = f"import * as C from {json.dumps(MODULE)};\n{bindings}console.log(JSON.stringify({expression}));"
    done = subprocess.run([NODE, "--input-type=module", "-e", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.filterwarnings("ignore:Precision loss")  # SciPy, on a side with no spread
def test_welch_matches_scipy():
    rng = np.random.default_rng(7)
    cases = [([1, 2, 3, 4], [2, 3, 5, 9, 11]), ([0.51, 0.49, 0.55], [0.40, 0.42]),
             ([1930.2, 1904.8, 1951.0, 1899.3], [1702.5, 1811.9, 1750.1]),
             ([5, 5, 5], [1, 2, 3]), ([10, 11], [10.5, 10.6, 10.4, 10.55, 10.45])]
    cases += [(list(rng.normal(0, 1, int(rng.integers(2, 12)))), list(rng.normal(rng.uniform(-2, 2), 3, int(rng.integers(2, 12)))))
              for _ in range(40)]
    got = js("cases.map(([a, b]) => C.welch(a, b))", cases=cases)
    for (a, b), res in zip(cases, got):
        want = stats.ttest_ind(a, b, equal_var=False)
        assert res["t"] == pytest.approx(want.statistic, rel=1e-9)
        assert res["df"] == pytest.approx(want.df, rel=1e-9)
        assert res["p"] == pytest.approx(want.pvalue, rel=1e-8, abs=1e-14)


def test_the_t_distribution_holds_in_the_tails_and_for_large_samples():
    grid = [(t, df) for t in (0.0, 0.3, 1.0, 2.2, 4.0, 9.0, 25.0) for df in (1.0, 1.7, 3.2, 10.0, 48.5, 400.0)]
    got = js("grid.map(([t, df]) => C.tTwoSided(t, df))", grid=grid)
    for (t, df), p in zip(grid, got):
        assert p == pytest.approx(2 * stats.t.sf(t, df), rel=1e-8, abs=1e-300)


def test_welch_needs_two_values_a_side_and_some_spread():
    assert js("[C.welch([1], [1, 2, 3]), C.welch([2, 2], [3, 3]), C.welch([], [])]") == [None, None, None]


def test_levels_are_shown_control_first_and_ages_in_developmental_order():
    order = js("C.levelOrder(samples)", samples=[{"meta": m} for m in (
        {"genotype": "KO", "age": "P100", "sex": "M"}, {"genotype": "Het", "age": "P22"},
        {"genotype": "WT", "age": "E18.5", "sex": "F"}, {"genotype": "cKO", "age": "P3"}, {"age": "8wk"})])
    assert order["genotype"] == ["WT", "Het", "cKO", "KO", ""]
    assert order["age"] == ["E18.5", "P3", "P22", "P100", "8wk"]
    assert order["sex"] == ["F", "M", ""]
    assert js("[C.levelName('sex', 'F'), C.levelName('sex', ''), C.levelName('age', 'P22')]") == [
        "Female", "Sex not set", "P22"]


COHORT = [
    ("WT_F1", "WT", "F", "P22", 1.50), ("WT_F2", "WT", "F", "P22", 1.56), ("WT_F3", "WT", "F", "P22", 1.47),
    ("WT_M1", "WT", "M", "P22", 1.61), ("WT_M2", "WT", "M", "P22", 1.66),
    ("KO_F1", "KO", "F", "P22", 1.21), ("KO_F2", "KO", "F", "P22", 1.30),
    ("KO_M1", "KO", "M", "P22", 1.35), ("KO_M2", "KO", "M", "P22", 1.28), ("KO_M3", "KO", "M", "P22", 1.33),
    ("WT_F4", "WT", "F", "P30", 1.70), ("WT_F5", "WT", "F", "P30", 1.74),
    ("KO_F4", "KO", "F", "P30", 1.41), ("KO_F5", "KO", "F", "P30", 1.52),
    ("KO_X1", "KO", "", "P30", 1.44),
]


def cohort():
    samples = [{"name": n, "meta": {"genotype": g, "sex": s, "age": a}} for n, g, s, a, _ in COHORT]
    values = {n: v for n, *_, v in COHORT}
    return samples, values


def run_comparisons(by):
    samples, values = cohort()
    return js("C.comparisons(samples, by, C.levelOrder(samples), (n) => values[n] ?? null)",
              samples=samples, by=by, values=values)


def test_knockout_against_wildtype_within_each_sex_and_age_then_with_sexes_pooled():
    rows = run_comparisons(["genotype", "sex", "age"])
    main = [r for r in rows if r["factor"] == "genotype"]
    # Pooled over sex but never over age, then each stratum.
    assert [(r["within"], r["na"], r["nb"]) for r in main] == [
        ("P22, sexes pooled", 5, 5), ("P30, sexes pooled", 2, 3),
        ("Female, P22", 3, 2), ("Female, P30", 2, 2), ("Male, P22", 2, 3)]
    assert all((r["a"], r["b"]) == ("WT", "KO") for r in main)
    by_name = {n: v for n, *_, v in COHORT}
    wt = [by_name[n] for n in ("WT_F1", "WT_F2", "WT_F3", "WT_M1", "WT_M2")]
    ko = [by_name[n] for n in ("KO_F1", "KO_F2", "KO_M1", "KO_M2", "KO_M3")]
    want = stats.ttest_ind(ko, wt, equal_var=False)
    assert main[0]["t"] == pytest.approx(want.statistic) and main[0]["p"] == pytest.approx(want.pvalue)
    # A specimen whose sex is not set is pooled with its age, but never makes a sex of its own.
    sexes = [r for r in rows if r["factor"] == "sex"]
    assert {(r["a"], r["b"]) for r in sexes} == {("Female", "Male")}
    assert [r["within"] for r in sexes] == ["WT, P22", "KO, P22"]
    ages = [r for r in rows if r["factor"] == "age"]
    assert [r["within"] for r in ages] == ["WT, Female", "KO, Female"]


def test_one_factor_compares_its_levels_and_pools_nothing():
    rows = run_comparisons(["genotype"])
    assert [(r["a"], r["b"], r["within"], r["pooled"]) for r in rows] == [("WT", "KO", "", False)]


def test_a_split_by_age_alone_repeats_the_comparison_per_age():
    rows = run_comparisons(["genotype", "age"])
    assert [(r["factor"], r["within"]) for r in rows] == [
        ("genotype", "P22"), ("genotype", "P30"), ("age", "WT"), ("age", "KO")]


def test_specimens_are_arranged_in_strata_of_the_splits():
    samples, _ = cohort()
    strata = js("C.arrange(samples, ['genotype', 'sex'], C.levelOrder(samples))", samples=samples)
    assert [s["label"] for s in strata] == ["Female", "Male", "Sex not set"]
    assert [[c["name"] for c in s["cells"]] for s in strata] == [["WT", "KO"], ["WT", "KO"], ["KO"]]
    assert strata[0]["cells"][1]["members"] == ["KO_F1", "KO_F2", "KO_F4", "KO_F5"]


def test_a_flag_leaves_a_specimen_out_of_the_analyses_it_names():
    got = js("[C.excludedFrom(null, 'molar'), C.excludedFrom({from: 'all'}, 'incisor'), "
             "C.excludedFrom({from: ['incisor']}, 'molar'), C.excludedFrom({from: ['incisor']}, 'incisor'), "
             "C.exclusionText({from: 'all'}), C.exclusionText({from: ['incisor', 'molar']}), "
             "C.exclusionText({from: ['segmentation']})]")
    assert got == [False, True, False, True, "all results", "Molar enamel and Incisor mineralization",
                   "Segmentation volumes"]
