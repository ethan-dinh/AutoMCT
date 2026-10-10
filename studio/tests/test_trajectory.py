"""
The incisor profile's cleanup, held to the R report it is ported from.

studio/trajectory.py ports the report's quality filters, its outlier rejection
against each specimen's own GAM trend, and its gap-aware running median. The
first test runs the report's R code -- copied from
longitudinal_incisor_analysis/R/longitudinal_analysis.rmd -- and the port on
the same synthetic profiles and asks for the same slices kept, the same
outliers, the same medians and the same crossover and plateau. The rest pin
each rule on its own, and hold the page's running median (web/js/trajectory.js,
run under Node) to the server's.

Run with:  python -m pytest studio/tests/test_trajectory.py
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio import trajectory as T  # noqa: E402

MODULE = (Path(__file__).resolve().parents[1] / "web" / "js" / "trajectory.js").as_uri()
NODE = shutil.which("node")
RSCRIPT = shutil.which("Rscript")


def synthetic(seed: int, n: int = 1100, tip_low: bool = False, top: float = 1.8) -> list[dict]:
    """A profile shaped like enamel_density.py's: a sigmoid rise, gaps, a coincident band, planted mode flips."""
    rng = np.random.default_rng(seed)
    z = np.arange(n)
    zp = 100 * z / (n - 1)
    if tip_low:
        zp = zp[::-1].copy()
    onset = rng.uniform(10, 18)
    f = np.clip((zp - onset) / (90 - onset), 0, 1)
    ratio = 0.62 + (top - 0.62) / (1 + np.exp(-10 * (f - 0.35))) + 0.01 * np.sin(zp / 7) + rng.normal(0, 0.0015, n)
    status = np.where(ratio > 1.03, "resolved_hi", np.where(ratio < 0.97, "resolved_lo", "coincident")).astype(object)
    status[zp < onset] = "pre_onset"
    ratio[zp < onset] = 0.0
    for _ in range(rng.integers(2, 6)):
        a = rng.integers(0, n - 40)
        status[a:a + rng.integers(3, 35)] = "unresolvable"
    status[zp > 97] = "saturated"
    measured = np.flatnonzero(np.isin(status, T.MEASURABLE))
    for i in rng.choice(measured, 6, replace=False):
        ratio[i] += rng.choice([-1, 1]) * rng.uniform(0.1, 0.6)
    for _ in range(2):
        a = rng.choice(measured[:-10])
        ratio[a:a + rng.integers(2, 7)] -= rng.uniform(0.09, 0.4)
    for i in rng.choice(measured, 4, replace=False):   # near the threshold, to test its edge
        ratio[i] += rng.choice([-1, 1]) * rng.uniform(0.06, 0.1)
    z_window = rng.integers(0, 5, n).astype(float)
    z_window[rng.choice(n, n // 80, replace=False)] = 15
    solidity = rng.uniform(0.2, 0.6, n)
    lo = np.flatnonzero(status == "resolved_lo")
    solidity[rng.choice(lo, max(1, lo.size // 20), replace=False)] = 0.04
    density = ratio * 13000 + rng.normal(0, 15, n)
    rows = []
    for i in range(n):
        empty = status[i] == "unresolvable"
        rows.append({"z": int(z[i]), "z_percent": round(float(zp[i]), 6), "status": status[i],
                     "enamel_ratio": "" if empty else round(float(ratio[i]), 6),
                     "enamel_density": "" if empty else round(float(0 if status[i] == "pre_onset" else density[i]), 3),
                     "z_window": int(z_window[i]), "solidity": round(float(solidity[i]), 4)})
    return rows


def as_profile(rows: list[dict]) -> dict:
    num = lambda v: float(v) if v not in ("", None) else None  # noqa: E731
    return {"z": [int(r["z"]) for r in rows], "z_percent": [num(r["z_percent"]) for r in rows],
            "status": [r["status"] for r in rows], "ratio": [num(r["enamel_ratio"]) for r in rows],
            "density": [num(r["enamel_density"]) for r in rows], "z_window": [num(r["z_window"]) for r in rows],
            "solidity": [num(r["solidity"]) for r in rows]}


# ---------------------------------------------------------------- against R

# Sections 2, 4, 5 and 6 of the report, as written there, run per specimen.
REPORT = r"""
suppressMessages({library(dplyr); library(readr); library(mgcv); library(zoo); library(tibble)})
args <- commandArgs(TRUE)
params <- list(median_k = 21, gam_k = 40, max_z_window = 12, outlier_abs = 0.08)
profiles <- read_csv(args[1], show_col_types = FALSE)
measured <- profiles |>
  filter(
    status %in% c("resolved_hi", "resolved_lo", "coincident"),
    z_window <= params$max_z_window,
    !(status == "resolved_lo" & (is.na(solidity) | solidity < 0.10))
  ) |>
  filter(!is.na(enamel_ratio))
fit_gam <- function(data, yvar = "enamel_ratio") {
  bam(reformulate(sprintf("s(z_percent, bs = 'tp', k = %d)", params$gam_k),
                  response = yvar),
      data = data, discrete = TRUE)
}
flag_outliers <- function(data, threshold = params$outlier_abs, max_iter = 10) {
  keep  <- rep(TRUE, nrow(data))
  resid <- rep(NA_real_, nrow(data))
  for (i in seq_len(max_iter)) {
    m     <- fit_gam(data[keep, , drop = FALSE])
    resid <- data$enamel_ratio - as.numeric(predict(m, newdata = data))
    new   <- abs(resid) <= threshold
    if (identical(new, keep)) break
    keep <- new
  }
  tibble(is_outlier = !keep, residual = resid)
}
scored <- measured |>
  group_by(sample) |>
  group_modify(~ bind_cols(.x, flag_outliers(.x))) |>
  ungroup()
clean <- scored |> filter(!is_outlier)
roll_med_gapped <- function(x, z, k) {
  if (length(x) < k) return(rep(NA_real_, length(x)))
  out <- zoo::rollapply(x, width = k, FUN = median, na.rm = TRUE,
                        fill = NA_real_, align = "center", partial = FALSE)
  zspan <- zoo::rollapply(z, width = k, FUN = function(v) max(v) - min(v),
                          fill = NA_real_, align = "center", partial = FALSE)
  out[is.na(zspan) | zspan > (k - 1) * 1.5] <- NA_real_
  out
}
smoothed <- clean |>
  arrange(sample, z) |>
  group_by(sample) |>
  mutate(
    ratio_med   = roll_med_gapped(enamel_ratio,   z, params$median_k),
    density_med = roll_med_gapped(enamel_density, z, params$median_k)
  ) |>
  ungroup()
crossing_of <- function(z, fit) {
  above <- fit >= 1
  idx   <- which(diff(above) == 1)
  if (length(idx) == 0) return(NA_real_)
  i <- idx[length(idx)]
  for (j in rev(idx)) {
    if (all(above[(j + 1):length(above)])) i <- j
  }
  z[i] + (1 - fit[i]) * (z[i + 1] - z[i]) / (fit[i + 1] - fit[i])
}
plateau_of <- function(z, fit) mean(fit[z >= 70 & z <= 95], na.rm = TRUE)
marks <- smoothed |> filter(!is.na(ratio_med)) |> arrange(sample, z_percent) |> group_by(sample) |>
  summarise(crossover = crossing_of(z_percent, ratio_med), plateau = plateau_of(z_percent, ratio_med), .groups = "drop")
coverage <- profiles |>
  mutate(measurable = status %in% c("resolved_hi", "resolved_lo", "coincident"),
         bin = cut(z_percent, breaks = seq(0, 100, 10), labels = FALSE, include.lowest = TRUE)) |>
  group_by(sample, bin) |> summarise(pct = 100 * mean(measurable), .groups = "drop")
write_csv(scored |> select(sample, z, is_outlier, residual) |>
            left_join(smoothed |> select(sample, z, ratio_med, density_med), by = c("sample", "z")),
          args[2], na = "")
write_csv(marks, args[3], na = "")
write_csv(coverage, args[4], na = "")
"""


def r_ready() -> bool:
    if RSCRIPT is None:
        return False
    probe = subprocess.run([RSCRIPT, "-e", "suppressMessages({library(dplyr); library(readr); library(mgcv); "
                                         "library(zoo); library(tibble)})"], capture_output=True, timeout=120)
    return probe.returncode == 0


@pytest.mark.skipif(not r_ready(), reason="needs R with mgcv, zoo, dplyr, readr and tibble to run the report's code")
def test_the_cleanup_keeps_and_smooths_what_the_r_report_does(tmp_path):
    # Past 1000 kept slices bam() discretizes the covariate, which the port must follow.
    cases = {"A": synthetic(1, 1100), "B": synthetic(2, 900, tip_low=True, top=1.2), "C": synthetic(4, 1400, tip_low=True)}
    with open(tmp_path / "profiles.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", *cases["A"][0]])
        writer.writeheader()
        for name, rows in cases.items():
            writer.writerows({"sample": name, **r} for r in rows)
    (tmp_path / "report.R").write_text(REPORT)
    done = subprocess.run([RSCRIPT, str(tmp_path / "report.R"), *(str(tmp_path / f) for f in
                           ("profiles.csv", "slices.csv", "marks.csv", "coverage.csv"))],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    r_slices = {(r["sample"], int(r["z"])): r for r in csv.DictReader(open(tmp_path / "slices.csv"))}
    r_marks = {r["sample"]: r for r in csv.DictReader(open(tmp_path / "marks.csv"))}
    r_cover = {(r["sample"], int(r["bin"])): float(r["pct"]) for r in csv.DictReader(open(tmp_path / "coverage.csv"))}
    num = lambda v: float(v) if v != "" else None  # noqa: E731

    for name, rows in cases.items():
        prof = as_profile(rows)
        got = T.clean(prof)
        scored = [i for i, z in enumerate(prof["z"]) if (name, z) in r_slices]
        # The same slices pass the quality filters...
        assert set(scored) == {i for i, (k, o) in enumerate(zip(got["kept"], got["outlier"])) if k or o}, name
        assert sum(got["outlier"]) >= 8, "the planted mode flips are there to be found"
        for i in scored:
            r = r_slices[(name, prof["z"][i])]
            # ...the same are outliers, against the same trend...
            assert got["outlier"][i] == (r["is_outlier"] == "TRUE"), (name, prof["z"][i])
            assert got["residual"][i] == pytest.approx(num(r["residual"]), abs=1e-6)
            # ...and the kept ones have the same running medians, empty in the same places.
            for key in ("ratio_med", "density_med"):
                want = num(r[key])
                assert (got[key][i] is None) == (want is None), (name, key, prof["z"][i])
                if want is not None:
                    assert got[key][i] == pytest.approx(want, abs=1e-9)
        pts = sorted((z, v) for z, v in zip(prof["z_percent"], got["ratio_med"]) if v is not None)
        zs, fit = [p[0] for p in pts], [p[1] for p in pts]
        assert T.crossing_of(zs, fit) == pytest.approx(float(r_marks[name]["crossover"]), abs=1e-9)
        assert T.plateau_of(zs, fit) == pytest.approx(float(r_marks[name]["plateau"]), abs=1e-9)
        assert [c for c in got["coverage"] if c is not None] == pytest.approx(
            [r_cover[(name, b)] for b in range(1, 11) if (name, b) in r_cover])


# ---------------------------------------------------------------- each rule


def test_only_resolved_or_coincident_slices_that_pass_the_filters_are_measured():
    status = ["resolved_hi", "resolved_lo", "resolved_lo", "coincident", "saturated", "pre_onset", "resolved_hi", "resolved_hi"]
    ratio = [1.5, 0.8, 0.8, 1.0, 1.9, 0.0, 1.4, 0.0]
    z_window = [2, 2, 2, 2, 2, 2, 13, 2]
    solidity = [0.0, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert T.quality(status, ratio, z_window, solidity).tolist() == [
        True, True, False, True, False, False, False, False]
    # A profile written without the two columns filters on the status alone.
    assert T.quality(status, ratio).tolist() == [True, True, True, True, False, False, True, False]


def sigmoid(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    zp = np.linspace(0, 100, n)
    return zp, 0.6 + 1.2 / (1 + np.exp(-(zp - 40) / 6)) + rng.normal(0, 0.0015, n)


def test_a_fit_on_the_wrong_mode_is_left_out_and_a_clean_curve_is_untouched():
    zp, y = sigmoid()
    flags, resid = T.flag_outliers(zp, y)
    assert not flags.any() and np.nanmax(np.abs(resid)) < 0.02
    bad = y.copy()
    bad[[120, 480, 777]] += [0.3, -0.5, 0.12]
    bad[600:606] -= 0.25                           # a short run on the dentin mode
    flags, _ = T.flag_outliers(zp, bad)
    assert np.flatnonzero(flags).tolist() == [120, 480, *range(600, 606), 777]


def test_the_running_median_leaves_the_ends_and_gaps_empty():
    z = np.concatenate([np.arange(0, 60), np.arange(72, 130)])      # 12 slices missing
    x = np.arange(z.size, dtype=float)
    med = T.rolling_median_gapped(x, z, 21)
    assert np.isnan(med[:10]).all() and np.isnan(med[-10:]).all(), "no partial windows at the ends"
    assert med[10] == 10.0 and med[40] == 40.0
    # A window reaching across the gap spans more than 30 slices, so it is dropped.
    assert np.isnan(med[55]) and np.isnan(med[64])
    assert med[80] == 80.0
    # Three slices missing are within the window's slack and bridged.
    z3 = np.concatenate([np.arange(0, 50), np.arange(53, 100)])
    assert not np.isnan(T.rolling_median_gapped(np.ones(z3.size), z3, 21)[45])
    # Every other slice measured: 21 slices are 11 points, the gap allowance a step of 2.
    assert T.median_points(2.0) == 11
    assert not np.isnan(T.rolling_median_gapped(np.ones(50), np.arange(0, 100, 2), 11, step=2.0)[20])


def test_the_crossover_is_the_last_rise_through_one():
    z = [20, 25, 30, 35, 40, 45, 50]
    wiggle = [0.8, 0.95, 1.05, 0.98, 1.02, 1.3, 1.6]
    assert T.crossing_of(z, wiggle) == pytest.approx(35 + (1 - 0.98) * 5 / (1.02 - 0.98))
    assert T.crossing_of(z, [0.8, 0.9, 0.95, 0.97, 0.99, 0.99, 0.98]) is None
    assert T.plateau_of([60, 70, 80, 95, 99], [1.0, 1.7, 1.8, 1.9, 5.0]) == pytest.approx(1.8)


def test_coverage_counts_tenths_as_r_cuts_them():
    z = [0.0, 10.0, 10.5, 55.0, 100.0]
    status = ["resolved_hi", "unresolvable", "coincident", "pre_onset", "resolved_lo"]
    assert T.coverage(z, status) == [50.0, 100.0, None, None, None, 0.0, None, None, None, 100.0]


def test_a_profile_from_before_the_onset_trim_is_marked_stale():
    zp, y = sigmoid(200)
    prof = {"z": list(range(200)), "z_percent": zp.tolist(), "ratio": y.tolist(), "density": (y * 13000).tolist(),
            "status": ["resolved_hi"] * 200}
    assert not T.clean(prof)["stale"]
    prof["status"][:5] = ["pre_onset"] * 5
    assert T.clean(prof)["stale"], "a pre-enamel slice with a ratio is an older build's"
    prof["ratio"][:5] = [0.0] * 5
    assert not T.clean(prof)["stale"]


# ---------------------------------------------------------------- the page's median


def js(expression: str, **inputs):
    """The JSON value of `expression`, evaluated with trajectory.js imported as J and each input bound."""
    bindings = "".join(f"const {k} = {json.dumps(v)};\n" for k, v in inputs.items())
    code = f"import * as J from {json.dumps(MODULE)};\n{bindings}console.log(JSON.stringify({expression}));"
    done = subprocess.run([NODE, "--input-type=module", "-e", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.skipif(NODE is None, reason="needs Node to run the page's own code")
def test_the_page_smooths_the_kept_slices_as_the_server_does():
    rows = synthetic(3, 800)
    prof = as_profile(rows)
    got = T.clean(prof)
    kept = [[prof["z_percent"][i], prof["ratio"][i], prof["z"][i]] for i in range(len(rows)) if got["kept"][i]]
    curve = js("J.smoothProfile(points, 21)", points=kept)
    server = sorted((prof["z_percent"][i], got["ratio_med"][i]) for i in range(len(rows)) if got["kept"][i])
    assert len(curve) == len(server) and sum(v is None for _, v in server) > 20, "the ends and gaps are empty"
    for (z_page, v_page), (z_server, v_server) in zip(curve, server):
        assert z_page == z_server and (v_page is None) == (v_server is None)
        if v_server is not None:
            assert v_page == pytest.approx(v_server)
    # Every other slice measured, with a gap: the page's window and gap rule match the server's.
    z = [*range(0, 120, 2), *range(150, 260, 2)]
    x = [float(np.sin(v / 9)) for v in z]
    want = T.rolling_median_gapped(np.array(x), np.array(z), T.median_points(2.0), step=2.0)
    page = js("J.gappedMedian(x, z, J.medianWindow(J.sliceStep(z), 21), J.sliceStep(z))", x=x, z=z)
    assert [None if np.isnan(v) else pytest.approx(v) for v in want] == page


@pytest.mark.skipif(NODE is None, reason="needs Node to run the page's own code")
def test_a_group_curve_starts_where_its_specimens_measure_half_their_slices():
    coverages = [[0, 20, 60, 90] + [100] * 6, [None, 40, 30, 95] + [100] * 6]
    assert js("J.coverageFloor(c)", c=coverages) == 30, "the second tenth averages 30%, the third 45%"
    assert js("J.coverageFloor(c)", c=[[10] * 10]) is None
    # A curve's empty stretch is never bridged.
    pts = [[0, 1.0], [10, None], [20, 3.0]]
    assert js("[J.valueAt(p, 0), J.valueAt(p, 5), J.valueAt(p, 15), J.valueAt(p, 20)]", p=pts) == [1.0, None, None, 3.0]
