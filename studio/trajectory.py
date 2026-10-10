"""
The R report's cleanup of one incisor profile, so the studio draws and
measures the trajectory the report does.

longitudinal_incisor_analysis/R/longitudinal_analysis.rmd cleans each
specimen's enamel/dentin profile before anything is averaged. This module ports
those steps with the report's own parameters:

1. Quality filtering (the report's section 2). Only a slice whose fit resolved
   enamel from dentin, or found the two coincident, carries a density:
   `resolved_hi`, `resolved_lo` and `coincident`, the last kept at its measured
   ratio (the report's "measured" rule, section 3). A slice pooled over more
   than 12 slices either side is dropped, and so is a `resolved_lo` slice whose
   low component is not solid enough (0.10) to be secretory matrix rather than
   partial-volume rim.
2. Outlier rejection (section 4). A slice more than 0.08 ratio units from its
   own specimen's trend is a fit that landed on the wrong mode. The trend is
   the report's fit_gam(): mgcv's thin-plate regression spline with 40 basis
   functions, fitted by REML with bam()'s discretization, and refitted on the
   kept slices until the kept set stops changing.
3. Smoothing (section 6). A 21-slice running median over the kept slices, left
   empty where the window would reach past either end or straddle a gap, so no
   value is made up where nothing was measured.

It also gives each specimen's measured share of each tenth of the tooth
(section 5), from which the results page keeps a group's curve off the
cervical stretch its specimens rarely measure, and the report's crossover and
plateau of a curve.

The report's group trend and bootstrap bands summarise a group rather than
clean a specimen; the results page averages the cleaned curves instead.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

# The report's params and constants.
MEASURABLE = ("resolved_hi", "resolved_lo", "coincident")
MAX_Z_WINDOW = 12          # params$max_z_window
MIN_SOLIDITY = 0.10        # resolved_lo floor: genuine matrix scores >= 0.10, rim 0.00-0.01
OUTLIER_ABS = 0.08         # params$outlier_abs, ratio units
GAM_K = 40                 # params$gam_k
MAX_ITER = 10              # flag_outliers(max_iter)
MEDIAN_K = 21              # params$median_k, slices
PLATEAU = (70.0, 95.0)     # plateau_of(): the mature stretch, % along the tooth
COVERAGE_FLOOR = 50.0      # % of slices measured before a group's curve is drawn
# bam(discrete = TRUE) rounds a covariate with more unique values than this onto an even grid.
DISCRETE_VALUES = 1000


def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


# ---------------------------------------------------------------- 1. quality


def quality(status, ratio, z_window=None, solidity=None) -> np.ndarray:
    """Which slices carry a measured ratio: the report's section 2 filters.

    `z_window` and `solidity` are per-slice lists, or None for a profile
    written without the column, which then filters on nothing.
    """
    # A ratio of exactly 0 is the pre-enamel zero, never a measurement.
    keep = np.array([s in MEASURABLE and _finite(r) and r != 0 for s, r in zip(status, ratio)], dtype=bool)
    if z_window is not None:
        keep &= np.array([_finite(w) and w <= MAX_Z_WINDOW for w in z_window], dtype=bool)
    if solidity is not None:
        lo = np.array([s == "resolved_lo" for s in status], dtype=bool)
        solid = np.array([_finite(v) and v >= MIN_SOLIDITY for v in solidity], dtype=bool)
        keep &= ~lo | solid
    return keep


# ---------------------------------------------------------------- 2. the trend


def _discretize(x: np.ndarray) -> np.ndarray:
    """mgcv's compress.df for one covariate: past 1000 unique values, each goes to the nearest of 1000 even steps."""
    if np.unique(x).size <= DISCRETE_VALUES:
        return x
    lo, hi = x.min(), x.max()
    grid = np.linspace(lo, hi, DISCRETE_VALUES)
    return grid[np.round((x - lo) / (grid[1] - grid[0])).astype(int)]


class ThinPlate:
    """mgcv's s(x, bs = "tp", k) for one covariate, fitted by REML (Wood 2003).

    The thin-plate spline's radial basis over the unique covariate values is cut
    to its k largest eigenvectors, with the penalty's null space (a constant and
    a slope) constrained out of them and added back as unpenalized columns.
    """

    def __init__(self, x: np.ndarray, k: int = GAM_K):
        self.shift = float(x.min())
        self.scale = float(x.max() - x.min()) or 1.0
        knots = np.unique(self._u(x))
        n = knots.size
        radial = np.abs(knots[:, None] - knots[None, :]) ** 3 / 12.0
        k = min(k, n)
        try:
            # mgcv finds the k largest-magnitude eigenvectors by Lanczos iteration too.
            vals, vecs = eigsh(radial, k=k, which="LM") if k < n else np.linalg.eigh(radial)
        except ArpackNoConvergence:
            vals, vecs = np.linalg.eigh(radial)
        order = np.argsort(-np.abs(vals))[:k]
        vals, vecs = vals[order], vecs[:, order]
        null = np.column_stack([np.ones(n), knots])
        q, _ = np.linalg.qr(vecs.T @ null, mode="complete")
        z = q[:, 2:]
        self.knots = knots
        self.basis = vecs @ z
        wiggly = z.T @ (vals[:, None] * z)
        wiggly = (wiggly + wiggly.T) / 2
        self.penalty = np.zeros((k, k))
        self.penalty[:k - 2, :k - 2] = wiggly
        self.beta = None

    def _u(self, x: np.ndarray) -> np.ndarray:
        # Rescaling the covariate changes the basis's conditioning, not the fitted curve.
        return (np.asarray(x, dtype=float) - self.shift) / self.scale

    def design(self, x: np.ndarray) -> np.ndarray:
        u = self._u(x)
        radial = np.abs(u[:, None] - self.knots[None, :]) ** 3 / 12.0
        return np.column_stack([radial @ self.basis, np.ones(u.size), u])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ThinPlate":
        """Penalized least squares, the smoothing parameter by REML with the scale profiled out."""
        X = self.design(x)
        y = np.asarray(y, dtype=float)
        xtx, xty, yty = X.T @ X, X.T @ y, float(y @ y)
        n, p = X.shape
        free = n - 2                      # residual dimension of the penalty's null space
        rank = p - 2
        s = self.penalty * (np.linalg.norm(xtx) / (np.linalg.norm(self.penalty) or 1.0))

        def solve(rho: float):
            a = xtx + math.exp(rho) * s
            try:
                c = cho_factor(a, lower=False, check_finite=False)
            except LinAlgError:
                return None
            return c, cho_solve(c, xty, check_finite=False)

        def reml(rho: float) -> float:
            got = solve(rho)
            if got is None:
                return math.inf
            c, beta = got
            rss = max(yty - 2 * beta @ xty + beta @ xtx @ beta, 0.0)
            phi = (rss + math.exp(rho) * beta @ s @ beta) / free
            if not phi > 0:
                return math.inf
            return 0.5 * free * math.log(phi) + np.log(np.diag(c[0])).sum() - 0.5 * rank * rho

        grid = np.arange(-30.0, 31.0, 1.0)
        scores = np.array([reml(r) for r in grid])
        best = int(np.argmin(scores))
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
        rho = minimize_scalar(reml, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6}).x
        if reml(rho) > scores[best]:
            rho = grid[best]
        got = solve(rho)
        # Only a design without full column rank fails every factorization.
        self.beta = got[1] if got is not None else np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.design(x) @ self.beta


def trend(z_percent, ratio, at=None, k: int = GAM_K) -> np.ndarray:
    """The report's fit_gam(): bam(ratio ~ s(z_percent, bs = "tp", k), discrete = TRUE), predicted at `at`.

    As predict.bam() does for a discrete fit, the points predicted at are
    discretized too, onto a grid over their own range.
    """
    x = _discretize(np.asarray(z_percent, dtype=float))
    model = ThinPlate(x, k).fit(x, np.asarray(ratio, dtype=float))
    return model.predict(_discretize(np.asarray(z_percent if at is None else at, dtype=float)))


def flag_outliers(z_percent, ratio, threshold: float = OUTLIER_ABS, max_iter: int = MAX_ITER,
                  k: int = GAM_K) -> tuple[np.ndarray, np.ndarray]:
    """(is_outlier, residual) for each slice, as the report's flag_outliers() finds them.

    The trend is refitted on the kept slices and every slice rescored against
    it until the kept set stops changing, so a gross outlier cannot drag the
    trend toward itself and mask its neighbours. A profile with no more slices
    than the trend has basis functions is not scored.
    """
    z = np.asarray(z_percent, dtype=float)
    y = np.asarray(ratio, dtype=float)
    keep = np.ones(z.size, dtype=bool)
    resid = np.full(z.size, np.nan)
    if np.unique(z).size <= k:
        return ~keep, resid
    for _ in range(max_iter):
        if np.unique(z[keep]).size <= k:
            break
        resid = y - trend(z[keep], y[keep], at=z, k=k)
        new = np.abs(resid) <= threshold
        if np.array_equal(new, keep):
            break
        keep = new
    return ~keep, resid


# ---------------------------------------------------------------- 3. smoothing


def rolling_median_gapped(x, z, k: int = MEDIAN_K, step: float = 1.0) -> np.ndarray:
    """The report's roll_med_gapped(): a centred k-point running median, NaN where it cannot be trusted.

    Empty at either end, where the window would be partial, and wherever the
    window's slices span more than 1.5 steps per point: a window straddling a
    run of missing slices is dropped rather than bridged. `step` is the slice
    spacing, 1 unless the profile was measured every Nth slice.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    out = np.full(x.size, np.nan)
    if x.size < k or k < 1:
        return out
    half = k // 2
    for i in range(half, x.size - (k - 1 - half)):
        lo, hi = i - half, i - half + k
        if z[lo:hi].max() - z[lo:hi].min() > (k - 1) * 1.5 * step:
            continue
        window = x[lo:hi]
        window = window[np.isfinite(window)]
        if window.size:
            out[i] = np.median(window)
    return out


def slice_step(z) -> float:
    """The slice spacing: the typical gap between consecutive measured slices."""
    gaps = np.diff(np.sort(np.asarray(z, dtype=float)))
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if gaps.size else 1.0


def median_points(step: float, slices: int = MEDIAN_K) -> int:
    """The running median's width in points: 21 slices, fewer points when every Nth slice was measured."""
    k = max(1, round(slices / step))
    return k if k % 2 else k + 1


# ---------------------------------------------------------------- landmarks


def crossing_of(z, fit) -> float | None:
    """Where a curve crosses ratio 1 for the last time, rising, interpolated: the report's crossing_of().

    Not the first time it touches 1: a wiggle in the steep crossover can nick
    1.0 and fall back. None when the curve never rises through 1.
    """
    pts = [(a, b) for a, b in zip(z, fit) if _finite(a) and _finite(b)]
    if len(pts) < 2:
        return None
    zs = np.array([p[0] for p in pts])
    fs = np.array([p[1] for p in pts])
    above = fs >= 1
    idx = np.flatnonzero(np.diff(above.astype(int)) == 1)
    if not idx.size:
        return None
    i = int(idx[-1])
    for j in idx[::-1]:
        if above[j + 1:].all():
            i = int(j)
    return float(zs[i] + (1 - fs[i]) * (zs[i + 1] - zs[i]) / (fs[i + 1] - fs[i]))


def plateau_of(z, fit) -> float | None:
    """The mean of a curve over the mature stretch, 70-95% along the tooth: the report's plateau_of()."""
    vals = [b for a, b in zip(z, fit) if _finite(a) and _finite(b) and PLATEAU[0] <= a <= PLATEAU[1]]
    return float(np.mean(vals)) if vals else None


def coverage(z_percent, status) -> list[float | None]:
    """The % of a specimen's slices measured in each tenth of the tooth: the report's section 5.

    Tenths are (0, 10], (10, 20], ... with 0 in the first, as R's cut() makes
    them. None for a tenth the profile has no slice in.
    """
    edges = np.arange(0.0, 101.0, 10.0)
    total = [0] * 10
    measured = [0] * 10
    for z, s in zip(z_percent, status):
        if not _finite(z):
            continue
        b = min(max(int(np.searchsorted(edges, z, side="left")) - 1, 0), 9)
        total[b] += 1
        measured[b] += s in MEASURABLE
    return [100.0 * m / t if t else None for m, t in zip(measured, total)]


# ---------------------------------------------------------------- one profile


def clean(profile: dict) -> dict:
    """The cleanup of one loaded profile, slice by slice, in the profile's own order.

      kept        the slice is measured and passes the quality filters and outlier rejection
      outlier     it passed the quality filters but sits more than 0.08 from its trend
      residual    its ratio less the trend's, for every slice the filters kept
      ratio_med   the 21-slice running median of the kept ratios, None where empty
      density_med the same of the kept enamel densities
      coverage    % of slices measured in each tenth of the tooth
      stale       pre-enamel slices carry a ratio: written by a build of
                  enamel_density.py from before the enamel-onset change
    """
    n = len(profile["status"])
    status, ratio = profile["status"], profile["ratio"]
    qc = quality(status, ratio, profile.get("z_window"), profile.get("solidity"))
    zp = np.array([v if _finite(v) else np.nan for v in profile["z_percent"]], dtype=float)
    qc &= np.isfinite(zp)
    y = np.array([v if _finite(v) else np.nan for v in ratio], dtype=float)
    outlier = np.zeros(n, dtype=bool)
    residual = np.full(n, np.nan)
    rows = np.flatnonzero(qc)
    if rows.size:
        flags, resid = flag_outliers(zp[rows], y[rows])
        outlier[rows] = flags
        residual[rows] = resid
    kept = qc & ~outlier

    # The running median runs along the slices in scan order, over the kept ones only.
    zs = profile.get("z") or [None] * n
    order = [i for i in np.flatnonzero(kept) if zs[i] is not None]
    order.sort(key=lambda i: zs[i])
    ratio_med = [None] * n
    density_med = [None] * n
    if order:
        zk = np.array([zs[i] for i in order], dtype=float)
        step = slice_step(zk)
        k = median_points(step)
        dens = np.array([profile["density"][i] if _finite(profile["density"][i]) else np.nan for i in order])
        for target, values in ((ratio_med, y[order]), (density_med, dens)):
            med = rolling_median_gapped(values, zk, k, step)
            for i, v in zip(order, med):
                target[i] = float(v) if math.isfinite(v) else None

    stale = any(s == "pre_onset" and _finite(r) and r != 0 for s, r in zip(status, ratio))
    return {
        "kept": kept.tolist(),
        "outlier": outlier.tolist(),
        "residual": [float(v) if math.isfinite(v) else None for v in residual],
        "ratio_med": ratio_med,
        "density_med": density_med,
        "coverage": coverage(profile["z_percent"], status),
        "stale": stale,
    }
