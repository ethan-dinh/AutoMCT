"""
Patches of enamel thinner, or less mineralized, than the enamel around them.

Thin enamel is not patchy enamel. A tooth whose enamel is a tenth thinner
everywhere has no patches; one whose enamel is full depth but for a few spots
where it is half as deep has. The thickness median sees the first and barely
moves for the second, and the spread of the thickness sees neither: on a molar
crown the spread is anatomy -- thick cusps, thinner fissure floors, and the
taper to nothing at the cervical margin. On every P22 TGM2 scan, knockout and
wildtype alike, the 5th percentile of the thickness is that taper (13-15 um)
and the median 56-59 um.

So each spot is held up against the enamel around it, on the same tooth.

What is judged
--------------
The points of the enamel-dentin junction where ``enamel_surfaces`` measured
the thickness, one tooth at a time, with two values at each:

- the enamel's thickness there, measured from the junction;
- the density of the column's core: the scan along the line the thickness was
  measured on, read only where it lies clear of both faces by the scan's blur
  (``enamel_surfaces.core_density``). Blur pulls a thin layer's brightness
  down, so without the clearance thin enamel would read as less mineralized.

The density is taken above the background, so a deficit is the share of the
mineral that is missing, as the thickness's is the share of the depth. It is
not taken above the dentin: the dentin's own mineral may change with the
genotype too.

The enamel around a point
-------------------------
A Gaussian-weighted fit of the value over the points near it, sigma
``around_um``. Two things make it a fit and not an average:

- It is local-linear: a plane through the values, read at the point. A local
  average flags the low side of every steady slope -- the taper toward the
  margin, a cusp's flank -- as thinner than its surroundings.
- After the first fit, each point counts in proportion to how little it falls
  short of the fit there (Tukey's biweight of the shortfall, nothing past
  ``weight_cut`` of the call). Only the shortfall: a cusp, thicker than around
  it, keeps its full say. Without it a patch pulls its own reference down: on a
  sheet of 6 um points, a disk 30% thinner and 60-90 um in radius read only
  13-21% thinner at its centre than "around" a plain average (two passes, the
  second without the points the first called), under a 25% call, and was
  never found. With it, disks to 90 um come out at their full depth.

Which points are judged
-----------------------
- Points measured to a contact between teeth are left out of everything:
  there the thickness runs to the cut between the teeth, not to air.
- Points within ``margin_um`` of the enamel's edge -- where it stops on bare
  dentin: the cervical edge, where the root surface runs on, and the rim of any
  bare area of the crown, a cusp tip without enamel -- help to place "around"
  but are not judged: the enamel tapers to nothing there. On
  P22_TGM2_WT_F_1, judged up to the bare cusp tips, nearly every thin patch was
  the ring of tapering enamel around one.
- "Around" must have been read on enough enamel: at least ``coverage`` of the
  area a flat sheet puts under the Gaussian. A straight edge of the measured
  enamel reads 0.5; a fragment of junction alone, far less.

A patch
-------
Judged points at least ``thinner_by`` thinner than around them (and at least
``thinner_by_voxels`` thinner, so thin enamel does not cross the bar on point
noise), or with at least ``less_by`` of their mineral missing, are joined when
they lie within ``link_voxels`` of one another. A patch covers at least
``min_area_mm2``. The bars -- 45% thinner, 12% less mineral -- are where 1% of
the wildtype enamel's judged junction falls (``_CALLS``): a patch is enamel
falling short of its surroundings as only 1% of wildtype enamel does.

Enamel thinner than about two voxels has no junction points: the surfaces need
enamel a couple of voxels deep to be placed. Neither has enamel more than about
a fifth less dense than the rest, which the enamel mask does not count as
enamel. A severe patch is then a ring of thin points around a hole, and the hole
it encloses is counted with it (``hole_mm2``), in the patch's own plane, where
no junction point lies.

What it cannot see
------------------
- Thinning wider than about 0.3 mm: "around" follows it. Patches out to about
  90 um in radius are found at full depth, larger ones partly. The thickness
  median and percentiles, and the core density, carry the broad changes.
- Fissure floors really are thinner than their walls, so wildtype teeth have
  patches too. A knockout is read against its wildtype, never against zero.
- The density can be judged only where the enamel is deep enough to have a
  core: about half the junction on 6 um scans, little on 8 um scans blurred over
  13-15 um.

Run the patch stage again on saved points (``--save-points``) without
measuring anything::

    python enamel_patches.py sweep A_junction_points.npz B_junction_points.npz \\
        --set around_um=40,60,90 --set thinner_by=0.2,0.25 -o sweep.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass, fields, replace

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Params:
    """How patches are found (see the module's description)."""

    around_um: float = 60.0         # sigma of the Gaussian "around" is read over
    smooth_voxels: float = 1.0      # sigma of the light smoothing of point noise, in voxels
    fits: int = 4                   # fits of "around"; every one after the first weighs points by their shortfall
    weight_cut: float = 0.6         # a point this share of the call short of the last fit has no say in the next
    coverage: float = 0.35          # judged only where "around" holds this share of a flat sheet's area
    margin_um: float = 100.0        # not judged this close to the enamel's edge (cervical, or a bare crown)
    min_area_mm2: float = 0.002     # the smallest patch, with the holes it encloses
    link_voxels: float = 1.5        # called points this close are one patch
    thinner_by: float = 0.45        # thin: this share thinner than around ... (see _CALLS)
    thinner_by_voxels: float = 1.5  # ... and at least this many voxels thinner
    less_by: float = 0.12           # less dense: this share of the mineral around missing (see _CALLS)
    core_blurs: float = 2.5         # a column's core lies this many blur sigmas from both faces
    fill_holes: bool = True         # count the holes a patch encloses with it


# _CALLS -- the two bars, set on the six usable P22 TGM2 wildtypes (6 um
# SCANCO; never P22_TGM2_WT_M_5) before any knockout was looked at. At 25%
# thinner and 5% less mineral, 5.2% and 37.8% of their judged junction fell
# past the bars: fissure floors, and the enamel's own density varying about 5%
# (root-mean-square) around the fit while point noise is 1.2-1.4%. By a rule
# fixed beforehand, each bar was raised until 1% of the pooled judged wildtype
# junction fell past it, "around" refitted at each bar: 43.8% thinner and 11.2%
# less mineral, each rounded up to a band of the map, 45% (0.9% past it) and 12%
# (0.4%). Leaving any one wildtype out moved them to 42.8-44.5% and 11.1-11.2%,
# which round the same way, so no wildtype is scored on a bar its own data set:
# 0.6-1.2% of its judged junction lies in thin patches, 0.2-0.5% in less dense.

# The levels at which the share of judged area falling short is also reported,
# so a threshold can be reconsidered without measuring again.
THIN_STEPS = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)
DENSITY_STEPS = (0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15)

# The clearances, in blur sigmas, the saved points keep a column's core density at.
CORE_BLURS = (2.0, 2.5, 3.0)

# Neighbours the light smoothing looks at: at a Gaussian of a voxel, the points
# within three of them.
_SMOOTH_NEIGHBOURS = 48

# Points handled per batch by the smoothing.
_CHUNK = 50_000

# "Around" is read on a grid of this share of its sigma, bounded in cells.
_CELLS_PER_SIGMA = 3.0
_MAX_CELLS = 6_000_000

# The local-linear fit: a ridge, as a share of the weight, on the moments of
# position (along the surface's normal the points have no spread, and the fit
# no slope); the plane is used only where the points spread along the surface
# at least this far (in sigma squared) and "around" holds at least this share
# of a flat sheet. Elsewhere the Gaussian average stands in.
_RIDGE = 0.05
_MIN_SPREAD = 0.1
_MIN_PLANE_COVERAGE = 0.2

# A patch's holes are looked for only when it is flat enough to have a plane:
# its third singular value under this share of its first.
_FLATNESS = 0.35

# A hole is at least this many cells of the patch's plane (each link_voxels on a side).
_MIN_HOLE_CELLS = 3

# The largest patches listed in the summary.
_LISTED = 25


# ---------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------

def point_areas(points_um: np.ndarray, centres_um: np.ndarray, tri_areas_um2: np.ndarray,
                reach_um: float, share: int = 4) -> np.ndarray:
    """The area (um2) each point stands for: every triangle's area shared among its nearest points within reach.

    The triangles are those of the junction found on the scan smoothed by a
    voxel, whose area ``enamel_surfaces.junction_area`` reports; so the areas
    sum to it, less what lies out of reach. Noise bubbles off the junction get
    none. Shared among a few nearest points, not given to one, so no point on
    the surface is left with nothing.
    """
    areas = np.zeros(len(points_um), dtype=np.float64)
    if not len(points_um) or not len(centres_um):
        return areas
    k = int(min(share, len(points_um)))
    d, j = cKDTree(points_um).query(centres_um, k=k, distance_upper_bound=reach_um)
    d, j = d.reshape(len(centres_um), k), j.reshape(len(centres_um), k)
    found = j < len(points_um)
    n = found.sum(axis=1)
    each = np.where(n > 0, tri_areas_um2 / np.maximum(n, 1), 0.0)
    return np.bincount(j[found], weights=np.repeat(each, k).reshape(-1, k)[found], minlength=len(points_um))


def barycentric_areas(points_um: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """A third of each triangle's area to each of its corners; a point in no triangle takes the median."""
    areas = np.zeros(len(points_um), dtype=np.float64)
    if len(faces):
        a, b, c = (points_um[faces[:, i]] for i in range(3))
        tri = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        for i in range(3):
            areas += np.bincount(faces[:, i], weights=tri / 3.0, minlength=len(points_um))
    lone = areas <= 0
    if lone.any() and (~lone).any():
        areas[lone] = float(np.median(areas[~lone]))
    return areas


# ---------------------------------------------------------------------------
# Smoothing, and the value around each point
# ---------------------------------------------------------------------------

def light_smooth(points_um: np.ndarray, values: np.ndarray, weights: np.ndarray, sigma_um: float,
                 k: int = _SMOOTH_NEIGHBOURS) -> np.ndarray:
    """A Gaussian-weighted average of ``values`` over each point's nearest neighbours with weight.

    NaN values and points without weight take no part; a point without
    neighbours within three sigma gets NaN.
    """
    n = len(points_um)
    out = np.full(n, np.nan)
    source = np.isfinite(values) & (weights > 0)
    if not n or not source.any():
        return out
    tree = cKDTree(points_um[source])
    vals, wts = values[source], weights[source]
    k = int(min(k, len(vals)))
    for start in range(0, n, _CHUNK):
        chunk = points_um[start:start + _CHUNK]
        d, j = tree.query(chunk, k=k, distance_upper_bound=3.0 * sigma_um)
        d, j = d.reshape(len(chunk), k), j.reshape(len(chunk), k)
        found = j < len(vals)
        jj = np.where(found, j, 0)
        g = np.where(found, np.exp(-0.5 * (np.where(found, d, 0.0) / sigma_um) ** 2), 0.0) * wts[jj]
        den = g.sum(axis=1)
        num = (g * vals[jj]).sum(axis=1)
        out[start:start + len(chunk)] = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    return out


class _Grid:
    """Gaussian sums over points, read back at the points: splatted to a grid, filtered, sampled.

    Each point's share goes to the eight cells around it (trilinear), the grid
    is filtered with the Gaussian, and the result is read at every point the
    same way. That is the sum of each quantity over the other points weighted by
    the Gaussian of their distance, made a little wider by the two trilinear
    steps (by 2% at three cells to the sigma).
    """

    def __init__(self, points_um: np.ndarray, sigma_um: float):
        self.sigma = float(sigma_um)
        lo, hi = points_um.min(axis=0), points_um.max(axis=0)
        cell = self.sigma / _CELLS_PER_SIGMA
        while True:
            pad = 4.0 * self.sigma + 2.0 * cell
            shape = np.ceil((hi - lo + 2.0 * pad) / cell).astype(np.int64) + 2
            if int(np.prod(shape)) <= _MAX_CELLS:
                break
            cell *= 1.25
        self.cell, self.shape = cell, tuple(int(s) for s in shape)
        self.size = int(np.prod(shape))
        self.at = (points_um - (lo - pad)) / cell                       # fractional cell coordinates
        self.u = (points_um - (lo + hi) / 2.0) / self.sigma             # positions in sigmas, centred
        base = np.floor(self.at).astype(np.int64)
        frac = self.at - base
        idx, wts = [], []
        for corner in itertools.product((0, 1), repeat=3):
            c = np.asarray(corner)
            idx.append(np.ravel_multi_index((base + c).T, self.shape))
            wts.append(np.prod(np.where(c == 1, frac, 1.0 - frac), axis=1))
        self.idx, self.wts = np.concatenate(idx), np.concatenate(wts)
        # What a flat sheet of unit area per unit area adds up to under the
        # filtered grid: across the sheet the Gaussian is widened by the two
        # trilinear steps, each a tent of variance cell**2 / 6.
        across = np.sqrt(self.sigma ** 2 + cell ** 2 / 3.0)
        self.flat = cell ** 3 / (np.sqrt(2.0 * np.pi) * across)

    def sums(self, quantity: np.ndarray) -> np.ndarray:
        grid = np.bincount(self.idx, weights=np.tile(quantity, 8) * self.wts, minlength=self.size)
        grid = ndi.gaussian_filter(grid.reshape(self.shape), self.sigma / self.cell, mode="constant", truncate=4.0)
        return ndi.map_coordinates(grid, self.at.T, order=1, mode="constant", cval=0.0, output=np.float64)

    def fit(self, values: np.ndarray, weights: np.ndarray):
        """At every point: (Gaussian weight, local-linear value, Gaussian average, spread along the surface)."""
        u, w = self.u, weights
        v = np.where(w > 0, values, 0.0)
        s0 = self.sums(w)
        s1 = np.stack([self.sums(w * u[:, i]) for i in range(3)], axis=1)
        s2 = np.empty((len(w), 3, 3))
        for i in range(3):
            for j in range(i, 3):
                s2[:, i, j] = s2[:, j, i] = self.sums(w * u[:, i] * u[:, j])
        t0 = self.sums(w * v)
        t1 = np.stack([self.sums(w * v * u[:, i]) for i in range(3)], axis=1)
        # The same moments about each point's own position.
        m1 = s1 - u * s0[:, None]
        m2 = (s2 - u[:, :, None] * s1[:, None, :] - s1[:, :, None] * u[:, None, :]
              + u[:, :, None] * u[:, None, :] * s0[:, None, None])
        n1 = t1 - u * t0[:, None]
        held = s0 > 1e-12 * max(float(s0.max()), 1e-300)
        average = np.where(held, t0 / np.where(held, s0, 1.0), np.nan)
        spread = np.zeros(len(w))
        linear = np.full(len(w), np.nan)
        if held.any():
            h0 = s0[held]
            cov = m2[held] / h0[:, None, None] - (m1[held] / h0[:, None])[:, :, None] * (m1[held] / h0[:, None])[:, None, :]
            spread[held] = np.linalg.eigvalsh(cov)[:, 1]
            a = np.zeros((int(held.sum()), 4, 4))
            a[:, 0, 0] = h0
            a[:, 0, 1:] = a[:, 1:, 0] = m1[held]
            a[:, 1:, 1:] = m2[held] + _RIDGE * h0[:, None, None] * np.eye(3)
            b = np.concatenate([t0[held][:, None], n1[held]], axis=1)
            linear[held] = np.linalg.solve(a, b[:, :, None])[:, 0, 0]
        return s0, linear, average, spread


def around(points_um: np.ndarray, values: np.ndarray, weights: np.ndarray, sigma_um: float, *,
           smoothed: np.ndarray | None = None, cut: float | None = None,
           fits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """The value around each point, and how much enamel it was read on.

    A Gaussian-weighted local-linear fit of ``values`` over the points with
    ``weights`` (their areas; zero for points that take no part), read at every
    point. After the first fit each point is weighed by Tukey's biweight of its
    shortfall -- how far its ``smoothed`` value falls under the last fit, as a
    share of it -- cut at ``cut``. Returns (reference, coverage): coverage is the
    first fit's weight over a flat sheet's.
    """
    n = len(points_um)
    reference = np.full(n, np.nan)
    coverage = np.zeros(n)
    base = np.where(np.isfinite(values) & (weights > 0), weights, 0.0)
    if not n or not (base > 0).any():
        return reference, coverage
    grid = _Grid(points_um, sigma_um)
    w = base
    for fit in range(max(1, int(fits))):
        s0, linear, average, spread = grid.fit(values, w)
        if fit == 0:
            coverage = s0 / grid.flat
        plane = np.isfinite(linear) & (spread >= _MIN_SPREAD) & (coverage >= _MIN_PLANE_COVERAGE)
        reference = np.where(plane, linear, average)
        if fit == fits - 1 or cut is None or smoothed is None:
            break
        ok = np.isfinite(reference) & (reference > 0) & np.isfinite(smoothed)
        short = np.where(ok, np.clip(1.0 - smoothed / np.where(ok, reference, 1.0), 0.0, None), 0.0)
        w = base * np.where(short < cut, (1.0 - (short / cut) ** 2) ** 2, 0.0)
    return reference, coverage


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------

def _plane_holes(pts: np.ndarray, cell: float, others: cKDTree | None) -> float:
    """The area (um2) of the holes ``pts`` encloses in its own plane where no other junction point lies."""
    if len(pts) < 8:
        return 0.0
    centre = pts.mean(axis=0)
    x = pts - centre
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if len(s) < 3 or s[0] <= 0 or s[2] / s[0] >= _FLATNESS:
        return 0.0
    uv = x @ vt[:2].T
    lo = uv.min(axis=0)
    ij = np.floor((uv - lo) / cell).astype(np.int64) + 2
    occupied = np.zeros(tuple(ij.max(axis=0) + 3), dtype=bool)
    occupied[ij[:, 0], ij[:, 1]] = True
    closed = ndi.binary_closing(occupied, structure=np.ones((3, 3), dtype=bool))
    holes = ndi.binary_fill_holes(closed) & ~closed
    if not holes.any():
        return 0.0
    if others is not None:
        cells = np.argwhere(holes)
        at = centre + (((cells - 2) + 0.5) * cell + lo) @ vt[:2]
        # A junction point at the cell -- within its size, or off the plane by
        # as much as the patch itself bends -- means enamel stands there.
        bend = 3.0 * s[2] / np.sqrt(len(pts))
        d, _ = others.query(at, distance_upper_bound=max(cell, bend))
        holes[tuple(cells[np.isfinite(d)].T)] = False
    labels, n = ndi.label(holes)
    if not n:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes[sizes >= _MIN_HOLE_CELLS].sum()) * cell * cell


def find_patches(points_um: np.ndarray, areas_um2: np.ndarray, deficit: np.ndarray, called: np.ndarray, *,
                 link_um: float, min_area_um2: float, fill_holes: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Called points joined into patches.

    Returns (each point's patch, -1 for none, numbered from the largest; one
    record per patch, largest first). A patch's area counts the holes it
    encloses where no junction point lies; it is kept when that is at least
    ``min_area_um2``.
    """
    patch = np.full(len(points_um), -1, dtype=np.int32)
    idx = np.flatnonzero(called)
    if not idx.size:
        return patch, []
    pts = points_um[idx]
    pairs = cKDTree(pts).query_pairs(link_um, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(idx.size, idx.size))
    n, comp = connected_components(graph, directed=False)
    own = np.bincount(comp, weights=areas_um2[idx], minlength=n)
    count = np.bincount(comp, minlength=n)
    # A patch too small even with holes, which need a ring of points around them, is never looked at.
    candidates = np.flatnonzero((own >= 0.25 * min_area_um2) | (count >= 12))
    others = None
    if fill_holes and candidates.size:
        rest = ~called
        others = cKDTree(points_um[rest]) if rest.any() else None
    kept = []
    order = np.argsort(comp, kind="stable")
    starts = np.searchsorted(comp[order], np.arange(n + 1))
    for c in candidates:
        members = idx[order[starts[c]:starts[c + 1]]]
        hole = _plane_holes(points_um[members], link_um, others) if fill_holes else 0.0
        area = float(areas_um2[members].sum())
        if area + hole < min_area_um2:
            continue
        kept.append((area + hole, members, hole))
    kept.sort(key=lambda k: -k[0])
    records = []
    for number, (total, members, hole) in enumerate(kept):
        patch[members] = number
        a = areas_um2[members]
        d = deficit[members]
        w = a if a.sum() > 0 else np.ones_like(a)
        p = points_um[members]
        centre = (p * w[:, None]).sum(axis=0) / w.sum()
        _, _, vt = np.linalg.svd(p - centre, full_matrices=False)
        extent = np.ptp((p - centre) @ vt[:2].T, axis=0) if len(p) > 1 else np.zeros(2)
        records.append({
            "area_um2": total, "hole_um2": hole, "own_um2": float(a.sum()),
            "depth": float(np.sum(w * d) / w.sum()), "deepest": float(np.max(d)),
            "length_um": float(extent[0]), "width_um": float(extent[1]),
            "centre_um": centre, "n_points": int(len(members)),
        })
    return patch, records


# ---------------------------------------------------------------------------
# One tooth, and the row
# ---------------------------------------------------------------------------

@dataclass
class Found:
    """One tooth's patches of one kind, and what the summary is built from."""

    smoothed: np.ndarray    # the value after the light smoothing
    reference: np.ndarray   # the value around each point
    deficit: np.ndarray     # 1 - smoothed / reference where judged, NaN elsewhere
    patch: np.ndarray       # each point's patch (-1: none)
    judged: np.ndarray
    sums: dict
    records: list


def _empty_sums(steps) -> dict:
    return {"total": 0.0, "judged": 0.0, "in_patches": 0.0, "holes": 0.0, "count": 0, "largest": 0.0,
            "depth_num": 0.0, "depth_den": 0.0, "depth_abs_num": 0.0, "missing_num": 0.0, "missing_den": 0.0,
            "short_num": 0.0, "var_num": 0.0, "noise_num": 0.0, "noise_den": 0.0,
            "at_least": np.zeros(len(steps)), "steps": tuple(steps), "value_num": 0.0, "value_den": 0.0}


def tooth_patches(points_um: np.ndarray, areas_um2: np.ndarray, values: np.ndarray, usable: np.ndarray,
                  eligible: np.ndarray, *, by: float, by_abs: float, params: Params, voxel_um: float,
                  steps=THIN_STEPS) -> Found:
    """One tooth's patches of one kind.

    ``values`` are the thickness (um) or the core density above the
    background (raw units) at each point; ``usable`` the points that place
    "around", ``eligible`` those that may be judged. A point is called where it
    falls at least ``by`` (a share) and ``by_abs`` (the value's units) short of
    the value around it.
    """
    n = len(points_um)
    weights = np.where(usable, areas_um2, 0.0)
    smoothed = light_smooth(points_um, values, weights, params.smooth_voxels * voxel_um)
    reference, coverage = around(points_um, values, weights, params.around_um, smoothed=smoothed,
                                 cut=params.weight_cut * by, fits=params.fits)
    judged = (eligible & (coverage >= params.coverage) & np.isfinite(reference) & (reference > 0)
              & np.isfinite(smoothed))
    deficit = np.full(n, np.nan)
    deficit[judged] = 1.0 - smoothed[judged] / reference[judged]
    called = judged & (deficit >= by) & (reference - smoothed >= by_abs)
    patch, records = find_patches(points_um, areas_um2, np.nan_to_num(deficit), called,
                                  link_um=params.link_voxels * voxel_um,
                                  min_area_um2=params.min_area_mm2 * 1e6, fill_holes=params.fill_holes)

    s = _empty_sums(steps)
    a = areas_um2
    j = judged
    inside = patch >= 0
    d = deficit
    s["total"] = float(a.sum())
    s["judged"] = float(a[j].sum())
    s["in_patches"] = float(a[inside].sum())
    s["holes"] = float(sum(r["hole_um2"] for r in records))
    s["count"] = len(records)
    s["largest"] = float(records[0]["area_um2"]) if records else 0.0
    s["depth_num"] = float(np.sum(a[inside] * d[inside]))
    s["depth_den"] = float(a[inside].sum())
    s["depth_abs_num"] = float(np.sum(a[inside] * (reference[inside] - smoothed[inside])))
    s["missing_num"] = s["depth_abs_num"]
    s["missing_den"] = float(np.sum(a[j] * reference[j]))
    s["short_num"] = float(np.sum(a[j] * np.clip(d[j], 0.0, None) ** 2))
    s["var_num"] = float(np.sum(a[j] * d[j] ** 2))
    raw = j & np.isfinite(values)
    s["noise_num"] = float(np.sum(a[raw] * ((values[raw] - smoothed[raw]) / reference[raw]) ** 2))
    s["noise_den"] = float(a[raw].sum())
    s["at_least"] = np.array([float(a[j & (d >= t)].sum()) for t in steps]) if j.any() else np.zeros(len(steps))
    held = usable & np.isfinite(values)
    s["value_num"] = float(np.sum(a[held] * values[held]))
    s["value_den"] = float(a[held].sum())
    return Found(smoothed, reference, deficit, patch, judged, s, records)


def _ratio(num: float, den: float):
    return float(num / den) if den > 0 else None


def numbers(s: dict) -> dict:
    """The reported numbers from a tooth's sums, or the row's."""
    judged = s["judged"] + s["holes"]
    has = s["judged"] > 0
    return {
        "share_judged": _ratio(s["judged"], s["total"]),
        "judged_mm2": s["judged"] / 1e6,
        "share": _ratio(s["in_patches"] + s["holes"], judged),
        "patches": int(s["count"]),
        "patches_per_mm2": _ratio(s["count"], judged / 1e6) if has else None,
        "depth": _ratio(s["depth_num"], s["depth_den"]),
        "largest_mm2": s["largest"] / 1e6 if has else None,
        "hole_mm2": s["holes"] / 1e6,
        "missing": _ratio(s["missing_num"], s["missing_den"]),
        "shortfall": float(np.sqrt(s["short_num"] / s["judged"])) if has else None,
        "variation": float(np.sqrt(s["var_num"] / s["judged"])) if has else None,
        "noise": float(np.sqrt(s["noise_num"] / s["noise_den"])) if s["noise_den"] > 0 else None,
        "at_least": {"deficit": list(s["steps"]),
                     "share": [_ratio(float(v), s["judged"]) for v in s["at_least"]]},
    }


def combine(sums: list[dict]) -> dict:
    """The row's sums from its teeth's."""
    out = _empty_sums(sums[0]["steps"] if sums else THIN_STEPS)
    for s in sums:
        for key, value in s.items():
            if key == "steps":
                continue
            if key == "largest":
                out[key] = max(out[key], value)
            else:
                out[key] = out[key] + value
    return out


# ---------------------------------------------------------------------------
# Both kinds, on a molar row's junction
# ---------------------------------------------------------------------------

def _fill_specks(points_um: np.ndarray, values: np.ndarray, flags: np.ndarray,
                 reach_um: float) -> tuple[np.ndarray, np.ndarray]:
    """For the map only: an unjudged point among judged ones takes their mean, and their patch if most are in one.

    About one junction point in twelve has no thickness (no straight line
    through enamel reached the outer surface from it), scattered through the
    enamel; the outer surface, painted from the junction point each of its
    points was measured to, would show them as a grey speckle. A point
    counts as among judged ones with at least three of them within reach;
    the unjudged bands along the enamel's edges and contacts stay grey. The
    numbers never see these values.
    """
    hole = ~np.isfinite(values)
    if not hole.any() or hole.all():
        return values, flags
    ok = ~hole
    d, j = cKDTree(points_um[ok]).query(points_um[hole], k=6, distance_upper_bound=reach_um)
    found = j < int(ok.sum())
    count = found.sum(axis=1)
    take = count >= 3
    jj = np.where(found, j, 0)
    mean = np.where(found, values[ok][jj], 0.0).sum(axis=1) / np.maximum(count, 1)
    share = np.where(found, flags[ok][jj].astype(np.float64), 0.0).sum(axis=1) / np.maximum(count, 1)
    out_values, out_flags = values.copy(), flags.copy()
    idx = np.flatnonzero(hole)[take]
    out_values[idx] = mean[take]
    out_flags[idx] = share[take] >= 0.5
    return out_values, out_flags


def _clean(value):
    """JSON-ready: numpy scalars to Python, NaN and infinity to None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def measure_patches(points_vox: np.ndarray, voxel_um, *, areas_um2: np.ndarray, tooth: np.ndarray | None,
                    names: dict[int, str] | None, thickness_um: np.ndarray, to_contact: np.ndarray,
                    edge_um: np.ndarray, core_raw: np.ndarray | None, background: float,
                    edge_blur_um: float | None, clear_um: float | None, params: Params = Params(),
                    origin=(0, 0, 0), triangles: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[dict, dict]:
    """Thin and less-dense patches on a molar row's junction, per tooth and for the row.

    ``points_vox`` are the junction's points in the frame's voxels, ``origin``
    where that frame starts in the input volume (the patches' centres are given
    in the input's voxels). ``edge_um`` is each point's distance to the
    enamel's edge (``enamel_surfaces.bare_edges``). ``core_raw`` is each
    column's core density (raw,
    NaN where the column has no core), ``background`` the scan's level outside
    the tooth (and, for the map only, ``<kind>_shown``/``<kind>_patch_shown``:
    the same with the scattered unmeasured points among judged ones filled in
    from their neighbours, ``_fill_specks``). ``triangles`` (centres in the
    frame's voxels, areas in um2) is
    the junction found on the scan smoothed by a voxel: each kind shares its
    area among the points that kind can use, so a point it cannot judge holds
    none. Where dim enamel brings its junction level close to the noise, the
    raw junction doubles back into the enamel there: on a phantom column
    missing a fifth of its mineral, half its junction points lay inside the
    enamel with no core to read, and with areas shared among all points the
    column's patch came out half its size. Without ``triangles``,
    ``areas_um2``. Returns (the JSON-ready block, arrays per point for the
    map: each kind's deficit and whether the point lies in a counted patch).
    """
    voxel_um = np.asarray(voxel_um, dtype=np.float64)
    size = float(voxel_um.mean())
    pts = np.asarray(points_vox, dtype=np.float64) * voxel_um
    n = len(pts)
    groups = ([(int(label), name) for label, name in sorted((names or {}).items(), key=lambda kv: kv[1])]
              if tooth is not None and names else [(None, None)])
    measured = np.isfinite(thickness_um) & ~to_contact
    eligible_place = edge_um >= params.margin_um
    kinds = {
        "thin": dict(values=np.asarray(thickness_um, dtype=np.float64), usable=measured,
                     by=params.thinner_by, by_abs=params.thinner_by_voxels * size, steps=THIN_STEPS),
    }
    if core_raw is not None:
        above = np.asarray(core_raw, dtype=np.float64) - float(background)
        kinds["low_density"] = dict(values=above, usable=measured & np.isfinite(above) & (above > 0),
                                    by=params.less_by, by_abs=0.0, steps=DENSITY_STEPS)
    reach = 3.0 * size
    centres_um = None
    if triangles is not None and len(triangles[0]):
        centres_um = np.asarray(triangles[0], dtype=np.float64) * voxel_um
        tri_um2 = np.asarray(triangles[1], dtype=np.float64)
    block = {"method": "each point against the enamel around it, on the same tooth (local-linear, "
                       "one-sided weights)",
             "around_um": params.around_um, "smallest_patch_mm2": params.min_area_mm2,
             "margin_um": params.margin_um, "coverage": params.coverage}
    shades = {}
    for kind, spec in kinds.items():
        deficit = np.full(n, np.nan)
        in_patch = np.zeros(n, dtype=bool)
        tooth_sums, teeth, listed = [], [], []
        # The junction's area, shared among the points this kind can use.
        mine = np.zeros(n)
        if centres_um is not None and spec["usable"].any():
            mine[spec["usable"]] = point_areas(pts[spec["usable"]], centres_um, tri_um2, reach)
        else:
            mine = np.where(spec["usable"], areas_um2, 0.0)
        for label, name in groups:
            sel = np.ones(n, dtype=bool) if label is None else (tooth == label)
            if not sel.any():
                continue
            found = tooth_patches(pts[sel], mine[sel], spec["values"][sel], spec["usable"][sel] & (mine[sel] > 0),
                                  spec["usable"][sel] & (mine[sel] > 0) & eligible_place[sel], by=spec["by"],
                                  by_abs=spec["by_abs"], params=params, voxel_um=size, steps=spec["steps"])
            # Its share judged is of the whole junction there, every point's area.
            found.sums["total"] = float(areas_um2[sel].sum())
            deficit[sel] = found.deficit
            in_patch[sel] = found.patch >= 0
            tooth_sums.append(found.sums)
            if name is not None:
                teeth.append({"name": name, **_kind_numbers(kind, found.sums, background)})
            for r in found.records:
                listed.append({"tooth": name, "area_mm2": r["area_um2"] / 1e6, "hole_mm2": r["hole_um2"] / 1e6,
                               "depth": r["depth"], "deepest": r["deepest"], "length_um": r["length_um"],
                               "width_um": r["width_um"],
                               "centre_zyx": [float(v) for v in r["centre_um"] / voxel_um + np.asarray(origin)],
                               "n_points": r["n_points"]})
        listed.sort(key=lambda r: -r["area_mm2"])
        row = combine(tooth_sums) if tooth_sums else _empty_sums(spec["steps"])
        entry = {**_kind_numbers(kind, row, background), "teeth": teeth, "largest": listed[:_LISTED]}
        if kind == "thin":
            entry = {"thinner_by": params.thinner_by, "thinner_by_um": params.thinner_by_voxels * size, **entry}
        else:
            entry = {"less_by": params.less_by, "clear_um": clear_um, "edge_blur_um": edge_blur_um,
                     "background": float(background), **entry}
        block[kind] = entry
        shades[kind] = deficit
        shades[f"{kind}_patch"] = in_patch
        shades[f"{kind}_shown"], shades[f"{kind}_patch_shown"] = _fill_specks(pts, deficit, in_patch, 1.5 * size)
    if "low_density" not in block:
        block["low_density"] = None
    return _clean(block), shades


def _kind_numbers(kind: str, sums: dict, background: float = 0.0) -> dict:
    out = numbers(sums)
    if kind == "low_density":
        # In the scan's units: the patches' mean depth (a difference, so the
        # background drops out) and the mean core density over the measured
        # junction. The caller converts both when the scan is calibrated.
        out["depth_raw"] = _ratio(sums["depth_abs_num"], sums["depth_den"])
        mean_above = _ratio(sums["value_num"], sums["value_den"])
        out["core_density_raw"] = None if mean_above is None else mean_above + float(background)
    else:
        out["depth_um"] = _ratio(sums["depth_abs_num"], sums["depth_den"])
    return out


# ---------------------------------------------------------------------------
# Saved points, and the patch stage again on them
# ---------------------------------------------------------------------------

def core_key(blurs: float) -> str:
    return f"core_{blurs:.1f}".replace(".", "_")


def patches_from_points(saved, params: Params = Params()) -> tuple[dict, dict]:
    """The patch stage on points saved with ``--save-points`` (an open npz or a dict of its arrays)."""
    key = core_key(params.core_blurs)
    if key not in saved:
        raise ValueError(f"The saved points hold the core density at {', '.join(str(b) for b in CORE_BLURS)} "
                         f"blur sigmas, not {params.core_blurs}")
    names = json.loads(str(saved["names"])) if "names" in saved else {}
    names = {int(k): v for k, v in names.items()}
    tooth = saved["tooth"] if names else None
    edge = float(saved["edge_blur_um"]) if np.isfinite(saved["edge_blur_um"]) else None
    clear = float(saved[f"clear_{key}"]) if f"clear_{key}" in saved else None
    triangles = (saved["tri_centre"], saved["tri_area_um2"]) if "tri_centre" in saved else None
    return measure_patches(saved["verts"], saved["voxel_um"], areas_um2=saved["area_um2"], tooth=tooth, names=names,
                           thickness_um=saved["thickness_um"], to_contact=saved["to_contact"].astype(bool),
                           edge_um=saved["edge_um"], core_raw=saved[key],
                           background=float(saved["background"]), edge_blur_um=edge, clear_um=clear,
                           params=params, origin=saved["origin"], triangles=triangles)


def _parse_sets(sets: list[str]) -> list[dict]:
    """``key=v1,v2`` settings to every combination of them, as Params changes."""
    names = {f.name: f.type for f in fields(Params)}
    axes = []
    for item in sets:
        key, _, values = item.partition("=")
        key = key.strip()
        if key not in names:
            raise SystemExit(f"Unknown setting {key!r}; known: {', '.join(names)}")
        kind = type(getattr(Params(), key))
        parse = (lambda s: s.strip().lower() in ("1", "true", "yes")) if kind is bool else kind
        axes.append([(key, parse(v)) for v in values.split(",") if v.strip()])
    return [dict(combo) for combo in itertools.product(*axes)] if axes else [{}]


def sweep(paths: list[str], changes: list[dict]) -> list[dict]:
    """The patch stage on each saved file under each set of changes to the defaults."""
    out = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            saved = {k: data[k] for k in data.files}
        sample = os.path.basename(path).replace("_junction_points.npz", "")
        for change in changes:
            block, _ = patches_from_points(saved, replace(Params(), **change))
            out.append({"sample": sample, "set": change, "patches": block})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the enamel patch stage again on saved junction points.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("sweep", help="the patch stage on saved points, under settings changed from the defaults")
    run.add_argument("points", nargs="+", help="<sample>_junction_points.npz files (molar_enamel.py --save-points)")
    run.add_argument("--set", action="append", default=[], metavar="KEY=V1,V2",
                     help="a setting and the values to try; every combination is run")
    run.add_argument("-o", "--output", help="write every result here as JSON")
    args = parser.parse_args(argv)
    results = sweep(args.points, _parse_sets(args.set))
    for r in results:
        thin = r["patches"]["thin"] or {}
        low = r["patches"]["low_density"] or {}
        by_tooth = {t["name"]: t for t in thin.get("teeth", [])}
        low_tooth = {t["name"]: t for t in low.get("teeth", [])}

        def pct(v):
            return "  -  " if v is None else f"{100 * v:5.2f}"
        print(f"{r['sample']:28s} {json.dumps(r['set']):40s} thin {pct(thin.get('share'))} "
              f"(M1 {pct((by_tooth.get('M1') or {}).get('share'))})  less dense {pct(low.get('share'))} "
              f"(M1 {pct((low_tooth.get('M1') or {}).get('share'))})")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
