"""
Enamel thickness between the enamel's two surfaces, measured in 3-D.

The enamel is a layer with two faces: the outer surface, where it meets the
air or soft tissue around the tooth, and the enamel-dentin junction, where it
rests on the dentin. Each face is placed where the scan's intensity is halfway
between the materials on its two sides -- enamel and background outside,
enamel and dentin inside -- which is where a blurred step edge truly is.

Why two levels
--------------
A single threshold puts the two faces at different fractions of their edges.
The enamel threshold sits between dentin and enamel, right for the junction;
but the outer edge falls from enamel all the way to background, and its
halfway point is far lower. The blurred voxels there read like dentin, and a
mask cut at the enamel threshold loses them. On WT-M-1 (6 um voxels) that dim
outer rim measured as pure blur -- the enamel-background edge is no wider
than the enamel-dentin edge -- and the halfway surfaces put the outer face
about a voxel outside the enamel mask.

Levels read where each face is
------------------------------
Halfway needs the intensities on either side, and on a scan they are not
constants. Blur flattens a thin layer's peak: on P82_Odam-Cre_WT_2 (SkyScan,
8 um voxels, edges blurred over sigma 13-15 um) the enamel peaks at 175 where
it is 30-45 um deep and at 220 where it is 90-120 um. And the scan dims along
the jaw: under M3 the enamel peaks at 210 over dentin at 134, under M1 at 181
over 120. So each level is read where its face is (``read_levels``). Every
voxel follows the intensity uphill to the enamel's local peak, and the dentin
under the enamel is read just past the junction, pooled along the jaw. The
outer surface lies halfway between that peak and the background, the junction
halfway between it and the dentin under it.

The enamel itself
-----------------
``enamel_layer`` gives the enamel as voxels: the molar's voxels between its
two faces. Each is put on a face the way the surface points are, by what lies
a few voxels downhill from it, and is enamel if it is brighter than that
face's level. One threshold cannot do this. Set where the enamel is bright,
it drops thin or dimmed enamel whole: on P82_Odam-Cre_WT_2 the histogram's
threshold sat at 188, M1's median local peak, and missed 45% of M1's crown.

Which face is which
-------------------
Every point on the enamel's boundary is labelled by what lies beyond it, read
a few voxels out along the direction the intensity falls: background means
the outer surface, dentin means the junction. The very next voxel would not
do: one voxel out, the blurred outer edge still reads like dentin.

Thickness
---------
At each point of the junction: the distance to the nearest point of the outer
surface that a straight line through enamel reaches -- the radius of a ball
grown from that point until it touches the outer surface. On a folded crown
the nearest outer point can lie across dentin or a groove; such a pairing is
refused and the next nearest tried (on WT-M-1, 19% of the raw nearest
pairings). The reported thickness is measured from the junction: its shape
does not change with wear, and it is the usual reference in enamel studies.
The same measurement from the outer surface -- the depth of enamel beneath
each point of the tooth's surface -- is what the map paints on the outer
surface. On a curved crown the two are not symmetric (WT-M-1: 62 vs 57 um
median), so every number says which it is.

Where teeth touch
-----------------
Two touching molars have no outer surface between them: the enamel of one runs
into the enamel of the next. There the cut between the teeth (their labels,
``molar_enamel.split_teeth``) stands in for the outer surface of each, so the
enamel under a contact is measured to the contact, not along the fused enamel
to wherever it next meets the air.

Patches
-------
Each junction point's thickness, and the density of the enamel's core above
it (``core_density``), are held up against the enamel around them on the same
tooth by ``enamel_patches``: how much of the junction lies under enamel thinner,
or less mineralized, than its surroundings. The core is the line the thickness
was measured along, read only where it lies clear of both faces by the scan's
blur (``edge_blur``), so a thin column does not read dim. The enamel's edges,
where it tapers to nothing -- the cervical edge, found from the bare root
surface beside it, and any bare area of the crown (``bare_edges``) -- are left
unjudged.

The measurement needs the unmasked scan around the tooth. The molar volume is
zero outside its mask, so neither the background level nor the lower half of
the outer edge is in it; without the scan the background is taken as zero,
and a warning says so.
"""

from __future__ import annotations

import base64
import datetime
import gzip
import html
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

import enamel_patches  # this directory, importable as the script's own

logger = logging.getLogger(__name__)

# How far past a surface point its far side is read, in voxels: beyond the
# blur (sigma 1.2-1.5 voxels on WT-M-1), so the reading is the material there,
# not the edge.
_LOOK_OUT_VOXELS = 4.0

# How far into the enamel its near side is read, confirming it is enamel.
# Short, so thin enamel near the cervical margin still qualifies.
_LOOK_IN_VOXELS = 2.0

# Surfaces are looked for only this close to the enamel mask, in voxels
# (chessboard distance).
_NEAR_ENAMEL_VOXELS = 4

# The background level is the median of the scan in this shell outside the
# enamel and outside the tooth, in voxels: past the blurred edge, before the
# next structure.
_BACKGROUND_SHELL = (4, 8)

# Nearest outer points tried per junction point when the nearest cannot be
# reached through enamel.
_CANDIDATES = 64

# Samples along each line tested for running through enamel, and how close to
# its ends (voxels) a sample may dip below the junction level: the ends sit on
# the blurred faces themselves.
_SIGHT_SAMPLES = 40
_SIGHT_END_VOXELS = 1.5

# Lines tested per batch; bounds the memory of the sampled points.
_CHUNK = 50_000

# The viewer's two surfaces together stay under this many triangles; the
# marching-cubes step for display is raised until they do.
_DISPLAY_TRIANGLES = 600_000

# Tooth labels are widened by this many voxels before the surface points look
# them up: the outer surface lies just outside the molar mask.
_TOOTH_REACH = 3

# Smoothing, in voxels, of the scan the levels are read on. An 8-bit SkyScan
# scan steps in whole units, and noise would otherwise pick each voxel's peak.
_LEVEL_SMOOTH = 1.0

# How far uphill, in voxels, each voxel looks for the enamel's local peak. From
# the junction the peak lies about half the enamel's depth in: 4-5 voxels on
# P82_Odam-Cre_WT_2 (8 um voxels, enamel about 70 um deep).
_PEAK_REACH_VOXELS = 6

# A local peak is enamel only if it stands this share of the dentin's own
# contrast (dentin minus background) above the dentin under it. On
# P82_Odam-Cre_WT_1 and _2 and P22_TGM2_WT_F_1 and KO_F_1, the brightest 0.1%
# of the dentin more than 8 voxels from any enamel peaks 0.06-0.12 of that
# contrast above it, the dimmest 1% of the enamel 0.21-0.27. The dentin's
# contrast, not the enamel's, so a knockout's dimmer enamel is held to the
# same bar as its wildtype's.
_MIN_PEAK_CONTRAST = 0.15

# The dentin under the enamel is read only beside enamel beyond doubt: a local
# peak at least this share of the histogram's enamel-dentin contrast above its
# dentin peak. Read beside dentin with no enamel on it as well -- the far end
# of M3 on P82_Odam-Cre_WT_2, whose dentin peaks about 0.3 of that contrast
# above the dentin peak -- it came out at the pulp's level there, and that
# stretch's dentin all passed for enamel.
_SURE_PEAK_CONTRAST = 0.5

# A voxel is at the junction, where the dentin beyond it is read, when it lies
# within this share of its local enamel-dentin contrast of the halfway level.
_JUNCTION_BAND = 0.15

# The dentin under the enamel is pooled over this many slices along the jaw
# (0.24-0.32 mm at 6-8 um), from at least _DENTIN_MIN_SAMPLES junction voxels;
# a stretch with fewer takes its neighbours' level. Each of the passes reads
# the junction from the last pass's levels.
_DENTIN_WINDOW_SLICES = 40
_DENTIN_MIN_SAMPLES = 200
_DENTIN_PASSES = 3

# How far uphill, in voxels, a voxel of the outer face's blurred rim looks for
# the enamel's bright core. Not one point two voxels in: where the enamel
# stands little above its dentin, the core's level lies close to the enamel's
# peak, and two voxels inside a blurred outer edge the scan has not reached it
# (P42_C67B6_WT_1: the whole outer surface but the cusp tips refused).
_RIM_REACH_VOXELS = 3

# Voxels read per batch along the gradient; bounds the memory.
_POINT_CHUNK = 1_000_000

# A face's level where no enamel stands: above anything a scan holds, so no
# surface runs there.
_NO_LEVEL = np.float32(1e30)

# The blur at the outer face is read on the profile straight into the enamel
# from points with at least this much enamel beneath them (voxels), so the
# plateau lies past the blur and short of the junction's; from this many of
# them, sampled this far out and in (voxels) at this step.
_BLUR_MIN_DEPTH_VOXELS = 10.0
_BLUR_POINTS = 20_000
_BLUR_RANGE = (-6.0, 9.0)
_BLUR_STEP = 0.25

# Where the blur cannot be read, the core of a column is kept this many
# voxels from both faces: 2.3-2.9 sigma of the 6 um scans' blur.
_FALLBACK_CLEAR_VOXELS = 3.5

# Whatever the blur, the core lies at least this far (voxels) from both faces,
# and a column has a core when at least this length (voxels) of it qualifies.
_MIN_CLEAR_VOXELS = 2.0
_MIN_CORE_VOXELS = 1.0

# Steps along a column, in voxels, and columns read per batch.
_CORE_STEP = 0.5
_CORE_CHUNK = 20_000

# The root surface: bare tooth surface (no enamel on it) in a piece at least
# this large (mm2) that reaches at least this far (voxels) from the enamel. A
# speck of crown coating, or a small defect closed in by enamel, does neither.
# Its edge is the part of it within this distance (voxels) of the enamel.
_ROOT_PIECE_MM2 = 0.02
_ROOT_REACH_VOXELS = 8
_EDGE_NEAR_VOXELS = 4

# A bare area of the crown at least this large (mm2) is an edge of the enamel
# too: the enamel tapers to nothing around it as it does at the cervix. On
# P22_TGM2_WT_F_1 most cusp tips (and the middle of M3's crown, barely erupted)
# carry no enamel the mask counts, and nearly every "thin patch" was the ring of
# tapering enamel around one. Smaller bare specks are left out: each would
# take a whole margin band with it.
_BARE_PIECE_MM2 = 0.002


@dataclass
class Levels:
    """The intensities either side of the enamel's two faces.

    ``enamel`` and ``dentin`` are the molar histogram's peaks. With ``peak``
    (the enamel's local peak at each voxel of the frame where it was read, the
    histogram's enamel peak elsewhere) and ``dentin_z`` (the dentin under the
    enamel, per slice along axis 0), each face's level is read where the face
    is (``read_levels``); without them, it is the histogram's, one for the
    whole tooth. Where the local peak stands less than ``floor`` above the
    dentin there is no enamel, and neither face has a level: read off the
    dentin's own texture, a junction level there runs all through the dentin
    near the enamel (35% of P22_TGM2_KO_F_1's junction points, before it was
    held to the floor).
    """

    enamel: float
    dentin: float
    background: float
    background_from_scan: bool
    peak: np.ndarray | None = None
    dentin_z: np.ndarray | None = None
    floor: float = 0.0

    @property
    def local(self) -> bool:
        return self.peak is not None and self.dentin_z is not None

    @property
    def outer(self) -> float:
        """The outer surface: halfway between enamel and background."""
        return (self.enamel + self.background) / 2.0

    @property
    def junction(self) -> float:
        """The enamel-dentin junction: halfway between enamel and dentin."""
        return (self.enamel + self.dentin) / 2.0

    @property
    def split(self) -> float:
        """Below this, the far side of a face is background; above it, tooth."""
        return (self.background + self.dentin) / 2.0

    # The same three levels where each face is: as fields over the frame, or
    # at points. Without local levels, the histogram's. Where no enamel
    # stands, a face's field is _NO_LEVEL, which no scan reaches, and its
    # value at a point NaN, which no comparison passes.

    def _dentin_field(self):
        return self.dentin_z[:, None, None] if self.local else self.dentin

    def _held(self, field: np.ndarray) -> np.ndarray:
        standing = self.peak - self._dentin_field() >= self.floor
        return np.where(standing, field, _NO_LEVEL).astype(np.float32)

    def outer_field(self):
        return self._held((self.peak + self.background) / 2.0) if self.local else self.outer

    def junction_field(self):
        return self._held((self.peak + self._dentin_field()) / 2.0) if self.local else self.junction

    def split_field(self):
        return ((self.background + self._dentin_field()) / 2.0).astype(np.float32) if self.local else self.split

    def peak_at(self, points: np.ndarray) -> np.ndarray:
        if not self.local:
            return np.full(len(points), self.enamel)
        peak = ndi.map_coordinates(self.peak, np.asarray(points, dtype=np.float64).T, order=1, mode="nearest")
        return np.where(peak - self.dentin_at(points) >= self.floor, peak, np.nan)

    def dentin_at(self, points: np.ndarray) -> np.ndarray:
        if not self.local:
            return np.full(len(points), self.dentin)
        z = np.asarray(points, dtype=np.float64)[:, 0]
        return np.interp(z, np.arange(len(self.dentin_z)), self.dentin_z)

    def outer_at(self, points: np.ndarray) -> np.ndarray:
        return (self.peak_at(points) + self.background) / 2.0

    def junction_at(self, points: np.ndarray) -> np.ndarray:
        return (self.peak_at(points) + self.dentin_at(points)) / 2.0

    def split_at(self, points: np.ndarray) -> np.ndarray:
        return (self.background + self.dentin_at(points)) / 2.0


@dataclass
class Surface:
    verts: np.ndarray   # (n, 3) voxel coordinates
    faces: np.ndarray   # (m, 3) vertex indices

    @classmethod
    def empty(cls) -> "Surface":
        return cls(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64))


# ---------------------------------------------------------------------------
# Levels read where each face is, and the enamel between the faces
# ---------------------------------------------------------------------------

@dataclass
class Reading:
    """What each voxel of a region reads along the gradient of the smoothed scan."""

    points: np.ndarray   # (n, 3) voxel indices, in the region's C order
    value: np.ndarray    # the smoothed scan at the voxel
    peak: np.ndarray     # the highest value uphill, within _PEAK_REACH_VOXELS
    far: np.ndarray      # the value _LOOK_OUT_VOXELS downhill: what lies beyond the voxel's face
    inside: np.ndarray   # the highest value within _RIM_REACH_VOXELS uphill


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    return ndi.map_coordinates(image, points.T, order=1, mode="nearest")


def _read_along(smooth: np.ndarray, region: np.ndarray) -> Reading:
    """Read the smoothed scan uphill and downhill of every voxel of ``region``."""
    points = np.argwhere(region).astype(np.int32)
    value = smooth[region]
    peak = np.empty(len(points), dtype=np.float32)
    far = np.empty(len(points), dtype=np.float32)
    inside = np.empty(len(points), dtype=np.float32)
    upper = np.array(smooth.shape) - 1
    for start in range(0, len(points), _POINT_CHUNK):
        p = points[start:start + _POINT_CHUNK]
        grad = np.empty(p.shape, dtype=np.float32)
        for axis in range(3):
            hi, lo = p.copy(), p.copy()
            hi[:, axis] = np.minimum(hi[:, axis] + 1, upper[axis])
            lo[:, axis] = np.maximum(lo[:, axis] - 1, 0)
            grad[:, axis] = smooth[tuple(hi.T)] - smooth[tuple(lo.T)]
        up = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
        at = p.astype(np.float32)
        best = np.full(len(p), -np.inf, dtype=np.float32)
        chunk = slice(start, start + len(p))
        for k in range(1, _PEAK_REACH_VOXELS + 1):
            np.maximum(best, _sample(smooth, at + k * up), out=best)
            if k == _RIM_REACH_VOXELS:
                inside[chunk] = best
        peak[chunk] = np.maximum(best, value[chunk])
        far[chunk] = _sample(smooth, at - _LOOK_OUT_VOXELS * up)
    return Reading(points, value, peak, far, inside)


def _faces(read: Reading, dentin: np.ndarray, background: float) -> tuple[np.ndarray, np.ndarray]:
    """Each voxel's junction level, and whether it lies on the outer face.

    On the outer face when background lies beyond it and the enamel's bright
    core within a few voxels inside it; on the junction otherwise.
    """
    junction = (read.peak + dentin) / 2.0
    outer = (read.far < (background + dentin) / 2.0) & (read.inside >= junction)
    return junction, outer


def _dentin_under_enamel(read: Reading, n_slices: int, *, background: float, dentin_level: float,
                         enamel_level: float) -> np.ndarray:
    """The dentin under the enamel, per slice along axis 0.

    Read just past the junction -- the value a few voxels downhill of each
    junction voxel beside sure enamel, with tissue there rather than the
    background or the pulp -- and pooled over a stretch of the jaw; a stretch
    without enough of them takes its neighbours' level, and a tooth with none
    at all the histogram's dentin peak. Each pass finds the junction from the
    levels of the last. Kept a quarter of the histogram's contrast below its
    enamel peak; not held to its dentin peak, which a molar row whose enamel
    dims along the jaw can put on the dimmer enamel (on a phantom; on the
    scans, dentin is most of the molar and its peak).
    """
    contrast = enamel_level - dentin_level
    tissue_beyond = read.far >= (background + dentin_level) / 2.0
    high = enamel_level - 0.25 * contrast
    z = read.points[:, 0]
    slices = np.arange(n_slices)
    per_slice = np.full(n_slices, float(dentin_level))
    for _ in range(_DENTIN_PASSES):
        dentin = per_slice[z]
        # Sure enamel stands well above the dentin under it as read so far:
        # the histogram's dentin peak, the bulk dentin and the roots, lies
        # below the crown's dentin (P42_C67B6_WT_1: 107 under 126), and held
        # to it the crown's own dentin humps passed for sure enamel.
        sure = read.peak >= dentin + _SURE_PEAK_CONTRAST * (enamel_level - dentin)
        junction, outer = _faces(read, dentin, background)
        at = (sure & tissue_beyond & ~outer & (read.far < junction)
              & (np.abs(read.value - junction) < _JUNCTION_BAND * (read.peak - dentin)))
        zs, beyond = z[at], read.far[at]
        if zs.size < _DENTIN_MIN_SAMPLES:
            break
        order = np.argsort(zs, kind="stable")
        zs, beyond = zs[order], beyond[order]
        starts = np.searchsorted(zs, slices - _DENTIN_WINDOW_SLICES // 2)
        ends = np.searchsorted(zs, slices + _DENTIN_WINDOW_SLICES // 2, side="right")
        pooled = np.full(n_slices, np.nan)
        for s in np.flatnonzero(ends - starts >= _DENTIN_MIN_SAMPLES):
            pooled[s] = np.median(beyond[starts[s]:ends[s]])
        held = np.isfinite(pooled)
        if not held.any():
            break
        per_slice = np.minimum(np.interp(slices, slices[held], pooled[held]), high)
    return per_slice


def read_levels(image: np.ndarray, region: np.ndarray, *, background: float, dentin_level: float,
                enamel_level: float, background_from_scan: bool, dentin_z: np.ndarray | None = None,
                smooth: np.ndarray | None = None) -> tuple[Levels, Reading]:
    """The levels each face is placed at, read at every voxel of ``region``.

    The enamel's local peak is the highest value of the scan (smoothed by a
    voxel) up the gradient within a few voxels. The dentin under the enamel
    is ``dentin_z`` when given -- as the enamel mask read it, so the surfaces
    and the mask agree -- and otherwise read here. ``dentin_level`` and
    ``enamel_level`` are the molar histogram's peaks, which the reading
    starts from.
    """
    if smooth is None:
        smooth = ndi.gaussian_filter(np.asarray(image, dtype=np.float32), _LEVEL_SMOOTH)
    read = _read_along(smooth, region)
    if dentin_z is None:
        dentin_z = _dentin_under_enamel(read, image.shape[0], background=background,
                                        dentin_level=dentin_level, enamel_level=enamel_level)
    peak = np.full(image.shape, float(enamel_level), dtype=np.float32)
    peak[region] = read.peak
    dentin_z = np.asarray(dentin_z, dtype=np.float64)
    levels = Levels(float(enamel_level), float(dentin_level), float(background), background_from_scan,
                    peak=peak, dentin_z=dentin_z, floor=contrast_floor(float(np.median(dentin_z)), background))
    return levels, read


def contrast_floor(dentin: float, background: float) -> float:
    """How far above the dentin under it a peak must stand to be enamel: ``_MIN_PEAK_CONTRAST`` of the dentin's contrast."""
    return _MIN_PEAK_CONTRAST * (dentin - background)


def _spread(values: np.ndarray) -> dict | None:
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {"median": float(p50), "p5": float(p5), "p95": float(p95)}


def enamel_layer(image: np.ndarray, molar: np.ndarray, *, background: float, dentin_level: float,
                 enamel_level: float, background_from_scan: bool = True) -> tuple[np.ndarray, Levels, dict]:
    """The enamel: the voxels of ``molar`` between its outer surface and its junction with the dentin.

    Every voxel brighter than its junction level -- halfway between the
    enamel's local peak and the dentin under it -- is enamel, and so is the
    blurred rim beyond it on the outer face: a voxel with background a few
    voxels downhill and enamel just uphill, brighter than halfway between that
    peak and the background. A peak standing less than ``_MIN_PEAK_CONTRAST``
    of the dentin's own contrast above that dentin is no enamel at all. Each
    voxel's own value is compared, not the smoothed one the levels are read
    on: smoothing would pull a sharp corner of enamel under its level.

    The rim is a shell a voxel or two thick over the enamel. Beside the
    enamel's edge, where the root's surface runs on from the crown or a worn
    cusp lays the dentin bare, the dentin reads the same -- it is as dim, air
    lies beyond it and enamel beside it -- but it is part of the thick dentin
    body, which an opening keeps and the thin rim does not survive.

    ``image`` is the scan around the molar (or the molar volume, zero outside
    its mask, with ``background`` 0); ``dentin_level`` and ``enamel_level``
    are the molar histogram's peaks. Returns (the enamel, the levels it was
    found at, a JSON-ready record of them).
    """
    levels, read = read_levels(image, molar, background=background, dentin_level=dentin_level,
                               enamel_level=enamel_level, background_from_scan=background_from_scan)
    value = np.asarray(image, dtype=np.float32)[molar]
    dentin = levels.dentin_z[read.points[:, 0]]
    junction, outer = _faces(read, dentin, background)
    outer_level = (read.peak + background) / 2.0
    floor = levels.floor
    standing = read.peak - dentin >= floor
    core = standing & (value >= junction)
    rim = standing & ~core & outer & (value >= outer_level)
    # The body: the tissue that is not enamel, on the tooth's side of the
    # outer surface. The molar mask reaches further out into the blur (on a
    # phantom blurred as P82 is, 1.4 voxels), and with the rim that band is
    # thick enough to survive the opening.
    tissue = np.zeros(molar.shape, dtype=bool)
    tissue[molar] = ~core & (value >= outer_level)
    body = ndi.binary_opening(tissue, structure=np.ones((3, 3, 3), dtype=bool))
    del tissue
    found = core | (rim & ~body[molar])
    del body
    enamel = np.zeros(molar.shape, dtype=bool)
    enamel[molar] = found

    # Where the faces were placed: the levels along the enamel's own edge.
    edge = (enamel & ~ndi.binary_erosion(enamel))[molar] & found
    on_rim = found & ~core
    record = {
        "method": "local-halfway",
        "background": float(background),
        "background_from": "scan" if background_from_scan else "molar volume (zero outside its mask)",
        "dentin_under_enamel": _spread(dentin[edge & ~outer]),
        "junction_level": _spread(junction[edge & ~outer]),
        "outer_level": _spread(outer_level[edge & outer]),
        "enamel_peak": _spread(read.peak[found]),
        "min_peak_contrast": float(floor),
        "outer_rim_voxels": int(on_rim.sum()),
    }
    del read
    junction_level = record["junction_level"] or {}
    logger.info("Enamel between its halfway faces: %d voxels; junction level %.5g (%.5g-%.5g), over dentin "
                "at %.5g; a local peak counts from %.5g above it", int(enamel.sum()),
                junction_level.get("median", float("nan")), junction_level.get("p5", float("nan")),
                junction_level.get("p95", float("nan")),
                (record["dentin_under_enamel"] or {}).get("median", float("nan")), floor)
    return enamel, levels, record


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

def background_level(image: np.ndarray, molar: np.ndarray, dist: np.ndarray) -> float | None:
    """Median intensity in a shell just outside the enamel-covered surface."""
    lo, hi = _BACKGROUND_SHELL
    shell = (dist >= lo) & (dist <= hi) & ~molar
    values = image[shell]
    return float(np.median(values)) if values.size else None


def _inward(smooth: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Unit vectors toward brighter intensity at each vertex: into the enamel."""
    grad = np.empty(verts.shape, dtype=np.float64)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0
        grad[:, axis] = (ndi.map_coordinates(smooth, (verts + step).T, order=1, mode="nearest")
                         - ndi.map_coordinates(smooth, (verts - step).T, order=1, mode="nearest"))
    return grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)


def _subset(verts: np.ndarray, faces: np.ndarray, kept: np.ndarray) -> Surface:
    index = np.full(len(verts), -1, dtype=np.int64)
    index[kept] = np.arange(int(kept.sum()))
    whole = kept[faces].all(axis=1)
    return Surface(verts[kept], index[faces[whole]])


def find_surface(image: np.ndarray, smooth: np.ndarray, level, near: np.ndarray,
                 keep, step: int = 1) -> Surface:
    """The ``level`` isosurface near the enamel, cut to the points ``keep(far, near_side, verts, inward)`` accepts.

    ``level`` is one intensity, or one per voxel (a ``Levels`` field): the
    surface then runs where the scan crosses the level of its own place.
    ``far`` is the intensity a few voxels out of the enamel from each point,
    ``near_side`` a couple of voxels in, ``inward`` the unit vector into it.
    """
    from skimage.measure import marching_cubes

    field, at = (image - level, 0.0) if np.ndim(level) else (image, float(level))
    around = field[near]
    if not around.size or not float(around.min()) < at < float(around.max()):
        return Surface.empty()
    del around
    try:
        verts, faces, _, _ = marching_cubes(field, level=at, step_size=step, mask=near)
    except (ValueError, RuntimeError) as exc:
        logger.warning("No surface at its level: %s", exc)
        return Surface.empty()
    del field
    if not len(verts):
        return Surface.empty()
    inward = _inward(smooth, verts)
    far = ndi.map_coordinates(image, (verts - _LOOK_OUT_VOXELS * inward).T, order=1, mode="nearest")
    near_side = ndi.map_coordinates(image, (verts + _LOOK_IN_VOXELS * inward).T, order=1, mode="nearest")
    return _subset(verts, faces, keep(far, near_side, verts, inward))


def _keepers(levels: Levels, enamel: np.ndarray | None = None):
    """Which points of each face's isosurface belong to the enamel's two surfaces.

    Both need enamel just inside them: with ``enamel`` (the mask, as 0/1
    floats) the point a couple of voxels in lies in it; without, it is
    brighter than the junction's level. The outer surface has background
    beyond it, the junction tissue that is not enamel.
    """
    def enamel_inside(near_side, verts, inward):
        if enamel is None:
            return near_side > levels.junction_at(verts)
        return _sample(enamel, verts + _LOOK_IN_VOXELS * inward) >= 0.5

    def outer(far, near_side, verts, inward):
        return (far < levels.split_at(verts)) & enamel_inside(near_side, verts, inward)

    def junction(far, near_side, verts, inward):
        split, junction_level = levels.split_at(verts), levels.junction_at(verts)
        return (far > split) & (far < junction_level) & enamel_inside(near_side, verts, inward)
    return outer, junction


# ---------------------------------------------------------------------------
# Thickness
# ---------------------------------------------------------------------------

def _through_enamel(image: np.ndarray, a: np.ndarray, b: np.ndarray, level: float) -> np.ndarray:
    """Whether each straight line a->b (voxel coordinates) stays enamel-bright away from its ends."""
    out = np.empty(len(a), dtype=bool)
    fr = np.linspace(0.0, 1.0, _SIGHT_SAMPLES)
    from_end = np.minimum(fr, 1.0 - fr)
    for start in range(0, len(a), _CHUNK):
        aa, bb = a[start:start + _CHUNK], b[start:start + _CHUNK]
        span = bb - aa
        pts = aa[:, None, :] + fr[None, :, None] * span[:, None, :]
        vals = ndi.map_coordinates(image, pts.reshape(-1, 3).T, order=1, mode="nearest").reshape(len(aa), -1)
        interior = from_end[None, :] * np.linalg.norm(span, axis=1)[:, None] > _SIGHT_END_VOXELS
        out[start:start + _CHUNK] = ~np.any(interior & (vals < level), axis=1)
    return out


def contact_points(labels: np.ndarray | None) -> np.ndarray:
    """Where two teeth touch: the midpoints between face neighbours of different teeth (voxel coordinates)."""
    if labels is None:
        return np.empty((0, 3))
    points = []
    for axis in range(3):
        lo = tuple(slice(None, -1) if i == axis else slice(None) for i in range(3))
        hi = tuple(slice(1, None) if i == axis else slice(None) for i in range(3))
        a, b = labels[lo], labels[hi]
        at = np.argwhere((a != b) & (a > 0) & (b > 0)).astype(np.float64)
        at[:, axis] += 0.5
        points.append(at)
    return np.concatenate(points)


def nearest_through_enamel(src: np.ndarray, dst: np.ndarray, image: np.ndarray, level: float,
                           voxel_um: np.ndarray, return_index: bool = False):
    """For each ``src`` point, the distance (um) to the nearest ``dst`` point a line through enamel reaches.

    NaN where none of the ``_CANDIDATES`` nearest can be reached. With
    ``return_index``, also which ``dst`` point that is (-1 for none).
    """
    dist = np.full(len(src), np.nan)
    index = np.full(len(src), -1, dtype=np.int64)
    if not len(src) or not len(dst):
        return (dist, index) if return_index else dist
    tree = cKDTree(dst * voxel_um)
    d, j = tree.query(src * voxel_um)
    ok = _through_enamel(image, src, dst[j], level)
    dist[ok] = d[ok]
    index[ok] = j[ok]
    rest = np.flatnonzero(~ok)
    k = min(_CANDIDATES, len(dst))
    if rest.size and k > 1:
        dk, jk = tree.query(src[rest] * voxel_um, k=k)
        pending = np.arange(rest.size)
        for col in range(1, k):
            if not pending.size:
                break
            good = _through_enamel(image, src[rest[pending]], dst[jk[pending, col]], level)
            dist[rest[pending[good]]] = dk[pending[good], col]
            index[rest[pending[good]]] = jk[pending[good], col]
            pending = pending[~good]
    return (dist, index) if return_index else dist


def thickness_stats(values: np.ndarray) -> dict:
    """Median, mean and percentiles (um) of the finite values, with how many there were."""
    v = values[np.isfinite(values)]
    if not v.size:
        return {"median": None, "mean": None, "p5": None, "p25": None, "p75": None, "p95": None, "n_points": 0}
    p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    return {"median": float(p50), "mean": float(v.mean()), "p5": float(p5), "p25": float(p25),
            "p75": float(p75), "p95": float(p95), "n_points": int(v.size)}


def _share(values: np.ndarray) -> float | None:
    return float(np.isfinite(values).mean()) if values.size else None


# ---------------------------------------------------------------------------
# What the patches need from the scan
# ---------------------------------------------------------------------------

def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """Where ``y`` first rises through ``level``, interpolated along ``x``."""
    above = y >= level
    if above[0] or not above.any():
        return None
    i = int(np.argmax(above))
    y0, y1 = float(y[i - 1]), float(y[i])
    return float(x[i - 1] + (level - y0) / (y1 - y0) * (x[i] - x[i - 1])) if y1 > y0 else float(x[i])


def edge_blur(image: np.ndarray, outer_verts: np.ndarray, inward: np.ndarray, depth_voxels: np.ndarray,
              background: float, seed: int = 0) -> float | None:
    """The scan's blur at the enamel's outer face, as a Gaussian sigma in voxels; None if it cannot be read.

    Read on the median profile straight into the enamel from outer points with
    enamel deep beneath them, scaled from the background (0) to the enamel's
    plateau (1): a blurred step climbs from 0.16 to 0.84 over two sigma. The
    plateau is read 3-4.5 voxels in at first, then 2.5-3.5 sigma past the
    halfway point once a sigma is known, from points whose junction lies a
    further 2.5 sigma beyond.
    """
    ok = np.flatnonzero(np.isfinite(depth_voxels) & (depth_voxels >= _BLUR_MIN_DEPTH_VOXELS))
    if ok.size < 200:
        return None
    if ok.size > _BLUR_POINTS:
        ok = np.random.default_rng(seed).choice(ok, _BLUR_POINTS, replace=False)
    steps = np.arange(_BLUR_RANGE[0], _BLUR_RANGE[1] + 1e-9, _BLUR_STEP)
    pts = outer_verts[ok, None, :] + steps[None, :, None] * inward[ok, None, :]
    profile = ndi.map_coordinates(image, pts.reshape(-1, 3).T, order=1, mode="nearest").reshape(ok.size, -1)
    depth = depth_voxels[ok]
    window, use, sigma = (3.0, 4.5), np.ones(ok.size, dtype=bool), None
    for _ in range(3):
        inside = (steps >= window[0]) & (steps <= window[1])
        if not inside.any() or use.sum() < 200:
            break
        plateau = np.median(profile[use][:, inside], axis=1)
        scale = plateau - background
        good = scale > 0
        if good.sum() < 200:
            break
        median = np.median((profile[use][good] - background) / scale[good, None], axis=0)
        x16, x50, x84 = (_crossing(steps, median, v) for v in (0.16, 0.5, 0.84))
        if x16 is None or x50 is None or x84 is None or x84 <= x16:
            break
        sigma = (x84 - x16) / 2.0
        window = (x50 + 2.5 * sigma, min(x50 + 3.5 * sigma, float(steps[-1])))
        use = depth >= window[1] + 2.5 * sigma
    return None if sigma is None else float(sigma)


def core_density(image: np.ndarray, junction_verts: np.ndarray, partner_verts: np.ndarray, voxel_um: np.ndarray,
                 clear_voxels: dict, faces_um: np.ndarray) -> dict:
    """Each column's core density: the scan along the line from a junction point to the outer point it was
    measured to, read only where the line lies at least a clearance from both faces.

    Blur pulls a thin layer's brightness down: within a couple of sigma of a
    face the scan reads part enamel and part what lies beyond. So only the
    stretch at least ``clear`` voxels from every point of either face counts
    -- measured to the nearest face point (``faces_um``: the outer surface,
    the contacts and the junction), not along the line, since beside a fissure
    or at a cusp the other face is nearer sideways. ``clear_voxels`` maps a
    name to a clearance; returns {name: mean raw value per column}, NaN where
    under ``_MIN_CORE_VOXELS`` of the column qualifies or it has no partner.
    """
    n = len(junction_verts)
    out = {name: np.full(n, np.nan) for name in clear_voxels}
    if not n or not clear_voxels:
        return out
    size = float(np.mean(voxel_um))
    span = partner_verts - junction_verts
    length = np.linalg.norm(span, axis=1)
    least = min(clear_voxels.values())
    columns = np.flatnonzero(np.isfinite(length) & (length >= 2.0 * least + _MIN_CORE_VOXELS))
    if not columns.size:
        return out
    tree = cKDTree(faces_um)
    reach = (max(clear_voxels.values()) + 1.0) * size
    need = int(np.ceil(_MIN_CORE_VOXELS / _CORE_STEP)) + 1
    for start in range(0, columns.size, _CORE_CHUNK):
        cols = columns[start:start + _CORE_CHUNK]
        n_steps = int(np.ceil(length[cols].max() / _CORE_STEP)) + 1
        t = np.arange(n_steps) * _CORE_STEP
        along = t[None, :] <= length[cols, None]
        unit = span[cols] / length[cols, None]
        pos = junction_verts[cols, None, :] + t[None, :, None] * unit[:, None, :]
        flat = pos[along]
        values = np.full(along.shape, np.nan)
        values[along] = ndi.map_coordinates(image, flat.T, order=1, mode="nearest")
        clearance = np.zeros(along.shape)
        clearance[along], _ = tree.query(flat * voxel_um, distance_upper_bound=reach)
        for name, clear in clear_voxels.items():
            core = along & (clearance >= clear * size)
            count = core.sum(axis=1)
            total = np.where(core, values, 0.0).sum(axis=1)
            out[name][cols] = np.where(count >= need, total / np.maximum(count, 1), np.nan)
    return out


def bare_edges(molar: np.ndarray, enamel: np.ndarray, dist: np.ndarray, points: np.ndarray,
               voxel_um: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """How far each point lies (um) from the enamel's edges, where it stops on bare dentin.

    The tooth's surface where no enamel covers it is the dentin body -- the
    molar less its enamel, opened so the blurred rim the molar mask keeps a
    voxel or two beyond the enamel drops out -- on the molar's surface. Its
    pieces large enough and reaching far enough from the enamel are the roots'
    surfaces, and their edge near the enamel is the cervical edge. The pulp's
    walls, if the mask leaves the pulp out, join the roots' piece but lie far
    from the enamel. A bare area of the crown -- a cusp tip without enamel, a
    worn cusp -- is an edge of the enamel too once it covers
    ``_BARE_PIECE_MM2``; a speck smaller than that is not. Returns (distance to
    any of those edges, distance to the cervical edge alone, a record);
    infinite where there is no such edge.
    """
    n = len(points)
    nowhere = np.full(n, np.inf)
    body = ndi.binary_opening(molar & ~enamel, structure=np.ones((3, 3, 3), dtype=bool))
    bare = body & ~ndi.binary_erosion(molar)
    del body
    labels, count = ndi.label(bare, structure=np.ones((3, 3, 3), dtype=bool))
    record = {"pieces": 0, "edge_points": 0}
    if not count or not n:
        return nowhere, nowhere.copy(), record
    ids = np.arange(1, count + 1)
    sizes = ndi.sum_labels(bare, labels, ids)
    reach = ndi.maximum(dist, labels, ids)
    near_count = ndi.sum_labels(bare & (dist <= _EDGE_NEAR_VOXELS), labels, ids)
    face_mm2 = float(np.mean(voxel_um)) ** 2 / 1e6
    is_root = (sizes * face_mm2 >= _ROOT_PIECE_MM2) & (reach >= _ROOT_REACH_VOXELS)
    is_crown = ~is_root & (sizes * face_mm2 >= _BARE_PIECE_MM2) & (near_count > 0)
    near = bare & (dist <= _EDGE_NEAR_VOXELS)
    del bare
    at = points * voxel_um

    def distance(mask: np.ndarray) -> np.ndarray:
        where = np.argwhere(mask)
        if not len(where):
            return np.full(n, np.inf)
        return cKDTree(where * voxel_um).query(at)[0]

    cervical = near & np.isin(labels, ids[is_root])
    edge = near & np.isin(labels, ids[is_root | is_crown])
    # The bare crown's area counts its surface voxels: about its area, more on
    # surfaces running slantwise to the grid.
    record = {"pieces": int(is_root.sum()), "edge_points": int(cervical.sum()),
              "bare_crown_pieces": int(is_crown.sum()),
              "bare_crown_mm2": float(sizes[is_crown].sum() * face_mm2)}
    out = distance(edge), distance(cervical)
    del labels, near, cervical, edge
    return out[0], out[1], record


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def measure(image: np.ndarray, molar: np.ndarray, enamel: np.ndarray, spacing: tuple[float, float, float], *,
            dentin_level: float | None, enamel_level: float | None, from_scan: bool = True,
            labels: np.ndarray | None = None, names: dict[int, str] | None = None,
            display: bool = True, dentin_z: np.ndarray | None = None,
            patches: enamel_patches.Params | None = enamel_patches.Params(), origin=(0, 0, 0),
            points_out: dict | None = None) -> tuple[dict | None, dict | None, list[str]]:
    """Two-surface enamel thickness, and the patches of thinner or less dense enamel.

    ``image`` is the scan (or, without one, the molar volume) cropped with a
    margin around the tooth; ``molar``, ``enamel`` and ``labels`` share its
    frame, which starts at ``origin`` in the input volume. Each surface lies at
    its level where it is (``read_levels``); ``dentin_z``, the dentin under the
    enamel per slice of this frame as the enamel mask read it, keeps the
    junction where the mask put it. ``patches`` is how the patches are found
    (None: not looked for); ``points_out``, when given, is filled with each
    junction point's measurements for ``enamel_patches.sweep``. Returns
    (summary, display, warnings): the JSON-ready summary with a per-tooth
    block and the patches, the meshes for the map (None unless ``display``),
    and anything the reader should know.
    """
    warnings: list[str] = []
    if dentin_level is None or enamel_level is None or not enamel.any():
        warnings.append("Enamel thickness skipped: no dentin or enamel peak to place the surfaces by")
        return None, None, warnings
    image = np.asarray(image, dtype=np.float32)
    voxel_um = np.asarray(spacing, dtype=np.float64) * 1000.0
    dist = ndi.distance_transform_cdt(~enamel, metric="chessboard")
    near = dist <= _NEAR_ENAMEL_VOXELS

    measured = background_level(image, molar, dist) if from_scan else None
    if measured is None:
        warnings.append("Enamel thickness: no scan around the tooth, so the background outside the "
                        "enamel was taken as zero; pass the reoriented scan (--scan) to measure it")
    levels = Levels(float(enamel_level), float(dentin_level), 0.0 if measured is None else measured,
                    measured is not None)
    if not levels.background < levels.dentin < levels.enamel:
        warnings.append(f"Enamel thickness skipped: background {levels.background:.4g}, dentin "
                        f"{levels.dentin:.4g} and enamel {levels.enamel:.4g} are not in rising order")
        return None, None, warnings

    smooth = ndi.gaussian_filter(image, _LEVEL_SMOOTH)
    # The levels where each face is: the enamel's local peak, read a voxel
    # beyond where the surfaces are looked for (marching cubes reads each
    # cube's far corners), and the dentin under the enamel.
    band = ndi.binary_dilation(near, structure=np.ones((3, 3, 3), dtype=bool))
    levels, _ = read_levels(image, band, background=levels.background, dentin_level=levels.dentin,
                            enamel_level=levels.enamel, background_from_scan=levels.background_from_scan,
                            dentin_z=dentin_z, smooth=smooth)
    del band
    outer_level, junction_level = levels.outer_field(), levels.junction_field()
    inside = enamel.astype(np.float32)
    keep_outer, keep_junction = _keepers(levels, inside)
    outer = find_surface(image, smooth, outer_level, near, keep_outer)
    junction = find_surface(image, smooth, junction_level, near, keep_junction)
    placed = {"outer_surface": _spread(levels.outer_at(outer.verts)),
              "junction": _spread(levels.junction_at(junction.verts)),
              "dentin_under_enamel": _spread(levels.dentin_at(junction.verts))}
    logger.info("Enamel surfaces: outer %d points at %s, junction %d points at %s, over dentin at %s "
                "(background %.5g%s)", len(outer.verts), _levels_text(placed["outer_surface"]),
                len(junction.verts), _levels_text(placed["junction"]), _levels_text(placed["dentin_under_enamel"]),
                levels.background, "" if levels.background_from_scan else ", assumed")
    if not len(outer.verts) or not len(junction.verts):
        warnings.append("Enamel thickness skipped: the outer surface or the enamel-dentin junction was not found")
        return None, None, warnings

    # Where teeth touch, the contact is the outer surface each tooth has there.
    contact = contact_points(labels)
    if len(contact):
        logger.info("Enamel thickness: %d points where teeth touch stand in for their outer surface", len(contact))
    ends = np.concatenate([outer.verts, contact]) if len(contact) else outer.verts
    # A line runs through enamel while it stays in the enamel away from its
    # ends. Not "while it stays brighter than the junction's level": near its
    # outer end every line crosses the blurred rim, which reads below that
    # level for 0.85 sigma of blur (1.5 voxels on P82's 8 um SkyScan scans),
    # and on P82_Odam-Cre_WT_2 that refused 32% of the junction's pairings.
    from_junction, partner = nearest_through_enamel(junction.verts, ends, inside, 0.5, voxel_um, return_index=True)
    from_outer, outer_partner = nearest_through_enamel(outer.verts, junction.verts, inside, 0.5, voxel_um,
                                                       return_index=True)
    # Measured to a contact: the cut between two teeth lies at the far end, not the air.
    to_contact = partner >= len(outer.verts)
    size = float(voxel_um.mean())

    # What the patches need from the scan: its blur at the outer face, and the
    # density of each column's core, read that far clear of both faces.
    blur, clear, core = None, {}, {}
    if patches is not None:
        blur = edge_blur(image, outer.verts, _inward(smooth, outer.verts), from_outer / size, levels.background)
        if blur is None:
            warnings.append("Enamel patches: the blur at the enamel's outer face could not be read, so the "
                            f"enamel's core is read {_FALLBACK_CLEAR_VOXELS:g} voxels from both faces")
        clear = {b: (max(b * blur, _MIN_CLEAR_VOXELS) if blur is not None else _FALLBACK_CLEAR_VOXELS)
                 for b in (*enamel_patches.CORE_BLURS, patches.core_blurs)}
        far_end = np.full(junction.verts.shape, np.nan)
        reached = (partner >= 0) & ~to_contact
        far_end[reached] = ends[partner[reached]]
        core = core_density(image, junction.verts, far_end, voxel_um, clear,
                            np.concatenate([ends, junction.verts]) * voxel_um)
        del far_end
    del ends

    tooth_lookup = _tooth_lookup(labels)
    area_total, area_by_label, smoothed_junction, triangles_mm2 = junction_area(
        smooth, levels, near, voxel_um, tooth_lookup, junction_level, keep_junction, return_surface=True)
    del inside
    del junction_level
    tooth_at = tooth_lookup(junction.verts) if tooth_lookup is not None else None
    teeth = []
    if tooth_at is not None and names:
        for label, name in sorted(names.items(), key=lambda kv: kv[1]):
            mine = from_junction[tooth_at == label]
            teeth.append({"label": int(label), "name": name, **thickness_stats(mine),
                          "share_measured": _share(mine),
                          "junction_area_mm2": float(area_by_label.get(int(label), 0.0))})

    summary = {
        "method": "nearest point through enamel, between halfway surfaces",
        "direction": "enamel-dentin junction to outer surface",
        **thickness_stats(from_junction),
        "share_measured": _share(from_junction),
        "contact_points": int(len(contact)),
        "junction_area_mm2": area_total,
        "from_outer_surface": {**thickness_stats(from_outer), "share_measured": _share(from_outer)},
        # The histogram's peaks, the background, and the levels each surface
        # was placed at where it runs: the median over its points, and their
        # 5th-95th percentile.
        "levels": {
            "enamel": levels.enamel, "dentin": levels.dentin, "background": levels.background,
            "background_from": "scan" if levels.background_from_scan else "assumed zero (no scan)",
            "outer_surface": (placed["outer_surface"] or {}).get("median"),
            "junction": (placed["junction"] or {}).get("median"),
            "placed": placed,
        },
        "teeth": [{k: v for k, v in t.items() if k != "label"} for t in teeth],
    }
    logger.info("Enamel thickness (from the junction): median %.1f um, middle half %.1f-%.1f um, "
                "%.0f%% of %d junction points measured; junction area %.4f mm2",
                summary["median"] or float("nan"), summary["p25"] or float("nan"), summary["p75"] or float("nan"),
                100 * (summary["share_measured"] or 0), len(junction.verts), area_total)

    shades = None
    if patches is not None:
        # Each junction point stands for its share of the junction found on the
        # scan smoothed by a voxel, whose area is the one reported.
        centres = (smoothed_junction.verts[smoothed_junction.faces].mean(axis=1) if len(smoothed_junction.faces)
                   else np.zeros((0, 3)))
        areas = enamel_patches.point_areas(junction.verts * voxel_um, centres * voxel_um, triangles_mm2 * 1e6,
                                           reach_um=3.0 * size)
        triangles = (centres.astype(np.float32), (triangles_mm2 * 1e6).astype(np.float32))
        if not areas.sum():
            areas = enamel_patches.barycentric_areas(junction.verts * voxel_um, junction.faces)
            areas *= (area_total * 1e6 / areas.sum()) if areas.sum() and area_total else 1.0
            triangles = None
        del smoothed_junction
        edge_um, cervical_um, margin = bare_edges(molar, enamel, dist, junction.verts, voxel_um)
        if not margin["edge_points"]:
            warnings.append("Enamel patches: no cervical edge found (no bare root surface beside the enamel), so "
                            "the taper to the margin is judged with the rest")
        blur_um = None if blur is None else blur * size
        block, shades = enamel_patches.measure_patches(
            junction.verts, voxel_um, areas_um2=areas, tooth=tooth_at if names else None, names=names,
            thickness_um=from_junction, to_contact=to_contact, edge_um=edge_um,
            core_raw=core.get(patches.core_blurs), background=levels.background, edge_blur_um=blur_um,
            clear_um=clear[patches.core_blurs] * size, params=patches, origin=origin, triangles=triangles)
        block["margin"] = margin
        summary["edge_blur_um"] = blur_um
        summary["patches"] = block
        _log_patches(block)
        if points_out is not None:
            points_out.update({
                "verts": junction.verts.astype(np.float32), "faces": junction.faces.astype(np.int32),
                "voxel_um": voxel_um, "origin": np.asarray(origin, dtype=np.float64),
                "background": np.float64(levels.background),
                "edge_blur_um": np.float64(np.nan if blur_um is None else blur_um),
                "area_um2": areas, "thickness_um": from_junction, "to_contact": to_contact,
                "partner": partner.astype(np.int32), "cervical_um": cervical_um, "edge_um": edge_um,
                "dentin": levels.dentin_at(junction.verts),
                "tooth": (tooth_at if tooth_at is not None else np.zeros(len(junction.verts))).astype(np.uint8),
                "names": np.array(json.dumps({str(k): v for k, v in (names or {}).items()})),
                **({"tri_centre": triangles[0], "tri_area_um2": triangles[1]} if triangles is not None else {}),
                **{enamel_patches.core_key(b): core[b] for b in enamel_patches.CORE_BLURS},
                **{f"clear_{enamel_patches.core_key(b)}": np.float64(clear[b] * size)
                   for b in enamel_patches.CORE_BLURS},
            })

    shown = None
    if display:
        shown = _display(image, smooth, levels, near, dist, molar, enamel, voxel_um, tooth_lookup, names or {},
                         outer, junction, from_outer, from_junction, shades=shades, outer_partner=outer_partner)
    return summary, shown, warnings


def _log_patches(block: dict) -> None:
    for kind, label in (("thin", "thinner"), ("low_density", "less dense")):
        entry = block.get(kind)
        if not entry:
            continue
        share = entry.get("share")
        logger.info("Enamel patches %s than the enamel around them: %s of the %.3f mm2 judged (%s of the "
                    "junction), %d patches%s", label, "-" if share is None else f"{100 * share:.2f}%",
                    entry.get("judged_mm2") or 0.0,
                    "-" if entry.get("share_judged") is None else f"{100 * entry['share_judged']:.0f}%",
                    entry.get("patches") or 0,
                    "".join(f"; {t['name']} {100 * t['share']:.2f}%" for t in entry.get("teeth", [])
                            if t.get("share") is not None))


def _levels_text(spread: dict | None) -> str:
    if not spread:
        return "-"
    return f"{spread['median']:.5g} ({spread['p5']:.5g}-{spread['p95']:.5g})"


def _triangle_areas(verts: np.ndarray, faces: np.ndarray, voxel_um: np.ndarray) -> np.ndarray:
    p = verts * (voxel_um / 1000.0)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def junction_area(smooth: np.ndarray, levels: Levels, near: np.ndarray, voxel_um: np.ndarray,
                  tooth_lookup, level=None, keep=None, return_surface: bool = False):
    """Area (mm2) of the enamel-dentin junction, in all and by tooth label.

    Measured on the junction of the scan smoothed by one voxel: on the raw
    scan, noise roughens the surface and inflates its area with the scan's
    noise (+13% at a phantom's realistic contrast-to-noise). Smoothed, it reads
    0-6% low -- the most on tightly curved surfaces (a sphere of 12 voxels'
    radius) -- and stays within a percent or two of that across a sevenfold
    range of noise, so scans of differing noise compare. Like the thickness,
    it covers the junction where enamel lies over dentin, so it stops a couple
    of voxels short of the enamel's edge near the roots. ``level`` is the
    junction's level field and ``keep`` its points' test when already at hand.
    With ``return_surface``, also the surface and its triangles' areas (mm2).
    """
    surf = find_surface(smooth, smooth, levels.junction_field() if level is None else level, near,
                        _keepers(levels)[1] if keep is None else keep)
    if not len(surf.faces):
        return (0.0, {}, surf, np.zeros(0)) if return_surface else (0.0, {})
    areas = _triangle_areas(surf.verts, surf.faces, voxel_um)
    by_label: dict[int, float] = {}
    if tooth_lookup is not None:
        centres = surf.verts[surf.faces].mean(axis=1)
        label = tooth_lookup(centres)
        sums = np.bincount(label.astype(np.int64), weights=areas)
        by_label = {i: float(v) for i, v in enumerate(sums) if i > 0}
    if return_surface:
        return float(areas.sum()), by_label, surf, areas
    return float(areas.sum()), by_label


def _tooth_lookup(labels: np.ndarray | None):
    if labels is None or not labels.any():
        return None
    # Only the voxels outside every tooth take a widened label: inside, a
    # tooth's own label stands, even beside a contact with a higher one.
    widened = np.where(labels > 0, labels, ndi.grey_dilation(labels, size=2 * _TOOTH_REACH + 1))
    upper = np.array(widened.shape) - 1

    def lookup(verts: np.ndarray) -> np.ndarray:
        idx = np.clip(np.round(verts).astype(np.int64), 0, upper)
        return widened[tuple(idx.T)]
    return lookup


# ---------------------------------------------------------------------------
# Display meshes
# ---------------------------------------------------------------------------

def _display(image, smooth, levels, near, dist, molar, enamel, voxel_um, tooth_lookup, names,
             outer: Surface, junction: Surface, from_outer, from_junction, *, shades: dict | None = None,
             outer_partner: np.ndarray | None = None) -> dict:
    """Meshes for the map, in a viewer frame: x along the jaw, y toward the crowns, in um.

    Two solids. The enamel, closed -- its outer surface, its underside on the
    junction and its thin edge near the roots -- coloured everywhere by the
    thickness measured nearest each point. And beneath it the rest of the
    tooth, grey, except where it meets the enamel: there it carries the
    thickness of the enamel above, so taking the enamel away shows it on the
    dentin.

    ``shades`` are the patches' values at the junction's points (each kind's
    deficit, NaN where not judged, and whether the point lies in a counted
    patch); the outer surface takes those of the junction point it was
    measured to (``outer_partner``), so the colour runs through the enamel.
    """
    triangles = len(outer.faces) + len(junction.faces)
    step = max(1, int(np.ceil(np.sqrt(triangles / _DISPLAY_TRIANGLES))))
    reach = 3.0 * float(voxel_um.max())

    # The enamel solid: the enamel mask plus the blurred outer rim that is
    # still more enamel than background (brighter than the outer surface's
    # level, next to background) -- a mask cut at one threshold stops a voxel
    # short; one between the halfway faces (``enamel_layer``) only where the
    # molar mask itself does.
    background = image < levels.split_field()
    by_background = ndi.distance_transform_cdt(~background, metric="chessboard") <= 2
    rim = ndi.binary_dilation(enamel, iterations=2) & ~enamel & (image >= levels.outer_field()) & by_background
    solid = enamel | rim
    del background, by_background, rim
    cap = _solid_mesh(ndi.gaussian_filter(solid.astype(np.float32), 1.0), step)
    measured = np.concatenate([outer.verts, junction.verts])
    values = np.concatenate([from_outer, from_junction])
    cap_values = _nearest_values(cap.verts, measured, values, voxel_um, reach)

    # The rest of the tooth, pulled back a voxel from the enamel so it never
    # shows through it.
    body = molar & ~ndi.binary_dilation(solid, iterations=1)
    core = _solid_mesh(ndi.gaussian_filter(body.astype(np.float32), 1.0), step + 1)
    del body
    core_values = _nearest_values(core.verts, junction.verts, from_junction, voxel_um, 1.2 * reach)

    # Each patch kind's deficit, and whether a point lies in a counted patch
    # (NaN where it was not judged), on both meshes.
    extra = {"enamel": {}, "dentin": {}}
    for kind in ("thin", "low_density"):
        at_junction = (shades or {}).get(f"{kind}_shown", (shades or {}).get(kind))
        if at_junction is None:
            continue
        flags = shades.get(f"{kind}_patch_shown", shades.get(f"{kind}_patch"))
        flag_junction = np.where(np.isfinite(at_junction), flags.astype(np.float64), np.nan)
        partner_ok = outer_partner is not None
        for name, junction_values in ((kind, at_junction), (f"{kind}_patch", flag_junction)):
            if partner_ok:
                linked = outer_partner >= 0
                at_outer = np.full(len(outer.verts), np.nan)
                at_outer[linked] = junction_values[outer_partner[linked]]
            else:
                at_outer = np.full(len(outer.verts), np.nan)
            extra["enamel"][name] = _nearest_values(cap.verts, measured, np.concatenate([at_outer, junction_values]),
                                                    voxel_um, reach)
            if len(core.verts):
                extra["dentin"][name] = _nearest_values(core.verts, junction.verts, junction_values, voxel_um,
                                                        1.2 * reach)

    frame = _viewer_frame(molar, enamel, voxel_um)
    layers = {}
    for key, surf, vals in (("enamel", cap, cap_values), ("dentin", core, core_values)):
        if not len(surf.faces):
            continue
        pos = frame(surf.verts)
        layer = {"pos": pos, "faces": _outward(pos, surf.faces).astype(np.uint32),
                 "value": np.asarray(vals, dtype=np.float32)}
        if tooth_lookup is not None:
            layer["tooth"] = tooth_lookup(surf.verts).astype(np.uint8)
        layer.update(extra[key])
        layers[key] = layer
    logger.info("Thickness map: %s (display step %d)",
                ", ".join(f"{k} {len(v['faces'])} triangles" for k, v in layers.items()), step)
    return {"layers": layers, "teeth": {str(k): v for k, v in names.items()}, "step": step,
            "voxel_um": [float(v) for v in voxel_um]}


def _solid_mesh(field: np.ndarray, step: int) -> Surface:
    """The 0.5 isosurface of a smoothed 0/1 solid: closed wherever the solid is."""
    from skimage.measure import marching_cubes

    try:
        verts, faces, _, _ = marching_cubes(field, level=0.5, step_size=step)
    except (ValueError, RuntimeError):
        return Surface.empty()
    return Surface(verts, faces)


def _nearest_values(verts: np.ndarray, points: np.ndarray, values: np.ndarray, voxel_um: np.ndarray,
                    reach_um: float) -> np.ndarray:
    """Each vertex takes the nearest measured value within ``reach_um``; NaN beyond."""
    out = np.full(len(verts), np.nan, dtype=np.float32)
    ok = np.isfinite(values)
    if not len(verts) or not ok.any():
        return out
    d, j = cKDTree(points[ok] * voxel_um).query(verts * voxel_um)
    near = d <= reach_um
    out[near] = values[ok][j[near]]
    return out


def _outward(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces wound so their normals point out of the solid (counter-clockwise seen from outside).

    The viewer draws a see-through enamel with its back faces culled, which
    needs the winding to agree. Marching cubes winds every face of a mesh the
    same way, so one test settles it: a closed surface's signed volume is
    positive when its normals point out.
    """
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    signed = np.einsum("ij,ij->i", a.astype(np.float64), np.cross(b, c).astype(np.float64)).sum()
    return faces if signed >= 0 else faces[:, [0, 2, 1]]


def _viewer_frame(molar: np.ndarray, enamel: np.ndarray, voxel_um: np.ndarray):
    """Voxel (z, y, x) -> viewer (along jaw, toward crowns, across), um, centred on the tooth.

    Built in physical x-y-z so the picture is not mirrored: the array's
    (z, y, x) order is a reflection of it. Array axis 0 runs along the jaw
    after the pipeline's reorientation; "toward the crowns" is from the
    tooth's centroid to the enamel's, with its along-jaw part removed.
    """
    def physical(v):
        return np.asarray(v, dtype=np.float64)[..., ::-1] * voxel_um[::-1]

    jaw = np.array([0.0, 0.0, 1.0])
    up = physical(np.array(ndi.center_of_mass(enamel))) - physical(np.array(ndi.center_of_mass(molar)))
    up -= jaw * up.dot(jaw)
    if np.linalg.norm(up) < 1e-9:
        up = np.array([0.0, 1.0, 0.0])
    up /= np.linalg.norm(up)
    across = np.cross(jaw, up)
    centre = physical(np.array(ndi.center_of_mass(molar)))
    basis = np.stack([jaw, up, across])

    def frame(verts: np.ndarray) -> np.ndarray:
        return ((physical(verts) - centre) @ basis.T).astype(np.float32)
    return frame


# ---------------------------------------------------------------------------
# The interactive map
# ---------------------------------------------------------------------------

_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thickness_viewer.html")


# The patch kinds' deficits are stored as tenths of a percent in 16 signed
# bits, this value meaning "not judged"; their patches as 0 or 1 in a byte.
_NO_DEFICIT = -32768
PATCH_KINDS = ("thin", "low_density")


def _pack(layers: dict) -> tuple[bytes, dict]:
    """The meshes as one little-endian blob plus where each array sits in it.

    Positions are quantized to 16 bits over the shared bounding box (a few
    hundredths of a micrometre on a molar row); thickness is stored in tenths
    of a micrometre, 65535 meaning no measurement. Each patch kind's deficit
    is stored in tenths of a percent, signed (``_NO_DEFICIT``: not judged),
    and whether a point lies in a counted patch as a byte.
    """
    allpos = np.concatenate([layer["pos"] for layer in layers.values()])
    lo, hi = allpos.min(axis=0), allpos.max(axis=0)
    scale = np.where(hi > lo, (hi - lo) / 65535.0, 1.0)
    blob = bytearray()

    def add(arr: np.ndarray) -> int:
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(np.ascontiguousarray(arr).tobytes())
        return offset

    layout = {"bbox_min": [float(v) for v in lo], "scale": [float(v) for v in scale], "layers": {}}
    for name, layer in layers.items():
        q = np.round((layer["pos"] - lo) / scale).astype("<u2")
        entry = {"n_vertices": int(len(q)), "n_faces": int(len(layer["faces"])),
                 "pos": add(q), "faces": add(layer["faces"].astype("<u4"))}
        if "value" in layer:
            v = layer["value"]
            enc = np.where(np.isfinite(v), np.clip(np.round(np.nan_to_num(v) * 10.0), 0, 65534), 65535)
            entry["value"] = add(enc.astype("<u2"))
        if "tooth" in layer:
            entry["tooth"] = add(layer["tooth"].astype("u1"))
        for kind in PATCH_KINDS:
            if kind in layer:
                d = np.asarray(layer[kind], dtype=np.float64)
                enc = np.where(np.isfinite(d), np.clip(np.round(np.nan_to_num(d) * 1000.0), -32767, 32767), _NO_DEFICIT)
                entry[kind] = add(enc.astype("<i2"))
                flag = np.asarray(layer.get(f"{kind}_patch", np.zeros(len(d))), dtype=np.float64)
                entry[f"{kind}_patch"] = add((np.nan_to_num(flag) >= 0.5).astype("u1"))
        layout["layers"][name] = entry
    return bytes(blob), layout


def _patch_meta(block: dict | None) -> dict | None:
    """What the map's panel shows of the patches: each kind's call and its numbers, row and per tooth."""
    if not block:
        return None
    keys = ("share", "patches", "patches_per_mm2", "depth", "share_judged", "largest_mm2")
    out = {"around_um": block.get("around_um"), "margin_um": block.get("margin_um")}
    for kind in PATCH_KINDS:
        entry = block.get(kind)
        if not entry:
            continue
        out[kind] = {"by": entry.get("thinner_by", entry.get("less_by")),
                     **{k: entry.get(k) for k in keys},
                     "teeth": [{"name": t["name"], **{k: t.get(k) for k in keys}} for t in entry.get("teeth", [])]}
    return out


def write_viewer(path: str, display: dict, *, sample: str, summary: dict, vmax: float) -> str:
    """A self-contained HTML page: the thickness map in 3-D, with its legend."""
    blob, layout = _pack(display["layers"])
    payload = base64.b64encode(gzip.compress(blob, 6)).decode("ascii")
    meta = {
        "sample": sample, "vmax": float(vmax), "layout": layout, "teeth": display["teeth"],
        "reported": {k: summary.get(k) for k in ("median", "p25", "p75", "n_points", "share_measured",
                                                 "junction_area_mm2")},
        "from_outer": {k: (summary.get("from_outer_surface") or {}).get(k) for k in ("median", "p25", "p75")},
        "per_tooth": summary.get("teeth") or [],
        "background_from": (summary.get("levels") or {}).get("background_from"),
        "voxel_um": display["voxel_um"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "patches": _patch_meta(summary.get("patches")),
        "modes": ["thickness"] + [k for k in PATCH_KINDS
                                  if any(k in layer for layer in display["layers"].values())],
    }
    with open(_TEMPLATE, encoding="utf-8") as handle:
        page = handle.read()
    page = (page.replace("__SAMPLE__", html.escape(sample))
                .replace('"__META__"', json.dumps(meta).replace("</", "<\\/"))
                .replace("__DATA__", payload))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return path


# ---------------------------------------------------------------------------
# The still figure
# ---------------------------------------------------------------------------

# Sequential blue, dark to light: on the dark stage, thicker enamel is brighter.
RAMP = ("#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec",
        "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb")
STAGE, STAGE_INK, STAGE_INK_2, STAGE_INK_3 = "#0e1213", "#e2e9e7", "#a3aeac", "#74807e"
NO_DATA, CONTEXT = "#8a8378", "#4b5557"


def write_figure(path: str, display: dict, *, sample: str, summary: dict, vmax: float) -> str:
    """The enamel's thickness map seen from above and from one side, with a colour bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    layers = display["layers"]
    cmap = LinearSegmentedColormap.from_list("thickness", RAMP)
    norm = Normalize(0.0, vmax)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), facecolor=STAGE, gridspec_kw={"hspace": 0.22})
    # Viewer frame: x along the jaw, y toward the crowns, z across. From above
    # the screen shows (x, -z) and depth is y; from the side (x, y), depth z.
    views = (("Seen from above", lambda p: (p[:, 0], -p[:, 2]), lambda p: p[:, 1]),
             ("Seen from the side", lambda p: (p[:, 0], p[:, 1]), lambda p: p[:, 2]))
    for ax, (title, project, depth) in zip(axes, views):
        pts, colors = [], []
        for key, missing in (("dentin", CONTEXT), ("enamel", NO_DATA)):
            layer = layers.get(key)
            if layer is None:
                continue
            vals = layer["value"]
            rgba = cmap(norm(np.nan_to_num(vals)))
            rgba[~np.isfinite(vals)] = matplotlib.colors.to_rgba(missing)
            pts.append(layer["pos"])
            colors.append(rgba)
        if pts:
            p = np.concatenate(pts)
            c = np.concatenate(colors)
            order = np.argsort(depth(p))
            x, y = project(p[order])
            ax.scatter(x / 1000, y / 1000, s=0.5, c=c[order], lw=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_facecolor(STAGE)
        ax.set_title(title, color=STAGE_INK_2, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        x0, x1 = ax.get_xlim()
        y0, _ = ax.get_ylim()
        ax.plot([x0, x0 + 0.5], [y0, y0], color=STAGE_INK_2, lw=2, clip_on=False)
        ax.text(x0 + 0.25, y0, "0.5 mm", color=STAGE_INK_2, fontsize=8, ha="center", va="top")
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.02, extend="max")
    cb.set_label("Enamel thickness (µm)", color=STAGE_INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=STAGE_INK_3)
    median = summary.get("median")
    fig.suptitle(f"{sample}: enamel thickness" + (f", median {median:.0f} µm" if median is not None else ""),
                 color=STAGE_INK, x=0.125, ha="left", fontsize=12)
    fig.text(0.125, 0.04, "Colour: enamel thickness, measured along lines through enamel between its outer "
             "surface and the enamel-dentin junction. Grey: dentin and roots. Median: from the junction.",
             color=STAGE_INK_3, fontsize=8)
    fig.savefig(path, dpi=150, facecolor=STAGE, bbox_inches="tight")
    plt.close(fig)
    return path
