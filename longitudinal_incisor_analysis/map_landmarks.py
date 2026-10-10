"""
Map anatomical landmarks onto the incisor mineralization trajectory.

Author: Ethan Dinh
Date:   2026-09-14

`enamel_density.py` reports density against position along the incisor, but
that position is an index into a resliced volume -- it carries no anatomy. This
tool answers the question that makes the profile interpretable: *what is at
34 % along the tooth?* Scroll to a slice in napari, mark it, and get back where
it falls on the trajectory.

Landmarks are z, not points
---------------------------
A landmark here is a **z-plane**, or a pair of them: the first molar occupies a
range of z, eruption happens at one z. So picking is done with the slice slider
and a keypress rather than by clicking a 3-D position -- the y and x of a click
would be discarded anyway, and asking for them suggests they matter.

Two kinds of row come out of it:

  * **event** -- one z. Eruption.
  * **span**  -- a start and an end z, marked with two separate keys and
    recombined on write. M1, M2, M3.

How it works
------------
`cmpr.py` resamples the incisor along its centerline and writes the geometry
to `<sample>_cmpr_transform.npz`. Mapping a marked z onto the trajectory is
then a search for the centerline point sitting closest to that z-plane:

    marked z (in the original scan)
        -> nearest centerline point by z    (the crop origin is applied on load)
        -> slice index -> z_percent         (matched to the density profile)

Volume z is *not* arc length along the tooth. The incisor curves, so equal
steps in z are unequal steps along the centerline; that conversion is the
point of the mapping. `z_residual_vox` reports how far the matched centerline
point sits from the requested plane -- normally sub-voxel, and large only where
the centerline runs steeply through z, where a z-plane is a poor description of
position anyway.

What you pick on
----------------
The volume stage 1 reslices is the *incisor* volume, which is masked to the
incisor alone -- every other structure is zero. Molars and alveolar bone are
simply not in it. So picking happens on the **reoriented volume** instead: the
same array shape, the same coordinate frame, with all the tissue still present.
It is found automatically next to the transform, and the molar and bone
segmentations are overlaid on it when they are there.

Usage
-----
    # Pick landmarks interactively, then write the mapping
    python map_landmarks.py \\
        -t ".../Segments/Odam-CRE-A/Odam-CRE-A_cmpr_transform.npz" \\
        --profile ".../RESULTS/enamel_density_profile.csv" \\
        -o ".../RESULTS/Odam-CRE-A_landmarks.csv"

    # Re-map a saved landmark CSV without opening the viewer
    python map_landmarks.py -t ... --points saved_landmarks.csv -o out.csv

    # M1-M3 from the molar step's tooth split; pick only what it cannot place
    python map_landmarks.py -t ... --molars ".../<sample>_molar_enamel.json" -o out.csv

Molars from the tooth split
---------------------------
The molar step (`quantify_molar/molar_enamel.py`) splits the molar row into
M1, M2 and M3 and records where each crown -- the tooth's enamel -- runs along
the jaw, in this same volume z. Where an oblique contact lets two crowns
overlap in z, the slices go to the tooth in front, as they are marked by eye:
M1 keeps its whole crown and M2 starts where M1's ends. (On P22_TGM2_KO_F_1
that places M1-M3 within 1-4 slices of the hand-marked ends, and M3's far end
within 7.) `--molars` takes those spans for any molar not picked by hand (a
pick always wins), and only when the split found all three teeth. Each row of
the output says where it came from: `source` is "teeth" or "manual".
`detect_landmarks.py` places them from the split itself, with eruption, and
MicroCT Studio's Landmarks step runs that.

In the viewer: scroll to a slice and press that boundary's key --
`Q` eruption, `G`/`H` for where M1 starts and ends, `J`/`K` for M2, `N`/`W` for
M3. Every key marks exactly one z, and pressing it again just moves that mark.
Each one draws itself as a coloured plane across the tooth, visible in both
slice view and 3-D (`Shift-U` toggles), so what you have marked is always on
screen. `U` undoes. Close the window to write the CSV.

The keys look arbitrary because they are constrained: napari binds most single
letters and the digits 1-4 to layer-mode switches, and those win while the image
layer is selected -- which it must be for the scroll wheel to move through z.
Only G H J K N Q U W are free, so the keys are grouped by hand position instead
of by mnemonic. `assert_keys_available()` re-checks this against napari's own
shortcut table.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.view_napari import (  # noqa: E402
    auto_contrast,
    downsample,
    load_sample,
)

log = logging.getLogger("map_landmarks")

# Landmarks offered as defaults in the viewer. These are the ones the incisor
# trajectory is usually read against; any name can be typed instead.
# One key per landmark. Pressing a key marks the current z for that landmark;
# pressing it a second time records the end of its extent. Keys are the
# landmark's own initial where that is free, so there is nothing to memorise.
# One key per boundary. Every key marks exactly one z, and pressing it again
# just moves that z -- there is no press count to track and no hidden "waiting
# for the other end" state, which is what made a two-press span mode hard to
# use: nothing on screen distinguished "start recorded, waiting" from "nothing
# recorded", and a third press silently discarded the pair.
#
# A molar's extent is two named boundaries rather than one span: `G` marks where
# M1 starts, `H` where it ends. They are recombined into a single span row on
# write.
#
# KEY CHOICE IS CONSTRAINED. napari binds most single letters and the digits
# 1-4 to layer-mode switches at the *viewer* level -- `1` and `2` are
# activate_image_pan_zoom_mode and activate_image_transform_mode, `E`, `A` and
# `D` are mode switches too -- and those win over a viewer keybinding while the
# image layer is selected, which it must be for the scroll wheel to move
# through z. Only G H J K N Q U W are free in a default install, so the keys
# below are drawn from that set and grouped by hand position rather than chosen
# as mnemonics. `assert_keys_available()` re-checks this at import.
LANDMARK_KEYS = (
    ("q", "eruption", None),
    ("g", "M1", "start"),
    ("h", "M1", "end"),
    ("j", "M2", "start"),
    ("k", "M2", "end"),
    ("n", "M3", "start"),
    ("w", "M3", "end"),
)

# Viewer controls, also from the free set.
UNDO_KEY = "u"
TOGGLE_3D_KEY = "Shift-U"

# Colour per landmark, so a marked plane is identifiable in the 3-D view without
# having to read its label. Both boundaries of a molar share its colour.
LANDMARK_COLORS = {
    "eruption": "yellow",
    "M1": "lime",
    "M2": "cyan",
    "M3": "magenta",
}
FALLBACK_COLOR = "white"

DEFAULT_LANDMARKS = tuple(dict.fromkeys(label for _, label, _e in LANDMARK_KEYS))

# Where to look for a display volume when `--volume` is not given, relative to
# the sample root. The reoriented volume comes first deliberately: it is the
# whole hemimandible, so molars and bone are visible in it. The incisor volume
# is the fallback -- it is the same frame, but everything except the incisor has
# been masked to zero, which leaves nothing to pick a molar against.
DISPLAY_VOLUME_PATTERNS = (
    "{sample}_reoriented_volume.nrrd",
    "{sample}_reoriented_volume.tif",
    "volumes/{sample}_incisor_volume.nrrd",
    "volumes/{sample}_incisor_volume.tif",
)

# Companion volumes worth overlaying, and the colormap for each. These are the
# segmentation pipeline's own outputs, in the same frame as the reoriented
# volume, so a molar landmark can be placed on the molar segmentation rather
# than judged from greyscale.
OVERLAY_VOLUMES = (
    ("molar", "volumes/{sample}_molar_volume", "magenta"),
    ("bone", "volumes/{sample}_bone_volume", "bop blue"),
)

CSV_COLUMNS = [
    "landmark",
    "kind",              # "span" or "event"
    "z_percent_start",
    "z_percent_end",
    "z_percent_mid",
    "extent_percent",
    "z_start", "z_end",
    "slice_start", "slice_end",
    "arc_length_start_mm", "arc_length_end_mm", "length_mm",
    "z_residual_vox",
    "source",            # "teeth": placed from the molar step's split; "manual": picked
]

# The landmarks the molar step's tooth split can place: each molar from where
# its crown starts to where it ends along the jaw, in the same volume z.
CROWN_LANDMARKS = ("M1", "M2", "M3")



# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class CMPRTransform:
    """
    The geometry `cmpr.py` used to build a reslice, loaded back.

    Holds the centerline in *original volume* coordinates (the crop origin is
    added back on load, so callers never have to remember it) together with the
    arc length at each point, which is what makes "position along the tooth"
    a physical quantity rather than a slice count.
    """

    def __init__(self, path: str | Path):
        data = np.load(str(path))
        self.path = Path(path)
        self.bbox_min = data["bbox_min"].astype(np.float64)

        # Back into original-volume coordinates. Everything downstream works in
        # that frame, so the offset is applied exactly once, here.
        self.centerline = data["centerline"].astype(np.float64) + self.bbox_min

        self.frames = data["frames"].astype(np.float64)
        self.slice_size = int(data["slice_size"])
        self.spacing = float(data["spacing"])
        self.volume_shape = tuple(int(v) for v in data["volume_shape"])

        vs = data["voxel_spacing"].astype(np.float64)
        self.voxel_spacing = None if np.all(np.isnan(vs)) else tuple(vs)

        # Cumulative arc length, in voxels, along the centerline.
        steps = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        self.arc_length = np.concatenate([[0.0], np.cumsum(steps)])

    @property
    def n_slices(self) -> int:
        return len(self.centerline)

    def tip_at_index_0(self, volume: "np.ndarray | None" = None) -> bool:
        """
        Which end of the centerline is the incisal tip?

        `cmpr.py` traces the centerline from one end of the incisor to the
        other, but does not promise which end it starts from -- and on this
        dataset it starts at the TIP. Everything downstream that reports a
        position as "% along the tooth" needs to know, so it is measured here
        rather than assumed.

        The discriminator is MINERALIZATION, not size. Enamel at the incisal
        tip is fully mature and is the densest tissue in the scan; the cervical
        loop is soft-tissue-density matrix that has not mineralized yet. So the
        tip end carries much the higher upper-percentile intensity.

        Area was tried first and is not reliable here: the incisor flares near
        the cervical loop but the crop and the curve's obliquity make the
        apparent cross-section vary enough to flip the comparison between
        specimens. The density contrast is ~190 vs ~119 and does not.

        Bands are taken a short way in from each end, since the outermost
        slices are partial-volume slivers at both ends. Falls back to centerline
        geometry when no volume is supplied, which suffices for fixtures.
        """
        n = self.n_slices
        if n < 20:
            return False

        if volume is None:
            # Without intensities, use distance from the mid-arc point: the tip
            # end of a curved incisor is the far end of the arc.
            mid = self.centerline[n // 2]
            d0 = float(np.linalg.norm(self.centerline[0] - mid))
            d1 = float(np.linalg.norm(self.centerline[-1] - mid))
            return d0 > d1

        band = max(5, n // 20)
        axis0 = np.clip(self.centerline[:, 0].astype(int), 0, volume.shape[0] - 1)

        def peak_density(lo: int, hi: int) -> float:
            planes = np.unique(axis0[lo:hi])
            if planes.size == 0:
                return -1.0
            return float(np.percentile(volume[planes], 99.5))

        start = peak_density(band, 2 * band)
        end = peak_density(n - 2 * band, n - band)
        return start > end

    @property
    def mm_per_voxel(self) -> float:
        """
        Isotropic voxel size for reporting distances in mm.

        The mean of the three axes: the centerline runs obliquely through the
        volume, so no single axis spacing describes a step along it. Returns
        NaN when the source volume carried no spacing, which propagates into
        the mm columns rather than silently reporting voxels as millimetres.
        """
        if self.voxel_spacing is None:
            return float("nan")
        return float(np.mean(self.voxel_spacing))

    def project_z(self, z: float) -> dict:
        """
        Find where a volume z-plane crosses the centerline.

        A landmark read off the viewer's z-slider is a *plane*, not a point:
        "the first molar starts at z = 412" says nothing about y or x. So the
        match is on the z coordinate alone -- the centerline point whose own z
        is closest to the given plane.

        In the reoriented frame axis 0 is the anterior-posterior long axis, so
        scrolling z really does travel down the tooth. But z in the volume is
        not arc length along the centerline: the incisor curves, so equal steps
        in z are unequal steps along the tooth. That conversion is exactly what
        this does.

        `z_residual` is how far the matched centerline point's z sits from the
        requested plane. It is normally sub-voxel. It grows where the centerline
        runs steeply through z -- near a sharply curving tip -- and there a
        z-plane is a poor description of position along the tooth. Reported so
        that case is visible rather than silent.
        """
        z = float(z)
        dz = np.abs(self.centerline[:, 0] - z)
        idx = int(np.argmin(dz))

        mm = self.mm_per_voxel
        return {
            "slice_index": idx,
            "arc_length_vox": float(self.arc_length[idx]),
            "arc_length_mm": float(self.arc_length[idx] * mm),
            "z_residual_vox": float(dz[idx]),
            "curve_point": self.centerline[idx],
        }

    def z_span(self) -> tuple[float, float]:
        """Range of volume z the centerline actually covers."""
        return float(self.centerline[:, 0].min()), float(self.centerline[:, 0].max())


def z_to_percent(z: int, depth: int, orientation: str) -> float:
    """
    Map a slice index to 0 % (cervical loop) .. 100 % (tip).

    Mirrors `enamel_density.py`'s function of the same name. The two must agree
    exactly or landmarks land on the wrong end of the profile they annotate --
    so this is a deliberate duplicate of five lines rather than an import that
    would drag the whole analysis module (and its SciPy fitting stack) in to
    read one convention.
    """
    if depth <= 1:
        return 0.0
    if orientation == "tip-low-z":
        return 100.0 * (depth - 1 - z) / (depth - 1)
    return 100.0 * z / (depth - 1)


def centerline_index_to_percent(idx: int, depth: int, tip_at_index_0: bool) -> float:
    """
    Map a CENTERLINE index to 0 % (cervical loop) .. 100 % (tip).

    Distinct from `z_to_percent`, and the distinction matters. `z_to_percent`
    converts a *volume z-plane* using the orientation `enamel_density.py`
    recorded. This converts a *centerline index*, whose direction is a property
    of how `cmpr.py` traced the curve and has nothing to do with that flag.

    `cmpr.py` does NOT guarantee the trace starts at the cervical loop -- on
    this dataset it starts at the tip -- so the direction has to be measured,
    not assumed. `CMPRTransform.tip_at_index_0` does that from the volume
    geometry; pass its value here.

    Getting this backwards mirrors every landmark: molars come out ordered
    M3, M2, M1 and the eruption plane lands cervical to all of them, which is
    anatomically impossible. See `test_tip_low_z_not_mirrored`.
    """
    if depth <= 1:
        return 0.0
    frac = idx / (depth - 1)
    return 100.0 * (1.0 - frac) if tip_at_index_0 else 100.0 * frac


def orientation_from_profile(csv_path: str | Path) -> str | None:
    """
    Recover the orientation `enamel_density.py` used, from its output CSV.

    The profile stores both `z` and `z_percent`, so the convention is implied
    by whether they run together or opposed. Reading it back beats asking the
    user to remember: the orientation is auto-detected per specimen, and the
    reference specimen's tip is at low z -- the reverse of the plotting
    convention -- so a guess is wrong about as often as it is right.
    """
    try:
        with open(csv_path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("z_percent")]
    except (OSError, csv.Error) as exc:
        log.warning("Could not read orientation from %s: %s", csv_path, exc)
        return None

    pairs = []
    for row in rows:
        try:
            pairs.append((float(row["z"]), float(row["z_percent"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(pairs) < 2:
        log.warning("Not enough rows in %s to infer orientation", csv_path)
        return None

    (z0, p0), (z1, p1) = pairs[0], pairs[-1]
    if z1 == z0:
        return None
    return "tip-low-z" if (p1 - p0) / (z1 - z0) < 0 else "tip-high-z"


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def map_landmarks(
    transform: CMPRTransform,
    marks: list[dict],
    orientation: str = "tip-high-z",
    volume: "np.ndarray | None" = None,
    tip_at_index_0: bool | None = None,
) -> list[dict]:
    """
    Turn picked z-marks into positions on the trajectory.

    `marks` is a list of ``{"landmark": name, "z_start": z, "z_end": z|None}``.
    A span carries both ends; an event carries only `z_start`.

    Spans are reported start-to-end **in trajectory order**, not in the order
    the z values were marked. Those differ whenever the specimen is
    `tip-low-z`: there, increasing volume z is *decreasing* z_percent, so the
    mark made at lower z is the span's end. Getting this wrong silently
    reverses every molar's extent.

    `orientation` describes the VOLUME, and is recorded for provenance. It does
    not enter the z_percent conversion: positions come from centerline indices,
    whose direction is measured from the tooth's own geometry (see
    `CMPRTransform.tip_at_index_0`). Ordering of a span's two ends falls out of
    the computed percentages, so it stays correct either way.

    Pass `volume` so the tip end can be measured, or `tip_at_index_0` directly
    if it is already known.
    """
    depth = transform.n_slices
    if tip_at_index_0 is None:
        tip_at_index_0 = transform.tip_at_index_0(volume)
    log.info("Centerline runs %s",
             "tip -> cervical loop" if tip_at_index_0 else "cervical loop -> tip")

    total_arc_mm = float(transform.arc_length[-1]) * transform.mm_per_voxel

    def arc_from_cervical(p: dict) -> float:
        """Arc length measured from the cervical loop, so it rises with z_percent."""
        return total_arc_mm - p["arc_length_mm"] if tip_at_index_0 else p["arc_length_mm"]

    rows = []

    for mark in marks:
        name = mark["landmark"]
        z0 = float(mark["z_start"])
        z1 = mark.get("z_end")
        is_span = z1 is not None

        p0 = transform.project_z(z0)
        p1 = transform.project_z(float(z1)) if is_span else p0

        # `slice_index` indexes the CENTERLINE. Its direction is a property of
        # how cmpr.py traced the curve -- NOT of the volume's z orientation --
        # so it is measured from the tooth's geometry, not taken from the
        # `orientation` flag. Using that flag here mirrored every landmark.
        pct0 = centerline_index_to_percent(p0["slice_index"], depth, tip_at_index_0)
        pct1 = centerline_index_to_percent(p1["slice_index"], depth, tip_at_index_0)

        # Order by position along the tooth, not by volume z.
        if is_span and pct1 < pct0:
            p0, p1 = p1, p0
            pct0, pct1 = pct1, pct0
            z0, z1 = float(z1), z0

        mm = transform.mm_per_voxel
        length_mm = abs(p1["arc_length_mm"] - p0["arc_length_mm"]) if is_span else None
        residual = max(p0["z_residual_vox"], p1["z_residual_vox"])

        rows.append({
            "landmark": name,
            "kind": "span" if is_span else "event",
            "z_percent_start": round(pct0, 3),
            "z_percent_end": round(pct1, 3) if is_span else "",
            "z_percent_mid": round((pct0 + pct1) / 2.0, 3) if is_span else round(pct0, 3),
            "extent_percent": round(abs(pct1 - pct0), 3) if is_span else "",
            "z_start": round(z0, 2),
            "z_end": round(float(z1), 2) if is_span else "",
            "slice_start": p0["slice_index"],
            "slice_end": p1["slice_index"] if is_span else "",
            # Arc length is reported FROM THE CERVICAL LOOP, so it rises with
            # z_percent. The raw cumulative arc runs from centerline index 0,
            # which on a tip-first trace is the tip -- flip it so the two
            # columns cannot disagree about which way the tooth runs.
            "arc_length_start_mm": round(arc_from_cervical(p0), 4),
            "arc_length_end_mm": round(arc_from_cervical(p1), 4) if is_span else "",
            "length_mm": round(length_mm, 4) if is_span else "",
            "z_residual_vox": round(residual, 3),
            "source": mark.get("source") or "manual",
        })

    rows.sort(key=lambda r: r["z_percent_start"])
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d landmark(s) to %s", len(rows), path)


def read_marks_csv(path: str | Path) -> list[dict]:
    """
    Read back a landmark CSV as marks, so a point set can be re-mapped.

    Reads `z_start`/`z_end` -- the raw picks -- rather than the derived
    percentages, so re-mapping against a new reslice recomputes positions
    instead of copying stale ones.
    """
    marks = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                z_end = row.get("z_end", "")
                marks.append({
                    "landmark": row["landmark"],
                    "z_start": float(row["z_start"]),
                    "z_end": float(z_end) if z_end not in ("", None) else None,
                    "source": row.get("source") or "manual",
                })
            except (KeyError, TypeError, ValueError):
                log.warning("Skipping unreadable row: %s", row)
    return marks


def marks_from_molars(path: str | Path) -> list[dict]:
    """
    M1-M3 as the molar step's tooth split placed them: each crown along the
    jaw, less the slices where the crown in front of it still shows
    (`landmark_z_start`/`landmark_z_end`), so neighbours meet as they are
    marked by hand.

    `path` is the molar step's `<sample>_molar_enamel.json`. Nothing unless the
    split found all three teeth -- with two, a fused pair may carry a molar's
    name, and a mark from it would be wrong.
    """
    import json

    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("n_teeth") != len(CROWN_LANDMARKS):
        log.warning("The molar split in %s found %s teeth, not %d: no molar placed from it",
                    Path(path).name, summary.get("n_teeth"), len(CROWN_LANDMARKS))
        return []
    return [{"landmark": t["name"], "z_start": float(t["landmark_z_start"]), "z_end": float(t["landmark_z_end"]),
             "source": "teeth"}
            for t in summary.get("teeth") or []
            if t.get("name") in CROWN_LANDMARKS and t.get("landmark_z_start") is not None]


def merge_molar_marks(marks: list[dict], molars: list[dict]) -> list[dict]:
    """The marks with the molars from the tooth split added; a molar picked by hand keeps its pick."""
    kept = [m for m in marks if m.get("source") != "teeth"]
    picked = {m["landmark"] for m in kept}
    return kept + [m for m in molars if m["landmark"] not in picked]


# ---------------------------------------------------------------------------
# Keybinding safety
# ---------------------------------------------------------------------------

def napari_reserved_keys() -> set[str]:
    """
    Single keystrokes napari already binds, normalised to upper case.

    Read from napari's own shortcut settings rather than hard-coded, so a
    version that moves its mode switches is reflected instead of assumed.
    Returns an empty set if the settings cannot be read -- a missing guard is
    better than refusing to start.
    """
    try:
        from napari.settings import get_settings
        shortcuts = get_settings().shortcuts.shortcuts
    except Exception as exc:  # pragma: no cover - depends on napari internals
        log.debug("Could not read napari shortcuts: %s", exc)
        return set()

    reserved = set()
    for bindings in shortcuts.values():
        for binding in bindings:
            reserved.add(str(binding).strip().upper().replace("SHIFT-", "SHIFT+"))
    return reserved


def _normalise_key(key: str) -> str:
    """
    Normalise one keystroke the way napari will when it binds it.

    Used instead of a plain string compare because napari's normalisation is
    lossy in a way that matters: a bare shifted symbol such as "!" comes back
    as an empty key, so "!", "@" and "#" all collapse together and only the
    last of them binds. Falls back to a simple upper-casing if napari's parser
    is unavailable.
    """
    try:
        from napari.utils.key_bindings import bind_key
        keymap: dict = {}
        bind_key(keymap, key, lambda _v: None, overwrite=True)
        return "|".join(str(k) for k in keymap).upper()
    except Exception:  # pragma: no cover - depends on napari internals
        return key.strip().upper().replace("SHIFT-", "SHIFT+")


def assert_keys_available() -> list[str]:
    """
    Check this module's keys against napari's own, and report any collision.

    This exists because a collision is invisible at runtime: napari's binding
    simply wins, the key silently does something else (`1` switched the image
    layer to pan/zoom mode), and the landmark never gets marked. Nothing raises,
    so without this check the only symptom is a key that appears not to work.

    Returns a list of human-readable collision descriptions; empty means clear.
    """
    reserved = napari_reserved_keys()
    if not reserved:
        return []

    ours = [(key, label if end is None else f"{label} {end}")
            for key, label, end in LANDMARK_KEYS]
    ours += [(UNDO_KEY, "undo"), (TOGGLE_3D_KEY, "toggle 3-D")]

    collisions = []
    for key, what in ours:
        norm = key.strip().upper().replace("SHIFT-", "SHIFT+")
        if norm in reserved:
            collisions.append(f"{key!r} ({what}) is already a napari shortcut")

    # Our own keys must also be distinct from each other *after napari's own
    # normalisation*, which is stricter than a string compare: a bare shifted
    # symbol ("!") normalises to an empty key, so several such keys collapse
    # into one and only the last one binds. Ask napari to normalise rather than
    # reimplementing its rules.
    seen: dict[str, str] = {}
    for key, what in ours:
        norm = _normalise_key(key)
        if norm in seen:
            collisions.append(
                f"{key!r} ({what}) normalises to the same key as {seen[norm]!r}")
        seen[norm] = key
    return collisions


# ---------------------------------------------------------------------------
# Finding what to display
# ---------------------------------------------------------------------------

def find_display_volume(sample_root: Path, sample: str) -> Path | None:
    """
    Pick the volume to pick landmarks on, preferring the whole hemimandible.

    The incisor volume is what stage 1 reslices, but it is masked to the
    incisor alone -- every other structure is zero. Landmarks like the molars
    and the alveolar bone are simply not in it, so picking them there is
    impossible. The reoriented volume is the same array shape and the same
    coordinate frame, with all the tissue still present, which is what makes
    those landmarks visible.
    """
    for pattern in DISPLAY_VOLUME_PATTERNS:
        candidate = sample_root / pattern.format(sample=sample)
        if candidate.exists():
            return candidate
    return None


def infer_sample_root(transform_path: Path, sample: str) -> Path | None:
    """
    Guess the segmentation output folder for a sample, from the transform path.

    Checks the transform's own directory and its parent, then the same two with
    a `<sample>/` suffix -- covering both `cmpr.py --per-sample-dir` layouts and
    a CMPR directory sitting beside the segmentation outputs.
    """
    bases = [transform_path.parent, transform_path.parent.parent]
    candidates = [b / sample for b in bases] + bases
    for base in candidates:
        if (base / "volumes").is_dir() or any(
            (base / p.format(sample=sample)).exists()
            for p in DISPLAY_VOLUME_PATTERNS
        ):
            return base
    return None


def load_overlays(
    sample_root: Path, sample: str, expected_shape: tuple[int, int, int],
) -> list[tuple[str, np.ndarray, str]]:
    """
    Load the molar and bone segmentations, if they are sitting next to the scan.

    Each is checked against the display volume's shape and skipped on a
    mismatch rather than displayed at the wrong scale -- an overlay that does
    not line up is worse than no overlay, because it looks authoritative while
    pointing at the wrong voxels.
    """
    found = []
    for label, pattern, colormap in OVERLAY_VOLUMES:
        for suffix in (".nrrd", ".tif"):
            path = sample_root / (pattern.format(sample=sample) + suffix)
            if not path.exists():
                continue
            try:
                volume, _ = load_sample(path)
            except (OSError, ValueError) as exc:
                log.warning("Could not read %s overlay: %s", label, exc)
                break
            if volume.shape != expected_shape:
                log.warning(
                    "Skipping %s overlay: shape %s does not match display "
                    "volume %s", label, volume.shape, expected_shape,
                )
                break
            log.info("Overlay: %s (%s)", label, path.name)
            found.append((label, volume, colormap))
            break
    return found


# ---------------------------------------------------------------------------
# Interactive picking
# ---------------------------------------------------------------------------

def pick_landmarks(
    volume: np.ndarray,
    spacing: tuple[float, float, float] | None,
    transform: CMPRTransform,
    name: str,
    contrast_limits: tuple[float, float],
    scale_factor: int = 1,
    overlays: list[tuple[str, np.ndarray, str]] | None = None,
    start_3d: bool = False,
) -> list[dict]:
    """
    Open napari and return the z values the user marked.

    One key per boundary. Scroll to the slice, press the key -- that is the
    whole interaction, and pressing a key again simply moves its mark. Each one
    draws itself as a plane across the specimen, coloured by landmark; nothing
    is drawn by hand, and the layer is held read-only so a stray click cannot
    add a plane no mark corresponds to.

    Pressing a landmark's key twice records the second z as that landmark's
    end, which is how a molar's extent gets captured without a separate
    span mode to keep track of.

    Returns marks in *full-resolution* volume z, undoing any display
    downsample.
    """
    import napari

    for problem in assert_keys_available():
        log.warning("Keybinding conflict: %s -- that key will not mark.", problem)

    display_scale = spacing if spacing is not None else (1.0, 1.0, 1.0)
    depth = volume.shape[0]

    viewer = napari.Viewer(title=f"Pick landmarks - {name}")
    viewer.add_image(
        volume,
        name=name,
        scale=display_scale,
        contrast_limits=contrast_limits,
        colormap="gray",
        rendering="attenuated_mip",
    )

    for label, overlay, colormap in overlays or []:
        viewer.add_image(
            overlay,
            name=label,
            scale=display_scale,
            colormap=colormap,
            blending="additive",
            opacity=0.55,
            rendering="attenuated_mip",
        )

    viewer.add_points(
        transform.centerline / scale_factor,
        name="centerline",
        size=2,
        face_color="cyan",
        opacity=0.45,
        scale=display_scale,
        out_of_slice_display=True,
    )

    # Marked planes. A shapes layer renders in both 2-D and 3-D, so this is the
    # slice-view readout and the 3-D overview at once -- one representation, no
    # second copy to keep in sync. ndim=3 is required because the layer starts
    # empty: with no data there is nothing to infer dimensionality from, and a
    # 2-D layer rejects the 3-element scale.
    _, ny, nx = volume.shape
    planes = viewer.add_shapes(
        name="landmark planes",
        ndim=3,
        scale=display_scale,
        opacity=0.35,
        blending="additive",
        features={"label": []},
        text={"string": "{label}", "size": 9, "color": "white",
              "anchor": "upper_left"},
    )

    # (landmark, end) -> z. Each key owns exactly one slot, so pressing a key
    # again moves that mark rather than advancing a state machine.
    marked: dict[tuple[str, str | None], int] = {}
    press_order: list[tuple[str, str | None]] = []

    def current_z() -> int:
        return int(viewer.dims.current_step[0])

    def summary() -> str:
        if not marked:
            keys = "  ".join(f"{k}={label}" + (f"\u00b7{end}" if end else "")
                             for k, label, end in LANDMARK_KEYS)
            return f"no marks yet  |  {keys}"

        parts = []
        for label in DEFAULT_LANDMARKS:
            a = marked.get((label, "start"))
            b = marked.get((label, "end"))
            point = marked.get((label, None))
            if point is not None:
                parts.append(f"{label} {point}")
            elif a is not None and b is not None:
                parts.append(f"{label} {min(a, b)}-{max(a, b)}")
            elif a is not None:
                parts.append(f"{label} start {a} (no end)")
            elif b is not None:
                parts.append(f"{label} end {b} (no start)")
        return "  |  ".join(parts) if parts else "no marks yet"

    def redraw():
        """
        Rebuild the plane layer from `marked`.

        Called after every change rather than incrementally patched: the state
        is a handful of z values, so a full rebuild is both cheap and immune to
        the layer drifting out of step with the marks it is meant to show.
        """
        planes.data = []
        rects, colors, labels = [], [], []
        for (label, end), z in marked.items():
            rects.append(np.array(
                [[z, 0, 0], [z, 0, nx], [z, ny, nx], [z, ny, 0]], dtype=float))
            colors.append(LANDMARK_COLORS.get(label, FALLBACK_COLOR))
            labels.append(label if end is None else f"{label} {end}")
        if rects:
            planes.add_rectangles(rects, edge_color=colors, face_color=colors)
            planes.features = {"label": labels}
        # Re-assert read-only here, not once at setup: assigning `.data` resets
        # `editable` back to True, so a layer locked before the first redraw
        # quietly becomes editable again on the next one.
        planes.mode = "pan_zoom"
        planes.editable = False

    def record(label: str, end: str | None):
        z = current_z()
        key = (label, end)
        moved = key in marked
        marked[key] = z
        if not moved:
            press_order.append(key)
        name = label if end is None else f"{label} {end}"
        log.info("%-12s z=%-5d %s", name, z, "(moved)" if moved else "")
        viewer.status = summary()
        redraw()

    def bind(key: str, label: str, end: str | None):
        @viewer.bind_key(key, overwrite=True)
        def _handler(_v, _l=label, _e=end):
            record(_l, _e)
        _handler.__doc__ = (f"Mark {label}" if end is None
                            else f"Mark {label} {end}") + " at this z"

    for key, label, end in LANDMARK_KEYS:
        bind(key, label, end)

    @viewer.bind_key(UNDO_KEY, overwrite=True)
    def undo(_v):
        """Remove the most recent mark."""
        while press_order:
            key = press_order.pop()
            if key in marked:
                del marked[key]
                name = key[0] if key[1] is None else f"{key[0]} {key[1]}"
                log.info("removed %s", name)
                break
        viewer.status = summary()
        redraw()

    @viewer.bind_key(TOGGLE_3D_KEY, overwrite=True)
    def toggle_3d(_v):
        """Switch between slice view and 3-D."""
        viewer.dims.ndisplay = 2 if viewer.dims.ndisplay == 3 else 3
        log.info("display: %dD", viewer.dims.ndisplay)

    # The plane layer is a readout, not a drawing surface (`redraw` keeps it in
    # pan/zoom so a stray click cannot add a rectangle no mark corresponds to).
    # Leave the image layer selected so the scroll wheel still moves through z.
    viewer.layers.selection.active = viewer.layers[0]

    viewer.dims.ndisplay = 3 if start_3d else 2
    viewer.dims.set_current_step(0, depth // 2)
    viewer.scale_bar.visible = spacing is not None
    viewer.scale_bar.unit = "mm"
    viewer.status = summary()
    redraw()

    log.info("")
    log.info("Scroll to a slice, then press:")
    for key, label, end in LANDMARK_KEYS:
        log.info("  %-2s %s", key, label if end is None else f"{label} {end}")
    log.info("Each key marks one z; pressing it again just moves that mark.")
    log.info("  %-2s undo last    %s toggle 2-D / 3-D", UNDO_KEY, TOGGLE_3D_KEY)
    log.info("Close the window when done.")
    log.info("")

    napari.run()

    # Recombine each landmark's boundaries into one row. A molar marked at only
    # one end is still reported -- as a point, with a warning -- rather than
    # dropped: a half-marked molar is far more likely to be an interrupted
    # session than a mistake worth discarding.
    marks = []
    for label in DEFAULT_LANDMARKS:
        point = marked.get((label, None))
        a = marked.get((label, "start"))
        b = marked.get((label, "end"))

        if point is not None:
            marks.append({"landmark": label,
                          "z_start": point * scale_factor, "z_end": None})
        elif a is not None and b is not None:
            lo, hi = (a, b) if a <= b else (b, a)
            marks.append({"landmark": label,
                          "z_start": lo * scale_factor,
                          "z_end": hi * scale_factor})
        elif a is not None or b is not None:
            only = a if a is not None else b
            which = "start" if a is not None else "end"
            log.warning(
                "%s has only its %s marked (z=%d); recording it as a single "
                "position, not a span.", label, which, only,
            )
            marks.append({"landmark": label,
                          "z_start": only * scale_factor, "z_end": None})
    return marks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map anatomical landmarks onto the incisor trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--transform", "-t", type=Path, required=True,
                   help="<sample>_cmpr_transform.npz written by cmpr.py")
    p.add_argument("--volume", "-v", type=Path, default=None,
                   help="Volume to pick landmarks on. Default: the sample's "
                        "reoriented volume, found next to the transform")
    p.add_argument("--sample-root", type=Path, default=None,
                   help="Segmentation output folder for this sample, if it is "
                        "not discoverable from the transform's location")
    p.add_argument("--no-overlays", dest="overlays", action="store_false",
                   help="Do not overlay the molar and bone segmentations")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="CSV to write (default: <transform dir>/<sample>_landmarks.csv)")
    p.add_argument("--points", type=Path, default=None,
                   help="Re-map a saved landmark CSV instead of opening the viewer")
    p.add_argument("--molars", type=Path, default=None,
                   help="The molar step's <sample>_molar_enamel.json: M1-M3 from where its split "
                        "teeth's crowns start and end, for any molar not picked or in --points")
    p.add_argument("--start-3d", action="store_true",
                   help="Open in 3-D view (toggle any time with 'd')")
    p.add_argument("--profile", type=Path, default=None,
                   help="enamel_density_profile.csv, to read the orientation from")
    p.add_argument("--orientation", default=None,
                   choices=("tip-low-z", "tip-high-z"),
                   help="Override the orientation instead of reading it from --profile")
    p.add_argument("--downsample", type=int, default=1, metavar="N",
                   help="Subsample the display volume by N on each axis")
    p.add_argument("--percentiles", type=float, nargs=2, default=(2.0, 99.8),
                   metavar=("LOW", "HIGH"), help="Contrast percentiles")
    p.add_argument("--verbose", action="store_true", help="Debug logging")
    return p


def resolve_orientation(args) -> str:
    """
    Decide which end of the array is the tip.

    Explicit flag wins; otherwise read it from the density profile, which
    recorded what the analysis actually used.

    This is now provenance only. z_percent is derived from centerline indices,
    which are in cervical-loop -> tip order whatever the volume does, so an
    unknown orientation no longer mirrors the output -- it just leaves the
    recorded value less precise.
    """
    if args.orientation:
        log.info("Orientation: %s (from --orientation)", args.orientation)
        return args.orientation

    if args.profile is not None:
        found = orientation_from_profile(args.profile)
        if found:
            log.info("Orientation: %s (read from %s)", found, args.profile.name)
            return found

    log.info(
        "No orientation given and none readable from a profile; recording "
        "tip-high-z. z_percent is unaffected -- it comes from centerline "
        "order, not from volume z."
    )
    return "tip-high-z"


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.transform.exists():
        log.error("Transform '%s' does not exist.", args.transform)
        log.error("It is written by cmpr.py next to the resliced volume.")
        sys.exit(1)
    if args.downsample < 1:
        log.error("--downsample must be >= 1 (got %d).", args.downsample)
        sys.exit(1)

    transform = CMPRTransform(args.transform)
    log.info(
        "Transform: %d centerline points, original volume %s, voxel size %s mm",
        transform.n_slices, transform.volume_shape,
        "unknown" if transform.voxel_spacing is None
        else f"{transform.mm_per_voxel:.5g}",
    )

    orientation = resolve_orientation(args)

    if args.points is not None:
        landmarks = read_marks_csv(args.points)
        log.info("Read %d landmark(s) from %s", len(landmarks), args.points)
    else:
        sample = args.transform.name.replace("_cmpr_transform.npz", "")
        sample_root = args.sample_root or infer_sample_root(args.transform, sample)

        volume_path = args.volume
        if volume_path is None:
            if sample_root is None:
                log.error(
                    "Could not find the sample's volumes from the transform's "
                    "location. Pass --volume, or --sample-root <sample folder>."
                )
                sys.exit(1)
            volume_path = find_display_volume(sample_root, sample)
            if volume_path is None:
                log.error("No display volume for '%s' under %s.", sample, sample_root)
                log.error("Looked for: %s", ", ".join(DISPLAY_VOLUME_PATTERNS))
                sys.exit(1)
            log.info("Display volume: %s", volume_path)

        if not volume_path.exists():
            log.error("Volume '%s' does not exist.", volume_path)
            sys.exit(1)

        volume, spacing = load_sample(volume_path)

        if volume.shape != transform.volume_shape:
            log.error(
                "Volume shape %s does not match the transform's original shape %s. "
                "This transform belongs to a different scan.",
                volume.shape, transform.volume_shape,
            )
            sys.exit(1)

        if "incisor_volume" in volume_path.name:
            log.warning(
                "Picking on the incisor volume: everything except the incisor "
                "is masked to zero, so molars and bone are not visible. Use the "
                "reoriented volume to place those landmarks."
            )

        overlays = []
        if args.overlays and sample_root is not None:
            overlays = load_overlays(sample_root, sample, volume.shape)

        if args.downsample > 1:
            volume = downsample(volume, args.downsample)
            overlays = [(n, downsample(v, args.downsample), c) for n, v, c in overlays]
            if spacing is not None:
                spacing = tuple(s * args.downsample for s in spacing)
            log.info("Downsampled display by %dx to %s", args.downsample, volume.shape)

        limits = auto_contrast(volume, percentiles=tuple(args.percentiles))
        landmarks = pick_landmarks(
            volume, spacing, transform,
            name=volume_path.stem,
            contrast_limits=limits,
            scale_factor=args.downsample,
            overlays=overlays,
            start_3d=args.start_3d,
        )

    if args.molars is not None:
        molars = marks_from_molars(args.molars)
        landmarks = merge_molar_marks(landmarks, molars)
        log.info("Placed %s from the molar split in %s",
                 ", ".join(m["landmark"] for m in landmarks if m.get("source") == "teeth") or "no molar",
                 args.molars.name)

    if not landmarks:
        log.warning("No landmarks picked; nothing written.")
        return

    # Which end of the centerline is the tip is measured from the tooth itself.
    # In the --points path no volume has been loaded, so load one just for that
    # measurement; if none can be found, fall back to centerline geometry.
    tip_volume = locals().get("volume")
    if tip_volume is None:
        try:
            tip_sample = args.transform.name.replace("_cmpr_transform.npz", "")
            tip_root = args.sample_root or infer_sample_root(args.transform, tip_sample)
            vp = find_display_volume(tip_root, tip_sample) if tip_root else None
            if vp is not None and vp.exists():
                tip_volume, _ = load_sample(vp)
                log.info("Read %s to locate the incisal tip", vp.name)
        except Exception as exc:                       # noqa: BLE001
            log.debug("Could not load a volume for tip detection: %s", exc)

    rows = map_landmarks(transform, landmarks, orientation=orientation,
                         volume=tip_volume)

    out = args.output
    if out is None:
        sample = args.transform.name.replace("_cmpr_transform.npz", "")
        out = args.transform.parent / f"{sample}_landmarks.csv"
    write_csv(rows, out)

    log.info("")
    log.info("%-16s %-6s %8s %8s %9s", "landmark", "kind", "z% start", "z% end", "length")
    for row in rows:
        if row["kind"] == "span":
            log.info("%-16s %-6s %8.2f %8.2f %7.3fmm",
                     row["landmark"], row["kind"], row["z_percent_start"],
                     row["z_percent_end"], row["length_mm"])
        else:
            log.info("%-16s %-6s %8.2f %8s %9s",
                     row["landmark"], row["kind"], row["z_percent_start"], "-", "-")


if __name__ == "__main__":
    main()
