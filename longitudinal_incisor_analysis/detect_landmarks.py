"""
Find the incisor trajectory's landmarks from the segmentation: eruption, and M1-M3.

Author: Ethan Dinh
Date:   2026-09-30

`map_landmarks.py` puts landmarks -- planes across the jaw -- on the density
trajectory, but somebody had to pick them first. This places them from what
the segmentation already holds, so a whole cohort is marked in one run;
MicroCT Studio's Landmarks step runs it for each specimen. A mark set by hand
always wins: detection fills in only the landmarks nobody has marked, and the
`source` column says which is which (`auto` or `manual`).

Eruption
--------
Where the incisor's enamel comes out of the bone. The tooth is followed
along its centerline (the straightening's transform), and in each
cross-section rays radiate outward from its enamel -- the labial band of its
surface, the brightest tissue in the tooth -- looking for the jaw bone.
Inside the jaw every one of them meets bone. Toward the tip the bone over the
enamel opens, and eruption is the first cross-section, walking tipward from
the last one where the bone covered all of the enamel, where no ray from the
enamel touches bone any more, nor does for the next 15 cross-sections
(0.09 mm at 6 um), so that a gap in a thin plate of bone does not free it
early.

A ray starts on enamel where the tooth's outer 0.024 mm along it reaches the
incisor's enamel threshold: the valley under the enamel peak of its own
intensity histogram, found as the molars' is (a flatter peak than a molar's:
its log prominence is 0.16-0.54 on the P22 scans, so 0.1 is asked for). The
3 degrees at each edge of the enamel band are left out, where the band thins
to nothing and whether a ray there counts as enamel is chance.

On P22_TGM2_KO_F_1 eruption falls at z = 1615, six slices from the plane
marked there by eye (1621); counting the band's edges too, the bone they
clip keeps it until 1667. On the 12 P22 TGM2 specimens with a usable scan it
falls 424 +/- 23 slices past the front of M1's crown.

It needs the bone to be bone. Masks written before the segmentation learnt
to leave out the soft tissue round the teeth hold a sleeve of it round the
incisor, touching the enamel long after the bone has stopped. So the bone is
cleaned here the way the segmentation now cleans it (`clean_bone_mask`); a
mask already clean comes through unchanged.

Molars
------
M1-M3 from the molar step's own split (`molar_enamel.molar_crowns`): each
crown along the jaw, less the slices where the crown in front of it still
shows, so neighbours meet as they are marked by hand. Placed only when the
split finds all three teeth. The molar step itself need not have run.

Usage
-----
    python detect_landmarks.py -t <sample>_cmpr_transform.npz \\
        --incisor-volume <sample>_incisor_volume.nrrd \\
        --bone-volume <sample>_bone_volume.nrrd \\
        --molar-volume <sample>_molar_volume.nrrd \\
        --marks <sample>_marks.csv --marks-out <sample>_marks.csv \\
        -o <sample>_landmarks.csv --report <sample>_landmarks.json

`--marks` are the marks so far, whose hand-set ones are kept; `--marks-out`
gets them with the detected ones, and `-o` the lot mapped onto the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO / "quantify_molar"))
sys.path.insert(0, str(_REPO / "segment_mandible"))

from data_io import load_nrrd  # noqa: E402
from map_landmarks import (  # noqa: E402
    CROWN_LANDMARKS,
    CMPRTransform,
    map_landmarks,
    orientation_from_profile,
    read_marks_csv,
    write_csv,
)
from molar_enamel import molar_crowns  # noqa: E402
from segmentation.molar_bone import _find_enamel_threshold, clean_bone_mask  # noqa: E402

log = logging.getLogger("detect_landmarks")

VERSION = 2   # 2: eruption from rays out of the enamel (1: a third of the circle open)

AUTO, MANUAL = "auto", "manual"
# What an earlier run placed, which a new run replaces. "teeth" is what the
# molar step called the M1-M3 it placed, before this placed them.
AUTO_SOURCES = (AUTO, "teeth")

# The marks file, as MicroCT Studio reads and writes it.
MARKS_COLUMNS = ("landmark", "kind", "z_start", "z_end", "source")

# Eruption (see the module docstring). Rays round the tooth in each
# cross-section; the tooth's outer layer whose brightest voxel says whether a
# ray starts on enamel; the band's edges left out; the log prominence asked of
# the incisor's enamel peak; the fewest rays on enamel for a cross-section to
# be judged at all; the share of them meeting bone that counts as covered;
# how far along the tooth none may touch bone; and the median window, in
# cross-sections, the share is read through.
N_DIRECTIONS = 120
ENAMEL_DEPTH_MM = 0.024
ENAMEL_EDGE_DEG = 3.0
ENAMEL_MIN_LOG_PROMINENCE = 0.1
MIN_ENAMEL_RAYS = 5
COVERED_SHARE = 0.95
CLEAR_RUN_MM = 0.09
SMOOTH_SECTIONS = 9

# How far out a ray is followed: from the centerline to the tooth's outer
# surface (the last incisor voxel within TOOTH_REACH_MM, past the pulp), then
# REACH_MM beyond it for bone. In the P22 sockets the bone lies a median
# 0.12 mm from the tooth, 99 % of rays within 0.34 mm. Sampled every STEP_VOX
# voxels.
TOOTH_REACH_MM = 0.96
REACH_MM = 0.72
STEP_VOX = 0.75

# Voxel size assumed when the transform records none (the lab's 6 um).
FALLBACK_MM = 0.006


# ---------------------------------------------------------------------------
# Eruption
# ---------------------------------------------------------------------------

def ray_survey(centerline: np.ndarray, frames: np.ndarray, incisor: np.ndarray, bone: np.ndarray,
               n_directions: int = N_DIRECTIONS, tooth_reach_vox: float = TOOTH_REACH_MM / FALLBACK_MM,
               reach_vox: float = REACH_MM / FALLBACK_MM, depth_vox: float = ENAMEL_DEPTH_MM / FALLBACK_MM,
               step: float = STEP_VOX) -> tuple[np.ndarray, np.ndarray]:
    """Rays out from the tooth in every cross-section along the centerline: where they meet bone, and what they start on.

    ``centerline`` is (n, 3) in the volume's own voxel coordinates and
    ``frames`` (n, 2, 3) the two in-plane axes of each cross-section, as the
    straightening's transform holds them. ``incisor`` is the incisor volume:
    its intensities, zero outside the tooth. Returns (hits, surface), each
    (n, n_directions): whether a ray meets bone past the tooth's surface, and
    the brightest the tooth is in its outer ``depth_vox`` there (0 where the
    ray meets no tooth).
    """
    bone = np.ascontiguousarray(bone, dtype=np.uint8)
    angles = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)
    radii = np.arange(0.0, tooth_reach_vox + reach_vox, step)
    in_tooth_reach = radii <= tooth_reach_vox
    depth = max(int(round(depth_vox / step)), 0)
    along = np.arange(len(radii))[None, :]
    hits = np.zeros((len(centerline), n_directions), dtype=bool)
    surface = np.zeros((len(centerline), n_directions), dtype=np.float32)
    for i, (point, (u, v)) in enumerate(zip(centerline, frames)):
        directions = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v            # (d, 3)
        coords = (point + radii[None, :, None] * directions[:, None, :]).reshape(-1, 3).T
        values = ndi.map_coordinates(incisor, coords, order=0, mode="constant", cval=0).reshape(n_directions, -1)
        tooth = (values > 0) & in_tooth_reach
        found = ndi.map_coordinates(bone, coords, order=0, mode="constant", cval=0)
        found = found.reshape(n_directions, -1).astype(bool)
        # The tooth's outer surface along each ray: its last voxel there.
        outer = np.where(tooth.any(axis=1), tooth.shape[1] - 1 - np.argmax(tooth[:, ::-1], axis=1), 0)
        layer = tooth & (along <= outer[:, None]) & (along >= (outer - depth)[:, None])
        surface[i] = np.where(layer, values, 0).max(axis=1)
        beyond = (along > outer[:, None]) & (radii[None, :] <= radii[outer][:, None] + reach_vox)
        hits[i] = (found & beyond).any(axis=1)
    return hits, surface


def enamel_threshold(values: np.ndarray) -> float:
    """The intensity from which the incisor is enamel: the valley under the enamel peak of its log histogram."""
    return float(_find_enamel_threshold(np.asarray(values, dtype=np.float32),
                                        min_log_prominence=ENAMEL_MIN_LOG_PROMINENCE))


def enamel_rays(surface: np.ndarray, threshold: float, edge: int = 1) -> np.ndarray:
    """Which rays start on enamel, less ``edge`` rays at each edge of every stretch of it round the circle."""
    enamel = np.asarray(surface) >= threshold
    core = enamel.copy()
    for shift in range(1, edge + 1):
        core &= np.roll(enamel, shift, axis=1) & np.roll(enamel, -shift, axis=1)
    return core


def enamel_touching(hits: np.ndarray, enamel: np.ndarray, min_rays: int = MIN_ENAMEL_RAYS) -> np.ndarray:
    """The share of each cross-section's enamel rays that meet bone; NaN with too little enamel to tell."""
    n = enamel.sum(axis=1)
    return np.where(n >= min_rays, (enamel & hits).sum(axis=1) / np.maximum(n, 1), np.nan)


def find_eruption(hits: np.ndarray, enamel: np.ndarray, tip_at_index_0: bool, run_sections: int = 15,
                  covered: float = COVERED_SHARE, smooth: int = SMOOTH_SECTIONS) -> dict:
    """Where no ray from the enamel touches bone any more.

    ``hits`` are ``ray_survey``'s and ``enamel`` ``enamel_rays``', a row per
    centerline point; ``tip_at_index_0`` says which end of the centerline is
    the tip. Returns {"index", "covered_until", "reason"}: the eruption's
    centerline index, the last one before it where bone covered the enamel,
    and, when there is no eruption to give, why (index None).
    """
    order = np.arange(len(hits))[::-1] if tip_at_index_0 else np.arange(len(hits))   # cervical loop to tip
    share = enamel_touching(hits, enamel)[order]
    known = ~np.isnan(share)
    if not known.any():
        return {"index": None, "covered_until": None, "reason": "No enamel found on the incisor's surface"}
    along = np.arange(len(share))
    # A cross-section with too little enamel to judge takes its neighbours' share.
    share = ndi.median_filter(np.interp(along, along[known], share[known]), size=smooth, mode="nearest")
    covered_at = np.flatnonzero((share >= covered) & (along >= along[known][0]))
    if not covered_at.size:
        return {"index": None, "covered_until": None,
                "reason": "The bone never covers the incisor's enamel: check the incisor and bone segmentations"}
    last = int(covered_at.max())
    clear = share <= 0.0
    for k in range(last, len(share) - run_sections + 1):
        if clear[k:k + run_sections].all():
            return {"index": int(order[k]), "covered_until": int(order[last]), "reason": None}
    return {"index": None, "covered_until": int(order[last]), "reason": "The enamel still touches bone at the tip"}


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

def molar_marks(molar_volume: np.ndarray) -> tuple[list[dict], dict]:
    """M1-M3 from the molar split, and what the split found. No molar is placed unless all three were found."""
    crowns, how = molar_crowns(molar_volume)
    info = {"split": how, "n_teeth": len(crowns), "crowns": [asdict(c) for c in crowns]}
    if len(crowns) != len(CROWN_LANDMARKS):
        return [], info
    return [{"landmark": c.name, "z_start": float(c.landmark_z_start), "z_end": float(c.landmark_z_end),
             "source": AUTO}
            for c in crowns if c.name in CROWN_LANDMARKS and c.landmark_z_start is not None], info


def merge_marks(marks: list[dict], detected: list[dict]) -> list[dict]:
    """The marks set by hand, and the detected ones for every landmark not set by hand."""
    kept = [m for m in marks if (m.get("source") or MANUAL) not in AUTO_SOURCES]
    by_hand = {m["landmark"] for m in kept}
    return kept + [m for m in detected if m["landmark"] not in by_hand]


def write_marks(path: Path, marks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MARKS_COLUMNS)
        writer.writeheader()
        for mark in marks:
            writer.writerow({"landmark": mark["landmark"],
                             "kind": "span" if mark.get("z_end") is not None else "event",
                             "z_start": mark["z_start"], "z_end": "" if mark.get("z_end") is None else mark["z_end"],
                             "source": mark.get("source") or MANUAL})


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _load(path: Path, what: str) -> np.ndarray:
    volume = load_nrrd(str(path))
    if volume is None:
        raise SystemExit(f"Could not read the {what} {path}")
    return volume


def detect_eruption(transform: CMPRTransform, incisor: np.ndarray, bone_volume: np.ndarray,
                    tip_at_index_0: bool) -> tuple[dict | None, dict]:
    """(the eruption mark or None, what was found on the way). ``incisor`` is the incisor volume."""
    log.info("Cleaning the bone mask")
    bone, cleanup = clean_bone_mask(bone_volume, bone_volume > 0)
    log.info("Bone: floor %s, %d soft-tissue and %d island voxels left out, %d kept",
             "none needed" if cleanup["floor"] is None else f"{cleanup['floor']:.0f}",
             cleanup["soft_tissue"], cleanup["islands"], cleanup["kept"])
    threshold = enamel_threshold(incisor[incisor > 0])
    mm = transform.mm_per_voxel if np.isfinite(transform.mm_per_voxel) else FALLBACK_MM
    log.info("Following the incisor through %d cross-sections, from its enamel (%.0f and brighter)",
             transform.n_slices, threshold)
    hits, surface = ray_survey(transform.centerline, transform.frames, incisor, bone,
                               tooth_reach_vox=TOOTH_REACH_MM / mm, reach_vox=REACH_MM / mm,
                               depth_vox=ENAMEL_DEPTH_MM / mm)
    del bone
    enamel = enamel_rays(surface, threshold, edge=int(round(ENAMEL_EDGE_DEG * N_DIRECTIONS / 360.0)))
    run = max(int(round(CLEAR_RUN_MM / mm)), 1)
    found = find_eruption(hits, enamel, tip_at_index_0, run_sections=run)
    touching = enamel_touching(hits, enamel)
    info = {
        "enamel_threshold": threshold, "enamel_edge_deg": ENAMEL_EDGE_DEG, "run_sections": run,
        "bone_cleanup": cleanup,
        "profile": {"z": [round(float(z), 1) for z in transform.centerline[:, 0]],
                    "enamel_rays": [int(n) for n in enamel.sum(axis=1)],
                    "enamel_touching_bone": [None if np.isnan(t) else round(float(t), 3) for t in touching],
                    "touching_bone": [round(float(s), 3) for s in hits.mean(axis=1)]},
    }
    if found["index"] is None:
        info.update(found=False, reason=found["reason"])
        log.warning("No eruption found: %s", found["reason"])
        return None, info
    z = float(transform.centerline[found["index"], 0])
    info.update(found=True, index=found["index"], z=round(z, 1),
                covered_until_z=round(float(transform.centerline[found["covered_until"], 0]), 1))
    log.info("Eruption at z = %.0f (centerline point %d): no ray from the enamel touches bone; the bone covered "
             "it until z = %.0f", z, found["index"], info["covered_until_z"])
    return {"landmark": "eruption", "z_start": float(round(z)), "z_end": None, "source": AUTO}, info


def detect(transform: CMPRTransform, incisor: np.ndarray, tip_at_index_0: bool, bone_path: Path | None,
           molar_path: Path | None, eruption: bool = True, molars: bool = True) -> tuple[list[dict], dict]:
    """(the landmarks found, what was found on the way) -- the report's body. ``incisor`` is the incisor volume."""
    detected: list[dict] = []
    report: dict = {"warnings": []}
    if eruption:
        if bone_path is None:
            report["warnings"].append("No bone volume: eruption not looked for")
        else:
            log.info("Reading the bone")
            mark, report["eruption"] = detect_eruption(transform, incisor, _load(bone_path, "bone volume"),
                                                       tip_at_index_0)
            if mark is None:
                report["warnings"].append(f"No eruption found: {report['eruption']['reason']}")
            else:
                detected.append(mark)
    if molars:
        if molar_path is None:
            report["warnings"].append("No molar volume: M1-M3 not placed")
        else:
            log.info("Splitting the molars")
            found, report["molars"] = molar_marks(_load(molar_path, "molar volume"))
            log.info("Teeth split (%s): %d found", report["molars"]["split"], report["molars"]["n_teeth"])
            if not found:
                report["warnings"].append(f"The molar split found {report['molars']['n_teeth']} teeth, not 3: "
                                          "M1-M3 not placed")
            detected += found
    return detected, report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Place eruption and M1-M3 from the segmentation, then map them "
                                            "onto the incisor trajectory.")
    p.add_argument("-t", "--transform", type=Path, required=True, help="<sample>_cmpr_transform.npz")
    p.add_argument("--incisor-volume", type=Path, required=True, help="The segmentation's incisor volume")
    p.add_argument("--bone-volume", type=Path, help="The segmentation's bone volume (for eruption)")
    p.add_argument("--molar-volume", type=Path, help="The segmentation's molar volume (for M1-M3)")
    p.add_argument("--marks", type=Path, help="The marks so far; those set by hand are kept")
    p.add_argument("--marks-out", type=Path, required=True, help="Where to write the marks")
    p.add_argument("-o", "--output", type=Path, required=True, help="The marks mapped onto the trajectory")
    p.add_argument("--report", type=Path, help="What was found, as JSON")
    p.add_argument("--profile", type=Path, help="enamel_density_profile.csv, to record the orientation from")
    p.add_argument("--no-eruption", dest="eruption", action="store_false", help="Do not look for eruption")
    p.add_argument("--no-molars", dest="molars", action="store_false", help="Do not place M1-M3")
    p.add_argument("--map-only", action="store_true",
                   help="Find nothing: map the marks as they are (after adjusting them by hand)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    transform = CMPRTransform(args.transform)
    log.info("Transform: %d centerline points, volume %s", transform.n_slices, transform.volume_shape)
    existing = read_marks_csv(args.marks) if args.marks and args.marks.is_file() else []
    by_hand = sorted({m["landmark"] for m in existing if (m.get("source") or MANUAL) not in AUTO_SOURCES})
    if by_hand:
        log.info("Set by hand, and kept: %s", ", ".join(by_hand))

    log.info("Reading the incisor")
    incisor_volume = _load(args.incisor_volume, "incisor volume")
    if incisor_volume.shape != transform.volume_shape:
        raise SystemExit(f"The incisor volume's shape {incisor_volume.shape} is not the transform's "
                         f"{transform.volume_shape}: re-run the straightening")
    tip_at_index_0 = transform.tip_at_index_0(incisor_volume)

    if args.map_only:
        marks = existing
        log.info("Mapping %d mark(s) as they are", len(marks))
    else:
        detected, found = detect(transform, incisor_volume, tip_at_index_0, args.bone_volume, args.molar_volume,
                                 eruption=args.eruption, molars=args.molars)
        marks = merge_marks(existing, detected)
        for message in found["warnings"]:
            log.warning(message)
        # What was detected, a landmark set by hand too, so the viewer can go back to it.
        report = {"version": VERSION, "sample": args.transform.name.replace("_cmpr_transform.npz", ""),
                  "detected": detected, "marks": marks, **found}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    del incisor_volume
    write_marks(args.marks_out, marks)

    orientation = (orientation_from_profile(args.profile) if args.profile else None) or "tip-high-z"
    rows = map_landmarks(transform, marks, orientation=orientation, tip_at_index_0=tip_at_index_0) if marks else []
    write_csv(rows, args.output)
    for row in rows:
        log.info("%-9s %-6s %s", row["landmark"], row["source"],
                 f"{row['z_percent_start']:.2f}-{row['z_percent_end']:.2f} %" if row["kind"] == "span"
                 else f"{row['z_percent_start']:.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
