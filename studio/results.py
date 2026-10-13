"""
Results gathered across samples, for the charts and the exports.

Everything here reads files the pipeline already wrote -- density profiles,
landmark tables, molar summaries, preview voxel counts -- and caches each by
modification time, so the results page stays fast however often it is
reopened. The page groups and compares the specimens itself, by whichever of
their details the reader picks, so nothing here decides what is compared.

A specimen can be flagged: left out of every analysis, or only some of them,
with a note saying why. The page drops it from its charts and statistics and
lists it with its note; the exports drop it too, except the metadata, which
lists every specimen with its note, and the R specimen list, which keeps it
commented out.

An incisor profile's densities are in the scan's stored units. They are sent
with the HA calibration that converts them to mg HA/cm³, the one the molar step
uses for the same specimen, so the two report density on one scale.

Each incisor profile is cleaned as the R report cleans it (trajectory.py):
quality filters, outliers left out against the specimen's own trend, and a
running median that leaves gaps empty. The pages draw, and the metrics read,
the cleaned curve; the export carries every slice with what the cleanup made
of it.
"""

from __future__ import annotations

import csv
import io
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import trajectory
from .paths import TOOLS
from .steps import (STEPS, calibration_sources, needs_conversion, raw_format, resolve_params, validation_summary,
                    view_dir)
from .workspace import Workspace, group_of, read_json

# The analyses a flagged specimen can be left out of, as the results page splits them.
EXCLUSION_SCOPES = {"molar": "Molar enamel", "incisor": "Incisor mineralization",
                    "segmentation": "Segmentation volumes"}

# Enamel density, converted with the scan's HA calibration and raw.
DENSITY_METRICS = ("enamel_mean_mgHA", "enamel_low_mgHA", "enamel_high_mgHA",
                   "enamel_mean_raw", "enamel_low_raw", "enamel_high_raw")

_cache: dict[tuple[str, float], Any] = {}


def _cached(path: Path, loader):
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    if key not in _cache:
        if len(_cache) > 2000:
            _cache.clear()
        try:
            _cache[key] = loader(path)
        except (OSError, ValueError, KeyError):
            _cache[key] = None
    return _cache[key]


def _float(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_profile(path: Path) -> dict:
    rows = _read_rows(path)
    rows.sort(key=lambda r: _float(r.get("z_percent")) or 0.0)
    columns = set(rows[0]) if rows else set()
    profile = {
        "z": [int(float(r["z"])) if r.get("z") not in (None, "") else None for r in rows],
        "z_percent": [_float(r.get("z_percent")) for r in rows],
        "ratio": [_float(r.get("enamel_ratio")) for r in rows],
        "density": [_float(r.get("enamel_density")) for r in rows],
        # The ratio's denominator: the molar's dentin, or the traced core.
        "dentin": [_float(r.get("dentin_reference")) for r in rows],
        # The dentin peak fitted in each slice, which the tool's own figure draws.
        "dentin_mode": [_float(r.get("dentin_mode")) for r in rows],
        "status": [r.get("status") or "" for r in rows],
    }
    # The report's cleanup, once per version of the file. The two columns its
    # quality filters read are not kept: nothing else needs them.
    profile.update(trajectory.clean({
        **profile,
        "z_window": [_float(r.get("z_window")) for r in rows] if "z_window" in columns else None,
        "solidity": [_float(r.get("solidity")) for r in rows] if "solidity" in columns else None,
    }))
    return profile


def incisor_profile(ws: Workspace, record: dict) -> dict | None:
    path = ws.artifact(record, "incisor_profile")
    prof = _cached(path, _load_profile) if path else None
    if prof is None:
        return None
    # The run record names the constants the script ran with. One without
    # OPENING_RISE predates the withdrawal of the dentin readings from the
    # enamel's opening stretch, so the profile can still open high and drop.
    run_path = ws.artifact(record, "incisor_run")
    run = _cached(run_path, read_json) if run_path else None
    if run and "OPENING_RISE" not in (run.get("constants") or {}):
        return {**prof, "stale": True}
    return prof


CALIBRATION_KEYS = ("source", "description", "slope", "intercept", "divisor", "offset", "units")
_calibrations: dict[tuple, dict | None] = {}


def _from_files(paths: list[Path]) -> dict | None:
    try:
        key = tuple((str(p), p.stat().st_mtime) for p in paths)
    except OSError:
        return None
    if key not in _calibrations:
        if str(TOOLS) not in sys.path:
            sys.path.insert(0, str(TOOLS))
        from scanco_calibration import find_calibration  # noqa: PLC0415
        if len(_calibrations) > 2000:
            _calibrations.clear()
        _calibrations[key] = find_calibration(paths)[0]
    return _calibrations[key]


def ha_calibration(ws: Workspace, record: dict, molar_summary: dict | None = None) -> dict | None:
    """The HA calibration a specimen's stored values convert to mg HA/cm³ with, or None.

    The molar step's is used when it has run: it may have been entered by hand
    or had its mu scaling overridden, and the incisor then reads on the same
    scale as the molar. Before that, the molar step's settings are applied as
    that step would apply them: a calibration entered there, or else the one in
    the scan's own files with the settings' mu scaling.
    """
    if molar_summary is None:
        molar_summary = molar(ws, record)
    used = (molar_summary or {}).get("calibration")
    if used and all(used.get(k) is not None for k in ("slope", "intercept", "divisor")):
        return used
    defaults = (ws.settings().get("step_defaults") or {}).get("molar_quant")
    p = resolve_params(STEPS["molar_quant"].params, {}, defaults)
    if p["ha_slope"] is not None and p["ha_intercept"] is not None:
        divisor = float(p["mu_scaling"] or 1)
        return {"source": None, "description": "entered in the Molar enamel settings",
                "slope": float(p["ha_slope"]), "intercept": float(p["ha_intercept"]),
                "divisor": divisor, "offset": 0.0, "units": "mg HA/ccm"}
    found = _from_files(calibration_sources(ws, record))
    if not found:
        return None
    out = {k: found.get(k) for k in CALIBRATION_KEYS}
    if p["mu_scaling"]:
        out["divisor"] = float(p["mu_scaling"])
    return out


def view_calibration(ws: Workspace, record: dict, view: str) -> dict | None:
    """The HA calibration one of a specimen's views converts to mg HA/cm³ with, or None.

    Every view made from the volume the segmentation read -- the conversion,
    the hemimandible, the structures cut out of it, the straightened incisor --
    keeps that volume's stored values, so the specimen's own calibration
    converts them all. A raw scan that was converted first holds the scanner's
    values from before any rescaling to 1/cm or noise-floor shift: an ISQ
    answers to the calibration in its own header, any other raw scan to none.
    """
    if view == "source" and needs_conversion(record):
        raw = ws.artifact(record, "raw")
        found = _from_files([raw]) if raw is not None and raw_format(record) == "isq" else None
        return {k: found.get(k) for k in CALIBRATION_KEYS} if found else None
    return ha_calibration(ws, record)


def to_mg_ha(values: list[float | None], cal: dict | None) -> list[float | None]:
    """Stored values in mg HA/cm³, as scanco_calibration.to_density converts them."""
    if not cal:
        return [None] * len(values)
    return [None if v is None else (v + (cal.get("offset") or 0.0)) / cal["divisor"] * cal["slope"] + cal["intercept"]
            for v in values]


def exclusion(record: dict) -> dict | None:
    """A specimen's flag -- {from: "all" or [scopes], note, at} -- or None when it is not flagged."""
    flag = record.get("excluded")
    if not isinstance(flag, dict):
        return None
    scopes = flag.get("from", "all")
    if scopes != "all":
        scopes = [k for k in EXCLUSION_SCOPES if k in (scopes if isinstance(scopes, list) else [])]
        if not scopes or len(scopes) == len(EXCLUSION_SCOPES):
            scopes = "all"
    return {"from": scopes, "note": str(flag.get("note") or ""), "at": flag.get("at")}


def excluded_from(record: dict, scope: str) -> bool:
    """Whether a flag leaves the specimen out of one analysis."""
    flag = exclusion(record)
    return flag is not None and (flag["from"] == "all" or scope in flag["from"])


def _median_curve(profile: dict, key: str) -> tuple[list[float], list[float]] | None:
    """One of the cleaned running medians as (z %, value) in order along the tooth, or None when under 10 points."""
    pts = sorted((z, v) for z, v in zip(profile["z_percent"], profile[key]) if z is not None and v is not None)
    return ([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 10 else None


def incisor_metrics(profile: dict | None, cal: dict | None = None) -> dict:
    """The cleaned curve's landmarks -- where it crosses 1, its mature plateau, its peak -- and the slices behind it.

    The crossover and plateau are the report's crossing_of() and plateau_of(),
    read off the specimen's own running median rather than a group's trend.
    The plateau and peak are read off the enamel density's running median too,
    as stored (`_raw`) and converted with the HA calibration `cal` (`_mgHA`),
    with the ratio's denominator beside them: the dentin reference, its median
    over the kept slices.
    """
    if not profile:
        return {}
    kept = sum(profile["kept"])
    outliers = sum(profile["outlier"])
    measurable = sum(s in trajectory.MEASURABLE and r is not None for s, r in zip(profile["status"], profile["ratio"]))
    out = {"kept_slices": kept, "outlier_slices": outliers, "filtered_slices": measurable - kept - outliers}
    curve = _median_curve(profile, "ratio_med")
    if curve is None:
        return out
    zs, fit = curve
    out.update({"crossover_z": trajectory.crossing_of(zs, fit), "plateau_ratio": trajectory.plateau_of(zs, fit),
                "peak_ratio": float(max(fit))})
    density = _median_curve(profile, "density_med")
    dentin = [d for d, k in zip(profile["dentin"], profile["kept"]) if k and d is not None]
    raw = {"plateau": trajectory.plateau_of(*density) if density else None,
           "peak": float(max(density[1])) if density else None,
           "dentin": float(np.median(dentin)) if dentin else None}
    for key, v in raw.items():
        out[f"{key}_raw"] = v
        out[f"{key}_mgHA"] = to_mg_ha([v], cal)[0]
    return out


def landmarks(ws: Workspace, record: dict) -> list[dict]:
    """The mapped landmarks: where each falls along the incisor (z %), and the straightened incisor's slices it maps to."""
    path = ws.artifact(record, "landmarks")
    rows = _cached(path, _read_rows) if path else None
    out = []
    for r in rows or []:
        out.append({
            "landmark": r.get("landmark"), "kind": r.get("kind"),
            "start": _float(r.get("z_percent_start")), "end": _float(r.get("z_percent_end")),
            "mid": _float(r.get("z_percent_mid")), "length_mm": _float(r.get("length_mm")),
            "slice_start": _float(r.get("slice_start")), "slice_end": _float(r.get("slice_end")),
            "source": "manual" if (r.get("source") or "manual") == "manual" else "auto",
        })
    return out


def per_junction_area(thickness_um, area_mm2):
    """The median enamel thickness over the enamel–dentin junction's area, in µm per mm²."""
    if thickness_um is None or not area_mm2:
        return None
    return thickness_um / area_mm2


def per_sqrt_junction_area(thickness_um, area_mm2):
    """The mean enamel thickness over the square root of the enamel–dentin junction's area, both in µm.

    A pure number: the junction's square root is its linear size, so a tooth
    scaled up in every direction keeps the same ratio, unlike the thickness
    per mm², which falls with the square of size. 0.06 means the enamel is, on
    average, 6% as thick as the junction is across.
    """
    if thickness_um is None or not area_mm2 or area_mm2 <= 0:
        return None
    return thickness_um / (math.sqrt(area_mm2) * 1000.0)


# Patches of enamel thinner, or less dense, than the enamel around them, by
# the names Results and the export use; summaries before version 6 have none.
PATCH_KEYS = ("thin_patch_share", "thin_patches_per_mm2", "thin_patch_depth", "thickness_shortfall",
              "low_density_patch_share", "low_density_patches_per_mm2", "low_density_patch_depth",
              "density_shortfall", "core_density_raw", "core_density_mgHA")


def patch_numbers(block: dict | None) -> dict:
    """The patches of the whole molar row, by ``PATCH_KEYS``, from the summary's ``enamel.patches``."""
    thin = (block or {}).get("thin") or {}
    low = (block or {}).get("low_density") or {}
    return {
        "thin_patch_share": thin.get("share"), "thin_patches_per_mm2": thin.get("patches_per_mm2"),
        "thin_patch_depth": thin.get("depth"), "thickness_shortfall": thin.get("shortfall"),
        "low_density_patch_share": low.get("share"), "low_density_patches_per_mm2": low.get("patches_per_mm2"),
        "low_density_patch_depth": low.get("depth"), "density_shortfall": low.get("shortfall"),
        "core_density_raw": low.get("core_density_raw"), "core_density_mgHA": low.get("core_density_mg_ha"),
    }


def molar(ws: Workspace, record: dict) -> dict | None:
    path = ws.artifact(record, "molar_quant")
    data = _cached(path, lambda p: read_json(p)) if path else None
    if not data:
        return None
    thickness = (data.get("enamel") or {}).get("thickness_um") or {}
    # Each tooth's thickness block: summaries before version 7 keep the per-tooth
    # mean only there, not in ``teeth``.
    measured_tooth = {t.get("name"): t for t in thickness.get("teeth") or []}

    def tooth_stat(t, key, stat):
        value = t.get(key)
        return value if value is not None else (measured_tooth.get(t["name"]) or {}).get(stat)
    density = (data.get("enamel") or {}).get("density") or {}
    raw = density.get("raw") or {}
    mg = density.get("mg_ha_cm3") or {}
    calibration = data.get("calibration") or None
    return {
        # Summaries written before the density block still carry the raw mean.
        "enamel_mean_raw": raw.get("mean", data["enamel"].get("mean_intensity")),
        "enamel_low_raw": raw.get("low"),
        "enamel_high_raw": raw.get("high"),
        "enamel_mean_mgHA": mg.get("mean"),
        "enamel_low_mgHA": mg.get("low"),
        "enamel_high_mgHA": mg.get("high"),
        "density_percentiles": density.get("percentiles"),
        "calibration": {k: calibration.get(k) for k in CALIBRATION_KEYS} if calibration else None,
        "enamel_fraction": data["enamel"]["fraction"],
        "enamel_mm3": data["enamel"]["volume_mm3"],
        "total_mm3": data["total"]["volume_mm3"],
        "dentin_mm3": data["dentin"]["volume_mm3"],
        "density_ratio": data.get("enamel_to_dentin_density"),
        "enamel_minus_dentin": data.get("enamel_minus_dentin"),
        "enamel_median": data["enamel"].get("median_intensity"),
        "dentin_mode": data["dentin"].get("mode"),
        # Summaries before version 3 carry the enamel mask's distance-map ridge
        # here, with no percentiles or junction area.
        "thickness_um": thickness.get("median"),
        # Mean and standard deviation over the measured junction points (summaries from version 7).
        "thickness_mean_um": thickness.get("mean"),
        "thickness_sd_um": thickness.get("sd"),
        "thickness_p25_um": thickness.get("p25"),
        "thickness_p75_um": thickness.get("p75"),
        "thickness_measured": thickness.get("share_measured"),
        "thickness_background_from": (thickness.get("levels") or {}).get("background_from"),
        "junction_area_mm2": thickness.get("junction_area_mm2"),
        "thickness_um_per_mm2": per_junction_area(thickness.get("median"), thickness.get("junction_area_mm2")),
        "thickness_mean_per_sqrt_junction": per_sqrt_junction_area(thickness.get("mean"),
                                                                   thickness.get("junction_area_mm2")),
        "edge_blur_um": thickness.get("edge_blur_um"),
        **patch_numbers((data.get("enamel") or {}).get("patches")),
        "threshold": data["threshold"]["value"],
        "threshold_method": data["threshold"]["method"],
        # How the enamel was found (summaries before version 5: one threshold),
        # and the levels its junction was placed at along the tooth.
        "enamel_mask": (data.get("enamel_mask") or {}).get("method", "threshold"),
        "junction_level": (data.get("enamel_mask") or {}).get("junction_level"),
        "dentin_under_enamel": (data.get("enamel_mask") or {}).get("dentin_under_enamel"),
        "teeth": [{"name": t["name"], "enamel_fraction": t["enamel_fraction"],
                   "volume_mm3": t["volume_mm3"], "enamel_mm3": t["enamel_volume_mm3"],
                   "thickness_um": t.get("thickness_median_um"),
                   "thickness_mean_um": tooth_stat(t, "thickness_mean_um", "mean"),
                   "thickness_sd_um": tooth_stat(t, "thickness_sd_um", "sd"),
                   "junction_area_mm2": t.get("junction_area_mm2"),
                   "thickness_um_per_mm2": per_junction_area(t.get("thickness_median_um"), t.get("junction_area_mm2")),
                   "thickness_mean_per_sqrt_junction": per_sqrt_junction_area(
                       tooth_stat(t, "thickness_mean_um", "mean"), t.get("junction_area_mm2")),
                   **{k: t.get(k) for k in PATCH_KEYS}}
                  for t in data.get("teeth") or []],
        "warnings": data.get("warnings") or [],
        "histogram": data.get("histogram"),
    }


def segmentation(ws: Workspace, record: dict) -> dict | None:
    """Structure volumes from the hemimandible preview's full-resolution voxel counts."""
    out: dict[str, Any] = {}
    meta = read_json(view_dir(ws, record["name"], "mandible") / "view.json")
    if meta and meta.get("spacing_known"):
        voxel = float(np.prod(meta["spacing_mm"]))
        counts = meta.get("counts") or {}
        for part in ("incisor", "bone", "molar"):
            if part in counts:
                out[f"{part}_mm3"] = counts[part] * voxel
    mq = molar(ws, record)
    if mq:
        out["molar_mm3"] = mq["total_mm3"]
    validation = record.get("validation") or validation_summary(ws.artifact(record, "validation"), record["name"])
    if validation:
        out["validation"] = validation.get("status")
        out["problems"] = len(validation.get("problems") or [])
    return out or None


def gather(ws: Workspace) -> dict:
    """Every specimen's details and measurements, for the results page to group and compare."""
    samples, incisor, marks, molars, seg = [], {}, {}, {}, {}
    for rec in ws.all():
        meta = rec.get("meta") or {}
        samples.append({"name": rec["name"], "group": group_of(meta), "meta": meta, "excluded": exclusion(rec),
                        "folder": rec.get("folder") or ""})
        mq = molar(ws, rec)
        if mq:
            molars[rec["name"]] = {k: v for k, v in mq.items() if k != "histogram"}
        prof = incisor_profile(ws, rec)
        if prof:
            # The residuals are for the export; the page draws the kept slices and their medians.
            incisor[rec["name"]] = {**{k: v for k, v in prof.items() if k != "residual"},
                                    "calibration": ha_calibration(ws, rec, mq)}
        lm = landmarks(ws, rec)
        if lm:
            marks[rec["name"]] = lm
        sg = segmentation(ws, rec)
        if sg:
            seg[rec["name"]] = sg
    return {
        "samples": samples,
        "incisor": incisor,
        "incisor_metrics": {n: incisor_metrics(p, p["calibration"]) for n, p in incisor.items()},
        "landmarks": marks,
        "molar": molars,
        "segmentation": seg,
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def _csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return buf.getvalue()


META_COLUMNS = ["sample", "group", "genotype", "age", "line", "replicate", "sex", "side",
                "study", "scan_date", "scanner", "voxel_um", "notes"]
# What each measurement export leads with: enough to group by in R without a join.
ID_COLUMNS = ["sample", "group", "genotype", "age", "sex"]
EXCLUSION_COLUMNS = ["excluded_from", "exclusion_note", "flagged_on"]


def _exclusion_columns(record: dict) -> dict:
    flag = exclusion(record)
    if flag is None:
        return {}
    return {"excluded_from": flag["from"] if flag["from"] == "all" else ";".join(flag["from"]),
            "exclusion_note": flag["note"], "flagged_on": (flag.get("at") or "")[:10]}


def export(ws: Workspace, kind: str, names: list[str] | None = None) -> tuple[str, str]:
    """(filename, text) for one export: of every specimen, or of `names` alone (those the results page shows)."""
    records = ws.all()
    if names is not None:
        wanted = set(names)
        records = [r for r in records if r["name"] in wanted]
    base = [{"sample": r["name"], "group": group_of(r.get("meta") or {}), **(r.get("meta") or {}),
             **_exclusion_columns(r)} for r in records]
    meta_by = {b["sample"]: b for b in base}
    # A flagged specimen's measurements stay out of the exports of the analyses it is left out of.
    kept = lambda scope: [r for r in records if not excluded_from(r, scope)]  # noqa: E731

    if kind == "metadata":
        return "metadata.csv", _csv(base, META_COLUMNS + EXCLUSION_COLUMNS)
    if kind == "excluded":
        rows = [meta_by[r["name"]] for r in records if exclusion(r)]
        return "excluded.csv", _csv(rows, ID_COLUMNS + EXCLUSION_COLUMNS)
    if kind == "specimens":
        # The R report's params block: "sample | genotype | age". The report is
        # the incisor's, so a specimen left out of it stays in the list, commented
        # out with its note, where whoever edits the params will see why.
        width = max((len(b["sample"]) for b in base), default=10)
        lines = []
        for r, b in zip(records, base):
            line = f'- "{b["sample"]:<{width}} | {b.get("genotype") or "":<9} | {b.get("age") or ""}"'
            if excluded_from(r, "incisor"):
                note = " ".join((exclusion(r)["note"] or "flagged").split())
                line = f"# {line}  # excluded: {note}"
            lines.append(f"    {line}")
        return "specimens.txt", "  specimens:\n" + "\n".join(lines) + "\n"
    if kind == "molar":
        rows = []
        for r in kept("molar"):
            mq = molar(ws, r)
            if not mq:
                continue
            row = {**meta_by[r["name"]], **{k: v for k, v in mq.items() if not isinstance(v, (list, dict))}}
            for t in mq["teeth"]:
                row[f"{t['name']}_enamel_fraction"] = t["enamel_fraction"]
                row[f"{t['name']}_thickness_um"] = t.get("thickness_um")
                row[f"{t['name']}_thickness_mean_um"] = t.get("thickness_mean_um")
                row[f"{t['name']}_thickness_sd_um"] = t.get("thickness_sd_um")
                row[f"{t['name']}_thickness_mean_per_sqrt_junction"] = t.get("thickness_mean_per_sqrt_junction")
                row[f"{t['name']}_junction_area_mm2"] = t.get("junction_area_mm2")
                row[f"{t['name']}_thickness_um_per_mm2"] = t.get("thickness_um_per_mm2")
                for key in PATCH_KEYS:
                    row[f"{t['name']}_{key}"] = t.get(key)
            if mq["density_percentiles"]:
                row["density_percentiles"] = "/".join(f"{p:g}" for p in mq["density_percentiles"])
            if mq["calibration"]:
                row["calibration"] = os.path.basename(str(mq["calibration"].get("source") or ""))
            if mq.get("junction_level"):
                row["junction_level_median"] = mq["junction_level"]["median"]
            rows.append(row)
        cols = ID_COLUMNS + ["enamel_fraction", "enamel_mm3", "dentin_mm3", "total_mm3",
                                   "density_ratio", "enamel_minus_dentin", "enamel_median", "dentin_mode",
                                   *DENSITY_METRICS, "density_percentiles", "calibration",
                                   "thickness_um", "thickness_mean_um", "thickness_sd_um", "thickness_p25_um",
                                   "thickness_p75_um", "thickness_measured",
                                   "thickness_background_from", "junction_area_mm2", "thickness_um_per_mm2",
                                   "thickness_mean_per_sqrt_junction", "threshold", "threshold_method", "enamel_mask", "junction_level_median",
                                   "M1_enamel_fraction", "M2_enamel_fraction", "M3_enamel_fraction",
                                   "M1_thickness_um", "M2_thickness_um", "M3_thickness_um",
                                   "M1_thickness_mean_um", "M2_thickness_mean_um", "M3_thickness_mean_um",
                                   "M1_thickness_sd_um", "M2_thickness_sd_um", "M3_thickness_sd_um",
                                   "M1_junction_area_mm2", "M2_junction_area_mm2", "M3_junction_area_mm2",
                                   "M1_thickness_um_per_mm2", "M2_thickness_um_per_mm2", "M3_thickness_um_per_mm2",
                                   "M1_thickness_mean_per_sqrt_junction", "M2_thickness_mean_per_sqrt_junction",
                                   "M3_thickness_mean_per_sqrt_junction",
                                   "edge_blur_um", *PATCH_KEYS,
                                   *(f"{tooth}_{key}" for key in PATCH_KEYS for tooth in ("M1", "M2", "M3"))]
        return "molar_enamel.csv", _csv(rows, cols)
    if kind == "incisor":
        rows = []
        for r in kept("incisor"):
            prof = incisor_profile(ws, r)
            if not prof:
                continue
            m = meta_by[r["name"]]
            cal = ha_calibration(ws, r)
            mg = {"enamel_density_mgHA": to_mg_ha(prof["density"], cal),
                  "density_med_mgHA": to_mg_ha(prof["density_med"], cal),
                  "dentin_reference_mgHA": to_mg_ha(prof["dentin"], cal),
                  "dentin_mode_mgHA": to_mg_ha(prof["dentin_mode"], cal)}
            if cal:
                # Before enamel onset the tool writes 0 for "no enamel", which stays 0: it is
                # not a stored value to convert.
                mg["enamel_density_mgHA"] = [0.0 if st == "pre_onset" else v
                                             for v, st in zip(mg["enamel_density_mgHA"], prof["status"])]
            source = os.path.basename(str(cal.get("source") or "")) if cal else ""
            for i in range(len(prof["z"])):
                rows.append({**{k: m.get(k) for k in ID_COLUMNS}, "z": prof["z"][i],
                             "z_percent": prof["z_percent"][i], "status": prof["status"][i],
                             "enamel_ratio": prof["ratio"][i], "enamel_density": prof["density"][i],
                             "dentin_reference": prof["dentin"][i], "dentin_mode": prof["dentin_mode"][i],
                             # What the report's cleanup made of the slice, as R reads a logical.
                             "kept": "TRUE" if prof["kept"][i] else "FALSE",
                             "outlier": "TRUE" if prof["outlier"][i] else "FALSE",
                             "residual": prof["residual"][i], "ratio_med": prof["ratio_med"][i],
                             "density_med": prof["density_med"][i],
                             **{k: v[i] for k, v in mg.items()}, "calibration": source})
        cols = ID_COLUMNS + ["z", "z_percent", "status", "enamel_ratio", "enamel_density", "dentin_reference",
                             "dentin_mode", "kept", "outlier", "residual", "ratio_med", "density_med",
                             "enamel_density_mgHA", "density_med_mgHA", "dentin_reference_mgHA", "dentin_mode_mgHA",
                             "calibration"]
        return "incisor_profiles.csv", _csv(rows, cols)
    if kind == "landmarks":
        rows = []
        for r in kept("incisor"):
            path = ws.artifact(r, "landmarks")
            if not path:
                continue
            m = meta_by[r["name"]]
            for row in _read_rows(path):
                rows.append({**row, **{k: m.get(k) for k in ID_COLUMNS}})
        cols = ["landmark", "kind", "z_percent_start", "z_percent_end", "z_percent_mid", "extent_percent",
                "z_start", "z_end", "slice_start", "slice_end", "arc_length_start_mm", "arc_length_end_mm",
                "length_mm", "z_residual_vox", "sample", "genotype", "age", "sex", "group"]
        return "landmarks.csv", _csv(rows, cols)
    if kind == "segmentation":
        rows = []
        for r in kept("segmentation"):
            sg = segmentation(ws, r) or {}
            rows.append({**meta_by[r["name"]], **sg})
        cols = ID_COLUMNS + ["incisor_mm3", "bone_mm3", "molar_mm3", "validation", "problems"]
        return "segmentation.csv", _csv(rows, cols)
    raise KeyError(kind)


def file_size(path: str | os.PathLike) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None
