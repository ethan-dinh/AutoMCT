"""
What each step runs, what it needs, and what it leaves behind.

A step here is a thin description of one of the repository's own tools: the
parameters worth exposing (with the tool's own defaults), the command line to
build from them, the log lines that mark its progress, and which files it
wrote. Nothing re-implements an algorithm; changing a tool's behaviour means
changing the tool, and the studio follows.

Pipeline steps belong to a sample and run in this order:

    convert -> segment -> cmpr -> incisor_quant
                       |       \\-> landmarks (with the segmentation)
                       \\-> molar_quant
                       \\-> ml_segment (turned off for now)

An NRRD scan skips convert: it already is what the segmentation reads, so it
is segmented as it is, with any crop box applied in memory. Every other format
(ISQ, DICOM, BMP, TIFF) is converted to NRRD first.

Tools (RSQ motion check, histogram plot, DICOM->TIFF, model training, the
test suites) run on their own, outside any sample.
"""

from __future__ import annotations

import csv
import datetime
import glob
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .jobs import Job, Launch
from .paths import ML, PYTHON, REPO, SCRIPTS
from .volumes import view_is_current
from .workspace import Workspace, artifacts_in, now_iso, read_json, source_format

# Why the machine-learning step and tool are greyed out in the UI and refused
# by the server. Set to None to bring them back.
ML_OFF: str | None = "Machine learning is turned off for now."


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Param:
    key: str
    label: str
    type: str = "number"          # number | int | bool | select | text | path | paths
    default: Any = None
    help: str = ""
    options: list[tuple[Any, str]] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    group: str = ""
    advanced: bool = False
    placeholder: str = ""

    def spec(self) -> dict:
        data = asdict(self)
        if self.options is not None:
            data["options"] = [{"value": v, "label": label} for v, label in self.options]
        return data


def _coerce(param: Param, value: Any) -> Any:
    if value in (None, ""):
        return None if param.type not in ("bool",) else bool(param.default)
    if param.type == "bool":
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    if param.type == "int":
        return int(float(value))
    if param.type == "number":
        return float(value)
    if param.type == "paths":
        if isinstance(value, str):
            return [v.strip() for v in value.splitlines() if v.strip()]
        return list(value)
    return value


def resolve_params(params: list[Param], supplied: dict | None, defaults: dict | None = None) -> dict:
    """Tool defaults, overlaid with workspace defaults, overlaid with this run's values."""
    out = {}
    for p in params:
        value = p.default
        if defaults and p.key in defaults:
            value = defaults[p.key]
        if supplied and p.key in supplied:
            value = supplied[p.key]
        out[p.key] = _coerce(p, value)
    return out


# ---------------------------------------------------------------------------
# The step registry
# ---------------------------------------------------------------------------

@dataclass
class Step:
    id: str
    label: str
    summary: str
    lane: str = "heavy"
    params: list[Param] = field(default_factory=list)
    upstream: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    optional: bool = False
    interactive: str | None = None   # "crop" / "landmarks": needs picking in the UI first
    disabled: str | None = None      # why the step is switched off; shown, never run

    def spec(self) -> dict:
        return {
            "id": self.id, "label": self.label, "summary": self.summary, "lane": self.lane,
            "params": [p.spec() for p in self.params], "upstream": list(self.upstream),
            "produces": list(self.produces), "optional": self.optional,
            "interactive": self.interactive, "disabled": self.disabled,
        }


# The lab's conversion recipe, as the defaults:
#
#   convert_ISQ_NRRD.py <scan> --autocrop --autocrop-margin-mm 0.1 --target-voxel-um 6
#       --gauss-sigma 0.15 --nlm-h 0.3 --median-radius 0 --aniso-iterations 10
#       --aniso-kappa 600 --tv-weight 0.15 --unsharp-amount 0.04 --reorient
#       --reorient-tight --reorient-margin-mm 0.5 --save-raw --save-dtype int16
#       --clamp-negative
#
# Cropping to the specimen and cutting the rotated volume tight to it is what
# keeps a converted scan small: a whole 3168 x 3168 x 412 field of view is 8 GB.
CONVERT_PARAMS = [
    Param("autocrop", "Crop to the specimen automatically", "bool", True,
          "Used only when no crop box has been drawn. Otsu-thresholds the scan and keeps the "
          "largest piece, so the holder and debris are cut away.", group="Crop"),
    Param("autocrop_margin_mm", "Autocrop margin", "number", 0.1, unit="mm", min=0, step=0.1,
          group="Crop", advanced=True),
    Param("target_voxel_um", "Bin to voxel size", "number", 6.0,
          "Average blocks of voxels to reach roughly this voxel size. A scan already this size or "
          "coarser is left as it is. Leave blank to keep the scan's resolution.",
          unit="µm", min=0, step=1, group="Resolution", placeholder="native"),
    Param("bin_method", "Binning", "select", "mean",
          options=[("mean", "Mean (denoises)"), ("max", "Max (keeps thin dense structures)")],
          group="Resolution", advanced=True),
    Param("reorient", "Rotate onto the long axis", "bool", True,
          "Resample so Z runs along the specimen's long axis. With a tight crop after it, the "
          "rotated volume holds little but the specimen.", group="Orientation"),
    Param("reorient_tip_at", "Tip at", "select", "start", options=[("start", "Start of Z"), ("end", "End of Z")],
          group="Orientation", advanced=True),
    Param("reorient_tight", "Crop tight after rotating", "bool", True, group="Orientation", advanced=True),
    Param("reorient_margin_mm", "Margin after rotating", "number", 0.5, unit="mm", min=0, step=0.1,
          group="Orientation", advanced=True),
    Param("noise_floor", "Flatten background below", "number", None,
          "Voxels below this level are set to one constant; tissue keeps its exact values.",
          group="Background", advanced=True, placeholder="off"),
    Param("noise_floor_method", "Background level is", "select", "absolute",
          options=[("absolute", "An intensity"), ("percentile", "A percentile"), ("otsu", "Found by Otsu")],
          group="Background", advanced=True),
    Param("nlm_h", "Non-local means strength", "number", 0.3,
          "0 turns the filters' denoising off; with every filter at 0 the conversion keeps the "
          "scanner's values. The unfiltered copy keeps them either way.", min=0, max=5, step=0.1,
          group="Filters"),
    Param("median_radius", "Median filter radius", "int", 0, unit="voxels", min=0, max=5, group="Filters",
          advanced=True),
    Param("gauss_sigma", "Gaussian pre-smoothing", "number", 0.15, unit="voxels", min=0, step=0.05,
          group="Filters", advanced=True),
    Param("aniso_iterations", "Anisotropic diffusion iterations", "int", 10, min=0, max=50, group="Filters",
          advanced=True),
    Param("aniso_kappa", "Anisotropic diffusion edge level", "number", 600.0,
          "Intensity steps above this, in the scan's own units, are kept as edges; smaller ones "
          "are smoothed.", min=0, step=10, group="Filters", advanced=True),
    Param("tv_weight", "Total-variation weight", "number", 0.15, min=0, max=1, step=0.01, group="Filters",
          advanced=True),
    Param("unsharp_amount", "Unsharp mask amount", "number", 0.04, min=0, max=3, step=0.01, group="Filters",
          advanced=True),
    Param("apply_rescale", "Convert to physical units", "bool", False,
          "Apply the header's rescale (DICOM slope/intercept, ISQ mu scaling).", group="Output"),
    Param("save_dtype", "Voxel type on disk", "select", "int16",
          options=[("auto", "Automatic"), ("float32", "float32"), ("int16", "int16"), ("uint16", "uint16")],
          help="int16 takes half the space of float32; filtered values are rounded to whole scanner "
               "units. Automatic keeps an unfiltered ISQ scan as int16 and anything else as float32.",
          group="Output", advanced=True),
    Param("save_raw", "Keep an unfiltered copy", "bool", True,
          "Also write <name>_prefiltered.nrrd: cropped, binned and rotated like the converted scan "
          "but unfiltered, so the two compare voxel for voxel. A later conversion that changes "
          "only the filters starts from it instead of the raw scan.", group="Output", advanced=True),
    Param("enhance_contrast", "Write a CLAHE viewing copy", "bool", False, group="Output", advanced=True),
    Param("compression_level", "gzip level", "int", 1, min=0, max=9, group="Output", advanced=True),
    Param("trim_saturated", "Replace rail-saturated voxels (ISQ)", "bool", True, group="ISQ", advanced=True),
    Param("clamp_negative", "Clamp negative values (ISQ)", "bool", True,
          "Raise voxels below 0 to 0. Enamel and dentine are untouched.", group="ISQ", advanced=True),
]

# What the unfiltered copy already carries: the reading of the scan, the crop,
# the binning and the rotation. A later conversion reuses the copy, skipping
# those stages, only when it asks for the same ones.
CONVERT_GEOMETRY = ("apply_rescale", "trim_saturated", "clamp_negative", "autocrop", "autocrop_margin_mm",
                    "target_voxel_um", "bin_method", "reorient", "reorient_tip_at", "reorient_tight",
                    "reorient_margin_mm")

SEGMENT_PARAMS = [
    Param("output_format", "Save as", "select", "nrrd", options=[("nrrd", "NRRD"), ("tif", "TIFF")],
          help="NRRD is what CMPR, the density fit and landmark mapping read by default. "
               "Choose TIFF to feed the ML training set."),
    Param("incisor_margin_shrink", "Trim the incisor surface by", "number", 0.0, unit="mm", min=0, step=0.002,
          help="Pull the incisor mask back from the partial-volume rim for density work. Keep at or "
               "below one to two voxels (0.008-0.016 mm at 8 µm) and identical across a comparison."),
    Param("voxel_size_um", "Voxel size override", "number", None, unit="µm", min=0, step=0.1,
          help="Only for a scan whose file records a wrong voxel size, or none.", placeholder="from file"),
    Param("strict", "Stop at the first failed check", "bool", False,
          help="Otherwise the sample finishes and failed checks are flagged."),
    Param("use_cache", "Cache the denoised volume", "bool", False,
          help="Re-runs skip denoising, the slowest stage. Costs about 4 bytes per voxel of disk "
               "per sample, kept with the sample's large files."),
    Param("cache_molar_stage", "Also cache up to the molar stage", "bool", False,
          help="Freezes the incisor result. Only for tuning bone/molar segmentation.", advanced=True),
    Param("workers", "Worker threads", "int", None, min=1, advanced=True, placeholder="all cores"),
]

ML_RUNS = [(str(p.relative_to(ML)), p.name) for p in sorted(ML.glob("runs*")) if p.is_dir()]

ML_PARAMS = [
    Param("checkpoint", "Model", "select", ML_RUNS[0][0] if ML_RUNS else "runs",
          options=ML_RUNS or [("runs", "runs")], help="A runs folder: every fold's best model."),
    Param("ensemble", "Average every fold", "bool", True),
    Param("tta", "Test-time flips", "bool", False, help="8x slower; smoother masks."),
    Param("staged", "Two-stage decision", "bool", False),
    Param("bg_threshold", "Background gate", "number", 0.05, min=0, max=1, step=0.01, advanced=True),
    Param("device", "Device", "select", "auto",
          options=[("auto", "Automatic"), ("mps", "Apple GPU"), ("cuda", "CUDA"), ("cpu", "CPU")], advanced=True),
]

CMPR_PARAMS = [
    Param("spacing", "In-plane sampling", "number", 1.0, "Voxels per output pixel.", min=0.25, max=4, step=0.25),
    Param("margin", "Window safety factor", "number", 1.2, min=1, max=3, step=0.05, advanced=True),
    Param("slice_size", "Slice size", "int", None, unit="px", min=16, advanced=True, placeholder="auto"),
]

INCISOR_PARAMS = [
    Param("baseline", "Dentin reference", "select", "molar",
          options=[("molar", "Molar dentin (flat)"), ("trace", "Per-slice core trace")],
          help="The molar baseline is the default; it needs the segmentation's molar volume."),
    Param("stride", "Analyze every Nth slice", "int", 1, min=1, max=50,
          help="10 gives a quick preview of the profile."),
    Param("orientation", "Tip end", "select", "auto",
          options=[("auto", "Detect"), ("tip-low-z", "Low z"), ("tip-high-z", "High z")]),
    Param("erosion", "Rim erosion", "int", 2, min=1, max=6, advanced=True,
          help="Do not set to 0: the partial-volume rim would be fitted as a population."),
    Param("max_rim_fraction", "Max rim fraction", "number", 0.5, min=0.1, max=0.9, step=0.05, advanced=True),
    Param("gate_scale", "Significance gate scale", "number", 1.15, min=1, max=2, step=0.05, advanced=True),
    Param("max_baseline_delta", "Max molar/incisor disagreement", "number", 1.5, unit="σ", min=0.5, step=0.1,
          advanced=True),
    Param("trim_onset", "Trim slices before enamel onset", "bool", True, advanced=True),
    Param("plots", "Draw figures", "bool", True, advanced=True),
]

MOLAR_PARAMS = [
    Param("enamel_mask", "Enamel", "select", "local",
          options=[("local", "Between its halfway surfaces (levels read along the tooth)"),
                   ("threshold", "Every voxel above one threshold")],
          help="Between its halfway surfaces, the enamel is found by its shape: a layer brighter than the "
               "dentin it covers, however dim the scan reads it there, so a dimmer knockout enamel keeps its "
               "volume and shows in its density. One threshold misses thin or dimmed enamel: on "
               "P82_Odam-Cre_WT_2 it held 7.3% of the molar against 18.5%."),
    Param("method", "Histogram threshold", "select", "log-valley",
          options=[("log-valley", "Log-histogram valley (as segmentation)"), ("otsu", "Otsu")], advanced=True,
          help="Splits the molar histogram into its dentin and enamel peaks, where the halfway surfaces' "
               "levels start; with one threshold, the cut itself."),
    Param("threshold", "Fixed threshold", "number", None,
          help="One absolute value for every sample: the enamel is then every voxel above it, the volume above "
               "one density, for calibrated scans compared across genotypes.",
          placeholder="per sample"),
    Param("min_enamel_voxels", "Smallest enamel speck kept", "int", 50, unit="voxels", min=0, advanced=True),
    Param("split", "Split into M1-M3", "bool", True),
    Param("thickness", "Enamel thickness", "bool", True,
          help="Between the enamel's outer surface and its junction with the dentin, measured in 3-D and "
               "per tooth, with the junction's area. Reads the reoriented scan around the tooth."),
    Param("thickness_map", "3-D thickness map", "bool", True,
          help="The enamel coloured by its thickness, to turn and read in the browser."),
    Param("thickness_max", "Thickness map colour scale", "number", 120, unit="µm", min=10, step=10, advanced=True,
          help="The top of the map's colour scale. Keep one value across a cohort so the colours compare."),
    Param("patches", "Thin and less dense patches", "bool", True,
          help="Where the enamel is thinner, or its core less mineralized, than the enamel around it on the same "
               "tooth: the share of the enamel–dentin junction under such patches, per tooth, and a colouring of "
               "each in the 3-D map. Needs the thickness."),
    Param("patch_around_um", "Patches: compared with the enamel within", "number", 60, unit="µm", min=20, max=300,
          step=10, group="Patches", advanced=True,
          help="The sigma of the Gaussian each point's surroundings are read over: patches out to about this "
               "radius are found at their full depth, wider thinning is taken for the tooth's own shape."),
    Param("thinner_by", "Thin patch: at least", "number", 45, unit="% thinner", min=5, max=90, step=5,
          group="Patches", advanced=True,
          help="Set where 1% of the P22 TGM2 wildtypes' judged enamel falls: fissure floors are thinner than their "
               "walls on every tooth. Change it only for a reason, and the same for a whole cohort."),
    Param("less_dense_by", "Less dense patch: at least", "number", 12, unit="% less mineral", min=1, max=50, step=1,
          group="Patches", advanced=True,
          help="The share of the mineral around it a patch's core is missing (its density above the background). "
               "Set where 1% of the P22 TGM2 wildtypes' judged enamel falls: their enamel's density itself varies "
               "about 5% around its surroundings."),
    Param("patch_min_area_mm2", "Smallest patch", "number", 0.002, unit="mm²", min=0.0005, max=0.1, step=0.0005,
          group="Patches", advanced=True),
    Param("patch_margin_um", "Unjudged band at the enamel's edge", "number", 100, unit="µm", min=0, max=400,
          step=10, group="Patches", advanced=True,
          help="The enamel tapers to nothing toward its edge -- the cervical edge, and any bare area of the "
               "crown such as a cusp tip without enamel -- so it is left unjudged there."),
    Param("save_points", "Save every junction point", "bool", False, group="Patches", advanced=True,
          help="Writes <sample>_junction_points.npz, so the patch stage can be run again under other settings "
               "(enamel_patches.py sweep) without measuring."),
    Param("percentile_low", "Enamel density: lower percentile", "number", 5, min=0, max=50, step=1,
          help="Reported beside the mean, raw and in mg HA/cm³."),
    Param("percentile_high", "Enamel density: upper percentile", "number", 95, min=50, max=100, step=1),
    Param("ha_slope", "HA calibration slope", "number", None, group="Density calibration", advanced=True,
          placeholder="from the scan",
          help="Leave empty to use the calibration the Scanco scanner wrote into the ISQ. Set it, with the "
               "intercept, for a scan that carries none: mg HA/cm³ = value / mu scaling × slope + intercept."),
    Param("ha_intercept", "HA calibration intercept", "number", None, unit="mg HA/cm³",
          group="Density calibration", advanced=True, placeholder="from the scan"),
    Param("mu_scaling", "mu scaling", "number", None, group="Density calibration", advanced=True,
          placeholder="from the scan",
          help="What stored values are divided by before calibrating: 4096 for raw Scanco counts, "
               "1 for a volume already converted to 1/cm."),
    Param("plots", "Draw figure", "bool", True, advanced=True),
]

LANDMARK_PARAMS = [
    Param("eruption", "Find eruption", "bool", True,
          "Where the incisor's enamel comes out of the bone: walking along the tooth toward the tip, the first "
          "cross-section where no ray from the enamel touches bone."),
    Param("molars", "Place M1-M3", "bool", True,
          "Each molar's crown along the jaw, from the tooth split, less where the crown in front still shows. "
          "Placed only when all three teeth are found."),
]

STEPS: dict[str, Step] = {s.id: s for s in (
    Step("convert", "Convert", "ISQ, DICOM, BMP or TIFF scan to NRRD by the lab's recipe: cropped to the specimen, "
         "rotated onto its long axis and filtered, with an unfiltered copy kept.",
         params=CONVERT_PARAMS, produces=("converted",), interactive="crop"),
    Step("segment", "Segment", "Incisor, bone and molars, with automatic plausibility checks.",
         params=SEGMENT_PARAMS, upstream=("convert",),
         produces=("reoriented", "incisor_volume", "bone_volume", "molar_volume")),
    Step("ml_segment", "ML segmentation", "A second opinion from the trained 3-D U-Net.",
         params=ML_PARAMS, upstream=("segment",), produces=("ml_incisor_mask",), optional=True,
         disabled=ML_OFF),
    Step("cmpr", "Straighten incisor", "Curved reslice so every slice is a true cross-section.",
         params=CMPR_PARAMS, upstream=("segment",), produces=("cmpr_volume", "cmpr_transform")),
    Step("incisor_quant", "Incisor enamel", "Enamel density and enamel/dentin ratio along the tooth.",
         params=INCISOR_PARAMS, upstream=("cmpr", "segment"), produces=("incisor_profile",)),
    Step("landmarks", "Landmarks", "Eruption and M1-M3, found from the segmentation and placed on the density "
         "trajectory. Marks set by hand in the viewer are kept.",
         params=LANDMARK_PARAMS, lane="light", upstream=("segment", "cmpr"), produces=("landmarks",)),
    Step("molar_quant", "Molar enamel", "Enamel volume as a share of total molar volume, per tooth.",
         params=MOLAR_PARAMS, upstream=("segment",), produces=("molar_quant",)),
)}

PIPELINE_ORDER = ("convert", "segment", "ml_segment", "cmpr", "incisor_quant", "molar_quant", "landmarks")


# ---------------------------------------------------------------------------
# Views the viewer can open
# ---------------------------------------------------------------------------

VIEWS = {
    "source": {"label": "Raw scan", "base": "raw"},
    "converted": {"label": "Converted scan", "base": "converted"},
    "mandible": {"label": "Hemimandible", "base": "reoriented"},
    "incisor": {"label": "Incisor", "base": "incisor_volume"},
    "molar": {"label": "Molars", "base": "molar_volume"},
    "bone": {"label": "Bone", "base": "bone_volume"},
    "cmpr": {"label": "Straightened incisor", "base": "cmpr_volume"},
}

MANDIBLE_OVERLAYS = (
    ("incisor", "Incisor", "incisor_mask", "incisor_volume"),
    ("bone", "Bone", "bone_mask", "bone_volume"),
    ("molar", "Molars", "molar_mask", "molar_volume"),
    ("ml_incisor", "ML incisor", "ml_incisor_mask", None),
    ("ml_bone", "ML bone", "ml_bone_mask", None),
    ("ml_molar", "ML molars", "ml_molar_mask", None),
)

# The molar view's teeth: (overlay, label, number) -- each overlay is the
# voxels of the molar step's tooth-label map that hold that tooth's number.
TOOTH_OVERLAYS = (("m1", "M1", 1), ("m2", "M2", 2), ("m3", "M3", 3))


def _raw_viewable(record: dict) -> bool:
    return (record.get("source") or {}).get("kind") in ("isq", "dicom", "bmp", "volume")


def view_inputs(ws: Workspace, record: dict, view: str) -> tuple[Path | None, dict[str, Path]]:
    """(base file, {overlay name: file}) for a view, or (None, {}) if it cannot be built."""
    info = VIEWS[view]
    if view == "source" and not _raw_viewable(record):
        return None, {}
    base = ws.artifact(record, info["base"])
    overlays: dict[str, Path] = {}
    if view == "mandible":
        for name, _label, mask_key, volume_key in MANDIBLE_OVERLAYS:
            path = ws.artifact(record, mask_key) or (ws.artifact(record, volume_key) if volume_key else None)
            if path is not None:
                overlays[name] = path
    elif view == "molar":
        # The teeth before the enamel: the 3-D view colours a voxel by the
        # first overlay that covers it, so it shows which tooth is which.
        teeth = ws.artifact(record, "tooth_labels")
        if teeth is not None:
            for name, _label, _number in TOOTH_OVERLAYS:
                overlays[name] = teeth
        path = ws.artifact(record, "enamel_mask")
        if path is not None:
            overlays["enamel"] = path
    return base, overlays


def view_dir(ws: Workspace, name: str, view: str) -> Path:
    return ws.sample_dir(name) / "preview" / view


def view_spec(ws: Workspace, record: dict, view: str) -> dict:
    base, overlays = view_inputs(ws, record, view)
    if base is None:
        raise ValueError(f"{VIEWS[view]['label']} is not available for {record['name']} yet")
    kind = (record.get("source") or {}).get("kind")
    labels = {n: label for n, label, *_ in (*MANDIBLE_OVERLAYS, *TOOTH_OVERLAYS)} | {"enamel": "Enamel"}
    numbers = {n: number for n, _label, number in TOOTH_OVERLAYS}
    spec = {
        "out": str(view_dir(ws, record["name"], view)),
        "view": view,
        "base": {"path": str(base), "key": VIEWS[view]["base"]},
        "overlays": [{"name": n, "label": labels.get(n, n), "path": str(p), "value": numbers.get(n)}
                     for n, p in overlays.items()],
        "budget": int(ws.settings().get("preview_budget_voxels") or 24_000_000),
        "masked": view in ("incisor", "molar", "bone", "cmpr"),
        "z_first": view == "cmpr",
        "unwrap": view == "cmpr",
        "crop_to_tissue": view != "cmpr",
        "source_kind": kind if view == "source" and kind in ("isq", "dicom", "bmp") else None,
    }
    voxel_um = (record.get("meta") or {}).get("voxel_um")
    if voxel_um:
        spec["voxel_mm"] = [float(voxel_um) / 1000.0] * 3
    return spec


def views_for(ws: Workspace, record: dict) -> list[dict]:
    out = []
    for view, info in VIEWS.items():
        base, overlays = view_inputs(ws, record, view)
        if base is None:
            continue
        meta = read_json(view_dir(ws, record["name"], view) / "view.json")
        current = view_is_current(meta, base, overlays)
        out.append({
            "view": view, "label": info["label"], "base": str(base),
            "built": bool(meta), "current": current,
            "overlays": list(overlays),
            "shape": (meta or {}).get("shape"), "thumb": (meta or {}).get("thumb") if meta else None,
        })
    return out


# ---------------------------------------------------------------------------
# Status of a sample's pipeline
# ---------------------------------------------------------------------------

# Raw formats converted to NRRD before segmentation. An NRRD is not on the
# list: it already is what the segmentation reads, so it is segmented as it is.
CONVERTED_FORMATS = ("isq", "dicom", "bmp", "tiff")


def raw_format(record: dict) -> str | None:
    """The raw scan's format (isq, rsq, dicom, bmp, nrrd, tiff), "results", or None."""
    src = record.get("source") or {}
    if src.get("format"):
        return src["format"]
    return source_format(src.get("kind"), src.get("path") or (record.get("artifacts") or {}).get("raw"))


def needs_conversion(record: dict) -> bool:
    return raw_format(record) in CONVERTED_FORMATS


def segment_input(ws: Workspace, record: dict) -> tuple[Path | None, bool]:
    """The scan segmentation reads, and whether the crop box is still to be applied to it.

    An NRRD is read as it is, and cropped in memory as it loads. Anything
    else is read from its conversion, which the converter has already cropped.
    """
    raw = ws.artifact(record, "raw")
    if raw is not None and raw_format(record) == "nrrd":
        return raw, True
    return ws.artifact(record, "converted"), False


def calibration_sources(ws: Workspace, record: dict) -> list[Path]:
    """Where the HA calibration of the volume the segmentation read may be, in the order to try.

    Follows ``segment_input``. A conversion's metadata comes first: the
    converter copies the ISQ's calibration into it and records whether the
    values were rescaled to 1/cm. Conversions made before it copied the
    calibration still record the rescaling, so the ISQ itself comes next. An
    NRRD segmented as it is can only have its own converter's metadata beside it.
    """
    raw = ws.artifact(record, "raw")
    if raw is not None and raw_format(record) == "nrrd":
        candidates = [raw.parent / f"{raw.stem}_meta.json"]
    else:
        candidates = [ws.artifact(record, "convert_meta")]
        if raw is not None and raw_format(record) == "isq":
            candidates.append(raw)
    return [p for p in candidates if p is not None and p.is_file()]


def _skipped(record: dict, step: str) -> bool:
    """Steps a sample never needs, which the UI leaves out rather than showing as blocked."""
    return step == "convert" and (raw_format(record) in ("nrrd", "results"))


def _crop_step(record: dict) -> str | None:
    """The step that applies this sample's crop box."""
    if needs_conversion(record):
        return "convert"
    return "segment" if raw_format(record) == "nrrd" else None


def _done(ws: Workspace, record: dict, step: str) -> bool:
    keys = {
        "convert": ("converted",),
        "segment": ("incisor_volume", "molar_volume"),
        "ml_segment": ("ml_incisor_mask",),
        "cmpr": ("cmpr_volume",),
        "incisor_quant": ("incisor_profile",),
        "landmarks": ("landmarks",),
        "molar_quant": ("molar_quant",),
    }[step]
    return all(ws.artifact(record, k) is not None for k in keys)


def blocked_reason(ws: Workspace, record: dict, step: str) -> str | None:
    """Why a step cannot run yet, or None if its inputs are all there."""
    if STEPS[step].disabled:
        return STEPS[step].disabled
    fmt = raw_format(record)
    art = lambda k: ws.artifact(record, k)  # noqa: E731
    if step == "convert":
        if fmt == "nrrd":
            return "Not needed: an NRRD scan is segmented as it is"
        if fmt == "results":
            return "Imported as finished results"
        if fmt == "rsq":
            return "An RSQ file holds projections, not a scan; reconstruct it first"
        if fmt not in CONVERTED_FORMATS or art("raw") is None:
            return "No raw scan to convert"
        return None
    if step == "segment":
        if segment_input(ws, record)[0] is not None:
            return None
        return "Convert the scan first" if fmt in CONVERTED_FORMATS and art("raw") else "No scan to segment"
    if step == "ml_segment":
        return None if art("reoriented") else "Needs the segmentation's reoriented scan"
    if step == "cmpr":
        return None if art("incisor_volume") else "Needs the incisor segmentation"
    if step == "incisor_quant":
        return None if art("cmpr_volume") else "Needs the straightened incisor"
    if step == "landmarks":
        if not art("incisor_volume"):
            return "Needs the segmentation"
        return None if art("cmpr_transform") else "Needs the straightened incisor's transform"
    if step == "molar_quant":
        return None if art("molar_volume") else "Needs the molar segmentation"
    return None


def can_run_after(ws: Workspace, record: dict, step: str, planned: set[str]) -> bool:
    """Whether a step will have its inputs once the planned upstream steps finish."""
    if STEPS[step].disabled or _skipped(record, step):
        return False
    if blocked_reason(ws, record, step) is None:
        return True
    if step == "landmarks":
        # Two inputs from two steps: the segmentation's volumes, and the
        # straightened incisor's transform.
        return (ws.artifact(record, "incisor_volume") is not None or "segment" in planned) and \
            (ws.artifact(record, "cmpr_transform") is not None or "cmpr" in planned)
    produces = {"convert": {"segment"}, "segment": {"cmpr", "molar_quant", "ml_segment"},
                "cmpr": {"incisor_quant"}}
    return any(step in produces.get(p, set()) for p in planned)


def _upstream_closure(step: str) -> set[str]:
    out, todo = set(), list(STEPS[step].upstream)
    while todo:
        s = todo.pop()
        if s not in out:
            out.add(s)
            todo.extend(STEPS[s].upstream)
    return out


def plan(ws: Workspace, record: dict, requested: list[str]) -> list[tuple[str, list[str]]]:
    """Order the requested steps; for each, the requested steps it must wait for.

    Raises ValueError naming the first step whose inputs neither exist nor
    will be produced by an earlier step in the same run.
    """
    planned: list[str] = []
    out = []
    for sid in PIPELINE_ORDER:
        if sid not in requested:
            continue
        if not can_run_after(ws, record, sid, set(planned)):
            raise ValueError(f"{STEPS[sid].label}: {blocked_reason(ws, record, sid)}")
        out.append((sid, [u for u in planned if u in _upstream_closure(sid)]))
        planned.append(sid)
    return out


def default_run(ws: Workspace, record: dict) -> list[str]:
    """The steps "Run pipeline" queues by default: whatever is missing, in order."""
    wanted = []
    if needs_conversion(record) and not _done(ws, record, "convert"):
        wanted.append("convert")
    for sid in ("segment", "cmpr", "incisor_quant", "molar_quant"):
        if not _done(ws, record, sid):
            wanted.append(sid)
    # The landmarks, found automatically, once there is an incisor to find them on.
    if not _done(ws, record, "landmarks") and can_run_after(ws, record, "landmarks", set(wanted)):
        wanted.append("landmarks")
    try:
        return [s for s, _ in plan(ws, record, wanted)]
    except ValueError:
        return []


def pipeline_status(ws: Workspace, record: dict, active: list[Job]) -> list[dict]:
    steps_rec = record.get("steps") or {}
    by_step = {}
    for job in active:
        by_step.setdefault(job.step, job)
    out = []
    for sid in PIPELINE_ORDER:
        step = STEPS[sid]
        entry = steps_rec.get(sid) or {}
        job = by_step.get(sid)
        done = _done(ws, record, sid)
        if job is not None:
            status = job.status
        elif step.disabled:
            status = "disabled"
        elif _skipped(record, sid):
            status = "skipped"
        elif done:
            status = "done"
            finished = entry.get("finished") or ""
            for up in step.upstream:
                if _skipped(record, up) or STEPS[up].disabled:
                    continue
                up_finished = (steps_rec.get(up) or {}).get("finished") or ""
                if finished and up_finished and up_finished > finished:
                    status = "stale"
            # A crop box changed after the step that applies it ran. Runs
            # from before crop boxes were recorded carry no key, so no verdict.
            if sid == _crop_step(record) and "crop_box" in entry and \
                    (entry["crop_box"] or None) != (record.get("crop_box") or None):
                status = "stale"
        elif entry.get("status") == "failed":
            status = "failed"
        else:
            status = "blocked" if blocked_reason(ws, record, sid) else "ready"
        out.append({
            "id": sid, "label": step.label, "status": status,
            "reason": blocked_reason(ws, record, sid) if status in ("blocked", "skipped", "disabled") else None,
            "finished": entry.get("finished"), "duration_s": entry.get("duration_s"),
            "error": entry.get("error") if status == "failed" else None,
            "job": job.id if job else entry.get("job"),
            "progress": job.progress if job else None, "stage": job.stage if job else None,
            "optional": step.optional,
        })
    return out


# ---------------------------------------------------------------------------
# Preparing a job
# ---------------------------------------------------------------------------

def large_file_dir(ws: Workspace, record: dict) -> Path | None:
    """The folder beside the scan that takes this sample's large files, ready for writing.

    None means the workspace, where an uploaded or copied scan keeps them. A
    folder that cannot be written fails the step rather than quietly filling
    the workspace's drive instead.
    """
    storage = ws.storage_dir(record)
    if storage is None:
        if record.get("storage"):
            label = record["storage"][5:].split("/", 1)[0] if record["storage"].startswith("@vol/") else ""
            raise ValueError(f"The drive {label + ' ' if label else ''}that holds this specimen's large files "
                             "is not connected.")
        return None
    if not storage.parent.parent.is_dir():
        raise ValueError(f"The scan's folder {storage.parent.parent} is not available; is its drive connected?")
    try:
        storage.mkdir(parents=True, exist_ok=True)
        probe = storage / ".studio-write-test"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        raise ValueError(
            f"Cannot write beside the scan, in {storage}: {exc.strerror or exc}. Large files are kept on "
            "the scan's own drive; make that folder writable, or import the scan as a copy to keep "
            "everything in the workspace.") from exc
    return storage


def _module(module: str, spec: dict) -> list[str]:
    return [PYTHON, "-m", module, "--spec", json.dumps(spec)]


def _voxel_mm(record: dict, params: dict | None = None) -> float | None:
    um = (params or {}).get("voxel_size_um") or (record.get("meta") or {}).get("voxel_um")
    return float(um) / 1000.0 if um else None


CONVERT_STAGES = [
    (r"ISQ '|Assembling|Reading|Volume shape", 0.08, "Reading the scan"),
    (r"Crop box|Cropped volume|Autocrop", 0.25, "Cropping"),
    (r"Binning by", 0.3, "Binning"),
    (r"Reorient", 0.35, "Reorienting"),
    (r"Gaussian pre-smoothing|NLM filter|Median filter|diffusion|Richardson|TV \(Chambolle\)|Unsharp", 0.45,
     "Filtering"),
    (r"Wrote|Writing|write_nrrd|NRRD", 0.85, "Writing NRRD"),
    (r"Metadata", 0.97, "Recording metadata"),
]

CMPR_STAGES = [
    (r"Loading volume", 0.02, "Loading the incisor"),
    (r"Building the incisor mask", 0.05, "Masking"),
    (r"Filling holes", 0.08, "Filling holes"),
    (r"Determining centerline", 0.2, "Tracing the centerline"),
    (r"Building graph", 0.25, "Tracing the centerline"),
    (r"Computing local frames", 0.45, "Computing frames"),
    (r"Generating slices", 0.5, "Resampling slices"),
    (r"Saved reslice", 0.97, "Saving"),
]

INCISOR_STAGES = [
    (r"Loaded ", 0.05, "Loading the reslice"),
    (r"Global dentin anchor", 0.12, "Anchoring dentin"),
    (r"Calibrating null", 0.15, "Calibrating the significance test"),
    (r"Molar cross-check", 0.3, "Reading the molar baseline"),
    (r"Orientation", 0.35, "Finding the tip"),
    (r"Analyzing slices", 0.38, "Fitting each slice"),
    (r"Slice outcomes", 0.95, "Writing results"),
]

MOLAR_STAGES = [
    (r"Measuring", 0.05, "Loading the molars"),
    (r"Enamel threshold", 0.3, "Thresholding enamel"),
    (r"Teeth split|teeth", 0.4, "Separating teeth"),
    (r"Enamel surfaces", 0.6, "Measuring enamel thickness"),
    (r"Enamel patches", 0.75, "Finding thin and less dense patches"),
    (r"Thickness map", 0.8, "Drawing the thickness map"),
    (r"wrote", 0.9, "Writing results"),
]

LANDMARK_STAGES = [
    (r"Reading the incisor", 0.05, "Reading the incisor"),
    (r"Reading the bone|Cleaning the bone", 0.15, "Cleaning the bone"),
    (r"Following the incisor", 0.45, "Finding eruption"),
    (r"Splitting the molars", 0.6, "Splitting the molars"),
    (r"Centerline runs", 0.9, "Mapping onto the trajectory"),
    (r"Wrote", 0.97, "Writing"),
]


def prepare(ws: Workspace, job: Job) -> Launch:
    if job.step == "preview":
        record = ws.get(job.sample)
        spec = view_spec(ws, record, job.params["view"])
        return Launch(_module("studio.runners.preview", spec), str(REPO))
    if job.step == "import_copy":
        spec = {"src": job.params["src"], "dst": str(ws.sample_dir(job.sample) / "raw")}
        return Launch(_module("studio.runners.copy", spec), str(REPO))
    if job.step.startswith("tool:"):
        return prepare_tool(ws, job)

    record = ws.get(job.sample)
    if record is None:
        raise ValueError(f"Sample {job.sample} no longer exists")
    reason = blocked_reason(ws, record, job.step)
    if reason:
        raise ValueError(reason)
    name = record["name"]
    folder = ws.sample_dir(name)
    defaults = (ws.settings().get("step_defaults") or {}).get(job.step)
    p = resolve_params(STEPS[job.step].params, job.params, defaults)
    # What this run changed from the defaults. A later run of the step starts
    # from those changes and the defaults as they are then, so a run made
    # before the defaults changed does not keep repeating the old ones.
    base = resolve_params(STEPS[job.step].params, {}, defaults)
    job.context["custom"] = {k: v for k, v in p.items() if v != base.get(k)}
    job.params = {**job.params, **p}
    art = lambda k: ws.artifact(record, k)  # noqa: E731

    if job.step == "convert":
        kind = record["source"]["kind"]
        option_keys = [q.key for q in CONVERT_PARAMS if q.group not in ("ISQ",)]
        options = {k: p[k] for k in option_keys}
        options["crop_box"] = record.get("crop_box")
        large = large_file_dir(ws, record)
        raw = art("raw")
        # What an unfiltered copy must have been made with to be reused. The
        # scan is known by name and size, which do not change with the drive
        # letter or mount point, unlike its path.
        geometry = {k: p[k] for k in CONVERT_GEOMETRY}
        geometry.update(crop_box=record.get("crop_box"), voxel_mm=_voxel_mm(record),
                        source=raw.name if raw else None,
                        source_bytes=raw.stat().st_size if raw and raw.is_file() else None)
        spec = {
            "kind": kind, "source": str(raw), "out": str(large or folder / "converted"), "stem": name,
            "options": options,
            "isq": {"trim_saturated": p["trim_saturated"], "clamp_negative": p["clamp_negative"]},
            "voxel_mm": [_voxel_mm(record)] * 3 if _voxel_mm(record) else None,
            "geometry": geometry,
        }
        return Launch(_module("studio.runners.convert", spec), str(REPO), stages=CONVERT_STAGES,
                      context={"crop_box": record.get("crop_box"), "storage": str(large) if large else None})

    if job.step == "segment":
        source, crop_here = segment_input(ws, record)
        box = record.get("crop_box") if crop_here else None
        # The file's own spacing wins unless this run overrides it.
        voxel = p["voxel_size_um"] / 1000.0 if p.get("voxel_size_um") else None
        # The reoriented mandible is the one large output; the masked volumes
        # and masks the later steps read stay in the workspace.
        large = large_file_dir(ws, record)
        cache = large / "denoise-cache" if large else ws.cache_dir / "segment"
        if box:
            # The post-incisor cache is keyed by the input's path alone, so a
            # different crop of the same file must not find it.
            cache = cache / ("crop-" + "-".join(str(int(v)) for v in box))
        spec = {
            "input": str(source), "out_root": str(folder / "segmentation"), "name": name,
            "format": p["output_format"], "margin_shrink_mm": p["incisor_margin_shrink"] or 0.0,
            "voxel_size": [voxel] * 3 if voxel else None, "strict": p["strict"],
            "cache_dir": str(cache) if p["use_cache"] else None,
            "cache_molar_stage": p["cache_molar_stage"] and p["use_cache"],
            "workers": p["workers"], "crop_box": box,
            "reoriented_dir": str(large) if large else None,
        }
        return Launch(_module("studio.runners.segment", spec), str(REPO),
                      context={"crop_box": box, "storage": str(large) if large else None})

    if job.step == "ml_segment":
        runs = ML / p["checkpoint"]
        # predict.py takes the runs folder for an ensemble, but a single .pt
        # file otherwise; for one model, use the first fold that finished.
        checkpoint = runs
        if not p["ensemble"]:
            best = sorted(runs.glob("fold_*/best_model.pt"))
            if not best:
                raise ValueError(f"No fold in {runs} has a best_model.pt")
            checkpoint = best[0]
        argv = [PYTHON, str(SCRIPTS["ml_predict"]), "--checkpoint", str(checkpoint),
                "--input", str(art("reoriented")), "--output", str(folder / "ml"), "--device", p["device"]]
        argv += ["--ensemble"] if p["ensemble"] else []
        argv += ["--tta"] if p["tta"] else []
        argv += ["--staged", "--bg-threshold", str(p["bg_threshold"])] if p["staged"] else []
        return Launch(argv, str(ML / "src"), env={"KMP_DUPLICATE_LIB_OK": "TRUE"})

    if job.step == "cmpr":
        argv = [PYTHON, str(SCRIPTS["cmpr"]), "-i", str(art("incisor_volume")), "-o", str(folder / "cmpr"),
                "-f", "nrrd", "--spacing", str(p["spacing"]), "--margin", str(p["margin"]), "--log"]
        if p["slice_size"]:
            argv += ["--slice_size", str(p["slice_size"])]
        return Launch(argv, str(REPO / "longitudinal_incisor_analysis"), stages=CMPR_STAGES)

    if job.step == "incisor_quant":
        molar = art("molar_volume")
        baseline = p["baseline"] if (p["baseline"] != "molar" or molar) else "trace"
        argv = [PYTHON, str(SCRIPTS["enamel_density"]), "-i", str(art("cmpr_volume")),
                "-o", str(folder / "quant" / "incisor"), "--log", "--baseline", baseline,
                "--stride", str(p["stride"]), "--orientation", p["orientation"],
                "--erosion", str(p["erosion"]), "--max-rim-fraction", str(p["max_rim_fraction"]),
                "--gate-scale", str(p["gate_scale"]), "--max-baseline-delta", str(p["max_baseline_delta"])]
        if molar:
            argv += ["-m", str(molar)]
        if not p["trim_onset"]:
            argv.append("--no-trim-onset")
        if not p["plots"]:
            argv.append("--no-plot")
        return Launch(argv, str(REPO / "longitudinal_incisor_analysis"), stages=INCISOR_STAGES,
                      context={"baseline": baseline})

    if job.step == "landmarks":
        out = folder / "landmarks"
        marks = marks_path(ws, record)
        # Only mapping the marks as they are, after they were adjusted in the
        # viewer: a job setting rather than an option, so the next run from
        # the sample page still finds the landmarks.
        map_only = bool(job.params.get("map_only"))
        if map_only and not marks.is_file():
            raise ValueError("No landmarks to map: run the step to find them, or mark them in the viewer")
        argv = [PYTHON, str(SCRIPTS["detect_landmarks"]), "-t", str(art("cmpr_transform")),
                "--incisor-volume", str(art("incisor_volume")), "--marks-out", str(marks),
                "-o", str(out / f"{name}_landmarks.csv"), "--report", str(out / f"{name}_landmarks.json")]
        if marks.is_file():
            argv += ["--marks", str(marks)]
        if map_only:
            argv.append("--map-only")
        else:
            if not p["eruption"]:
                argv.append("--no-eruption")
            elif art("bone_volume"):
                argv += ["--bone-volume", str(art("bone_volume"))]
            if not p["molars"]:
                argv.append("--no-molars")
            elif art("molar_volume"):
                argv += ["--molar-volume", str(art("molar_volume"))]
        if art("incisor_profile"):
            argv += ["--profile", str(art("incisor_profile"))]
        return Launch(argv, str(REPO / "longitudinal_incisor_analysis"), stages=LANDMARK_STAGES)

    if job.step == "molar_quant":
        argv = [PYTHON, str(SCRIPTS["molar_enamel"]), "-i", str(art("molar_volume")),
                "-o", str(folder / "quant" / "molar"), "--log", "--method", p["method"],
                "--enamel-mask", p["enamel_mask"], "--min-enamel-voxels", str(p["min_enamel_voxels"])]
        if p["threshold"] is not None:
            argv += ["--threshold", str(p["threshold"])]
        if not p["split"]:
            argv.append("--no-split")
        # The enamel's outer face and the thickness read the background around
        # the tooth, which the molar volume has zeroed; the reoriented scan
        # still has it.
        scan = art("reoriented")
        if scan is not None and scan.is_file():
            argv += ["--scan", str(scan)]
        if not p["thickness"]:
            argv.append("--no-thickness")
        else:
            if not p["thickness_map"]:
                argv.append("--no-thickness-map")
            elif p["thickness_max"]:
                argv += ["--thickness-max", str(p["thickness_max"])]
            if not p["patches"]:
                argv.append("--no-patches")
            else:
                for key, flag, scale in (("patch_around_um", "--patch-around-um", 1.0),
                                         ("thinner_by", "--thinner-by", 0.01),
                                         ("less_dense_by", "--less-dense-by", 0.01),
                                         ("patch_min_area_mm2", "--patch-min-area-mm2", 1.0),
                                         ("patch_margin_um", "--patch-margin-um", 1.0)):
                    if p[key] is not None:
                        argv += [flag, f"{p[key] * scale:g}"]
                if p["save_points"]:
                    argv.append("--save-points")
        if not p["plots"]:
            argv.append("--no-plot")
        if p["percentile_low"] is not None and p["percentile_high"] is not None:
            argv += ["--percentiles", str(p["percentile_low"]), str(p["percentile_high"])]
        if p["ha_slope"] is not None and p["ha_intercept"] is not None:
            argv += ["--ha-slope", str(p["ha_slope"]), "--ha-intercept", str(p["ha_intercept"])]
        else:
            sources = calibration_sources(ws, record)
            if sources:
                argv += ["--calibration", *map(str, sources)]
        if p["mu_scaling"]:
            argv += ["--mu-scaling", str(p["mu_scaling"])]
        return Launch(argv, str(REPO), stages=MOLAR_STAGES)

    raise ValueError(f"Unknown step {job.step}")


# ---------------------------------------------------------------------------
# Landmark marks
# ---------------------------------------------------------------------------

# One row per landmark: where it sits in the reoriented scan's z (a molar, from
# z_start to z_end), and who put it there -- "auto" for one the Landmarks step
# found, "manual" for one marked in the viewer. The step writes this file too
# (longitudinal_incisor_analysis/detect_landmarks.py), in the same columns.
MARKS_COLUMNS = ("landmark", "kind", "z_start", "z_end", "source")

# Sources the Landmarks step replaces when it runs again. "teeth" is what the
# molar step called the M1-M3 it placed, before the Landmarks step placed them.
AUTO_SOURCES = ("auto", "teeth")


def marks_path(ws: Workspace, record: dict) -> Path:
    name = record["name"]
    return ws.artifact(record, "landmark_marks") or ws.sample_dir(name) / "landmarks" / f"{name}_marks.csv"


def read_marks(path: Path) -> list[dict]:
    """A marks file's marks. One written before marks had a source was marked by hand."""
    marks = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                z_end = row.get("z_end")
                marks.append({"landmark": row["landmark"], "z_start": float(row["z_start"]),
                              "z_end": float(z_end) if z_end not in ("", None) else None,
                              "source": row.get("source") or "manual"})
            except (KeyError, TypeError, ValueError):
                continue
    return marks


def write_marks(path: Path, marks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MARKS_COLUMNS)
        writer.writeheader()
        for mark in marks:
            writer.writerow({"landmark": mark["landmark"],
                             "kind": "span" if mark.get("z_end") is not None else "event",
                             "z_start": mark["z_start"], "z_end": "" if mark.get("z_end") is None else mark["z_end"],
                             "source": mark.get("source") or "manual"})


def detected_marks(report: dict | None) -> list[dict]:
    """The landmarks the Landmarks step found, from its report: where each goes back to if a hand mark is undone."""
    out = []
    for mark in (report or {}).get("detected") or []:
        try:
            out.append({"landmark": str(mark["landmark"]), "z_start": float(mark["z_start"]),
                        "z_end": None if mark.get("z_end") is None else float(mark["z_end"]), "source": "auto"})
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Recording what a job wrote
# ---------------------------------------------------------------------------

def _duration(job: Job) -> float | None:
    try:
        start = datetime.datetime.fromisoformat(job.started)
        return round((datetime.datetime.now() - start).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def validation_summary(path: Path | None, name: str) -> dict | None:
    report = read_json(path) if path else None
    if not report:
        return None
    entries = report.get("samples") or []
    entry = next((s for s in entries if s.get("sample") == name), entries[0] if len(entries) == 1 else None)
    if not entry:
        return None
    problems = [
        {"stage": st.get("stage"), "name": r.get("name"), "severity": r.get("severity"),
         "message": r.get("message")}
        for st in entry.get("stages") or [] for r in st.get("results") or []
        if str(r.get("severity", "")).upper() in ("WARN", "FAIL")
    ]
    checks = sum(len(st.get("results") or []) for st in entry.get("stages") or [])
    return {"status": entry.get("status"), "checks": checks, "problems": problems}


def finish(ws: Workspace, job: Job, submit_previews) -> None:
    """Record a finished job's outputs on its sample, and queue the previews they enable."""
    if job.step.startswith("tool:") or job.sample is None:
        return
    record = ws.get(job.sample)
    if record is None:
        return
    name = record["name"]
    folder = ws.sample_dir(name)
    updates: dict[str, str | None] = {}
    views: list[str] = []

    if job.step == "preview":
        return

    if job.step == "import_copy":
        copied = job.result.get("path")
        if not copied:
            raise ValueError("The copy reported no destination")

        def point_at_copy(rec):
            ws._set_artifact(rec, "raw", copied)  # noqa: SLF001
            rec.setdefault("source", {}).update({"path": ws.store_path(rec, copied), "mode": "copy"})
        ws.update(name, point_at_copy)
        if ws.settings().get("auto_preview", True):
            submit_previews(name, ["source"], after=None)
        return

    if job.step == "convert":
        res = job.result
        if not res.get("filtered"):
            raise ValueError("The converter reported no output volume")
        updates.update({"converted": res.get("filtered"), "prefiltered": res.get("prefiltered"),
                        "convert_meta": res.get("meta")})
        views = ["converted"]
    elif job.step == "segment":
        found = artifacts_in(folder / "segmentation" / name, name)
        if "incisor_volume" not in found:
            raise ValueError("No incisor volume was written")
        if job.result.get("reoriented"):
            found["reoriented"] = job.result["reoriented"]
        updates.update(found)
        views = ["incisor", "molar", "mandible"]
    elif job.step == "ml_segment":
        masks = folder / "ml" / "masks"
        for part in ("incisor", "bone", "molar"):
            path = masks / f"{part}_mask.tif"
            if path.exists():
                updates[f"ml_{part}_mask"] = str(path)
        if not updates:
            raise ValueError("The model wrote no masks")
        views = ["mandible"]
    elif job.step == "cmpr":
        out = folder / "cmpr"
        volume = out / f"{name}_resliced_volume.nrrd"
        transform = out / f"{name}_cmpr_transform.npz"
        if not volume.exists():
            raise ValueError(f"{volume.name} was not written")
        updates.update({"cmpr_volume": str(volume),
                        "cmpr_transform": str(transform) if transform.exists() else None})
        views = ["cmpr"]
    elif job.step == "incisor_quant":
        found = artifacts_in(folder / "quant" / "incisor", name, max_depth=1)
        if "incisor_profile" not in found:
            raise ValueError("No density profile was written")
        updates.update(found)
    elif job.step == "landmarks":
        path = folder / "landmarks" / f"{name}_landmarks.csv"
        if not path.exists():
            raise ValueError("No landmark table was written")
        updates["landmarks"] = str(path)
        marks = marks_path(ws, record)
        if marks.is_file():
            updates["landmark_marks"] = str(marks)
        report = folder / "landmarks" / f"{name}_landmarks.json"
        if report.is_file():
            updates["landmark_report"] = str(report)
    elif job.step == "molar_quant":
        found = artifacts_in(folder / "quant" / "molar", name, max_depth=1)
        if "molar_quant" not in found:
            raise ValueError("No molar summary was written")
        updates.update(found)
        views = ["molar"]

    def apply(rec):
        for key, path in updates.items():
            ws._set_artifact(rec, key, path)  # noqa: SLF001 -- same package
        entry = rec.setdefault("steps", {}).setdefault(job.step, {})
        entry.update({"status": "done", "finished": now_iso(), "job": job.id,
                      "duration_s": _duration(job), "params": job.params,
                      "custom": job.context.get("custom") or {}, "error": None})
        if job.step in ("convert", "segment"):
            # Which crop box this output was cut to, so a later change shows it as out of date.
            entry["crop_box"] = job.context.get("crop_box")
            if job.context.get("storage"):
                rec["storage"] = ws.store_path(rec, job.context["storage"])
        if job.step == "segment":
            rec["validation"] = validation_summary(ws.artifact(rec, "validation"), name)
    ws.update(name, apply)
    if job.context.get("storage"):
        _drop_workspace_copies(folder, name, job.step, updates)
    if views and ws.settings().get("auto_preview", True):
        submit_previews(name, views, after=None)


def _drop_workspace_copies(folder: Path, name: str, step: str, written: dict) -> None:
    """Delete the workspace copies that a run beside the scan has just replaced.

    A sample converted or segmented before its large files moved beside the
    scan still holds the old copies; once the new files are recorded they are
    superseded, and keeping them would fill the drive the move is meant to
    spare. Only the studio's own outputs inside the sample's workspace folder
    are removed, and never the file that was just written.
    """
    def inside(path, parent: Path) -> bool:
        return bool(path) and parent.resolve() in Path(path).resolve().parents

    if step == "convert":
        old = folder / "converted"
        if old.is_dir() and not inside(written.get("converted"), old):
            shutil.rmtree(old, ignore_errors=True)
    elif step == "segment":
        seg = folder / "segmentation" / name
        if not inside(written.get("reoriented"), seg):
            for stale in seg.glob(f"{name}_reoriented_volume.*"):
                stale.unlink(missing_ok=True)


def record_failure(ws: Workspace, job: Job) -> None:
    if job.sample is None or job.step in ("preview", "import_copy") or job.step.startswith("tool:"):
        return
    try:
        ws.record_step(job.sample, job.step, status="failed", finished=now_iso(), job=job.id,
                       error=job.error)
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# Standalone tools
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    id: str
    label: str
    summary: str
    params: list[Param]
    lane: str = "heavy"
    disabled: str | None = None      # why the tool is switched off; shown, never run

    def spec(self) -> dict:
        return {"id": self.id, "label": self.label, "summary": self.summary, "lane": self.lane,
                "params": [p.spec() for p in self.params], "disabled": self.disabled}


TEST_SUITES = [
    ("quantify_molar", "Molar enamel quantification", ["quantify_molar/tests"]),
    ("converters", "Converter pipeline", ["tools/tests"]),
    ("segmentation", "Segmentation checks",
     ["segment_mandible/tests/test_molar_background.py", "segment_mandible/tests/test_bone_cleanup.py",
      "segment_mandible/tests/test_validation.py", "segment_mandible/tests/test_shrink.py",
      "segment_mandible/tests/test_cache.py"]),
    ("landmarks", "Landmark detection", ["longitudinal_incisor_analysis/tests/test_detect_landmarks.py"]),
    ("studio", "Studio", ["studio/tests"]),
]

TOOLS: dict[str, Tool] = {t.id: t for t in (
    Tool("rsq_stability", "Check a raw scan for motion",
         "Reads a Scanco .rsq projection file and measures vibration, drift and slips before "
         "you spend time reconstructing it. Compare against known-good scans for a verdict.",
         [Param("rsq_files", "RSQ files to check", "paths", [], "One path per line.", placeholder="/path/to/scan.rsq"),
          Param("reference", "Known-good reference scans", "paths", [],
                "Two or more give a noise floor, and with it a verdict."),
          Param("col_lo", "First detector column", "int", None, advanced=True),
          Param("col_hi", "Last detector column", "int", None, advanced=True,
                help="Restrict to exclude a stationary holder."),
          Param("dump_sinogram", "Save sinogram images", "bool", False)]),
    Tool("density_histogram", "Plot a density histogram",
         "The intensity histogram of a converted volume, with air, soft tissue and mineral peaks.",
         [Param("nrrd_file", "Volume (.nrrd)", "path", "", placeholder="/path/to/volume.nrrd"),
          Param("bins", "Bins", "int", 512, min=16, max=4096),
          Param("mu_scaling", "mu scaling", "number", None, "Divide counts by this to plot 1/cm.",
                advanced=True),
          Param("linear", "Linear y axis", "bool", False)], lane="light"),
    Tool("convert_dicom_tif", "Convert DICOM to BigTIFF",
         "A DICOM series directory to one BigTIFF volume with spacing tags.",
         [Param("dicom_dir", "DICOM folder", "path", ""),
          Param("apply_rescale", "Convert to physical units", "bool", False)]),
    Tool("ml_train", "Train the segmentation model",
         "Leave-one-out training of the 3-D U-Net on segmented samples. Takes hours; runs in the "
         "background. Build the training set from this workspace first.",
         [Param("data", "Training set folder", "path", "", "Samples laid out as reoriented_volume.tif + masks/.",
                placeholder="use the workspace's training set"),
          Param("output", "Runs folder", "path", "", placeholder="ML-Segmentation/runs"),
          Param("epochs", "Max epochs", "int", 500, min=1),
          Param("folds", "Folds", "text", "", placeholder="all"),
          Param("batch", "Batch size", "int", 2, min=1, advanced=True),
          Param("lr", "Learning rate", "number", 1e-3, advanced=True),
          Param("patience", "Early-stop patience", "int", 50, advanced=True)],
         disabled=ML_OFF),
    Tool("tests", "Run the test suites",
         "The repository's synthetic-data tests. Real-data tests skip when the scans are not mounted.",
         [Param("suites", "Suites", "select", "all",
                options=[("all", "All")] + [(sid, label) for sid, label, _ in TEST_SUITES])],
         lane="light"),
)}


def tool_dir(ws: Workspace, tool: str, job_id: str) -> Path:
    return ws.root / "tools" / tool / job_id


def prepare_tool(ws: Workspace, job: Job) -> Launch:
    tool = TOOLS[job.step.split(":", 1)[1]]
    if tool.disabled:
        raise ValueError(tool.disabled)
    p = resolve_params(tool.params, job.params)
    out = tool_dir(ws, tool.id, job.id)
    out.mkdir(parents=True, exist_ok=True)
    job.context["out"] = str(out)

    if tool.id == "rsq_stability":
        files = p["rsq_files"] or []
        if not files:
            raise ValueError("Give at least one .rsq file")
        argv = [PYTHON, str(SCRIPTS["rsq_stability"]), *files, "--output-dir", str(out)]
        if p["reference"]:
            argv += ["--reference", *p["reference"]]
        if p["col_lo"] is not None and p["col_hi"] is not None:
            argv += ["--col-range", str(p["col_lo"]), str(p["col_hi"])]
        if p["dump_sinogram"]:
            argv.append("--dump-sinogram")
        return Launch(argv, str(REPO))
    if tool.id == "density_histogram":
        if not p["nrrd_file"]:
            raise ValueError("Choose a volume")
        target = out / (Path(p["nrrd_file"]).stem + "_histogram.png")
        argv = [PYTHON, str(SCRIPTS["density_histogram"]), p["nrrd_file"], "--out", str(target),
                "--bins", str(p["bins"])]
        if p["mu_scaling"]:
            argv += ["--mu-scaling", str(p["mu_scaling"])]
        if p["linear"]:
            argv.append("--linear")
        return Launch(argv, str(REPO))
    if tool.id == "convert_dicom_tif":
        if not p["dicom_dir"]:
            raise ValueError("Choose a DICOM folder")
        argv = [PYTHON, str(SCRIPTS["convert_dicom_tif"]), p["dicom_dir"], "--out", str(out)]
        if p["apply_rescale"]:
            argv.append("--apply-rescale")
        return Launch(argv, str(REPO))
    if tool.id == "ml_train":
        data = p["data"] or str(ws.root / "ml_training")
        output = p["output"] or str(ML / "runs")
        argv = [PYTHON, str(SCRIPTS["ml_train"]), "--data", data, "--output", output,
                "--epochs", str(p["epochs"]), "--batch", str(p["batch"]), "--lr", str(p["lr"]),
                "--patience", str(p["patience"])]
        if p["folds"]:
            argv += ["--folds", p["folds"]]
        return Launch(argv, str(ML), env={"KMP_DUPLICATE_LIB_OK": "TRUE"})
    if tool.id == "tests":
        chosen = [s for s in TEST_SUITES if p["suites"] in ("all", s[0])]
        paths = [path for _, _, paths in chosen for path in paths]
        return Launch([PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths], str(REPO))
    raise ValueError(f"Unknown tool {tool.id}")


def tool_outputs(job: Job) -> list[dict]:
    out = job.context.get("out")
    if not out or not os.path.isdir(out):
        return []
    files = []
    for path in sorted(glob.glob(os.path.join(out, "**", "*"), recursive=True)):
        if os.path.isfile(path):
            files.append({"name": os.path.relpath(path, out), "path": path, "size": os.path.getsize(path)})
    return files


def build_training_set(ws: Workspace) -> dict:
    """Lay out every TIFF-segmented sample the way ML-Segmentation's loader expects.

    The loader wants ``<sample>/reoriented_volume.tif`` and ``masks/<part>_mask.tif``
    with no sample prefix, while the pipeline now prefixes every file. Links are
    made rather than copies, so the training set costs no disk.
    """
    if TOOLS["ml_train"].disabled:
        raise ValueError(TOOLS["ml_train"].disabled)
    root = ws.root / "ml_training"
    root.mkdir(exist_ok=True)
    made, skipped = [], []
    for record in ws.all():
        parts = {"reoriented_volume.tif": ws.artifact(record, "reoriented")}
        for part in ("incisor", "bone", "molar"):
            parts[f"masks/{part}_mask.tif"] = ws.artifact(record, f"{part}_mask")
        if any(p is None or p.suffix.lower() not in (".tif", ".tiff") for p in parts.values()):
            skipped.append(record["name"])
            continue
        for rel, src in parts.items():
            dst = root / record["name"] / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            try:
                dst.symlink_to(src)
            except OSError:
                import shutil  # noqa: PLC0415
                shutil.copy2(src, dst)
        made.append(record["name"])
    return {"folder": str(root), "samples": made, "skipped": skipped}


def read_profile(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
