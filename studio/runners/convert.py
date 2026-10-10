"""
Convert a scan to NRRD through the converters' shared pipeline, unattended.

ISQ and DICOM go through the same readers ``convert_ISQ_NRRD.py`` and
``convert_DICOM_NRRD.py`` use; BMP stacks and existing .nrrd/.tif volumes are
read with the segmentation loaders and handed to the same ``convert_volume``,
so every format gets the identical crop -> bin -> reorient -> filter chain.
The crop box comes from the studio's crop view as ``crop_box``, which is what
replaces the napari prompt of ``--crop``.

The converter reuses the unfiltered copy a previous run left
(``<stem>_prefiltered.nrrd``) and skips the reading, crop, binning and rotation
it carries -- the right thing when only the filters change, and wrong when the
crop or voxel size did, since it keeps the old ones. The spec's ``geometry``
says what this run asks for; a note beside the copy says what it was made
with, and a copy whose note differs, or has none, is set aside so the scan is
read again.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

from studio.paths import TOOLS
from studio.runners._common import progress, read_spec, setup_logging

sys.path.insert(0, str(TOOLS))

from volume_pipeline import (  # noqa: E402
    PipelineOptions,
    convert_volume,
    directions_from_spacing,
)

log = logging.getLogger("studio.convert")

# Stages that change voxel values; with all of them off the conversion is a
# pure re-container, and an integer scan can stay integer on disk.
_FILTERS = ("gauss_sigma", "median_radius", "aniso_iterations", "rl_iterations",
            "tv_weight", "unsharp_amount", "nlm_h")


def _options(spec: dict) -> PipelineOptions:
    allowed = {f.name for f in dataclasses.fields(PipelineOptions)}
    fields = {k: v for k, v in (spec.get("options") or {}).items()
              if k in allowed and v is not None and v != ""}
    if fields.get("crop_box") is not None:
        fields["crop_box"] = tuple(int(v) for v in fields["crop_box"])
    if fields.get("save_dtype") in (None, "auto"):
        filtered = any(float(fields.get(k) or 0) > 0 for k in _FILTERS)
        integer_source = spec["kind"] == "isq" and not fields.get("apply_rescale")
        fields["save_dtype"] = "int16" if integer_source and not filtered and \
            fields.get("noise_floor") is None else "float32"
    return PipelineOptions(**fields)


def _generic_reader(kind: str, source: Path, voxel_mm):
    """Source tuple for a BMP stack or an existing volume file."""
    from studio.volumes import _segment_io, load_volume  # noqa: PLC0415

    def read():
        if kind == "bmp":
            volume = _segment_io().load_bmp_stack(str(source), exclude_pattern="*spr.bmp")
            spacing = None
        else:
            volume, spacing = load_volume(source)
        if voxel_mm:
            spacing = tuple(float(v) for v in voxel_mm)
        if spacing is None:
            log.warning("%s records no voxel size; writing 1 mm isotropic. Set the "
                        "specimen's voxel size to fix this.", source.name)
            spacing = (1.0, 1.0, 1.0)
        return volume, directions_from_spacing(*spacing), [0.0, 0.0, 0.0], {
            "source": str(source), "source_format": kind,
        }

    return read


def _prefiltered_note(out: Path, stem: str) -> Path:
    return out / f"{stem}_prefiltered.studio.json"


def set_aside_stale_copy(out: Path, stem: str, geometry: dict | None) -> bool:
    """Delete an unfiltered copy made with other geometry than ``geometry``; True if one was.

    The note goes with it, so a run that fails after writing a new copy
    leaves one with no note, which the next run does not trust either.
    """
    copy, note = out / f"{stem}_prefiltered.nrrd", _prefiltered_note(out, stem)
    made = None
    try:
        made = json.loads(note.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if copy.exists() and geometry is not None and made == geometry:
        return False
    stale = copy.exists()
    copy.unlink(missing_ok=True)
    note.unlink(missing_ok=True)
    return stale


def main() -> int:
    setup_logging()
    spec = read_spec(__doc__.splitlines()[1])
    kind, source, out, stem = spec["kind"], Path(spec["source"]), Path(spec["out"]), spec["stem"]
    opts = _options(spec)
    geometry = spec.get("geometry")
    if set_aside_stale_copy(out, stem, geometry):
        log.info("Reading the scan again: %s_prefiltered.nrrd was made with other reading, crop, "
                 "voxel size or rotation settings.", stem)
    log.info("Converting %s (%s) -> %s/%s.nrrd", source, kind, out, stem)
    log.info("Options: %s", {k: v for k, v in dataclasses.asdict(opts).items() if v not in (None, False, 0, 0.0)})
    progress(0.02, "Reading the scan")

    if kind == "isq":
        from convert_ISQ_NRRD import read_isq_volume  # noqa: PLC0415
        isq = spec.get("isq") or {}
        clamp = bool(isq.get("clamp_negative"))
        floor = int(isq.get("negative_floor") or 0)
        if clamp:
            opts = dataclasses.replace(opts, output_floor=float(floor))

        def read():
            return read_isq_volume(
                source, apply_scaling=opts.apply_rescale,
                trim_saturated_voxels=bool(isq.get("trim_saturated", True)),
                clamp_negative=clamp, negative_floor=floor,
            )
    elif kind == "dicom":
        from convert_DICOM_NRRD import read_dicom_series  # noqa: PLC0415

        def read():
            return read_dicom_series(source, apply_rescale=opts.apply_rescale, workers=opts.workers)
    elif kind in ("bmp", "volume"):
        read = _generic_reader(kind, source, spec.get("voxel_mm"))
    else:
        log.error("Cannot convert a %r source", kind)
        return 2

    outputs = convert_volume(output_dir=out, stem=stem, read_source=read, opts=opts)
    if outputs.get("prefiltered") and geometry is not None:
        _prefiltered_note(out, stem).write_text(json.dumps(geometry), encoding="utf-8")
    progress(1.0, "Converted")
    print("@@result " + json.dumps({k: str(v) for k, v in outputs.items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
