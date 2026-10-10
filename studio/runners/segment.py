"""
Segment one sample with the mandible pipeline, without its prompts.

``segment_mandible/main.py`` asks "Segment X? (y/n)" before every sample and
can open napari after it; neither works from a job queue. This calls the same
``segment_mandible()`` and saves through main.py's own ``_save_results``, so
the files on disk are exactly what the CLI writes, and adds the validation
report next to them.

An NRRD scan is segmented without being converted first, so its crop box
(``crop_box`` in the spec) is applied here, as the volume loads. For a scan
linked in place, the reoriented mandible (the one large output) is written
beside the scan instead, into ``reoriented_dir``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys

import numpy as np

from studio.paths import SEGMENT, TOOLS
from studio.runners._common import progress, read_spec, setup_logging

sys.path.insert(0, str(SEGMENT))

log = logging.getLogger("studio.segment")

# The loaders pipeline.py reads its input with, by the names it imports them as.
LOADERS = ("load_nrrd", "load_tiff", "load_dicom_series", "load_bmp_stack")


def crop_loaders(module, box) -> None:
    """Make ``module``'s loaders return only the crop box of what they read.

    This is how the converter's crop reaches a scan that is not converted: the
    loaders stay the pipeline's own and only their result is trimmed, so every
    stage after loading sees an ordinary, smaller volume.
    """
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from volume_pipeline import crop_box_slices  # noqa: PLC0415

    def wrap(load):
        def cropped(*args, **kwargs):
            out = load(*args, **kwargs)
            volume = out[0] if isinstance(out, tuple) else out
            if volume is None:
                return out
            volume = np.ascontiguousarray(volume[crop_box_slices(box, volume.shape)])
            return (volume, *out[1:]) if isinstance(out, tuple) else volume
        return cropped

    for name in LOADERS:
        setattr(module, name, wrap(getattr(module, name)))


def redirect_reoriented(module, folder: str) -> None:
    """Make ``module``'s save_volume write the reoriented volume into ``folder``.

    main.py's ``_save_results`` writes everything under one output root. The
    reoriented mandible is by far its largest file, so it alone is sent to the
    folder beside the scan, written there directly rather than copied later;
    the masked volumes the later steps read stay where they were.
    """
    save = module.save_volume
    parent, leaf = os.path.split(os.path.normpath(folder))

    def save_volume(volume, output_dir, name, *args, **kwargs):
        if name.endswith("_reoriented_volume"):
            output_dir = leaf
            if args:
                args = (parent, *args[1:])
            else:
                kwargs["base_dir"] = parent
        return save(volume, output_dir, name, *args, **kwargs)

    module.save_volume = save_volume


# Log lines the pipeline emits at each stage, mapped to overall progress. The
# pipeline disables its tqdm bars when stderr is not a terminal, so these are
# the only progress signal a job gets.
STAGES = (
    (r"Loaded (NRRD|TIFF|DICOM)|Loading", 0.03, "Loading the volume"),
    (r"Loaded normalized \+ denoised", 0.34, "Reusing the cached denoise"),
    (r"(?i)non.local|NLM", 0.06, "Denoising (non-local means)"),
    (r"(?i)total variation|TV denois", 0.22, "Denoising (total variation)"),
    (r"Finding minimum bone intensity", 0.35, "Finding the bone threshold"),
    (r"Removing small disconnected islands", 0.40, "Removing the background"),
    (r"Reorienting mandible", 0.44, "Reorienting"),
    (r"Applying CLAHE", 0.50, "Enhancing contrast"),
    (r"Isolating surrounding bone", 0.58, "Isolating the incisor"),
    (r"Segmenting incisor", 0.64, "Segmenting the incisor"),
    (r"Removing incisor from volume", 0.76, "Removing the incisor"),
    (r"Segmenting bone and molar", 0.80, "Segmenting bone and molars"),
    (r"Molars separated", 0.88, "Separating the molars"),
    (r"Background trim", 0.91, "Trimming crown background"),
    (r"Saving volume", 0.94, "Saving"),
)


class _StageReporter(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.best = 0.0
        self.patterns = [(re.compile(p), f, label) for p, f, label in STAGES]

    def emit(self, record):
        message = record.getMessage()
        for pattern, fraction, label in self.patterns:
            if fraction > self.best and pattern.search(message):
                self.best = fraction
                progress(fraction, label)
                break


def main() -> int:
    setup_logging()
    logging.getLogger().addHandler(_StageReporter())
    spec = read_spec("Segment one sample with the mandible pipeline")

    import main as main_module  # noqa: PLC0415  (segment_mandible/main.py)
    from main import _resolve_spacing, _save_results  # noqa: PLC0415
    import pipeline  # noqa: PLC0415
    from pipeline import segment_mandible  # noqa: PLC0415
    from validation import ValidationSummary  # noqa: PLC0415

    if spec.get("crop_box"):
        crop_loaders(pipeline, [int(v) for v in spec["crop_box"]])
    if spec.get("reoriented_dir"):
        redirect_reoriented(main_module, spec["reoriented_dir"])
    name, out_root = spec["name"], spec["out_root"]
    override = tuple(spec["voxel_size"]) if spec.get("voxel_size") else None
    summary = ValidationSummary()
    spacing_out: dict = {}
    progress(0.01, "Starting")
    log.info("Segmenting %s from %s", name, spec["input"])
    result = segment_mandible(
        spec["input"],
        spacing_out=spacing_out,
        incisor_margin_shrink_mm=float(spec.get("margin_shrink_mm") or 0.0),
        voxel_size_override=override,
        workers=spec.get("workers"),
        apply_dicom_rescale=bool(spec.get("apply_dicom_rescale")),
        validation=summary.sample(name),
        strict=bool(spec.get("strict")),
        cache_dir=spec.get("cache_dir"),
        cache_molar_stage=bool(spec.get("cache_molar_stage")),
    )
    sample_dir = os.path.join(out_root, name)
    report = os.path.join(sample_dir, "validation_report.json")
    if result is None:
        summary.log()
        if summary.samples:
            summary.write_json(report)
        log.error("Segmentation did not produce an output (see the errors above)")
        return 2

    fmt = spec.get("format") or "nrrd"
    _save_results(result, name, out_root, False, output_format=fmt,
                  spacing=_resolve_spacing(spacing_out, override))
    del result
    summary.log()
    summary.write_json(report)

    # _save_results logs and swallows write errors, so check the files it
    # should have written rather than trusting that it returned.
    reoriented = os.path.join(spec.get("reoriented_dir") or sample_dir, f"{name}_reoriented_volume.{fmt}")
    expected = [os.path.join(sample_dir, "volumes", f"{name}_{part}_volume.{fmt}")
                for part in ("incisor", "bone", "molar")] + [reoriented]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        log.error("Segmentation outputs missing: %s", ", ".join(missing))
        return 3
    if spec.get("reoriented_dir"):
        log.info("Reoriented mandible written beside the scan: %s", reoriented)
    status = summary.samples[0].status if summary.samples else "OK"
    progress(1.0, f"Segmented ({status})")
    print("@@result " + json.dumps({"status": status, "dir": sample_dir, "reoriented": reoriented}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
