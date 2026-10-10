"""
Segment on the upscaled grid; apply the masks to the original scan.

For each sample:

1. Load the scan (any format segment_mandible reads), optionally cropped to
   the specimen.
2. Train a model on it, unless one was given with ``--model``.
3. Super-resolve it into a temporary NRRD that overlays the original.
4. Run the unmodified classical pipeline (``segment_mandible``) on that NRRD.
5. Map the pipeline's masks back to the original voxel grid (see
   ``mask_mapping``) and apply them to the original, unmodified intensities.
6. Write everything in segment_mandible's own output layout, plus a label map
   in the scan's original frame and a JSON summary.

Nothing written here contains an upscaled intensity: the upscaled volume is
an input to step 4 and is deleted afterwards (keep it with ``--keep-upscaled``).
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .classical import load_classical_pipeline
from .config import TrainConfig
from .mask_mapping import (
    LABELS,
    STRUCTURES,
    coarse_record,
    count_fine_voxels,
    record_bbox,
    reorient,
    unreorient,
    vote,
)
from .model import describe_device, load_generator, select_device
from .train import MODEL_FILE, train_model
from .upscale import check_training_resolution, upscale_to_nrrd
from .volume_io import (
    Spacing,
    discover_samples,
    load_scan,
    memory_bytes,
    specimen_bounding_box,
    tissue_threshold,
    write_nrrd,
)

# pylint: disable=wrong-import-order
from data_io import (  # noqa: E402  (segment_mandible; on sys.path via .classical)
    OUTPUT_FORMATS,
    permute_spacing_for_record,
    save_mask,
    save_volume,
    write_spacing_sidecar,
)

logger = logging.getLogger(__name__)

# Peak memory of segment_mandible, per voxel of the volume it is given, over
# and above the loaded volume itself. Read off pipeline.py rather than
# measured: the non-local-means stage holds the normalised volume, its float32
# copy, the result and the per-slice results still referenced by their futures
# -- four float32 copies at once. Later stages work on the smaller
# reoriented crop. Treat it as an estimate, good to a factor of ~1.5.
PIPELINE_BYTES_PER_VOXEL = 16


@dataclass
class SegmentOptions:  # pylint: disable=too-many-instance-attributes
    """Settings for :func:`segment_sample`; mirrors the CLI's `segment` flags."""

    model: str | None = None
    out_root: str = "segmentation_results_sr"
    work_dir: str | None = None
    device: str = "auto"
    tile: int = 64
    halo: int = 10
    skip_air: bool = True
    crop: bool = False
    crop_margin: int = 24
    min_fraction: float = 0.5
    output_format: str = "tif"
    save_fractions: bool = False
    keep_upscaled: bool = False
    overwrite: bool = False
    voxel_size: Spacing | None = None
    apply_dicom_rescale: bool = False
    incisor_margin_shrink_mm: float = 0.0
    workers: int | None = None
    strict: bool = False
    cache_dir: str | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    resume_training: bool = False
    progress: bool = True


def estimate_pipeline_peak_bytes(voxels: int, itemsize: int) -> int:
    """Rough peak RAM of segment_mandible on a volume of ``voxels`` voxels."""
    return voxels * (itemsize + PIPELINE_BYTES_PER_VOXEL)


def _preflight(fine_shape: tuple[int, ...], itemsize: int) -> dict:
    voxels = math.prod(fine_shape)
    peak = estimate_pipeline_peak_bytes(voxels, itemsize)
    total, available = memory_bytes()
    logger.info(
        "Upscaled volume: %s = %.2f G voxels (%.1f GB on disk). Estimated peak RAM "
        "for segmentation: ~%.0f GB (this machine: %s total, %s available).",
        fine_shape, voxels / 1e9, voxels * itemsize / 1e9, peak / 1e9,
        f"{total / 1e9:.0f} GB" if total else "unknown",
        f"{available / 1e9:.0f} GB" if available else "unknown",
    )
    if total and peak > total:
        logger.warning(
            "Segmenting the upscaled volume will likely need more RAM than this "
            "machine has (~%.0f GB vs %.0f GB) and may be killed or swap heavily. "
            "Options: --crop to drop the air around the specimen, or a machine "
            "with more memory.", peak / 1e9, total / 1e9,
        )
    return {"fine_voxels": voxels, "estimated_peak_ram_bytes": peak, "ram_total_bytes": total}


def _jsonable_record(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, (tuple, list)):
            out[key] = [list(v) if isinstance(v, (tuple, list)) else v for v in value]
        else:
            out[key] = value
    return out


def outputs_exist(out_root: str, name: str) -> bool:
    """Whether a sample already has results, in either container format."""
    return any(
        os.path.exists(os.path.join(out_root, name, "masks", f"{name}_incisor_mask.{fmt}"))
        for fmt in OUTPUT_FORMATS
    )


def segment_sample(  # pylint: disable=too-many-locals,too-many-statements
    name: str,
    path: str,
    opts: SegmentOptions,
    validation=None,
) -> dict | None:
    """
    Run the full super-resolved segmentation for one sample.

    Returns the summary written to ``<out>/<name>/<name>_sr_summary.json``, or
    None when the classical pipeline failed (its own log says why).
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    out_root = os.path.abspath(opts.out_root)
    sample_dir = os.path.join(out_root, name)
    work_dir = os.path.abspath(opts.work_dir or os.path.join(out_root, "_sr_work"))
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    device = select_device(opts.device)

    # 1. Load, and choose the region to super-resolve --------------------------
    mark = time.perf_counter()
    scan = load_scan(path, voxel_size=opts.voxel_size, apply_dicom_rescale=opts.apply_dicom_rescale)
    full_shape = tuple(scan.data.shape)
    if opts.crop:
        box = specimen_bounding_box(scan.data, tissue_threshold(scan.data), opts.crop_margin)
        box = tuple(slice(s.start or 0, s.stop if s.stop is not None else n) for s, n in zip(box, full_shape))
        logger.info(
            "Cropped to the specimen: %s of %s (%.0f%% of the voxels)",
            tuple((s.start, s.stop) for s in box), full_shape,
            100 * math.prod(s.stop - s.start for s in box) / math.prod(full_shape),
        )
    else:
        box = tuple(slice(0, n) for n in full_shape)
    crop_start = tuple(s.start for s in box)
    region = scan.data[box]
    region_shape = tuple(region.shape)
    itemsize = region.dtype.itemsize
    region_geometry = scan.geometry.shifted(crop_start)
    timings["load"] = time.perf_counter() - mark

    # 2. Model -----------------------------------------------------------------
    mark = time.perf_counter()
    model_path = opts.model
    if model_path is None:
        model_dir = os.path.join(sample_dir, "sr_model")
        existing = os.path.join(model_dir, MODEL_FILE)
        if os.path.exists(existing) and not (opts.overwrite or opts.resume_training):
            logger.info("Reusing the model trained on this sample earlier: %s", existing)
            model_path = existing
        else:
            logger.info("No --model given: training one on %s (self-supervised)", name)
            model_path = train_model([scan], model_dir, opts.train, device, resume=opts.resume_training)
    generator, checkpoint = load_generator(model_path, device)
    scale = generator.scale
    if scan.spacing_from_file:
        check_training_resolution(checkpoint, region_geometry.spacing)
        fine_geometry_source = region_geometry
    else:
        # Inventing a spacing would change the segmentation, not just its
        # labelling: segment_mandible's incisor checks convert jumps to mm when
        # a spacing exists, and a failed check swaps in its fallback mask. So
        # the upscaled file gets no spacing either, like the original.
        logger.warning(
            "%s has no voxel size, so the upscaled volume has none either. "
            "segment_mandible's checks then count in voxels, and at scale %s these "
            "are smaller than the voxels the checks were tuned on. Pass --voxel-size.",
            name, "x".join(map(str, scale)),
        )
        fine_geometry_source = None
    timings["model"] = time.perf_counter() - mark
    logger.info("Model %s (scale %s) on %s", model_path, "x".join(map(str, scale)), describe_device(device))

    fine_shape = tuple(n * s for n, s in zip(region_shape, scale))
    resources = _preflight(fine_shape, itemsize)

    # 3. Super-resolve -----------------------------------------------------------
    mark = time.perf_counter()
    fine_path = os.path.join(work_dir, f"{name}_sr{'x'.join(map(str, scale))}.nrrd")
    upscale_stats = upscale_to_nrrd(
        region, fine_geometry_source, generator, fine_path, device=device,
        tile=opts.tile, halo=opts.halo, skip_air=opts.skip_air, progress=opts.progress,
    )
    timings["upscale"] = time.perf_counter() - mark

    # The pipeline is the memory peak of the whole run, so nothing it does not
    # need stays resident: the network, and the original scan (reloaded below).
    del generator, region
    scan.data = None  # type: ignore[assignment]
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        # 4. Classical segmentation on the fine grid ------------------------------
        mark = time.perf_counter()
        segment_mandible = load_classical_pipeline()
        spacing_out: dict = {}
        result = segment_mandible(
            fine_path,
            spacing_out=spacing_out,
            incisor_margin_shrink_mm=opts.incisor_margin_shrink_mm,
            workers=opts.workers,
            validation=validation,
            strict=opts.strict,
            cache_dir=opts.cache_dir,
        )
        timings["segment"] = time.perf_counter() - mark
        if result is None:
            logger.error("segment_mandible failed on the upscaled %s; see its messages above", name)
            return None

        # 5. Masks back to the original grid ---------------------------------------
        mark = time.perf_counter()
        record = dict(spacing_out.get("record") or {})
        # Only the three structure volumes are needed, and only as masks. Drop
        # the rest first and convert one at a time, to stay under the peak.
        structure_volumes = list(result[2:])
        del result
        gc.collect()
        fine_box = record_bbox(record, fine_shape)
        coarse = coarse_record(record, scale, fine_shape)
        counts: dict[str, np.ndarray] = {}
        for structure in STRUCTURES:
            volume = structure_volumes.pop(0)
            mask = volume != 0
            del volume
            counts[structure] = count_fine_voxels(unreorient(mask, record), fine_box, scale)
            del mask
            gc.collect()
        fine_per_voxel = math.prod(scale)
        masks_box = vote(counts, fine_per_voxel, opts.min_fraction)
        timings["map_back"] = time.perf_counter() - mark
    finally:
        if opts.keep_upscaled:
            logger.info("Kept the upscaled volume: %s", fine_path)
        elif os.path.exists(fine_path):
            os.remove(fine_path)

    # 6. Apply to the original and write -------------------------------------------
    mark = time.perf_counter()
    scan = load_scan(path, voxel_size=opts.voxel_size, apply_dicom_rescale=opts.apply_dicom_rescale)
    region = scan.data[box]
    original = np.ascontiguousarray(reorient(region, coarse))
    unboxed = {**coarse, "bbox": None}
    masks = {s: np.ascontiguousarray(reorient(masks_box[s], unboxed)) for s in STRUCTURES}
    for structure, mask in masks.items():
        if mask.shape != original.shape:
            raise RuntimeError(
                f"{structure} mask {mask.shape} does not match the reoriented original "
                f"{original.shape}; the reorientation record was not applied consistently."
            )

    spacing = (
        permute_spacing_for_record(region_geometry.spacing, coarse)
        if scan.spacing_from_file else None
    )
    fmt = opts.output_format
    save_volume(original, name, f"{name}_reoriented_volume",
                base_dir=out_root, output_format=fmt, spacing=spacing)
    for structure in STRUCTURES:
        save_volume(np.where(masks[structure], original, 0), os.path.join(name, "volumes"),
                    f"{name}_{structure}_volume", base_dir=out_root, output_format=fmt, spacing=spacing)
        save_mask(masks[structure], os.path.join(name, "masks"), f"{name}_{structure}_mask",
                  base_dir=out_root, output_format=fmt, spacing=spacing)
        if opts.save_fractions:
            fine_counts = np.ascontiguousarray(reorient(counts[structure], unboxed)).astype(np.int16)
            save_volume(fine_counts, os.path.join(name, "sr_fractions"),
                        f"{name}_{structure}_fine_voxels", base_dir=out_root,
                        output_format=fmt, spacing=spacing)
    write_spacing_sidecar(name, spacing, base_dir=out_root, name=f"{name}_voxel_size")

    # A label map in the scan's own frame and geometry, so it overlays the
    # original file directly in 3D Slicer (the outputs above are reoriented).
    labels = np.zeros(full_shape, dtype=np.uint8)
    labels_box = labels[box][tuple(slice(lo, hi) for lo, hi in coarse["bbox"])]
    for structure in STRUCTURES:
        labels_box[masks_box[structure]] = LABELS[structure]
    labels_path = os.path.join(sample_dir, f"{name}_labels_input_frame.nrrd")
    write_nrrd(labels_path, labels, scan.geometry if scan.spacing_from_file else None)
    timings["write"] = time.perf_counter() - mark

    voxel_mm3 = (
        float(abs(np.linalg.det(region_geometry.directions))) if scan.spacing_from_file else None
    )
    structures = {}
    for structure in STRUCTURES:
        voxels = int(masks[structure].sum())
        fine = int(counts[structure].sum(dtype=np.int64))
        structures[structure] = {
            "voxels": voxels,
            "volume_mm3": voxels * voxel_mm3 if voxel_mm3 else None,
            "fine_voxels": fine,
            "subvoxel_voxels": fine / fine_per_voxel,
            "subvoxel_volume_mm3": fine / fine_per_voxel * voxel_mm3 if voxel_mm3 else None,
        }
        logger.info(
            "%s: %d voxels; %.1f counting partial voxels%s", structure, voxels, fine / fine_per_voxel,
            f" ({fine / fine_per_voxel * voxel_mm3:.4g} mm^3)" if voxel_mm3 else "",
        )

    timings["total"] = time.perf_counter() - started
    summary = {
        "sample": name,
        "input": os.path.abspath(path),
        "model": os.path.abspath(model_path),
        "scale_zyx": list(scale),
        "min_fraction": opts.min_fraction,
        "original_shape_zyx": list(full_shape),
        "crop_box_zyx": [[s.start, s.stop] for s in box] if opts.crop else None,
        "upscaled_shape_zyx": list(fine_shape),
        "voxel_size_mm_zyx": list(region_geometry.spacing) if scan.spacing_from_file else None,
        "reorientation_record_original_grid": _jsonable_record(coarse),
        "structures": structures,
        "labels_input_frame": os.path.basename(labels_path),
        "label_values": LABELS,
        "upscale": upscale_stats,
        "resources": resources,
        "model_validation": checkpoint.get("validation"),
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    with open(os.path.join(sample_dir, f"{name}_sr_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved %s to %s in %.0f s", name, sample_dir, timings["total"])
    return summary


def segment_samples(input_path: str, opts: SegmentOptions, report_path: str | None = None) -> int:
    """
    Segment every sample under ``input_path`` (one file or folder, or a folder
    of them). Returns a process exit code: 0 only if every sample succeeded and
    passed its validation checks.
    """
    from validation import ValidationSummary  # pylint: disable=import-outside-toplevel

    samples = discover_samples(input_path)
    if not samples:
        logger.error("No samples found under %s", input_path)
        return 1
    summary = ValidationSummary()
    failed = []
    for index, (name, path) in enumerate(samples, start=1):
        if outputs_exist(os.path.abspath(opts.out_root), name) and not opts.overwrite:
            logger.info("Skipping %s: already segmented (pass --overwrite to redo)", name)
            continue
        logger.info("=== %s (%d of %d) ===", name, index, len(samples))
        try:
            if segment_sample(name, path, opts, validation=summary.sample(name)) is None:
                failed.append(name)
        except Exception:  # pylint: disable=broad-exception-caught
            # One bad sample must not cost the rest of a batch that may run for
            # days; the traceback is logged and the exit code reports it.
            logger.exception("Failed on %s", name)
            failed.append(name)

    summary.log()
    if summary.samples:
        path = report_path or os.path.join(os.path.abspath(opts.out_root), "validation_report.json")
        try:
            summary.write_json(path)
        except OSError as exc:
            logger.warning("Could not write validation report to %s: %s", path, exc)
    if failed:
        logger.error("Failed: %s", ", ".join(failed))
    return 1 if failed or summary.failed else 0
