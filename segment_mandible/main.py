"""
CLI entry point for the mandible segmentation pipeline.
"""

import argparse
import gc
import glob
import logging
import os
import pathlib

from data_io import (
    OUTPUT_FORMATS,
    permute_spacing_for_record,
    is_dicom_dir,
    load_bmp_stack,
    load_dicom_series,
    load_nrrd,
    load_tiff,
    save_mask,
    save_volume,
    write_spacing_sidecar,
)
from logging_setup import setup_logging
from pipeline import segment_mandible
from validation import ValidationSummary
from visualization import show_segmentation

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment mandible structures (incisor, bone, molar) from a BMP stack, DICOM series, or NRRD/TIFF volume."
    )
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="Path to a .nrrd/.tif file, a directory holding one sample's BMP "
             "stack or DICOM series, or a directory of per-sample sub-folders "
             "and/or loose .nrrd/.tif/.tiff files (one per sample).",
    )
    parser.add_argument(
        "--log", "-l", action="store_true",
        help="Write a timestamped run log into the output directory. The "
             "console is always logged; this adds the file, which always "
             "records DEBUG detail regardless of the console level.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log at DEBUG: per-slice tracking decisions, every individual "
             "check including the ones that passed, and richer tracebacks.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Log at WARNING: only problems and failed checks.",
    )
    parser.add_argument(
        "--no-color", dest="no_color", action="store_true",
        help="Disable coloured console output. Also implied when stderr is "
             "not a terminal or NO_COLOR is set.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Abort a sample as soon as a stage's validation checks fail, "
             "instead of writing an output that is known to be wrong. Checks "
             "run and are logged either way.",
    )
    parser.add_argument(
        "--validation-report", dest="validation_report", default=None,
        help="Write the full per-sample check results to this JSON path "
             "(default: <out>/validation_report.json).",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Open napari viewer after segmentation.",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Open napari viewer after every major pipeline step for inspection.",
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Output root directory (default: ./segmentation_results).",
    )
    parser.add_argument(
        "--format", "-f", dest="output_format",
        choices=list(OUTPUT_FORMATS), default="tif",
        help="Container for saved volumes and masks (default: tif). Both bake "
             "in the voxel size read from the input, so results open at true "
             "physical scale in 3D Slicer / ITK-SNAP / Fiji.",
    )
    parser.add_argument(
        "--voxel-size", dest="voxel_size", default=None,
        help="Override the voxel size, in mm. Either one number for isotropic "
             "voxels (e.g. 0.02) or three as dz,dy,dx (e.g. 0.02,0.01,0.01). "
             "Applies to the input frame; the axis order is tracked through "
             "reorientation. Use when the input file carries no spacing "
             "(e.g. a BMP stack) or its header is wrong.",
    )
    parser.add_argument(
        "--incisor-margin-shrink", dest="incisor_margin_shrink", type=float, default=0.0,
        help="Trim this much off the incisor surface, in mm, after segmentation "
             "(e.g. 0.05). The incisor/bone boundary is a partial-volume blend, "
             "so a small shrink pulls the mask back to unambiguously-incisor "
             "voxels for intensity-based measurements. Applies to the incisor "
             "only; bone and molar masks are unaffected. Default 0 (no shrink). "
             "Read as a voxel count if no voxel size is available.",
    )
    parser.add_argument(
        "--apply-dicom-rescale", dest="apply_dicom_rescale", action="store_true",
        help="For DICOM inputs, apply the header's RescaleSlope/RescaleIntercept "
             "so voxel values are in the series' physical units. Off by default: "
             "raw stored values are kept, matching tools/convert_DICOM_NRRD.py. "
             "The volume is min-max normalized before any threshold is taken, so "
             "this does not change the segmentation — it only changes the values "
             "written into the saved volumes.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Threads for the incisor margin shrink's distance transform "
             "(default: one per available core).",
    )
    parser.add_argument(
        "--cache-dir", dest="cache_dir", default=None,
        help="Reuse the normalized and denoised volume across runs, storing it "
             "in this directory. Denoising dominates the runtime and does not "
             "depend on any later step, so a second run on the same input "
             "starts at thresholding. The cache is keyed by the input's path "
             "and contents, so editing a scan rebuilds it; delete the "
             "directory to force a full recompute. Expect roughly four bytes "
             "per voxel per cached sample.",
    )
    parser.add_argument(
        "--cache-molar-stage", dest="cache_molar_stage", action="store_true",
        help="Additionally cache the pipeline state after incisor removal, so "
             "a re-run resumes at bone/molar segmentation. Requires "
             "--cache-dir. This freezes the incisor result, so use it only "
             "when iterating on molar/bone segmentation with the incisor "
             "settled -- otherwise incisor changes will appear to do nothing.",
    )
    return parser.parse_args()


def parse_voxel_size(value: str | None) -> tuple[float, float, float] | None:
    """Parse --voxel-size into a (dz, dy, dx) mm tuple, or None if not given."""
    if not value:
        return None
    parts = [p.strip() for p in value.replace(" ", ",").split(",") if p.strip()]
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"--voxel-size: could not parse {value!r} as numbers") from exc

    if len(numbers) == 1:
        numbers = numbers * 3
    if len(numbers) != 3:
        raise SystemExit(
            f"--voxel-size expects 1 or 3 values (got {len(numbers)} from {value!r})"
        )
    if any(n <= 0 for n in numbers):
        raise SystemExit(f"--voxel-size must be positive (got {value!r})")
    return (numbers[0], numbers[1], numbers[2])


def _resolve_spacing(
    spacing_out: dict,
    override: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    """
    Pick the voxel size to write: the --voxel-size override if given,
    otherwise whatever the pipeline read from the input file.

    The override is stated in the *input* frame, so it is mapped through the
    same reorientation record as a header-derived spacing -- an anisotropic
    override would otherwise land on the wrong axes of the output.
    """
    if override is not None:
        record = spacing_out.get("record")
        resolved = permute_spacing_for_record(override, record)
        if resolved != tuple(override):
            logger.info(
                "--voxel-size %s (input frame) maps to %s in the reoriented output frame",
                tuple(override), resolved,
            )
        return resolved
    return spacing_out.get("spacing")


def _save_results(
    result: tuple,
    sample_name: str,
    out_root: str,
    visualize: bool,
    *,
    output_format: str = "tif",
    spacing: tuple[float, float, float] | None = None,
) -> None:
    bmp_data, preprocessed, incisor, bone, molar = result

    volumes_dir = os.path.join(sample_name, "volumes")
    masks_dir   = os.path.join(sample_name, "masks")

    if spacing is None:
        logger.warning(
            "No voxel size available for %s — writing 1 mm isotropic. Pass "
            "--voxel-size to set the true spacing.", sample_name,
        )

    try:
        save_volume(bmp_data, sample_name, f"{sample_name}_reoriented_volume",
                    base_dir=out_root, output_format=output_format, spacing=spacing)
        save_volume(incisor, volumes_dir, f"{sample_name}_incisor_volume",
                    base_dir=out_root, output_format=output_format, spacing=spacing)
        save_volume(bone, volumes_dir, f"{sample_name}_bone_volume",
                    base_dir=out_root, output_format=output_format, spacing=spacing)
        save_volume(molar, volumes_dir, f"{sample_name}_molar_volume",
                    base_dir=out_root, output_format=output_format, spacing=spacing)
        save_mask(incisor > 0, masks_dir, f"{sample_name}_incisor_mask",
                  base_dir=out_root, output_format=output_format, spacing=spacing)
        save_mask(bone > 0, masks_dir, f"{sample_name}_bone_mask",
                  base_dir=out_root, output_format=output_format, spacing=spacing)
        save_mask(molar > 0, masks_dir, f"{sample_name}_molar_mask",
                  base_dir=out_root, output_format=output_format, spacing=spacing)
        write_spacing_sidecar(sample_name, spacing, base_dir=out_root,
                               name=f"{sample_name}_voxel_size")
        logger.info("Saved to %s/%s/ as .%s", out_root, sample_name, output_format)
    except Exception as e:
        logger.error("Error saving volumes: %s", e)

    if visualize:
        show_segmentation(preprocessed, incisor, bone, molar)

    del bmp_data, preprocessed, incisor, bone, molar
    gc.collect()


def _load_saved_volume(path: str):
    """Load a previously-saved output volume, dispatching on its extension."""
    if path.endswith(".nrrd"):
        return load_nrrd(path)
    return load_tiff(path)


def _existing_output_path(out_root: str, sample_name: str, name: str) -> str | None:
    """
    Path of an already-saved output for this sample, in whichever format it
    was written, or None if it has not been segmented yet.

    Checked across formats rather than against the current --format so that
    re-running with a different container still recognises previous work
    instead of silently re-segmenting it.
    """
    for fmt in OUTPUT_FORMATS:
        candidate = os.path.join(
            out_root, sample_name, "volumes", f"{sample_name}_{name}.{fmt}"
        )
        if os.path.exists(candidate):
            return candidate
    return None


def _visualize_existing(out_root: str, sample_name: str, original) -> None:
    """Re-open a previously-saved segmentation in napari."""
    paths = {
        key: _existing_output_path(out_root, sample_name, f"{key}_volume")
        for key in ("incisor", "bone", "molar")
    }
    incisor = _load_saved_volume(paths["incisor"]) if paths["incisor"] else None
    bone    = _load_saved_volume(paths["bone"])    if paths["bone"]    else None
    molar   = _load_saved_volume(paths["molar"])   if paths["molar"]   else None

    display_volume = original if original is not None else incisor
    if display_volume is None:
        logger.error("No volume available to visualize for %s", sample_name)
    else:
        show_segmentation(display_volume, incisor, bone, molar)
    del original, incisor, bone, molar, display_volume
    gc.collect()


def _process_single_sample(
    args: argparse.Namespace, out_root: str, summary: ValidationSummary
) -> None:
    """Handle a single .nrrd/.tif/.tiff file, or one sample's BMP stack or DICOM series."""
    input_path = args.input_path
    sample_name = pathlib.Path(input_path.rstrip(os.sep)).stem
    override_spacing = parse_voxel_size(args.voxel_size)

    if _existing_output_path(out_root, sample_name, "incisor_volume"):
        logger.info("Already segmented: %s", sample_name)
        if input(f"Visualize existing results for {sample_name}? (y/n): ").lower() == "y":
            original = _load_original_for_sample(input_path)
            _visualize_existing(out_root, sample_name, original)
        if input(f"Re-segment {sample_name}? (y/n): ").lower() != "y":
            return
    else:
        if input(f"Segment {sample_name}? (y/n): ").lower() != "y":
            return

    logger.info("Segmenting %s", sample_name)
    spacing_out: dict = {}
    result = segment_mandible(
        input_path, debug=args.debug, spacing_out=spacing_out,
        incisor_margin_shrink_mm=args.incisor_margin_shrink,
        voxel_size_override=override_spacing,
        workers=args.workers,
        apply_dicom_rescale=args.apply_dicom_rescale,
        validation=summary.sample(sample_name),
        strict=args.strict,
        cache_dir=args.cache_dir,
        cache_molar_stage=args.cache_molar_stage,
    )
    if result is not None:
        spacing = _resolve_spacing(spacing_out, override_spacing)
        _save_results(
            result, sample_name, out_root, args.visualize,
            output_format=args.output_format, spacing=spacing,
        )
    del result
    gc.collect()


def _load_original_for_sample(sample_path: str):
    """Load the original volume for visualization, regardless of sample type."""
    if sample_path.endswith(".nrrd"):
        return load_nrrd(sample_path)
    if sample_path.endswith((".tif", ".tiff")):
        return load_tiff(sample_path)
    if is_dicom_dir(sample_path):
        return load_dicom_series(sample_path)
    return load_bmp_stack(sample_path)


def _process_directory(
    args: argparse.Namespace, out_root: str, summary: ValidationSummary
) -> None:
    """
    Handle a directory of samples, where each sample is either a per-sample
    sub-folder (BMP stack, DICOM series, or itself containing a .nrrd/.tif) or
    a loose .nrrd/.tif/.tiff file sitting directly in the input directory.
    Hidden entries (name starting with ".") are ignored.

    Samples with no existing output are segmented automatically. Samples
    that already have output are only re-segmented if the user confirms.
    """
    entries = [name for name in os.listdir(args.input_path) if not name.startswith(".")]

    subfolder_samples = [
        (name, os.path.join(args.input_path, name))
        for name in entries
        if os.path.isdir(os.path.join(args.input_path, name)) and name != "Compressed"
    ]
    loose_file_samples = [
        (pathlib.Path(name).stem, os.path.join(args.input_path, name))
        for name in entries
        if name.endswith((".nrrd", ".tif", ".tiff"))
        and os.path.isfile(os.path.join(args.input_path, name))
    ]

    samples = subfolder_samples + loose_file_samples

    to_process = []

    override_spacing = parse_voxel_size(args.voxel_size)

    for sample_name, sample_path in samples:
        if _existing_output_path(out_root, sample_name, "incisor_volume"):
            logger.info("Already segmented: %s", sample_name)

            if input(f"Visualize existing results for {sample_name}? (y/n): ").lower() == "y":
                _visualize_existing(
                    out_root, sample_name, _load_original_for_sample(sample_path)
                )

            if input(f"Re-segment {sample_name}? (y/n): ").lower() != "y":
                continue

        to_process.append((sample_name, sample_path))

    for index, (sample_name, sample_path) in enumerate(to_process, start=1):
        logger.info("Segmenting %s (%d of %d)", sample_name, index, len(to_process))
        spacing_out: dict = {}
        result = segment_mandible(
            sample_path, debug=args.debug, spacing_out=spacing_out,
            incisor_margin_shrink_mm=args.incisor_margin_shrink,
            voxel_size_override=override_spacing,
            workers=args.workers,
            apply_dicom_rescale=args.apply_dicom_rescale,
            validation=summary.sample(sample_name),
            strict=args.strict,
            cache_dir=args.cache_dir,
            cache_molar_stage=args.cache_molar_stage,
        )
        if result is not None:
            _save_results(
                result, sample_name, out_root, args.visualize,
                output_format=args.output_format,
                spacing=_resolve_spacing(spacing_out, override_spacing),
            )
        del result
        gc.collect()


def _is_single_sample_dir(path: str) -> bool:
    """
    True if `path` is one sample's own slice directory rather than a directory
    of samples -- i.e. it holds loose .bmp slices or a DICOM series directly
    in it.

    Distinguishing the two matters because both are directories: a slice
    directory is segmented as a single sample, while anything else is walked
    for per-sample sub-folders and loose volume files.
    """
    if not os.path.isdir(path):
        return False
    return bool(glob.glob(os.path.join(path, "*.bmp"))) or is_dicom_dir(path)


def main() -> None:
    args = process_args()

    if args.verbose and args.quiet:
        raise SystemExit("--verbose and --quiet are mutually exclusive")

    out_root = str(pathlib.Path(args.out or "segmentation_results").resolve())

    setup_logging(
        out_root if args.log else None,
        verbose=args.verbose,
        quiet=args.quiet,
        color=not args.no_color,
    )

    summary = ValidationSummary()

    if os.path.isfile(args.input_path) or _is_single_sample_dir(args.input_path):
        _process_single_sample(args, out_root, summary)
    else:
        _process_directory(args, out_root, summary)

    summary.log()

    report_path = args.validation_report or os.path.join(out_root, "validation_report.json")
    if summary.samples:
        try:
            summary.write_json(report_path)
        except OSError as e:
            logger.warning("Could not write validation report to %s: %s", report_path, e)

    # A failed check means an output on disk is not trustworthy, so say so in
    # the exit status too -- a batch driven from a shell script should be able
    # to notice without parsing the log.
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
