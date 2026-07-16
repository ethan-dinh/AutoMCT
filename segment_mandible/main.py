"""
CLI entry point for the mandible segmentation pipeline.
"""

import argparse
import datetime
import gc
import logging
import os
import pathlib

from data_io import load_bmp_stack, load_nrrd, load_tiff, save_ct_volume_as_tiff, save_mask_as_tiff
from pipeline import segment_mandible
from visualization import create_3d_visualization


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def setup_logging(out_root: str) -> None:
    """Configure coloured console logging plus a plain-text .log file under out_root."""

    class ColorFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[94m",
            "INFO": "\033[92m",
            "WARNING": "\033[93m",
            "ERROR": "\033[91m",
            "CRITICAL": "\033[95m",
        }
        RESET = "\033[0m"

        def format(self, record):
            # Colorize a copy of the record so other handlers (e.g. the file
            # handler) sharing this same LogRecord see the plain levelname.
            color = self.COLORS.get(record.levelname, self.RESET)
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter("[%(levelname)s] - %(message)s"))

    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
    logging.info("Logging to %s", log_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment mandible structures (incisor, bone, molar) from a BMP stack or NRRD/TIFF volume."
    )
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="Path to a .nrrd/.tif file, or a directory containing per-sample BMP stack "
             "sub-folders and/or loose .nrrd/.tif/.tiff files (one per sample).",
    )
    parser.add_argument(
        "--log", "-l", action="store_true",
        help="Enable coloured logging output.",
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
        "--out", default=None,
        help="Output root directory (default: ./segmentation_results).",
    )
    return parser.parse_args()


def _save_results(
    result: tuple,
    sample_name: str,
    out_root: str,
    visualize: bool,
) -> None:
    bmp_data, preprocessed, incisor, bone, molar = result

    volumes_dir = os.path.join(sample_name, "volumes")
    masks_dir   = os.path.join(sample_name, "masks")

    try:
        save_ct_volume_as_tiff(bmp_data, sample_name,  "reoriented_volume", base_dir=out_root)
        save_ct_volume_as_tiff(incisor,  volumes_dir,  "incisor_volume",    base_dir=out_root)
        save_ct_volume_as_tiff(bone,     volumes_dir,  "bone_volume",       base_dir=out_root)
        save_ct_volume_as_tiff(molar,    volumes_dir,  "molar_volume",      base_dir=out_root)
        save_mask_as_tiff(incisor > 0, masks_dir, "incisor_mask", base_dir=out_root)
        save_mask_as_tiff(bone   > 0, masks_dir, "bone_mask",    base_dir=out_root)
        save_mask_as_tiff(molar  > 0, masks_dir, "molar_mask",   base_dir=out_root)
        logging.info("Saved to %s/%s/", out_root, sample_name)
    except Exception as e:
        logging.error("Error saving volumes: %s", e)

    if visualize:
        create_3d_visualization(
            preprocessed,
            additional_volumes={
                "Incisor": (incisor, "orange"),
                "Bone":    (bone,    "grey"),
                "Molar":   (molar,   "cyan"),
            },
        )

    del bmp_data, preprocessed, incisor, bone, molar
    gc.collect()


def _process_single_file(args: argparse.Namespace, out_root: str) -> None:
    """Handle a single .nrrd / .tif / .tiff input file."""
    input_path = args.input_path
    sample_name = pathlib.Path(input_path).stem

    incisor_tif = os.path.join(out_root, sample_name, "volumes", "incisor_volume.tif")
    if os.path.exists(incisor_tif):
        logging.info("Already segmented: %s", sample_name)
        if input(f"Visualize existing results for {sample_name}? (y/n): ").lower() == "y":
            original = load_tiff(input_path) if input_path.endswith((".tif", ".tiff")) else None
            incisor = load_tiff(os.path.join(out_root, sample_name, "volumes", "incisor_volume.tif"))
            bone    = load_tiff(os.path.join(out_root, sample_name, "volumes", "bone_volume.tif"))
            molar   = load_tiff(os.path.join(out_root, sample_name, "volumes", "molar_volume.tif"))
            display_volume = original if original is not None else incisor
            if display_volume is None:
                logging.error("No volume available to visualize for %s", sample_name)
            else:
                create_3d_visualization(
                    display_volume,
                    additional_volumes={
                        "Incisor": (incisor, "orange"),
                        "Bone":    (bone,    "grey"),
                        "Molar":   (molar,   "cyan"),
                    },
                )
            del original, incisor, bone, molar, display_volume
            gc.collect()
        if input(f"Re-segment {sample_name}? (y/n): ").lower() != "y":
            return
    else:
        if input(f"Segment {sample_name}? (y/n): ").lower() != "y":
            return

    logging.info("Segmenting %s", sample_name)
    result = segment_mandible(input_path, debug=args.debug)
    if result is not None:
        _save_results(result, sample_name, out_root, args.visualize)
    del result
    gc.collect()


def _load_original_for_sample(sample_path: str):
    """Load the original volume for visualization, regardless of sample type."""
    if sample_path.endswith(".nrrd"):
        return load_nrrd(sample_path)
    if sample_path.endswith((".tif", ".tiff")):
        return load_tiff(sample_path)
    return load_bmp_stack(sample_path)


def _process_directory(args: argparse.Namespace, out_root: str) -> None:
    """
    Handle a directory of samples, where each sample is either a per-sample
    sub-folder (BMP stack, or itself containing a .nrrd/.tif) or a loose
    .nrrd/.tif/.tiff file sitting directly in the input directory. Hidden
    entries (name starting with ".") are ignored.

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

    for sample_name, sample_path in samples:
        incisor_tif = os.path.join(out_root, sample_name, "volumes", "incisor_volume.tif")

        if os.path.exists(incisor_tif):
            logging.info("Already segmented: %s", sample_name)

            if input(f"Visualize existing results for {sample_name}? (y/n): ").lower() == "y":
                original = _load_original_for_sample(sample_path)
                incisor = load_tiff(os.path.join(out_root, sample_name, "volumes", "incisor_volume.tif"))
                bone    = load_tiff(os.path.join(out_root, sample_name, "volumes", "bone_volume.tif"))
                molar   = load_tiff(os.path.join(out_root, sample_name, "volumes", "molar_volume.tif"))
                display_volume = original if original is not None else incisor
                if display_volume is None:
                    logging.error("No volume available to visualize for %s", sample_name)
                else:
                    create_3d_visualization(
                        display_volume,
                        additional_volumes={
                            "Incisor": (incisor, "orange"),
                            "Bone":    (bone,    "grey"),
                            "Molar":   (molar,   "cyan"),
                        },
                    )
                del original, incisor, bone, molar, display_volume
                gc.collect()

            if input(f"Re-segment {sample_name}? (y/n): ").lower() != "y":
                continue

        to_process.append((sample_name, sample_path))

    for sample_name, sample_path in to_process:
        logging.info("Segmenting %s", sample_name)
        result = segment_mandible(sample_path, debug=args.debug)
        if result is not None:
            _save_results(result, sample_name, out_root, args.visualize)
        del result
        gc.collect()


def main() -> None:
    args = process_args()

    out_root = str(pathlib.Path(args.out or "segmentation_results").resolve())

    if args.log:
        setup_logging(out_root)

    if os.path.isfile(args.input_path):
        _process_single_file(args, out_root)
    else:
        _process_directory(args, out_root)


if __name__ == "__main__":
    main()
