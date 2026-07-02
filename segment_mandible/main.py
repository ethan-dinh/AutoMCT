"""
CLI entry point for the mandible segmentation pipeline.
"""

import argparse
import logging
import os
import pathlib

from data_io import load_bmp_stack, load_tiff, save_ct_volume_as_tiff, save_mask_as_tiff
from pipeline import segment_mandible
from visualization import create_3d_visualization


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def setup_logging() -> None:
    """Configure coloured console logging."""

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
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("[%(levelname)s] - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment mandible structures (incisor, bone, molar) from a BMP stack or NRRD/TIFF volume."
    )
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="Path to a .nrrd/.tif file, or a directory containing per-sample BMP stack sub-folders.",
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
                return
            create_3d_visualization(
                display_volume,
                additional_volumes={
                    "Incisor": (incisor, "orange"),
                    "Bone":    (bone,    "grey"),
                    "Molar":   (molar,   "cyan"),
                },
            )
        if input(f"Re-segment {sample_name}? (y/n): ").lower() != "y":
            return
    else:
        if input(f"Segment {sample_name}? (y/n): ").lower() != "y":
            return

    logging.info("Segmenting %s", sample_name)
    result = segment_mandible(input_path, debug=args.debug)
    if result is not None:
        _save_results(result, sample_name, out_root, args.visualize)


def _process_directory(args: argparse.Namespace, out_root: str) -> None:
    """Handle a directory of per-sample sub-folders (BMP stacks)."""
    input_folders = [
        name
        for name in os.listdir(args.input_path)
        if os.path.isdir(os.path.join(args.input_path, name)) and name != "Compressed"
    ]

    to_process = []

    for dir_name in input_folders:
        incisor_tif = os.path.join(out_root, dir_name, "volumes", "incisor_volume.tif")

        if os.path.exists(incisor_tif):
            logging.info("Already segmented: %s", dir_name)

            if input(f"Visualize existing results for {dir_name}? (y/n): ").lower() == "y":
                original = load_bmp_stack(os.path.join(args.input_path, dir_name))
                incisor = load_tiff(os.path.join(out_root, dir_name, "volumes", "incisor_volume.tif"))
                bone    = load_tiff(os.path.join(out_root, dir_name, "volumes", "bone_volume.tif"))
                molar   = load_tiff(os.path.join(out_root, dir_name, "volumes", "molar_volume.tif"))
                create_3d_visualization(
                    original,
                    additional_volumes={
                        "Incisor": (incisor, "orange"),
                        "Bone":    (bone,    "grey"),
                        "Molar":   (molar,   "cyan"),
                    },
                )

            if input(f"Re-segment {dir_name}? (y/n): ").lower() != "y":
                continue
        else:
            if input(f"Segment {dir_name}? (y/n): ").lower() != "y":
                continue

        to_process.append(dir_name)

    for dir_name in to_process:
        logging.info("Segmenting %s", dir_name)
        result = segment_mandible(os.path.join(args.input_path, dir_name), debug=args.debug)
        if result is not None:
            _save_results(result, dir_name, out_root, args.visualize)


def main() -> None:
    args = process_args()

    if args.log:
        setup_logging()

    out_root = str(pathlib.Path(args.out or "segmentation_results").resolve())

    if os.path.isfile(args.input_path):
        _process_single_file(args, out_root)
    else:
        _process_directory(args, out_root)


if __name__ == "__main__":
    main()
