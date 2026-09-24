"""
Interactive segmentation TUI with Napari visualization.

Mirrors main.py but adds a per-sample result menu (Accept / Re-segment / Skip)
so the user can inspect each segmentation before saving.

Usage (from segment_mandible/tests/):
    python test_pipeline.py --data /path/to/samples [--out /path/to/output]
    python run.py                  # zero-config wrapper
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pathlib
import sys
import traceback

# ---------------------------------------------------------------------------
# sys.path bootstrap — segment_mandible/ must be importable.
# This file lives in  segment_mandible/tests/  so the package root is one
# level up.  Setting this at module load time works whether the script is run
# directly or imported by run.py.
# ---------------------------------------------------------------------------
_PKG_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# pylint: disable=wrong-import-position
from data_io import save_ct_volume_as_tiff, save_mask_as_tiff           # noqa: E402
from logging_setup import setup_logging                                  # noqa: E402
from pipeline import segment_mandible                                    # noqa: E402
from validation import ValidationSummary                                 # noqa: E402
from visualization import show_segmentation                              # noqa: E402

# ---------------------------------------------------------------------------
# Optional TUI helper
# ---------------------------------------------------------------------------

try:
    import questionary  # type: ignore
    _HAS_QUESTIONARY = True
except ImportError:
    _HAS_QUESTIONARY = False


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

def _menu(prompt: str, choices: list[str]) -> str:
    if _HAS_QUESTIONARY:
        answer = questionary.select(prompt, choices=choices).ask()
        if answer is None:
            raise KeyboardInterrupt
        return answer
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    while True:
        raw = input("Choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"  Enter a number between 1 and {len(choices)}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# An interactive menu loop: each branch is one of the user's choices
# (accept / re-segment / skip / back / quit) at each of two prompts, so the
# count reflects the UI's shape rather than tangled logic.
# pylint: disable=too-many-branches
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive mandible segmentation TUI"
    )
    parser.add_argument("--data", "-i", required=True, help="Root directory of sample sub-folders")
    parser.add_argument("--log", "-l", action="store_true", help="Enable coloured logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Open napari viewer after every major pipeline step for inspection")
    parser.add_argument("--out", default=None, help="Output root (default: ./segmentation_results)")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Cache pre-molar-segmentation state here to skip loading/denoising/CLAHE/incisor "
             "segmentation on reruns of the same sample — speeds up iterating on molar/bone "
             "segmentation alone. Delete the directory to force a full rerun.",
    )
    args = parser.parse_args()

    if args.log:
        setup_logging()

    data_path = pathlib.Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Input path not found: {data_path}")
        sys.exit(1)

    out_root = str(pathlib.Path(args.out or "segmentation_results").resolve())

    _VOLUME_EXTS = {".nrrd", ".tif", ".tiff"}

    if data_path.is_file():
        # Single-file mode: treat the file itself as the only sample.
        data_dir = data_path.parent
        samples = [data_path.name]
    else:
        data_dir = data_path
        samples = sorted(
            name for name in os.listdir(data_dir)
            if not name.startswith(".")
            and (
                (os.path.isdir(data_dir / name) and name != "Compressed")
                or pathlib.Path(name).suffix.lower() in _VOLUME_EXTS
            )
        )
        if not samples:
            print(f"[INFO] No samples found under {data_dir} (looked for sub-folders and {', '.join(_VOLUME_EXTS)} files)")
            sys.exit(0)

    # -----------------------------------------------------------------------
    # Outer loop — file selection
    # -----------------------------------------------------------------------
    while True:
        choices = samples + ["[Quit]"]
        try:
            pick = _menu("Select a sample:", choices)
        except KeyboardInterrupt:
            print("\n[INFO] Exiting.")
            sys.exit(0)

        if pick == "[Quit]":
            print("[INFO] Exiting.")
            sys.exit(0)

        sample_path = str(data_dir / pick)

        # -------------------------------------------------------------------
        # Inner loop — segment / inspect / decide
        # -------------------------------------------------------------------
        while True:
            print(f"\n[INFO] Segmenting {pick} ...")
            try:
                sample_checks = ValidationSummary().sample(pick)
                result = segment_mandible(
                    sample_path, debug=args.debug, cache_dir=args.cache_dir,
                    validation=sample_checks,
                )
            except Exception:
                print(f"[ERROR] Pipeline failed for {pick}:")
                traceback.print_exc()
                try:
                    action = _menu("What next?", ["Retry", "Skip", "Back to list", "Quit"])
                except KeyboardInterrupt:
                    print("\n[INFO] Exiting.")
                    sys.exit(0)
                if action == "Retry":
                    continue
                if action == "Quit":
                    sys.exit(0)
                break  # Skip / Back to list

            if result is None:
                print("[ERROR] segment_mandible returned None — check pipeline logs.")
                try:
                    action = _menu("What next?", ["Retry", "Skip", "Back to list", "Quit"])
                except KeyboardInterrupt:
                    sys.exit(0)
                if action == "Retry":
                    continue
                if action == "Quit":
                    sys.exit(0)
                break

            reoriented, preprocessed, incisor, bone, molar = result

            show_segmentation(preprocessed, incisor, bone, molar)

            # Put the verdict in the prompt itself. The napari window above
            # shows the result but not what is wrong with it, and this is the
            # moment the accept/re-segment call is actually being made.
            status = sample_checks.status
            problems = sample_checks.problems
            if problems:
                print(f"\n[{status}] {len(problems)} check(s) flagged for {pick}:")
                for stage, name, message in problems:
                    print(f"  - [{stage}] {name}: {message}")
            else:
                print(f"\n[OK] All checks passed for {pick}.")

            try:
                action = _menu(
                    f"Result for {pick} (checks: {status}):",
                    ["Accept", "Re-segment", "Skip", "Back to list", "Quit"],
                )
            except KeyboardInterrupt:
                print("\n[INFO] Exiting.")
                sys.exit(0)

            if action == "Accept":
                volumes_dir = os.path.join(pick, "volumes")
                masks_dir   = os.path.join(pick, "masks")
                try:
                    save_ct_volume_as_tiff(reoriented, pick,         "reoriented_volume", base_dir=out_root)
                    save_ct_volume_as_tiff(incisor,    volumes_dir,  "incisor_volume",    base_dir=out_root)
                    save_ct_volume_as_tiff(bone,       volumes_dir,  "bone_volume",       base_dir=out_root)
                    save_ct_volume_as_tiff(molar,      volumes_dir,  "molar_volume",      base_dir=out_root)
                    save_mask_as_tiff(incisor > 0, masks_dir, "incisor_mask", base_dir=out_root)
                    save_mask_as_tiff(bone   > 0, masks_dir, "bone_mask",    base_dir=out_root)
                    save_mask_as_tiff(molar  > 0, masks_dir, "molar_mask",   base_dir=out_root)
                    print(f"[INFO] Saved to {out_root}/{pick}/")
                except Exception as e:
                    print(f"[ERROR] Could not save: {e}")
                break

            if action == "Re-segment":
                result = reoriented = preprocessed = incisor = bone = molar = None
                gc.collect()
                continue

            if action == "Quit":
                sys.exit(0)

            break  # Skip / Back to list

        # Release large arrays regardless of how the inner loop exited.
        result = reoriented = preprocessed = incisor = bone = molar = None
        gc.collect()


if __name__ == "__main__":
    main()
