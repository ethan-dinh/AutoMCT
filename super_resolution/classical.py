"""
Access to the classical ``segment_mandible`` code from this package.

``segment_mandible`` is not an installable package: its modules import each
other by bare name (``from data_io import ...``), so its directory has to be on
``sys.path``. That is done here, once, so the rest of this package can import
from it like any other module.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEGMENT_MANDIBLE_DIR = REPO_ROOT / "segment_mandible"


def add_segment_mandible_to_path() -> None:
    """Make ``segment_mandible``'s bare-name modules importable."""
    if not SEGMENT_MANDIBLE_DIR.is_dir():
        raise ImportError(
            f"segment_mandible not found at {SEGMENT_MANDIBLE_DIR}. This package "
            "uses its loaders and pipeline, so it has to sit next to it in the "
            "repository (the Docker image copies both)."
        )
    path = str(SEGMENT_MANDIBLE_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


add_segment_mandible_to_path()


def _headless_visualization(reason: BaseException) -> types.ModuleType:
    """
    Stand-in for ``segment_mandible/visualization`` when napari is missing.

    ``pipeline.py`` imports its napari viewers at module level even though it
    only opens them under ``debug=True``. The Docker image has no display and
    does not install napari, so without this stand-in the import alone would
    fail. Calling a viewer still fails, and says why.
    """
    module = types.ModuleType("visualization")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            f"napari is not available ({reason}); interactive viewers need the "
            "desktop conda environment, not the headless Docker image."
        )

    module.create_3d_visualization = _unavailable  # type: ignore[attr-defined]
    module.show_segmentation = _unavailable  # type: ignore[attr-defined]
    module.SEGMENTATION_COLORS = {}  # type: ignore[attr-defined]
    return module


def load_classical_pipeline():
    """
    Import and return ``segment_mandible.pipeline.segment_mandible``.

    Imported on demand rather than at module load, so ``train`` and ``upscale``
    work without the pipeline's heavier dependencies ever being touched.
    """
    try:
        importlib.import_module("visualization")
    except ImportError as exc:
        sys.modules["visualization"] = _headless_visualization(exc)
    from pipeline import segment_mandible  # pylint: disable=import-outside-toplevel

    return segment_mandible
