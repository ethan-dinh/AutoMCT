"""Where the repository's own tools live, so the studio calls them rather than copies of them."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"

# Jobs run under the same interpreter as the server, which is the one the
# `microct-analysis` environment provides.
PYTHON = sys.executable

TOOLS = REPO / "tools"
SEGMENT = REPO / "segment_mandible"
LONGITUDINAL = REPO / "longitudinal_incisor_analysis"
MOLAR = REPO / "quantify_molar"
ML = REPO / "ML-Segmentation"

SCRIPTS = {
    "cmpr": LONGITUDINAL / "cmpr.py",
    "enamel_density": LONGITUDINAL / "enamel_density.py",
    "map_landmarks": LONGITUDINAL / "map_landmarks.py",
    "detect_landmarks": LONGITUDINAL / "detect_landmarks.py",
    "molar_enamel": MOLAR / "molar_enamel.py",
    "convert_dicom_tif": TOOLS / "convert_DICOM_TIF.py",
    "rsq_stability": TOOLS / "analyze_RSQ_stability.py",
    "density_histogram": TOOLS / "plot_density_histogram.py",
    "view_napari": TOOLS / "view_napari.py",
    "ml_predict": ML / "src" / "predict.py",
    "ml_train": ML / "run_pipeline.py",
}

DEFAULT_WORKSPACE = Path.home() / "MicroCT-Studio"
