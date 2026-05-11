"""
Zero-config launcher for the interactive segmentation TUI.

Run from anywhere:
    python run.py

Paths are resolved relative to this file so the script works regardless of
the current working directory.

Edit DATA_DIR below if your data lives somewhere other than MicroCT-Analysis/data/.
"""

import pathlib
import sys

# ---------------------------------------------------------------------------
# Configuration — edit these two lines as needed
# ---------------------------------------------------------------------------

_here = pathlib.Path(__file__).resolve().parent          # segment_mandible/tests/

# Resolved to absolute paths so the script works from any CWD.
DATA_DIR = (_here.parent.parent / "data").resolve()      # MicroCT-Analysis/data/

# Accepted segmentations land in  segment_mandible/tests/segmentation_results/
OUT_DIR = _here / "segmentation_results"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

# Inject --data and --out so test_pipeline.main() picks them up via argparse.
sys.argv = [
    "test_pipeline.py", 
    "--data", str(DATA_DIR),
    "--out",  str(OUT_DIR),
    "--log",
    "--debug", # Uncomment to open Napari viewers at each pipeline step. Note: this is a lot of windows to click through if you have many samples, so use judiciously.
]

# tests/ must be on sys.path so `import test_pipeline` resolves.
# test_pipeline.py itself bootstraps segment_mandible/ onto sys.path.
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import test_pipeline  # noqa: E402

if __name__ == "__main__":
    test_pipeline.main()
