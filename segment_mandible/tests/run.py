"""
Zero-config launcher for the interactive segmentation TUI.

Run from anywhere:
    python run.py

Native directory pickers will prompt for the scan folder and the export folder.
"""

import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

# ---------------------------------------------------------------------------
# Prompt for data and output directories via native OS pickers
# ---------------------------------------------------------------------------

_here = pathlib.Path(__file__).resolve().parent          # segment_mandible/tests/

_root = tk.Tk()
_root.withdraw()                                          # hide the empty Tk window
_root.attributes("-topmost", True)                        # bring pickers to the front

_picked_data = filedialog.askdirectory(
    title="Select the folder containing your MicroCT scans",
    mustexist=True,
)
if not _picked_data:
    _root.destroy()
    print("[INFO] No scan folder selected — exiting.")
    sys.exit(0)

_picked_out = filedialog.askdirectory(
    title="Select the export/output folder",
    mustexist=False,
)
_root.destroy()

if not _picked_out:
    print("[INFO] No export folder selected — exiting.")
    sys.exit(0)

DATA_DIR = pathlib.Path(_picked_data).resolve()
OUT_DIR  = pathlib.Path(_picked_out).resolve()

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
