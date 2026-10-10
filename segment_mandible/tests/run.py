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

# Ask whether the user wants to pick a single volume file or a folder of samples.
_mode_win = tk.Toplevel(_root)
_mode_win.withdraw()
_mode_choice: list[str] = []

def _pick_file():
    _mode_choice.append("file")
    _mode_win.destroy()

def _pick_dir():
    _mode_choice.append("dir")
    _mode_win.destroy()

_mode_win.deiconify()
_mode_win.title("Input type")
_mode_win.attributes("-topmost", True)
tk.Label(_mode_win, text="What would you like to segment?", padx=20, pady=10).pack()
tk.Button(_mode_win, text="Single file (.nrrd / .tif)", width=28, command=_pick_file).pack(pady=4)
tk.Button(_mode_win, text="Folder of samples", width=28, command=_pick_dir).pack(pady=4)
_mode_win.protocol("WM_DELETE_WINDOW", lambda: (_mode_choice.append("cancel"), _mode_win.destroy()))
_mode_win.wait_window()

if not _mode_choice or _mode_choice[0] == "cancel":
    _root.destroy()
    print("[INFO] Cancelled — exiting.")
    sys.exit(0)

if _mode_choice[0] == "file":
    _picked_data = filedialog.askopenfilename(
        title="Select a MicroCT volume file",
        filetypes=[("Volume files", "*.nrrd *.tif *.tiff"), ("All files", "*.*")],
    )
else:
    _picked_data = filedialog.askdirectory(
        title="Select the folder containing your MicroCT scans",
        mustexist=True,
    )

if not _picked_data:
    _root.destroy()
    print("[INFO] No input selected — exiting.")
    sys.exit(0)

_picked_out = filedialog.askdirectory(
    title="Select the export/output folder",
    mustexist=False,
)

if not _picked_out:
    _root.destroy()
    print("[INFO] No export folder selected — exiting.")
    sys.exit(0)

# Ask whether to enable caching, which skips loading/denoising/CLAHE/incisor
# segmentation on reruns of the same sample — useful when iterating on
# molar/bone segmentation alone.
_cache_win = tk.Toplevel(_root)
_cache_win.withdraw()
_cache_choice: list[str] = []

def _use_cache():
    _cache_choice.append("yes")
    _cache_win.destroy()

def _no_cache():
    _cache_choice.append("no")
    _cache_win.destroy()

_cache_win.deiconify()
_cache_win.title("Caching")
_cache_win.attributes("-topmost", True)
tk.Label(
    _cache_win,
    text="Cache preprocessing so reruns skip straight to bone/molar segmentation?",
    padx=20, pady=10,
).pack()
tk.Button(_cache_win, text="Yes, use a cache folder", width=28, command=_use_cache).pack(pady=4)
tk.Button(_cache_win, text="No, run fresh every time", width=28, command=_no_cache).pack(pady=4)
_cache_win.protocol("WM_DELETE_WINDOW", lambda: (_cache_choice.append("no"), _cache_win.destroy()))
_cache_win.wait_window()

_picked_cache = None
if _cache_choice and _cache_choice[0] == "yes":
    _picked_cache = filedialog.askdirectory(
        title="Select (or create) a cache folder",
        mustexist=False,
    )

_root.destroy()

DATA_DIR = pathlib.Path(_picked_data).resolve()
OUT_DIR  = pathlib.Path(_picked_out).resolve()
CACHE_DIR = pathlib.Path(_picked_cache).resolve() if _picked_cache else None

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

if CACHE_DIR is not None:
    sys.argv += ["--cache-dir", str(CACHE_DIR)]

# tests/ must be on sys.path so `import test_pipeline` resolves.
# test_pipeline.py itself bootstraps segment_mandible/ onto sys.path.
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import test_pipeline  # noqa: E402

if __name__ == "__main__":
    test_pipeline.main()
