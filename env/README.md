# Environment Setup

This project uses a conda environment called **`microct-analysis`**.

It is supported on **Windows**, **macOS**, and **Linux**. Every dependency in
`environment.yml` resolves to a prebuilt binary on all three platforms for
Python 3.10, so no compiler is required.

---

## Prerequisites

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[Anaconda](https://www.anaconda.com/download) if you haven't already.

| Platform | Notes |
|---|---|
| **Windows** | Run the commands below from **Anaconda Prompt** (or PowerShell after running `conda init powershell` once). `conda` is not on `PATH` in a plain `cmd.exe` window. |
| **macOS** | Apple Silicon (M1–M4) and Intel are both supported. See [Intel Macs](#intel-macs) for one caveat. |
| **Linux** | Needs glibc ≥ 2.17 (any distribution from ~2014 onward). A desktop session is required for the napari/Tk GUI steps — see [Headless Linux](#headless-linux-servers). |

---

## Install the environment

From the root of this repository, run:

```bash
conda env create -f env/environment.yml
```

Then activate it:

```bash
conda activate microct-analysis
```

The same two commands work verbatim on all three platforms.

### Verify the install

```bash
python -c "import numpy, scipy, skimage, napari, nrrd, pydicom, torch, monai, questionary, tkinter; print('environment OK')"
```

---

## What gets installed

Everything comes from `conda-forge` except a handful of pip-only packages:

| Source | Packages |
|---|---|
| conda-forge | python 3.10, numpy, scipy, scikit-image, tifffile, networkx, plotly, tqdm, pillow, matplotlib, numba, rich, questionary, napari, pyqt6, tk |
| pip | pynrrd, pydicom, PyOpenGL-accelerate, torch, monai |

Two dependencies are worth calling out because they are easy to miss:

- **`tk`** — `segment_mandible/tests/run.py` imports `tkinter` for its native
  folder pickers. Conda's Python bundles Tk on all three platforms, but it is
  pinned explicitly so the dependency is visible. (Note: a *system* Python on
  Linux often needs a separate `python3-tk` package — using the conda
  environment avoids this entirely.)
- **`questionary`** — powers the interactive menus in
  `segment_mandible/tests/test_pipeline.py`. The code degrades to a plain
  numbered-list prompt if it is missing, so a broken install shows up as a
  downgraded UI rather than a crash.

---

## GPU Support (optional)

The `ML-Segmentation` module uses PyTorch. `environment.yml` installs the
default build, and the module selects a device automatically
(`cuda` → `mps` → `cpu`), so **no configuration is needed to get working
hardware acceleration on macOS**.

| Platform | Default build gives you | To get more |
|---|---|---|
| **Windows / Linux** | CPU only | Install a CUDA build (below) |
| **macOS (Apple Silicon)** | Metal/MPS acceleration out of the box | — |
| **macOS (Intel)** | CPU only | No GPU option; Apple dropped CUDA support |

### CUDA builds (Windows / Linux with an NVIDIA GPU)

Replace the PyTorch install **after** creating the environment:

```bash
conda activate microct-analysis

# CUDA 12.4 — recommended for CUDA 12.4+ drivers
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 — use if cu124 has compatibility issues
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121
```

Run `nvidia-smi` to see your driver's CUDA version. PyTorch wheels are built
against a specific CUDA toolkit version, but your driver only needs to be
>= that version — a CUDA 13.x driver supports cu124 and cu121 wheels.

To confirm which device will be used:

```bash
python -c "import numpy, torch; print('cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

---

## Platform notes

### Headless Linux (servers)

The conversion and segmentation pipelines run fine headless, but any step that
opens a viewer (`--visualize`, `--debug`, or `run.py`'s folder pickers) needs a
display. On a headless machine either omit those flags, or run under a virtual
framebuffer:

```bash
sudo apt-get install xvfb libgl1 libegl1        # once
xvfb-run -a python main.py --input_path /path/to/samples
```

Qt also needs `libgl1`/`libegl1` present even when using `xvfb-run`. If Qt
fails to start with an `xcb` plugin error, install `libxcb-cursor0`.

### Intel Macs

One dependency, `PyOpenGL-accelerate`, stopped publishing Intel-macOS wheels
after version 3.1.7, so a plain install tries to compile it from source. It is
a pure optimization for napari's rendering and is safe to skip:

```bash
conda env create -f env/environment.yml
pip uninstall -y PyOpenGL-accelerate      # if the build failed, this is a no-op
```

Also note PyTorch dropped Intel-macOS wheels after 2.2, so pip will resolve
`torch>=2.0` to 2.2.x there. That is sufficient for the `ML-Segmentation`
module.

### macOS: OpenMP runtime conflict

pip's PyTorch wheel bundles its own `libomp.dylib`, which collides with the one
conda already provides. If `torch` is the **first** thing a process imports,
macOS aborts with:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Every entry point in this repo imports numpy/scipy before torch, so this does
not occur in normal use. It only shows up in ad-hoc one-liners. Import `numpy`
first:

```bash
python -c "import numpy, torch; print(torch.__version__)"     # works
python -c "import torch; print(torch.__version__)"            # may abort
```

As a last resort `export KMP_DUPLICATE_LIB_OK=TRUE` suppresses the abort, but
it is explicitly unsupported by the OpenMP runtime — prefer the import order.

### Windows

- Long paths: deep sample directories can exceed the legacy 260-character
  limit. If you hit `FileNotFoundError` on paths that clearly exist, enable
  long paths:
  `Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1`
  (run as Administrator, then reboot).
- The parallel converters in `tools/` use the `spawn` start method, which
  re-imports the module in every worker. This is already handled by the
  `if __name__ == "__main__":` guards — preserve them in any new script that
  uses `ProcessPoolExecutor`.

---

## Update an existing environment

If `environment.yml` changes, update your local environment with:

```bash
conda env update -f env/environment.yml --prune
```

---

## Remove the environment

```bash
conda deactivate
conda env remove -n microct-analysis
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `conda: command not found` (Windows) | Use **Anaconda Prompt**, or run `conda init powershell` once and reopen the shell. |
| `ModuleNotFoundError: No module named 'tkinter'` | You are on a system Python, not the conda env. Run `conda activate microct-analysis`. On a system Python, install `python3-tk` (Debian/Ubuntu). |
| Napari opens a blank/black window | Update your GPU drivers. On Linux, confirm `libgl1`/`libegl1` are installed. |
| `qt.qpa.plugin: could not load the Qt platform plugin "xcb"` | Linux only — install `libxcb-cursor0` (and run under `xvfb-run` if headless). |
| `PyOpenGL-accelerate` fails to build | Intel Mac — see [Intel Macs](#intel-macs). It is optional. |
| Interactive menus show a numbered list instead of arrow-key selection | `questionary` is missing; run `conda env update -f env/environment.yml`. |
| `OMP: Error #15 ... libomp.dylib already initialized` | macOS — import `numpy` before `torch`; see [OpenMP runtime conflict](#macos-openmp-runtime-conflict). |
| Solver takes very long / conflicts | Use the faster libmamba solver: `conda config --set solver libmamba`. |
