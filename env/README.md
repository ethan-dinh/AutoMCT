# Environment Setup

This project uses a conda environment called **`microct-analysis`**.

## Prerequisites

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) if you haven't already.

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

---

## GPU Support (optional)

The `ML-Segmentation` module uses PyTorch. The default install in `environment.yml` pulls the CPU-only build. For GPU acceleration, replace the PyTorch install with the appropriate CUDA build **after** creating the environment:

```powershell
conda activate microct-analysis

# CUDA 12.4 — recommended for CUDA 13.x drivers (most recent wheels available)
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 — use if cu124 has compatibility issues
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121
```

Run `nvidia-smi` to see your driver's CUDA version. PyTorch wheels are built against
a specific CUDA toolkit version, but your driver only needs to be >= that version.
A CUDA 13.x driver supports cu124 and cu121 wheels.

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
