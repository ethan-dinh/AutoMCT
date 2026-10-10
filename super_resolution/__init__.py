"""
Generative super-resolution for segmenting micro-CT scans on a finer grid.

A 3-D GAN upscales a scan; the classical ``segment_mandible`` pipeline
segments the upscaled volume; the masks are brought back to the original voxel
grid and applied to the original, unmodified scan. The upscaled volume is only
ever an input to segmentation -- no measurement is taken from it.

Run ``python -m super_resolution --help`` from the repository root.
"""

import os

# Both must be set before torch is first imported, and every module in this
# package is imported through this file first, so this is the one place that
# is guaranteed to run early enough.
#
# - KMP_DUPLICATE_LIB_OK: the conda env links two OpenMP runtimes (torch's own
#   and conda-forge's), and on macOS a bare `import torch` aborts on the
#   duplicate. ML-Segmentation sets the same flag for the same reason.
# - PYTORCH_ENABLE_MPS_FALLBACK: an operator missing from the Apple-GPU backend
#   runs on the CPU instead of raising. Irrelevant on CUDA.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
