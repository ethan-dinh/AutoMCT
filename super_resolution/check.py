"""
``python -m super_resolution check``: is this machine ready to run the model?

Reports the accelerator PyTorch can see, the memory and disk available, and
whether segment_mandible imports, then pushes a tiny generator and critic
through a forward and backward pass on the chosen device. That last step is
what catches a CUDA build that does not match the driver, or an operator the
device does not support, in seconds rather than an hour into a run.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from importlib import metadata


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def run_check(device_name: str = "auto", work_dir: str = ".") -> int:
    """Print an environment report; return 0 when everything needed works."""
    # pylint: disable=import-outside-toplevel
    failures = 0

    def line(status: str, label: str, value: str) -> None:
        print(f"  [{status:^4}] {label:<22} {value}")

    print("super_resolution environment check")
    line("info", "platform", platform.platform())
    line("info", "python", sys.version.split()[0])
    for package in ("numpy", "scipy", "scikit-image", "pynrrd", "tifffile", "pydicom", "tqdm", "rich"):
        version = _version(package)
        missing = version == "not installed" and package != "rich"
        failures += missing
        line("FAIL" if missing else "ok", package, version)

    try:
        import torch
    except ImportError as exc:
        line("FAIL", "torch", f"cannot import ({exc})")
        return 1
    line("ok", "torch", f"{torch.__version__} (built for CUDA {torch.version.cuda or 'none'})")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            line("ok", f"cuda:{index}", f"{props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        line("warn", "cuda", "not available -- training and upscaling will be slow")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        line("ok", "mps", "available")

    from .model import (
        PatchCritic3d, SRGenerator3d, block_mean, describe_device, select_device,
    )
    from .volume_io import free_disk_bytes, memory_bytes

    try:
        device = select_device(device_name)
        line("ok", "device", describe_device(device))
    except RuntimeError as exc:
        line("FAIL", "device", str(exc))
        return 1

    total, available = memory_bytes()
    line("info", "RAM", f"{total / 1e9:.0f} GB total, {available / 1e9:.0f} GB available"
         if total and available else "unknown")
    try:
        free = free_disk_bytes(work_dir)
        line("info", "disk", f"{free / 1e9:.0f} GB free at {os.path.abspath(work_dir)}")
    except OSError as exc:
        line("warn", "disk", f"cannot read {work_dir}: {exc}")

    try:
        from .classical import load_classical_pipeline

        load_classical_pipeline()
        viewer = sys.modules.get("visualization")
        headless = viewer is not None and getattr(viewer, "__file__", None) is None
        line("ok", "segment_mandible", "pipeline imports" + (" (headless: no napari)" if headless else ""))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        failures += 1
        line("FAIL", "segment_mandible", f"{type(exc).__name__}: {exc}")

    try:
        start = time.perf_counter()
        torch.manual_seed(0)
        generator = SRGenerator3d(scale=2, features=8, growth=4, blocks=1).to(device)
        torch.nn.init.normal_(generator.tail.weight, std=0.01)
        critic = PatchCritic3d(features=8).to(device)
        low = torch.rand(2, 1, 12, 12, 12, device=device)
        high = generator(low)
        (high.mean() + critic(high).mean()).backward()
        error = float((block_mean(high.detach(), generator.scale) - low).abs().max())
        if tuple(high.shape) != (2, 1, 24, 24, 24) or error > 1e-4:
            raise RuntimeError(f"unexpected output: shape {tuple(high.shape)}, consistency error {error:.2e}")
        line("ok", "model smoke test", f"forward + backward on {device} in {time.perf_counter() - start:.2f} s")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        failures += 1
        line("FAIL", "model smoke test", f"{type(exc).__name__}: {exc}")

    print("All checks passed." if not failures else f"{failures} check(s) failed.")
    return 0 if not failures else 1
