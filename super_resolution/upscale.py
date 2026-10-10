"""
Applying a trained generator to a whole scan, tile by tile.

A scan is far too large for one forward pass, so it is cut into cubes. Each
cube is run with a margin (``halo``) of surrounding voxels and only its centre
is kept, so every kept voxel saw its neighbourhood as if the volume had been
processed whole; tiling is tested to agree with a single pass. The result
streams straight into a memory-mapped file.

Two things keep the classical pipeline's view of the scan unchanged:

- The output is clipped to the input's own [min, max]. The pipeline min-max
  normalises whatever it loads, so an overshoot anywhere (ringing at a sharp
  enamel edge, say) would rescale *every* threshold downstream. Together with
  the data-consistency projection this makes the upscaled volume's minimum and
  maximum exactly the original's.
- Tiles that are air throughout (not one voxel, halo included, reaches
  halfway from air to tissue) are filled by replicating the original voxels
  instead of running the network. Replication satisfies the consistency
  constraint exactly and keeps the air's intensity histogram -- which the
  pipeline's background threshold is read from -- identical to the original's.
"""

from __future__ import annotations

import logging
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from .model import SRGenerator3d
from .volume_io import (
    Geometry,
    IntensityScale,
    close_memmap,
    create_nrrd_memmap,
    air_and_tissue_levels,
    free_disk_bytes,
)

logger = logging.getLogger(__name__)


def _tiles(shape: tuple[int, ...], tile: int):
    """Corner and extent of each tile covering ``shape``."""
    for z in range(0, shape[0], tile):
        for y in range(0, shape[1], tile):
            for x in range(0, shape[2], tile):
                yield (z, y, x), (min(tile, shape[0] - z), min(tile, shape[1] - y), min(tile, shape[2] - x))


def air_ceiling(volume: np.ndarray) -> float:
    """
    Intensity a tile must not reach anywhere to be treated as air (stored units).

    Halfway between the typical air voxel and the typical tissue voxel (the
    medians either side of the Otsu split). The split alone is not used: when
    air and tissue are cleanly separated, Otsu's criterion is flat across the
    whole gap and it can return a value just above the air noise. A tile is
    skipped only if its *maximum* stays below this, so even a thin sliver of
    bone keeps it on the network; noise spikes in air cost speed, not quality.
    Faint soft tissue can fall below it and is replicated rather than
    super-resolved -- the pipeline's background threshold removes it anyway.
    """
    _, air, tissue = air_and_tissue_levels(volume)
    return air + 0.5 * (tissue - air)


def _nearest(block: np.ndarray, scale: tuple[int, int, int]) -> np.ndarray:
    for axis, s in enumerate(scale):
        if s > 1:
            block = np.repeat(block, s, axis=axis)
    return block


@torch.no_grad()
def upscale_volume(
    volume: np.ndarray,
    generator: SRGenerator3d,
    out: np.ndarray,
    *,
    device: torch.device,
    intensity: IntensityScale | None = None,
    tile: int = 64,
    halo: int = 10,
    skip_below: float | None = None,
    progress: bool = True,
) -> dict:
    """
    Super-resolve ``volume`` (Z, Y, X) into ``out``, which must be ``scale`` times its shape.

    ``out`` may be a memory map; its dtype is the output dtype (integers are
    rounded). ``skip_below`` (stored units) enables air skipping -- see
    :func:`air_ceiling`. Returns a few statistics for the log.
    """
    scale = generator.scale
    expected = tuple(n * s for n, s in zip(volume.shape, scale))
    if tuple(out.shape) != expected:
        raise ValueError(f"out has shape {out.shape}, expected {expected}")
    if tile < 1 or halo < 0:
        raise ValueError("tile must be >= 1 and halo >= 0")

    intensity = intensity or IntensityScale.fit(volume)
    vmin, vmax = volume.min(), volume.max()
    integer = np.issubdtype(out.dtype, np.integer)
    generator.eval()

    tiles = list(_tiles(volume.shape, tile))
    skipped = 0
    start = time.perf_counter()
    for corner, extent in tqdm(tiles, desc="Super-resolving", unit="tile", disable=not progress):
        lo = [max(c - halo, 0) for c in corner]
        hi = [min(c + e + halo, n) for c, e, n in zip(corner, extent, volume.shape)]
        region = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        inner = tuple(slice(c - l, c - l + e) for c, l, e in zip(corner, lo, extent))
        target = tuple(slice(c * s, (c + e) * s) for c, e, s in zip(corner, extent, scale))

        if skip_below is not None and region.max() < skip_below:
            out[target] = _nearest(region[inner], scale)
            skipped += 1
            continue

        low = torch.from_numpy(intensity.to_unit(region))[None, None].to(device)
        high = generator(low)[0, 0]
        keep = tuple(slice(i.start * s, i.stop * s) for i, s in zip(inner, scale))
        result = intensity.from_unit(high[keep].float().cpu().numpy())
        if integer:
            np.rint(result, out=result)
        np.clip(result, vmin, vmax, out=result)
        out[target] = result.astype(out.dtype, copy=False)

    seconds = time.perf_counter() - start
    stats = {
        "tiles": len(tiles),
        "tiles_skipped_as_air": skipped,
        "seconds": round(seconds, 1),
        "input_shape": list(volume.shape),
        "output_shape": list(out.shape),
    }
    logger.info(
        "Super-resolved %s -> %s in %.0f s (%d tiles, %d replicated as air)",
        volume.shape, tuple(out.shape), seconds, len(tiles), skipped,
    )
    return stats


def upscale_to_nrrd(
    volume: np.ndarray,
    geometry: Geometry | None,
    generator: SRGenerator3d,
    out_path: str,
    *,
    device: torch.device,
    tile: int = 64,
    halo: int = 10,
    skip_air: bool = True,
    progress: bool = True,
) -> dict:
    """
    Super-resolve ``volume`` into an NRRD at ``out_path`` that overlays the original.

    The file is raw (uncompressed) so it can be written in place tile by tile;
    it is ``scale^3`` times the input's size on disk. ``geometry=None`` (a scan
    with no known voxel size) writes none, rather than inventing one.
    """
    scale = generator.scale
    shape = tuple(n * s for n, s in zip(volume.shape, scale))
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    needed = math.prod(shape) * volume.dtype.itemsize
    free = free_disk_bytes(out_dir)
    if needed > free:
        raise OSError(
            f"The upscaled volume needs {needed / 1e9:.1f} GB but {out_dir} has "
            f"{free / 1e9:.1f} GB free. Point --work-dir at a larger disk, or use --crop."
        )

    intensity = IntensityScale.fit(volume)
    skip_below = air_ceiling(volume) if skip_air else None
    fine_geometry = geometry.upscaled(scale) if geometry is not None else None
    out = create_nrrd_memmap(out_path, shape, volume.dtype, fine_geometry)
    finished = False
    try:
        stats = upscale_volume(
            volume, generator, out, device=device, intensity=intensity,
            tile=tile, halo=halo, skip_below=skip_below, progress=progress,
        )
        finished = True
    finally:
        # Closed before any removal: Windows cannot delete a mapped file.
        close_memmap(out)
        del out
        if not finished:
            os.remove(out_path)
    stats["path"] = out_path
    stats["bytes"] = needed
    return stats


def check_training_resolution(checkpoint: dict, spacing: tuple[float, float, float]) -> None:
    """
    Warn when a scan's voxel size is far from what the model was trained on.

    The generator learnt what bone and enamel look like at the training scans'
    resolution. Applied to a scan with a very different voxel size it is
    extrapolating, and its output should be checked before it is trusted.
    """
    trained = [entry.get("spacing_mm_zyx") for entry in checkpoint.get("trained_on", [])]
    trained = [np.asarray(t, dtype=float) for t in trained if t]
    if not trained:
        return
    here = np.asarray(spacing, dtype=float)
    closest = min(trained, key=lambda t: float(np.max(np.abs(np.log(here / t)))))
    ratio = here / closest
    if np.any(ratio > 1.1) or np.any(ratio < 1 / 1.1):
        logger.warning(
            "This scan's voxel size %s mm differs from the training scans' (closest %s mm). "
            "The model is extrapolating to a new resolution; inspect the upscaled volume, "
            "or train on scans like this one.",
            tuple(round(float(v), 5) for v in here), tuple(round(float(v), 5) for v in closest),
        )
