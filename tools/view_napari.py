"""
Open a microCT sample in napari with a contrast range that actually shows the
specimen.

Accepts anything the project already produces: an .nrrd or .tif/.tiff volume,
or a directory of reconstruction slices (the ``*_rec_Tra*.bmp`` stacks under
data/). Voxel spacing is read off the file where the format carries it, so the
viewer opens the volume at true physical size instead of as a unitless voxel
grid.

Contrast: microCT reconstructions are dominated by air. A plain min/max window
therefore spends nearly its whole range on background and leaves bone and
enamel crushed into the top few percent. ``auto_contrast`` instead ignores the
background mode and stretches over the percentiles of the remaining signal, so
mineralised tissue fills the display range on open. Pass --percentiles to widen
or tighten that window, or --no-skip-background to include air in it.

Examples:
    python tools/view_napari.py data/KOP25-1_8um_2K_reoriented
    python tools/view_napari.py results/KOP25-1.nrrd --labels results/KOP25-1_mask.nrrd
    python tools/view_napari.py scan.tif --downsample 2 --percentiles 1 99.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Import the project's loaders so this viewer reads volumes exactly the way the
# segmentation pipeline does -- same (Z, Y, X) axis order, same spacing rules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment_mandible.data_io import (  # noqa: E402
    DEFAULT_SPACING_MM,
    load_bmp_stack,
    load_nrrd,
    load_tiff,
)

log = logging.getLogger("view_napari")

NRRD_SUFFIXES = (".nrrd", ".nhdr")
TIFF_SUFFIXES = (".tif", ".tiff")
SLICE_SUFFIXES = (".bmp", ".tif", ".tiff", ".png")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sample(path: Path) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """
    Load a volume from a file or a slice directory.

    Returns ``(volume_zyx, spacing_zyx_mm)``; spacing is None when the source
    carries none (BMP stacks always, TIFFs without resolution tags).
    """
    if path.is_dir():
        pattern = _slice_pattern(path)
        log.info("Loading slice stack from %s (%s)", path, pattern)
        # Sort numerically: the reconstruction index is zero-padded, but that
        # is not guaranteed across every export, and a plain lexical sort would
        # then interleave slices.
        return load_bmp_stack(str(path), file_pattern=pattern, sort_by="number"), None

    suffix = path.suffix.lower()
    if suffix in NRRD_SUFFIXES:
        volume, spacing = load_nrrd(str(path), return_spacing=True)
    elif suffix in TIFF_SUFFIXES:
        volume, spacing = load_tiff(str(path), return_spacing=True)
    else:
        raise ValueError(
            f"Unsupported input {path.name!r}; expected a directory of slices, "
            f"or a file ending in {', '.join(NRRD_SUFFIXES + TIFF_SUFFIXES)}"
        )

    if volume is None:
        raise FileNotFoundError(f"Could not read a volume from {path}")
    return volume, spacing


def _slice_pattern(directory: Path) -> str:
    """
    Pick the glob matching the image slices in a reconstruction directory.

    A BMP stack is the common case, but the same directory layout shows up with
    TIFF or PNG slices, so the extension is chosen by what is actually present
    rather than assumed.
    """
    for suffix in SLICE_SUFFIXES:
        if any(directory.glob(f"*{suffix}")):
            return f"*{suffix}"
    raise FileNotFoundError(
        f"No image slices ({', '.join(SLICE_SUFFIXES)}) found in {directory}"
    )


def downsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """
    Strided subsample by ``factor`` on every axis.

    Striding rather than averaging: this is only to keep a multi-gigabyte 3K
    stack inside GPU memory for viewing, and averaging would both cost a full
    pass over the array and blur exactly the fine mineral detail the display is
    meant to show.
    """
    if factor <= 1:
        return volume
    return volume[::factor, ::factor, ::factor]


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------

def auto_contrast(
    volume: np.ndarray,
    percentiles: tuple[float, float] = (2.0, 99.8),
    skip_background: bool = True,
    max_samples: int = 4_000_000,
) -> tuple[float, float]:
    """
    Choose a display window that puts the specimen across the visible range.

    Percentiles are computed on a random subsample -- a full sort of a 3K stack
    costs far more than the display decision is worth, and the quantiles of a
    few million voxels are stable well past the precision a contrast slider can
    show.

    With ``skip_background``, voxels at or below the background level are
    dropped before the percentiles are taken. Air is the single largest
    population in a microCT reconstruction, so including it drags the low
    percentile into the noise floor and compresses tissue into a sliver at the
    top of the range. The cut is placed at the first histogram minimum above
    the background peak, which is the valley between air and specimen.

    Returns a ``(low, high)`` contrast limit pair, always with ``high > low``.
    """
    finite = _sample_finite(volume, max_samples)
    if finite.size == 0:
        log.warning("Volume has no finite voxels; falling back to (0, 1)")
        return (0.0, 1.0)

    values = finite
    if skip_background:
        cut = _background_cut(finite)
        if cut is not None:
            foreground = finite[finite > cut]
            # Guard against a cut that leaves almost nothing: a scan that is
            # mostly specimen, or an unusual histogram, should fall back to the
            # full range rather than window onto a handful of voxels.
            if foreground.size >= max(finite.size * 0.001, 1000):
                values = foreground
                log.info(
                    "Background cut at %.4g; %.1f%% of voxels kept as signal",
                    cut, 100.0 * foreground.size / finite.size,
                )
            else:
                log.info("Background cut discarded (too little signal above it)")

    low, high = np.percentile(values, percentiles)
    low, high = float(low), float(high)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        # A constant volume: give napari a non-degenerate range so the slider
        # remains usable.
        high = low + 1.0

    log.info("Contrast limits: (%.4g, %.4g)", low, high)
    return (low, high)


def _sample_finite(volume: np.ndarray, max_samples: int) -> np.ndarray:
    """Flatten to finite voxels, randomly subsampled to at most ``max_samples``."""
    flat = np.asarray(volume).reshape(-1)
    if flat.size > max_samples:
        rng = np.random.default_rng(0)  # fixed seed: same file, same window
        flat = flat[rng.choice(flat.size, max_samples, replace=False)]
    flat = flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat
    return flat


def _background_cut(values: np.ndarray, bins: int = 256) -> float | None:
    """
    Intensity separating the air peak from the specimen.

    Takes the tallest histogram bin as background (air dominates the voxel
    count in every scan here) and walks up until the counts stop falling. That
    turning point is the valley between the air peak and the tissue
    distribution.

    Returns None when the background peak is not the lowest-intensity mode --
    an inverted or already-masked volume -- so the caller keeps the full range
    rather than cutting into the specimen.
    """
    counts, edges = np.histogram(values, bins=bins)
    peak = int(np.argmax(counts))

    # Background should sit in the dark end. If the dominant mode is up in the
    # bright half, this is not an air-dominated reconstruction.
    if peak > bins // 2:
        return None

    idx = peak
    while idx + 1 < counts.size and counts[idx + 1] < counts[idx]:
        idx += 1

    if idx >= counts.size - 1:
        return None
    return float(edges[idx + 1])


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def open_viewer(
    volume: np.ndarray,
    spacing: tuple[float, float, float] | None,
    name: str,
    contrast_limits: tuple[float, float],
    labels: np.ndarray | None = None,
    render_3d: bool = False,
):
    """Build the napari viewer with the image (and optional label) layer."""
    import napari

    scale = spacing if spacing is not None else DEFAULT_SPACING_MM

    viewer = napari.Viewer(title=f"microCT - {name}")
    viewer.add_image(
        volume,
        name=name,
        scale=scale,
        contrast_limits=contrast_limits,
        colormap="gray",
        # Attenuated MIP reads far better than plain MIP on a dense mineralised
        # sample, where a plain projection flattens the whole tooth into one
        # bright silhouette.
        rendering="attenuated_mip",
    )

    if labels is not None:
        viewer.add_labels(labels.astype(np.uint32), name=f"{name} labels", scale=scale)

    viewer.dims.ndisplay = 3 if render_3d else 2
    if not render_3d:
        # Open on the middle slice: the first slice of a reconstruction is
        # usually empty air.
        viewer.dims.set_current_step(0, volume.shape[0] // 2)

    viewer.scale_bar.visible = spacing is not None
    viewer.scale_bar.unit = "mm"
    return viewer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Open a microCT sample in napari with auto-contrast.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("sample", type=Path,
                   help="Path to an .nrrd/.tif volume, or a directory of slices")
    p.add_argument("--labels", type=Path, default=None,
                   help="Optional segmentation mask to overlay as a labels layer")
    p.add_argument("--downsample", type=int, default=1, metavar="N",
                   help="Subsample every Nth voxel on each axis before display")
    p.add_argument("--percentiles", type=float, nargs=2, default=(2.0, 99.8),
                   metavar=("LOW", "HIGH"),
                   help="Percentiles of the signal mapped to the display range")
    p.add_argument("--no-skip-background", dest="skip_background",
                   action="store_false",
                   help="Include air in the contrast percentiles")
    p.add_argument("--contrast-limits", type=float, nargs=2, default=None,
                   metavar=("LOW", "HIGH"),
                   help="Explicit display range, bypassing auto-contrast")
    p.add_argument("--3d", dest="render_3d", action="store_true",
                   help="Open in 3-D rendering mode instead of slice view")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    sample: Path = args.sample
    if not sample.exists():
        log.error("'%s' does not exist.", sample)
        sys.exit(1)
    if args.downsample < 1:
        log.error("--downsample must be >= 1 (got %d).", args.downsample)
        sys.exit(1)

    volume, spacing = load_sample(sample)

    if args.downsample > 1:
        volume = downsample(volume, args.downsample)
        if spacing is not None:
            # The voxels got physically larger; scale spacing to match or the
            # scale bar reports the wrong size.
            spacing = tuple(s * args.downsample for s in spacing)
        log.info("Downsampled by %dx to %s", args.downsample, volume.shape)

    labels = None
    if args.labels is not None:
        if not args.labels.exists():
            log.error("Label volume '%s' does not exist.", args.labels)
            sys.exit(1)
        labels, _ = load_sample(args.labels)
        labels = downsample(labels, args.downsample)
        if labels.shape != volume.shape:
            log.error(
                "Label shape %s does not match volume shape %s.",
                labels.shape, volume.shape,
            )
            sys.exit(1)

    if args.contrast_limits is not None:
        limits = (float(args.contrast_limits[0]), float(args.contrast_limits[1]))
        log.info("Using explicit contrast limits: %s", limits)
    else:
        limits = auto_contrast(
            volume,
            percentiles=tuple(args.percentiles),
            skip_background=args.skip_background,
        )

    log.info(
        "Volume %s, dtype %s, spacing (mm) %s",
        volume.shape, volume.dtype, spacing if spacing is not None else "unknown",
    )

    import napari

    open_viewer(
        volume,
        spacing,
        name=sample.stem or sample.name,
        contrast_limits=limits,
        labels=labels,
        render_3d=args.render_3d,
    )
    napari.run()


if __name__ == "__main__":
    main()
