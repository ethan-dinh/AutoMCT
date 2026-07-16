"""
Standalone prototype: enamel-disconnection threshold search.

Molars are hard to isolate directly because dentin (the bulk of a molar) and
bone sit at overlapping intensities, so a single global threshold either
fuses molars to bone or cuts into real tissue. Enamel is different: it is
strictly brighter than both bone and dentin, so it can always be isolated
cleanly. This prototype uses that asymmetry to find a working separation
threshold rather than to build the final molar mask directly:

1. Binarize the whole scan at intensity > 0 -> the full foreground
   (bone + molars, i.e. all mineralized tissue).
2. Binarize just the enamel -> the brightest voxels, thresholded well above
   the bone/dentin range.
3. Label the foreground at a candidate threshold and check whether any
   enamel voxels fall inside the same connected component as bone (the
   largest component by voxel count -- bone dominates by volume, so this
   is the simplest reliable way to identify it).
4. If enamel is still connected to bone, the threshold is too low (dentin
   is bridging molar to bone, or bone/dentin haven't split yet) -- raise it
   and try again.
5. Repeat until every enamel-containing component is disconnected from the
   bone component. At that point each enamel-containing component is a
   molar candidate.

This does NOT touch molar_bone.py -- it's purely for tuning the threshold
schedule and inspecting whether the resulting separation is usable before
wiring anything into the real pipeline.

Usage:
    python tests/prototype_molar_extraction.py --cache-dir /path/to/cache --data /path/to/sample
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pathlib
import sys
from typing import Optional

_PKG_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np

from segmentation.molar_bone import MIN_COMPONENT_SIZE
from segmentation.utils import label_3d_volume
from visualization import create_3d_visualization

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _binarize_foreground(volume: np.ndarray) -> np.ndarray:
    """Binary mask of the entire mineralized region (bone + molars)."""
    return volume > 0


def _binarize_enamel(volume: np.ndarray, enamel_threshold: float) -> np.ndarray:
    """Binary mask of enamel: the brightest voxels in the scan."""
    return volume > enamel_threshold


def _bone_label(counts: np.ndarray) -> int:
    """
    Identify the bone component: the largest connected component by voxel
    count. Bone is the easiest region to detect because it contains the
    most voxels; molars are smaller connected components.
    """
    return int(np.argmax(counts))


def _enamel_connected_to_bone(
    enamel_mask: np.ndarray,
    labeled_foreground: np.ndarray,
    bone_label: int,
) -> bool:
    """True if any enamel voxel lies inside the bone component."""
    return bool(np.any(labeled_foreground[enamel_mask] == bone_label))


def find_molar_separation_threshold(
    volume: np.ndarray,
    enamel_threshold: float,
    *,
    min_component_size: int = MIN_COMPONENT_SIZE,
    threshold_start: float | None = None,
    threshold_step: float = 0.01,
    max_iters: int = 200,
    visualize_every: Optional[int] = None,
) -> tuple[float, np.ndarray, int]:
    """
    Escalate the foreground intensity threshold until every enamel-bearing
    component is disconnected from the bone component.

    Parameters:
        volume: Intensity volume (incisor already removed upstream).
        enamel_threshold: Fixed intensity cutoff defining enamel -- enamel
            is always brighter than bone/dentin, so this does not need to
            escalate with the foreground threshold.
        min_component_size: Minimum voxels for a component to be considered
            real tissue (filters out small noise fragments).
        threshold_start: Starting foreground intensity threshold. Defaults
            to 0 -- ``volume`` (bone_molar_volume) has already had
            background stripped upstream by the pipeline's own valley
            threshold, so voxel value 0 already means "not tissue" here;
            recomputing a valley on this already-masked volume would just
            rediscover an inflated threshold from the upstream masking, not
            a real background/bone split.
        threshold_step: Amount to raise the threshold each iteration.
        max_iters: Safety cap on escalation steps.
        visualize_every: If set, opens a napari viewer every N iterations
            showing the original volume, the current thresholded candidate
            mask, and the enamel mask. Each viewer blocks until closed, so
            escalation resumes only after you close the window.

    Returns:
        (threshold, labeled_foreground, n_iters) where labeled_foreground is
        the connected-component labeling of the foreground at the returned
        threshold -- bone is the largest component; every other large
        component that contains enamel is a molar candidate.
    """
    enamel_mask = volume > enamel_threshold
    if not enamel_mask.any():
        logger.error("No voxels above enamel_threshold=%.4f -- cannot anchor molars", enamel_threshold)
        empty = np.zeros_like(volume, dtype=np.int32)
        return threshold_start or 0.0, empty, 0

    if threshold_start is None:
        threshold_start = 0.0
        logger.info("Using full foreground as starting point (threshold=0.0)")

    threshold = threshold_start

    for it in range(max_iters):
        candidate = (volume > threshold) & (volume > 0)
        labeled = label_3d_volume(candidate, connectivity=1)
        counts = np.bincount(labeled.ravel())
        counts[0] = 0

        if not counts.any():
            logger.warning("iter %3d: threshold=%.4f -> empty foreground; stopping", it, threshold)
            break

        bone_label = _bone_label(counts)
        still_connected = _enamel_connected_to_bone(enamel_mask, labeled, bone_label)

        n_large = int(np.sum(counts >= min_component_size))
        logger.info(
            "iter %3d: threshold=%.4f -> %d large component(s), bone=%d voxels, "
            "enamel connected to bone=%s",
            it, threshold, n_large, int(counts[bone_label]), still_connected,
        )

        if visualize_every and it % visualize_every == 0:
            create_3d_visualization(
                volume,
                additional_volumes={
                    "Thresholded candidate": (candidate, "red"),
                    "Enamel": (enamel_mask, "yellow"),
                },
                title=f"[prototype] iter {it}: threshold={threshold:.4f}, enamel connected to bone={still_connected}",
            )

        if not still_connected:
            logger.info(
                "Enamel disconnected from bone after %d iteration(s) at threshold=%.4f",
                it, threshold,
            )
            return threshold, labeled, it

        threshold += threshold_step

    logger.warning(
        "Enamel never disconnected from bone after %d iterations; returning last threshold %.4f",
        max_iters, threshold,
    )
    candidate = (volume > threshold) & (volume > 0)
    labeled = label_3d_volume(candidate, connectivity=1)
    return threshold, labeled, max_iters


def extract_molar_mask(
    labeled_foreground: np.ndarray,
    enamel_mask: np.ndarray,
    *,
    min_component_size: int = MIN_COMPONENT_SIZE,
) -> np.ndarray:
    """
    Given the disconnected labeling from find_molar_separation_threshold,
    return a boolean mask of every component that contains enamel and isn't
    the bone component -- i.e. all molar candidates.
    """
    counts = np.bincount(labeled_foreground.ravel())
    counts[0] = 0
    if not counts.any():
        return np.zeros_like(labeled_foreground, dtype=bool)

    bone_label = _bone_label(counts)
    large_labels = np.flatnonzero(counts >= min_component_size)

    molar_mask = np.zeros_like(labeled_foreground, dtype=bool)
    for lbl in large_labels:
        if lbl == bone_label:
            continue
        component = labeled_foreground == lbl
        if np.any(enamel_mask & component):
            molar_mask |= component

    return molar_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="cache_dir used with segment_mandible(cache_dir=...)")
    ap.add_argument("--data", required=True, help="original input_path used to build the cache (for the hash key)")
    ap.add_argument("--enamel-threshold", type=float, default=None,
                    help="Fixed intensity cutoff for enamel. Defaults to the 99.5th percentile of foreground voxels.")
    ap.add_argument("--threshold-start", type=float, default=None,
                    help="Starting foreground intensity threshold. Defaults to 0 (the full "
                         "bone_molar_volume foreground, since background is already stripped upstream).")
    ap.add_argument("--threshold-step", type=float, default=0.01)
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--visualize-every", type=int, default=10,
                    help="Open a napari viewer every N iterations showing the original volume, "
                         "the current threshold candidate mask, and the enamel mask. "
                         "Set to 0 to disable.")
    args = ap.parse_args()

    cache_key = hashlib.sha256(os.path.abspath(args.data).encode()).hexdigest()[:16]
    cache_path = os.path.join(args.cache_dir, f"molar_bone_cache_{cache_key}.npz")

    if not os.path.exists(cache_path):
        logger.error(
            "No cache found at %s -- run the pipeline once with cache_dir=%r first.",
            cache_path, args.cache_dir,
        )
        sys.exit(1)

    logger.info("Loading cached bone_molar_volume from %s", cache_path)
    cached = np.load(cache_path)
    bone_molar_volume = cached["bone_molar_volume"]

    foreground = _binarize_foreground(bone_molar_volume)
    if not foreground.any():
        logger.error("Empty foreground in cached bone_molar_volume -- nothing to do.")
        sys.exit(1)

    enamel_threshold = args.enamel_threshold
    if enamel_threshold is None:
        enamel_threshold = float(np.percentile(bone_molar_volume[foreground], 99.5))
        logger.info("Auto-selected enamel_threshold=%.4f (99.5th percentile of foreground)", enamel_threshold)

    enamel_mask = _binarize_enamel(bone_molar_volume, enamel_threshold)

    threshold, labeled_foreground, n_iters = find_molar_separation_threshold(
        bone_molar_volume, enamel_threshold,
        threshold_start=args.threshold_start,
        threshold_step=args.threshold_step,
        max_iters=args.max_iters,
        visualize_every=args.visualize_every,
    )

    molar_mask = extract_molar_mask(labeled_foreground, enamel_mask)
    bone_mask = (labeled_foreground > 0) & ~molar_mask

    logger.info(
        "Final separation at threshold=%.4f (%d iterations): molar voxels=%d, bone voxels=%d",
        threshold, n_iters, int(molar_mask.sum()), int(bone_mask.sum()),
    )

    create_3d_visualization(
        bone_molar_volume,
        additional_volumes={
            "Original foreground": (foreground, "gray"),
            "Enamel": (enamel_mask, "yellow"),
            "Molar candidates": (molar_mask, "red"),
            "Bone": (bone_mask, "blue"),
        },
        title=f"[prototype] enamel-disconnection molar extraction (threshold={threshold:.4f}, iters={n_iters})",
    )


if __name__ == "__main__":
    main()
