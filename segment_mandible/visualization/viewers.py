"""
3D visualization using napari.
"""

import gc
import logging
from typing import Mapping, Optional, Tuple

import numpy as np
import napari

logger = logging.getLogger(__name__)


def create_3d_visualization(
    volume: np.ndarray,
    *,
    labeled_volume: Optional[np.ndarray] = None,
    additional_volumes: Optional[Mapping[str, Tuple[Optional[np.ndarray], str]]] = None,
    slice_range: Optional[Tuple[int, int]] = None,
    ndisplay: int = 2,
    title: str = "napari",
) -> None:
    """
    Open a napari viewer with the CT volume and optional overlays.

    Parameters:
        volume: Raw 3D volume (Z, Y, X).
        labeled_volume: Optional integer label volume.
        additional_volumes: Mapping of name → (volume, colormap) to add as extra layers.
        slice_range: If given, crop the volume to (start, end) along axis 0.
        ndisplay: 2 for orthographic views, 3 for 3D rendering.
        title: Window title shown in the napari title bar.
    """
    viewer = None
    try:
        if slice_range:
            start, end = slice_range
            volume = volume[start:end]
            if labeled_volume is not None:
                labeled_volume = labeled_volume[start:end]

        viewer = napari.Viewer(ndisplay=ndisplay, title=title)
        viewer.add_image(volume, name="Original Volume", colormap="gray")

        if additional_volumes is not None:
            for name, (vol, color) in additional_volumes.items():
                viewer.add_image(vol, name=name, colormap=color, blending="additive")

        if labeled_volume is not None:
            viewer.add_labels(labeled_volume, name="Segmented Regions", opacity=0.35)

        logger.info("3D visualization opened in napari viewer")
        napari.run()

    except Exception as e:
        logger.error("Error creating 3D visualization: %s", e)
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except RuntimeError:
                # napari.run() already tore down the Qt widget when the user
                # closed the window, so the underlying C++ object is gone --
                # nothing left to close.
                pass
        del viewer
        gc.collect()


# The colours below are the convention across the CLI and the test TUI, so a
# result looks the same wherever it is opened. Defined once here rather than
# spelled out at each call site.
SEGMENTATION_COLORS = {
    "Incisor": "orange",
    "Bone": "grey",
    "Molar": "cyan",
}


def show_segmentation(
    background: np.ndarray,
    incisor: Optional[np.ndarray] = None,
    bone: Optional[np.ndarray] = None,
    molar: Optional[np.ndarray] = None,
    *,
    title: str = "napari",
) -> None:
    """
    Open the standard three-structure overlay on a grayscale background.

    Parameters:
        background: Volume to show underneath, usually the preprocessed or
            reoriented scan.
        incisor, bone, molar: Segmented volumes. Any that are None are skipped,
            so this also serves a partially-loaded result.
    """
    create_3d_visualization(
        background,
        additional_volumes={
            name: (volume, SEGMENTATION_COLORS[name])
            for name, volume in (
                ("Incisor", incisor), ("Bone", bone), ("Molar", molar)
            )
        },
        title=title,
    )
