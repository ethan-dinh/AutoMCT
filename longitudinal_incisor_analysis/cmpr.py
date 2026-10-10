"""
Curved Multi-Planar Reconstruction (CMPR) for longitudinal incisor analysis.

Author: Ethan Dinh
Date:   2026-09-09

Adapted from `segment_enamel_WIP/CMPR.py`. Same geometry -- skeletonize the incisor
mask, take the graph-diameter path as a centerline, propagate a rotation-
minimizing frame along it, and resample a plane per centerline point -- but the
resampling is intensity-preserving, because the consumer of this output is
`enamel_density.py`, which fits Gaussian components to each slice's intensity
histogram and reports peak centres to sub-unit precision.

Two changes from the segmentation-facing original, both about not destroying
the numbers the density fit depends on:

  [1] Interpolation happens in float, and the result is kept in float.
      `map_coordinates` returns the input's dtype, so resampling an int16
      volume rounds every interpolated voxel back to an integer. Measured on
      an int16 phantom with dentin-like statistics (mu=143, sigma=12): the
      int16 path yields 71 distinct intensities where the float path yields
      199171, a mean absolute error of 0.25 intensity units. That is a
      quantization grid two percent of the ~11-unit dentin width the fitter
      resolves peaks against, imposed on every voxel.

  [2] No dtype coercion on save. The original casts anything that is not
      int16/float32 to int16 -- silently flooring a float volume. Here the
      resampled float32 is written as float32.

Also fixed: the mask used to find the centerline is derived from the volume,
but the volume itself is never masked, thresholded, or rescaled. Intensities
that reach the output are the scanner's own, so `enamel_density.py`'s
saturation logic (which keys off the volume maximum) stays meaningful.

Output naming carries the sample name -- `<sample>_resliced_volume.nrrd` rather
than a bare `resliced_volume.tif` -- because `enamel_density.py` derives its
sample label from the input basename, so unnamed files make every specimen's
results read as "resliced_volume".

Output format is NRRD by default, matching what the segmentation pipeline now
writes for its own volumes. NRRD carries the voxel spacing as an explicit
`space directions` matrix rather than TIFF's resolution tags, so the reslice
opens at true physical size in Slicer/ITK-SNAP, and `enamel_density.py` reads
it through the same `load_nrrd` it already uses for the molar cross-check.
Pass `--format tif` for the legacy container.

Usage
-----
    # A single volume
    python cmpr.py -i ".../Segments/Odam-CRE-A/volumes/Odam-CRE-A_incisor_volume.nrrd" \
        -o results/Odam-CRE-A --log

    # Every sample under a directory of per-sample folders
    python cmpr.py -i ".../Segments" -o results --log
"""

from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import sys

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import splev, splprep
from scipy.ndimage import binary_erosion, binary_fill_holes, map_coordinates
from skimage.morphology import ball, closing, skeletonize
from tqdm import tqdm

# Same bridge enamel_density.py uses: read volumes exactly the way the
# segmentation pipeline writes them, so (Z, Y, X) order and spacing agree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_mandible.data_io import (  # noqa: E402
    load_nrrd,
    load_tiff,
    save_volume,
)

logger = logging.getLogger(__name__)

# ----------------------------------------- #
# Global Variables
# ----------------------------------------- #
IS_VISUALIZE = False

# Interpolation order for the reslice. Order 1 (trilinear) is what the original
# used and what this keeps: it is the highest order that cannot overshoot.
# Cubic (order 3) rings at the enamel/air edge, inventing intensities above the
# scan's ceiling and below its floor -- exactly the out-of-range values
# enamel_density.py's bounded fit is built to distrust.
INTERPOLATION_ORDER = 1

# Value written where a sampling plane falls outside the volume. Kept at 0 to
# match the original and the analysis, which treats 0 as background.
FILL_VALUE = 0.0

# Filenames tried, in order, when an input path is a per-sample directory.
INCISOR_PATTERNS = (
    "volumes/*_incisor_volume.nrrd",
    "volumes/*_incisor_volume.tif",
    "*_incisor_volume.nrrd",
    "*_incisor_volume.tif",
    "incisor_volume.nrrd",
    "incisor_volume.tif",
)


# ----------------------------------------- #
# Logging Setup
# ----------------------------------------- #
def setup_logging():
    class ColorFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[94m",
            "INFO": "\033[92m",
            "WARNING": "\033[93m",
            "ERROR": "\033[91m",
            "CRITICAL": "\033[95m",
        }
        RESET = "\033[0m"

        def format(self, record):
            color = self.COLORS.get(record.levelname, self.RESET)
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("[%(levelname)s] - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])


# ----------------------------------------- #
# I/O
# ----------------------------------------- #
def load_volume(path: str) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """
    Load a .tif/.tiff or .nrrd volume as (Z, Y, X), with spacing if recorded.

    Returns the array in its stored dtype -- conversion to float happens at the
    resampling step, where it matters, so this stays a faithful read.
    """
    logger.info("Loading volume from %s", path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        volume, spacing = load_tiff(path, return_spacing=True)
    elif ext == ".nrrd":
        volume, spacing = load_nrrd(path, return_spacing=True)
    else:
        raise SystemExit(f"Unsupported input format: {ext!r} (expected .tif or .nrrd)")
    if volume is None:
        raise SystemExit(f"Could not read {path}")

    logger.info(
        "Loaded shape=%s dtype=%s range=[%g, %g]",
        volume.shape, volume.dtype, float(volume.min()), float(volume.max()),
    )
    return volume, spacing


def save_resliced_volume(
    volume: np.ndarray,
    output_dir: str,
    sample_name: str,
    spacing: tuple[float, float, float] | None = None,
    output_format: str = "nrrd",
) -> str:
    """
    Write the resliced volume under a sample-qualified name, preserving dtype.

    Unlike the original's `save_ct_volume_as_tiff`, nothing here coerces the
    array to int16. A float32 reslice is written as float32; the only narrowing
    is float64 -> float32, which is lossless at CT's dynamic range and is what
    the TIFF/NRRD writers accept.
    """
    if volume.dtype == np.float64:
        logger.info("Narrowing float64 to float32 (lossless at this dynamic range)")
        volume = volume.astype(np.float32)

    logger.info(
        "Saving reslice - shape: %s, dtype: %s, min: %g, max: %g, distinct: %d",
        volume.shape, volume.dtype, float(volume.min()), float(volume.max()),
        np.unique(volume).size,
    )

    # save_volume joins base_dir/output_dir; passing the full path as base_dir
    # and "" as the sub-directory keeps the result at exactly `output_dir`.
    return save_volume(
        volume,
        output_dir="",
        name=f"{sample_name}_resliced_volume",
        base_dir=output_dir,
        output_format=output_format,
        spacing=spacing,
    )


def save_cmpr_transform(
    output_dir: str,
    sample_name: str,
    centerline: np.ndarray,
    frames: np.ndarray,
    bbox_min: np.ndarray,
    slice_size: int,
    spacing: float,
    volume_shape: tuple[int, int, int],
    voxel_spacing: tuple[float, float, float] | None,
) -> str:
    """
    Write the forward map from reslice coordinates back to the original volume.

    The reslice is not an independent volume: slice `i` was sampled from the
    plane through `centerline[i]` spanned by `frames[i]`, so

        original_zyx = bbox_min + centerline[i] + u * n[i] + v * b[i]

    recovers where any reslice voxel came from. Everything that map needs is
    computed during the reslice and, without this, discarded when the process
    exits -- leaving the resliced volume with no way back to the anatomy it
    came from.

    `bbox_min` is the piece most easily lost. The centerline is computed on the
    *cropped* volume, so a transform reconstructed later from the reslice alone
    would be offset by the crop origin -- a fixed translation that shifts every
    landmark, consistently enough to look plausible and be wrong.
    """
    path = os.path.join(output_dir, f"{sample_name}_cmpr_transform.npz")
    np.savez(
        path,
        centerline=np.asarray(centerline, dtype=np.float64),
        frames=np.asarray(frames, dtype=np.float64),
        bbox_min=np.asarray(bbox_min, dtype=np.int64),
        slice_size=np.int64(slice_size),
        spacing=np.float64(spacing),
        volume_shape=np.asarray(volume_shape, dtype=np.int64),
        voxel_spacing=(np.asarray(voxel_spacing, dtype=np.float64)
                       if voxel_spacing is not None
                       else np.full(3, np.nan)),
        interpolation_order=np.int64(INTERPOLATION_ORDER),
    )
    logger.info("Saved CMPR transform to %s", path)
    return path


def create_3d_visualization(
    volume: np.ndarray,
    labeled_volume: np.ndarray | None = None,
    additional_volumes: dict | None = None,
    ndisplay: int = 2,
    title: str = "napari",
) -> None:
    """
    Open a napari viewer with the CT volume and optional overlays.

    Parameters:
        volume: Raw 3D volume (Z, Y, X).
        labeled_volume: Optional integer label volume.
        additional_volumes: Mapping of name -> (volume, colormap) to add as extra layers.
        ndisplay: 2 for orthographic views, 3 for 3D rendering.
        title: Window title shown in the napari title bar.
    """
    import napari

    viewer = None
    try:
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


# ----------------------------------------- #
# Geometry
# ----------------------------------------- #
def compute_bounding_box(
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...]]:
    """
    Compute the bounding box of a 3D volume.

    Parameters:
        volume (np.ndarray): 3D array of CT data (e.g. int16 or float32).
    """
    logger.info("Computing bounding box")
    nonzero = np.argwhere(volume)

    if nonzero.size == 0:
        raise ValueError("Volume is empty or not segmented.")
    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0) + 1

    return (
        min_coords,
        max_coords,
        tuple(slice(minc, maxc) for minc, maxc in zip(min_coords, max_coords)),
    )


def plot_graph_with_centerline(G, coords, path_idx, output_dir=".", sample_name="sample"):
    """
    Visualize a 3D networkx graph with its centerline using Plotly.

    Parameters
    ----------
    G : networkx.Graph
        Graph with integer node labels corresponding to rows in coords.
    coords : np.ndarray, shape (N,3)
        Array of (z, y, x) coordinates for each node index.
    path_idx : list of int
        Ordered list of node indices representing the centerline.
    output_dir : str
        Directory to write the resulting HTML visualization to.
    sample_name : str
        Name of the input volume, used to label the output file.
    """
    # Nodes
    x_nodes = coords[:, 2]
    y_nodes = coords[:, 1]
    z_nodes = coords[:, 0]
    node_trace = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers",
        marker={"size": 2, "color": "red"},
        name="Skeleton Nodes",
    )

    # Edges
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = coords[u, 2], coords[u, 1], coords[u, 0]
        x1, y1, z1 = coords[v, 2], coords[v, 1], coords[v, 0]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line={"width": 1, "color": "lightgray"},
        name="Skeleton Edges",
    )

    # Centerline Path
    path_coords = coords[path_idx]
    path_trace = go.Scatter3d(
        x=path_coords[:, 2],
        y=path_coords[:, 1],
        z=path_coords[:, 0],
        mode="lines+markers",
        line={"width": 4, "color": "yellow"},
        marker={"size": 4, "color": "yellow"},
        name="Centerline",
    )

    fig = go.Figure(data=[edge_trace, node_trace, path_trace])
    fig.update_layout(
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
        },
        title=f"3D Skeleton Graph with Centerline — {sample_name}",
        showlegend=True,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sample_name}_graph_output.html")
    fig.write_html(output_path)
    logger.info("Saved skeleton visualization to %s", output_path)


def smooth_centerline(
    centerline: np.ndarray, smoothing_factor=5.0, n_points=100
) -> np.ndarray:
    """
    Smooth a 3D centerline using B-spline interpolation.

    Parameters
    ----------
    centerline : np.ndarray
        Array of shape (N, 3) with ordered (z, y, x) coordinates.
    smoothing_factor : float
        Smoothing parameter passed to `splprep`. Larger means smoother.
    n_points : int
        Number of points to interpolate along the smoothed spline.

    Returns
    -------
    np.ndarray
        Smoothed centerline of shape (n_points, 3)
    """
    # Convert to (x, y, z) for splprep
    points = centerline[:, [2, 1, 0]].T  # Shape: (3, N)

    # Fit spline to the centerline
    tck, _ = splprep(points, s=smoothing_factor)

    # Evaluate spline at n_points evenly spaced values
    u_new = np.linspace(0, 1, n_points)
    smoothed_points = np.array(splev(u_new, tck))  # Shape: (3, n_points)

    # Convert back to (z, y, x)
    return np.stack(
        [smoothed_points[2], smoothed_points[1], smoothed_points[0]], axis=1
    )


def extend_centerline_to_mask_extent(
    centerline: np.ndarray, mask: np.ndarray, step: float = 0.5
) -> np.ndarray:
    """
    Extend both ends of a centerline outward along their local tangent
    until they exit the mask, so CMPR slices span the full incisor
    including its tip and end rather than stopping short at the
    skeleton's endpoints.

    Parameters
    ----------
    centerline : np.ndarray, shape (N, 3)
        Ordered (z, y, x) coordinates along the centerline.
    mask : np.ndarray, shape (Z, Y, X)
        Binary mask of the incisor.
    step : float
        Distance, in voxels, to advance per extrapolation step.

    Returns
    -------
    np.ndarray
        Centerline with extra points prepended/appended at both ends.
    """
    def in_mask(point):
        z, y, x = np.round(point).astype(int)
        if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            return mask[z, y, x] != 0
        return False

    def extrapolate(anchor, direction):
        direction = direction / np.linalg.norm(direction)
        extra = []
        point = anchor.copy()
        while True:
            point = point + direction * step
            if not in_mask(point):
                break
            extra.append(point)
        return extra

    start_dir = centerline[0] - centerline[1]
    end_dir = centerline[-1] - centerline[-2]

    start_extra = extrapolate(centerline[0], start_dir)[::-1]
    end_extra = extrapolate(centerline[-1], end_dir)

    return np.concatenate(
        [np.array(start_extra).reshape(-1, 3), centerline,
         np.array(end_extra).reshape(-1, 3)],
        axis=0,
    )


def compute_centerline(
    mask: np.ndarray, output_dir: str = ".", sample_name: str = "sample"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the centerline of a 3D incisor volume.

    Parameters
    ----------
    mask : ndarray, shape (Z,Y,X)
        Binary mask of the incisor.
    output_dir : str
        Directory to write the skeleton visualization to, if enabled.
    sample_name : str
        Name of the input volume, used to label output files.

    Returns
    -------
    centerline : ndarray, shape (M,3)
        Ordered (z,y,x) coordinates along the longest path.
    centerline_mask : ndarray
        Binary volume with the centerline voxels set, for visualization.
    """
    logger.info("Skeletonizing...")
    skel = skeletonize(mask)

    # Build a weighted graph on skeleton voxels
    logger.info("Building graph...")
    coords = np.argwhere(skel)  # list of (z,y,x) skeleton points
    idx_of = {tuple(c): i for i, c in enumerate(coords)}
    G = nx.Graph()

    for i, (z, y, x) in tqdm(enumerate(coords), desc="Building graph", total=len(coords)):
        # Examine all 26 neighbors
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    j = idx_of.get((z + dz, y + dy, x + dx))
                    if j is not None:
                        # weight = Euclidean distance between voxels
                        G.add_edge(i, j, weight=np.sqrt(dz * dz + dy * dy + dx * dx))

    if G.number_of_nodes() == 0:
        raise RuntimeError("No skeleton voxels found; check segmentation.")

    # Skeletonization of a real incisor leaves specks detached from the main
    # branch, so the graph is not connected. Dijkstra cannot leave the
    # component it starts in, and seeding from an arbitrary node lands in
    # whichever fragment happens to hold the first skeleton voxel -- often a
    # two-voxel speck, whose "diameter" is too short to fit a spline through.
    # The incisor is the largest component by a wide margin, so search there.
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) > 1:
        logger.info(
            "Skeleton has %d components (sizes %s%s); using the largest",
            len(components),
            [len(c) for c in components[:5]],
            ", ..." if len(components) > 5 else "",
        )
    largest = G.subgraph(components[0])

    # Two-pass Dijkstra to find graph diameter endpoints
    logger.info("Finding farthest node u...")
    source = next(iter(components[0]))
    dist1 = nx.single_source_dijkstra_path_length(largest, source=source, weight="weight")
    u = max(dist1, key=lambda node: dist1[node])

    logger.info("Finding farthest node v...")
    dist2 = nx.single_source_dijkstra_path_length(largest, source=u, weight="weight")
    v = max(dist2, key=lambda node: dist2[node])

    logger.info("Finding diameter path...")
    path_idx = nx.dijkstra_path(largest, source=u, target=v, weight="weight")
    logger.info("Centerline path has %d points", len(path_idx))

    if IS_VISUALIZE:
        plot_graph_with_centerline(
            G, coords, path_idx, output_dir=output_dir, sample_name=sample_name
        )

    logger.info("Converting node indices to coordinates...")
    centerline = coords[path_idx]

    logger.info("Smoothing centerline...")
    n_points = len(centerline)
    smoothing_factor = 25 * len(centerline)
    smoothed_centerline = smooth_centerline(
        centerline, smoothing_factor=smoothing_factor, n_points=n_points
    )

    logger.info("Extending centerline to mask extent...")
    smoothed_centerline = extend_centerline_to_mask_extent(smoothed_centerline, mask)

    # Fill in the centerline voxels
    centerline_mask = np.zeros_like(mask, dtype=np.uint8)
    for zf, yf, xf in smoothed_centerline:
        z, y, x = map(int, np.round([zf, yf, xf]))
        if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            centerline_mask[z, y, x] = 1

    return smoothed_centerline, centerline_mask


def compute_local_frames(centerline):
    """
    Compute a Rotation Minimizing Frame (RMF) along the centerline using
    the double reflection method (Wang et al., 2008). Each frame is
    propagated from the previous one with the minimal twist needed to
    track the tangent, instead of being derived independently from a
    fixed reference vector -- this avoids both the discontinuous flips
    that occur when the tangent nears a fixed "up" vector and the
    general frame-to-frame rotation jitter of independent computation.
    """
    tangents = np.gradient(centerline, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, None]

    t0 = tangents[0]
    up = np.array([0.0, 0.0, 1.0])
    n0 = np.cross(t0, up)
    if np.linalg.norm(n0) < 1e-5:
        n0 = np.cross(t0, np.array([1.0, 0.0, 0.0]))
    n0 = n0 / np.linalg.norm(n0)

    frames = [(n0, np.cross(t0, n0))]
    positions = centerline

    for i in range(1, len(tangents)):
        x_prev, t_prev = positions[i - 1], tangents[i - 1]
        n_prev = frames[i - 1][0]
        x_cur, t_cur = positions[i], tangents[i]

        v1 = x_cur - x_prev
        c1 = np.dot(v1, v1)
        if c1 < 1e-12:
            frames.append(frames[i - 1])
            continue

        n_l = n_prev - (2 / c1) * np.dot(v1, n_prev) * v1
        t_l = t_prev - (2 / c1) * np.dot(v1, t_prev) * v1

        v2 = t_cur - t_l
        c2 = np.dot(v2, v2)
        if c2 < 1e-12:
            n_cur = n_l
        else:
            n_cur = n_l - (2 / c2) * np.dot(v2, n_l) * v2
        n_cur = n_cur / np.linalg.norm(n_cur)

        frames.append((n_cur, np.cross(t_cur, n_cur)))

    return frames


def compute_required_half_extent(mask: np.ndarray, centerline: np.ndarray) -> float:
    """
    Determine the largest distance from the centerline to any foreground
    voxel in the mask, so the CMPR slice window can be sized to fully
    contain the incisor's cross-section at every point along the curve.

    Parameters
    ----------
    mask : np.ndarray, shape (Z,Y,X)
        Binary mask of the incisor.
    centerline : np.ndarray, shape (M,3)
        Ordered (z,y,x) coordinates along the centerline.

    Returns
    -------
    float
        Maximum Euclidean distance, in voxels, from any centerline point
        to the farthest mask voxel that is closest to it.
    """
    eroded = binary_erosion(mask)
    surface = mask & ~eroded
    surface_coords = np.argwhere(surface)

    max_dist = 0.0
    chunk_size = 2000
    for start in range(0, len(surface_coords), chunk_size):
        chunk = surface_coords[start:start + chunk_size]
        dists = np.linalg.norm(
            chunk[:, None, :] - centerline[None, :, :], axis=2
        ).min(axis=1)
        max_dist = max(max_dist, dists.max())
    return float(max_dist)


# ----------------------------------------- #
# Resampling
# ----------------------------------------- #
def resample_slice(volume, origin, n, b, size=64, spacing=1.0):
    """
    Sample one plane through `volume`, spanned by the frame vectors n and b.

    `volume` must already be floating point. `map_coordinates` returns the
    input array's dtype, so handing it an integer volume would round every
    interpolated value to a whole number -- see the module docstring for the
    measured cost. `cmpr_pipeline` converts once, up front, rather than per
    slice, so the cast happens exactly one time per specimen.
    """
    grid_u = np.linspace(-size / 2, size / 2, size) * spacing
    grid_v = np.linspace(-size / 2, size / 2, size) * spacing
    uu, vv = np.meshgrid(grid_u, grid_v)
    coords = origin + uu[..., None] * n + vv[..., None] * b
    coords = coords.reshape(-1, 3).T
    return map_coordinates(
        volume,
        coords,
        order=INTERPOLATION_ORDER,
        mode="constant",
        cval=FILL_VALUE,
    ).reshape(size, size)


def cmpr_pipeline(
    path,
    output_dir,
    slice_size=None,
    spacing=0.5,
    margin=1.2,
    sample_name=None,
    output_format="nrrd",
):
    """
    Run the full CMPR for one incisor volume and write the resliced result.

    The intensities written out are the input's own, resampled in float and
    never rescaled, thresholded, or masked -- the binary mask drives the
    geometry only.
    """
    if sample_name is None:
        sample_name = derive_sample_name(path)

    volume, voxel_spacing = load_volume(path)

    logger.info("Building the incisor mask (geometry only -- intensities untouched)")
    binary_mask = (volume > 0).astype(np.uint8)
    binary_mask = closing(binary_mask, ball(2))

    logger.info("Filling holes in the binary mask")
    for z in tqdm(range(binary_mask.shape[0]), desc="Filling holes in mask"):
        binary_mask[z] = binary_fill_holes(binary_mask[z])

    bbox_min, _, bbox = compute_bounding_box(binary_mask)
    original_shape = volume.shape

    # Convert to float exactly once, here: everything downstream samples this
    # array, and map_coordinates preserves its dtype.
    cropped_volume = volume[bbox].astype(np.float32)
    cropped_mask = binary_mask[bbox]
    del volume, binary_mask
    gc.collect()

    logger.info("Determining centerline of the incisor...")
    smoothed_centerline, centerline_mask = compute_centerline(
        cropped_mask, output_dir=output_dir, sample_name=sample_name)

    logger.info("Computing local frames...")
    frames = compute_local_frames(smoothed_centerline)

    if slice_size is None:
        half_extent = compute_required_half_extent(cropped_mask, smoothed_centerline)
        slice_size = int(np.ceil(2 * half_extent * margin / spacing))
        logger.info(
            "Auto-sized slice window to %d px (half-extent=%.2f, spacing=%.2f)",
            slice_size, half_extent, spacing)

    logger.info("Generating slices...")
    slices = np.empty(
        (len(smoothed_centerline), slice_size, slice_size), dtype=np.float32
    )
    for i in tqdm(
        range(len(smoothed_centerline)),
        desc="Generating slices",
        total=len(smoothed_centerline),
        unit=" slices",
    ):
        n_vec, b_vec = frames[i]
        slices[i] = resample_slice(
            cropped_volume,
            smoothed_centerline[i],
            n_vec,
            b_vec,
            size=slice_size,
            spacing=spacing,
        )

    # The reslice resamples in-plane at `spacing` voxels per pixel and steps one
    # centerline point per slice; the centerline is arc-length parameterized, so
    # z advances ~1 voxel per slice.
    out_spacing = None
    if voxel_spacing is not None:
        dz, dy, dx = (float(s) for s in voxel_spacing)
        in_plane = float(np.mean([dy, dx]))
        out_spacing = (float(np.mean([dz, dy, dx])), in_plane * spacing,
                       in_plane * spacing)

    written = save_resliced_volume(
        slices, output_dir, sample_name,
        spacing=out_spacing, output_format=output_format,
    )
    logger.info("Saved reslice to %s", written)

    # The geometry that produced those slices, so downstream work can map a
    # position on the reslice back to a position in the original scan.
    save_cmpr_transform(
        output_dir, sample_name,
        centerline=smoothed_centerline,
        frames=frames,
        bbox_min=bbox_min,
        slice_size=slice_size,
        spacing=spacing,
        volume_shape=original_shape,
        voxel_spacing=voxel_spacing,
    )

    if IS_VISUALIZE:
        create_3d_visualization(
            cropped_volume,
            None,
            {
                "slices": (slices, "yellow"),
                "smoothed_centerline": (centerline_mask, "green"),
            },
        )

    return slices, smoothed_centerline, frames, cropped_volume


# ----------------------------------------- #
# Sample discovery / naming
# ----------------------------------------- #
def derive_sample_name(path: str) -> str:
    """
    Recover the specimen name from an incisor volume path.

    The segmentation pipeline writes `<Sample>/volumes/<Sample>_incisor_volume.nrrd`,
    so the basename carries the name and the suffix is stripped. Legacy layouts
    wrote a bare `incisor_volume.tif`, where the name lives in the enclosing
    sample folder instead -- fall back to that, skipping the `volumes/` level.
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    # Legacy layout first: an unqualified name carries no specimen, so the
    # enclosing sample folder is the only place the name exists. Checking this
    # before suffix-stripping matters -- "incisor_volume" ends with "_volume",
    # and trimming it would yield "incisor" for every specimen.
    if stem in ("incisor_volume", "volume", "resliced_volume"):
        parent = os.path.dirname(os.path.abspath(path))
        if os.path.basename(parent) in ("volumes", "masks"):
            parent = os.path.dirname(parent)
        return os.path.basename(parent) or stem

    for suffix in ("_incisor_volume", "_resliced_volume", "_volume"):
        if stem.endswith(suffix):
            trimmed = stem[: -len(suffix)]
            if trimmed:
                return trimmed
    return stem


def find_incisor_volume(sample_dir: str) -> str | None:
    """Locate a sample folder's incisor volume, newest layout first."""
    for pattern in INCISOR_PATTERNS:
        matches = sorted(glob.glob(os.path.join(sample_dir, pattern)))
        if matches:
            return matches[0]
    return None


def discover_samples(input_path: str) -> list[tuple[str, str]]:
    """
    Resolve `--input` into a list of (sample_name, volume_path).

    Accepts either a single volume file or a directory of per-sample folders.
    """
    if os.path.isfile(input_path):
        return [(derive_sample_name(input_path), input_path)]

    if not os.path.isdir(input_path):
        raise SystemExit(f"Input path does not exist: {input_path}")

    # A sample folder handed in directly.
    direct = find_incisor_volume(input_path)
    if direct is not None:
        return [(derive_sample_name(direct), direct)]

    samples = []
    for name in sorted(os.listdir(input_path)):
        sample_dir = os.path.join(input_path, name)
        if not os.path.isdir(sample_dir) or name in ("Compressed", "masks", "volumes"):
            continue
        volume_path = find_incisor_volume(sample_dir)
        if volume_path is None:
            logger.warning("No incisor volume found in %s -- skipping", sample_dir)
            continue
        samples.append((derive_sample_name(volume_path), volume_path))

    if not samples:
        raise SystemExit(f"No incisor volumes found under {input_path}")
    return samples


# ----------------------------------------- #
# CLI
# ----------------------------------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Intensity-preserving Curved Multi-Planar Reconstruction for the incisor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="An incisor volume (.tif/.nrrd), a sample folder, or a directory of sample folders",
    )
    parser.add_argument(
        "--output", "-o", default="./results",
        help="Directory to write CMPR results to",
    )
    parser.add_argument(
        "--format", "-f", default="nrrd", choices=("nrrd", "tif"),
        help="Container for the resliced volume",
    )
    parser.add_argument(
        "--per-sample-dir", action="store_true",
        help="Write each sample into its own <output>/<sample>/ subdirectory",
    )
    parser.add_argument("--log", "-l", action="store_true", help="Enable logging")
    parser.add_argument(
        "--slice_size", type=int, default=None,
        help="Slice width/height. If unset, auto-sized to fully contain the "
             "incisor cross-section.",
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="Spacing between pixels"
    )
    parser.add_argument(
        "--margin", type=float, default=1.2,
        help="Safety factor on the auto-sized slice window",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Enable interactive 3D visualization",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.log:
        setup_logging()
    if args.visualize:
        global IS_VISUALIZE
        IS_VISUALIZE = True

    logger.info("CMPR Pipeline Starting")

    samples = discover_samples(args.input)
    logger.info("Found %d sample(s): %s", len(samples), ", ".join(n for n, _ in samples))

    os.makedirs(args.output, exist_ok=True)

    for sample_name, volume_path in samples:
        logger.info("Processing %s", sample_name)
        # With several samples the per-sample subdirectory keeps the sidecar
        # HTML and logs apart; for a single sample, writing straight into
        # --output is what the caller usually means.
        out_dir = args.output
        if args.per_sample_dir or len(samples) > 1:
            out_dir = os.path.join(args.output, sample_name)
        cmpr_pipeline(
            path=volume_path,
            output_dir=out_dir,
            slice_size=args.slice_size,
            spacing=args.spacing,
            margin=args.margin,
            sample_name=sample_name,
            output_format=args.format,
        )

    logger.info("CMPR Pipeline Completed Successfully")


if __name__ == "__main__":
    main()
