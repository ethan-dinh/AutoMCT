"""
Curved Multi-Planar Reconstruction (CMPR) Module

This module provides functionality for performing curved multi-planar reconstruction 
on 3D volumes, particularly useful for incisor analysis. It extracts centerlines, 
computes local coordinate frames, and resamples volumes to create curved reformatted slices.

Author: Ethan Dinh
Date: August 4, 2025
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Any
from tqdm import tqdm

from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
import networkx as nx
import plotly.graph_objects as go

from .segmentation import morphological_closing, fill_holes

logger = logging.getLogger(__name__)


def _compute_bounding_box(volume: np.ndarray) -> Tuple[slice, slice, slice]:
    """
    Compute the bounding box of a non-zero volume.
    
    Args:
        volume: 3D volume array
        
    Returns:
        Tuple of (min_coords, max_coords, bbox_slices)
        
    Raises:
        ValueError: If volume is empty or not segmented
    """
    logger.info("Computing bounding box")
    nonzero = np.argwhere(volume)
    if nonzero.size == 0:
        raise ValueError("Volume is empty or not segmented.")
    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0) + 1
    bbox_slices = tuple(slice(minc, maxc) for minc, maxc in zip(min_coords, max_coords))
    return bbox_slices


def _smooth_centerline(centerline: np.ndarray, smoothing_factor: float = 5.0, n_points: int = 100) -> np.ndarray:
    """
    Smooth a 3D centerline using B-spline interpolation.

    Args:
        centerline: Array of shape (N, 3) with ordered (z, y, x) coordinates
        smoothing_factor: Smoothing parameter passed to splprep. Larger means smoother
        n_points: Number of points to interpolate along the smoothed spline

    Returns:
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
    smoothed_centerline = np.stack([smoothed_points[2], smoothed_points[1], smoothed_points[0]], axis=1)
    return smoothed_centerline


def _plot_graph_with_centerline(G: nx.Graph, 
                              coords: np.ndarray, 
                              path_idx: List[int],
                              output_path: str = "graph_output.html") -> None:
    """
    Visualize a 3D networkx graph with its centerline using Plotly.

    Args:
        G: Graph with integer node labels corresponding to rows in coords
        coords: Array of (z, y, x) coordinates for each node index
        path_idx: Ordered list of node indices representing the centerline
        output_path: Path to save the HTML visualization
    """
    # Nodes
    x_nodes = coords[:, 2]
    y_nodes = coords[:, 1]
    z_nodes = coords[:, 0]
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Skeleton Nodes'
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
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=1, color='lightgray'),
        name='Skeleton Edges'
    )

    # Centerline Path
    path_coords = coords[path_idx]
    x_path = path_coords[:, 2]
    y_path = path_coords[:, 1]
    z_path = path_coords[:, 0]
    path_trace = go.Scatter3d(
        x=x_path, y=y_path, z=z_path,
        mode='lines+markers',
        line=dict(width=4, color='yellow'),
        marker=dict(size=4, color='yellow'),
        name='Centerline'
    )

    fig = go.Figure(data=[edge_trace, node_trace, path_trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        title="3D Skeleton Graph with Centerline",
        showlegend=True
    )
    fig.write_html(output_path)


def _compute_centerline(mask: np.ndarray, smoothing_factor: float = 25, view_centerline: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the centerline of a 3D volume using skeletonization and graph analysis.

    Args:
        mask: Binary mask of the volume (Z, Y, X)
        smoothing_factor: Smoothing factor for centerline
        view_centerline: Whether to create visualization plots

    Returns:
        Tuple of (smoothed_centerline, centerline_mask)
        
    Raises:
        RuntimeError: If no skeleton voxels are found
    """
    logger.info("Skeletonizing...")
    skel = skeletonize(mask)

    # Build a weighted graph on skeleton voxels
    logger.info("Building graph...")
    coords = np.argwhere(skel)  # list of (z,y,x) skeleton points
    idx_of = {tuple(c): i for i, c in enumerate(coords)}
    G = nx.Graph()

    for i, (z, y, x) in tqdm(enumerate(coords), desc="Building graph"):
        # Examine all 26 neighbors
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    nb = (z + dz, y + dy, x + dx)
                    j = idx_of.get(nb)
                    if j is not None:
                        # weight = Euclidean distance between voxels
                        weight = np.sqrt(dz*dz + dy*dy + dx*dx)
                        G.add_edge(i, j, weight=weight)

    if G.number_of_nodes() == 0:
        raise RuntimeError("No skeleton voxels found; check segmentation.")

    # Two-pass Dijkstra to find graph diameter endpoints
    # Pass 1: from an arbitrary node (0) find farthest node u
    logger.info("Finding farthest node u...")
    dist1 = nx.single_source_dijkstra_path_length(G, source=0, weight='weight')
    u = max(dist1, key=dist1.get)

    # Pass 2: from u find farthest node v
    logger.info("Finding farthest node v...")
    dist2 = nx.single_source_dijkstra_path_length(G, source=u, weight='weight')
    v = max(dist2, key=dist2.get)

    # Extract the diameter path (list of node indices)
    logger.info("Finding diameter path...")
    path_idx = nx.dijkstra_path(G, source=u, target=v, weight='weight')

    # Plot the path in 3D if requested
    if view_centerline:
        _plot_graph_with_centerline(G, coords, path_idx, "./centerline.html")

    # Convert node indices back to (z,y,x) coordinates
    logger.info("Converting node indices to coordinates...")
    centerline = coords[path_idx]

    logger.info("Smoothing centerline...")

    # Determine the number of points as the number of slices
    n_points = len(centerline)  # or fewer for heavier subsampling
    smooth_factor = smoothing_factor * len(centerline)  # 10× to 50× stronger than default
    smoothed_centerline = _smooth_centerline(centerline, smoothing_factor=smooth_factor, n_points=n_points)

    # Fill in the centerline voxels
    centerline_mask = np.zeros_like(mask, dtype=np.uint8)
    for zf, yf, xf in smoothed_centerline:
        z, y, x = map(int, np.round([zf, yf, xf]))
        if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            centerline_mask[z, y, x] = 1

    return smoothed_centerline, centerline_mask


def _compute_local_frames(centerline: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute local coordinate frames along a centerline.
    
    Args:
        centerline: Array of shape (N, 3) with (z, y, x) coordinates
        
    Returns:
        List of (normal_vector, binormal_vector) tuples for each point
    """
    tangents = np.gradient(centerline, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1)[:, None]
    up = np.array([0, 0, 1])
    frames = []
    
    for t in tangents:
        n = np.cross(t, up)
        if np.linalg.norm(n) < 1e-5:
            n = np.array([1, 0, 0])
        n = n / np.linalg.norm(n)
        b = np.cross(t, n)
        frames.append((n, b))
    
    return frames


def _resample_slice(volume: np.ndarray, 
                  origin: np.ndarray, 
                  n: np.ndarray, 
                  b: np.ndarray, 
                  size: int = 64, 
                  spacing: float = 1.0) -> np.ndarray:
    """
    Resample a slice from a 3D volume using local coordinate frames.
    
    Args:
        volume: 3D volume to resample from
        origin: Origin point in (z, y, x) coordinates
        n: Normal vector
        b: Binormal vector
        size: Size of the output slice (width/height)
        spacing: Spacing between pixels
        
    Returns:
        Resampled 2D slice
    """
    grid_u = np.linspace(-size/2, size/2, size) * spacing
    grid_v = np.linspace(-size/2, size/2, size) * spacing
    uu, vv = np.meshgrid(grid_u, grid_v)
    coords = origin + uu[..., None] * n + vv[..., None] * b
    coords = coords.reshape(-1, 3).T
    return map_coordinates(volume, coords, order=1, mode='constant', cval=0).reshape(size, size)


def cmpr(volume: np.ndarray, slice_size: int = 512, spacing: float = 0.5, smoothing_factor: float = 25, view_centerline: bool = False) -> Dict[str, Any]:
    """
    Complete curved multi-planar reconstruction.
    
    Args:
        volume: 3D volume to process
        slice_size: Size of output slices
        spacing: Spacing between pixels in resampled slices
        smoothing_factor: Smoothing factor for centerline
        view_centerline: Whether to create visualizations
        
    Returns:
        Dictionary containing results and intermediate data
    """ 
    logger.info("Starting CMPR pipeline")
    
    # Create binary mask and apply morphological operations
    logger.info("Creating binary mask and applying morphological operations")
    binary_mask = (volume > 0).astype(np.uint8)
    binary_mask = morphological_closing(binary_mask, ball_radius=2)
    
    # Compute bounding box and crop
    logger.info("Computing bounding box and cropping volume")
    bbox = _compute_bounding_box(binary_mask)
    cropped_volume = volume[bbox]
    cropped_mask = binary_mask[bbox]

    # Fill holes in the mask
    logger.info("Filling holes in mask")
    for z in tqdm(range(cropped_mask.shape[0]), desc="Filling holes in mask"):
        cropped_mask[z] = fill_holes(cropped_mask[z])

    # Extract centerline
    logger.info("Extracting centerline")
    smoothed_centerline, centerline_mask = _compute_centerline(
        cropped_mask, smoothing_factor=smoothing_factor, view_centerline=view_centerline
    )

    # Compute local frames
    logger.info("Computing local coordinate frames")
    frames = _compute_local_frames(smoothed_centerline)

    # Generate resampled slices
    logger.info("Generating resampled slices")
    slices = []
    for i in tqdm(range(len(smoothed_centerline)), 
                  desc="Generating slices", 
                  total=len(smoothed_centerline), 
                  unit=" slices"):
        origin = smoothed_centerline[i]
        n_vec, b_vec = frames[i]
        sl = _resample_slice(cropped_volume, origin, n_vec, b_vec, 
                           size=slice_size, spacing=spacing)
        slices.append(sl)
        if (i + 1) % 10 == 0:
            logger.debug(f"Generated {i + 1}/{len(smoothed_centerline)} slices")

    # Stack slices into 3D volume
    slices_volume = np.stack(slices, axis=0)
    
    # Return results
    results = {
        'slices': slices,
        'slices_volume': slices_volume,
        'smoothed_centerline': smoothed_centerline,
        'frames': frames,
        'cropped_volume': cropped_volume,
        'centerline_mask': centerline_mask,
    }
    
    logger.info("CMPR pipeline completed successfully")
    return results
