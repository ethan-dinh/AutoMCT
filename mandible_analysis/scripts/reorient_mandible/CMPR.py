"""
Curved Multi-Planar Reconstruction (CMPR) for Incisor Analysis

This module provides functionality for performing curved multi-planar reconstruction on incisor volumes. It extracts the centerline of the incisor, computes local coordinate frames along the centerline, and resamples the volume to create curved reformatted slices.

Author: Ethan Dinh
Date: August 4, 2025
"""

import numpy as np
import logging
import tifffile as tiff
import argparse
from tqdm import tqdm
import os

from scipy.ndimage import map_coordinates
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev

import automct as mca
import networkx as nx
import plotly.graph_objects as go


IS_VISUALIZE = False

def setup_logging():
    class ColorFormatter(logging.Formatter):
        COLORS = {'DEBUG': '\033[94m', 'INFO': '\033[92m', 'WARNING': '\033[93m', 'ERROR': '\033[91m', 'CRITICAL': '\033[95m'}
        RESET = '\033[0m'
        def format(self, record):
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.RESET)
            record.levelname = f"{color}{levelname}{self.RESET}"
            return super().format(record)
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter('[%(levelname)s] - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

def load_volume(path):
    logging.info(f"Loading volume from {path}")
    return tiff.imread(path)

def save_ct_volume_as_tiff(volume: np.ndarray, output_dir: str, name: str):
    """
    Save a 3D CT volume as a multi-page TIFF.
    
    Parameters:
        volume (np.ndarray): 3D array of CT data (e.g. int16 or float32).
        output_dir (str): Directory to save the TIFF file.
        name (str): Base name for the output file (without extension).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Log the volume details
    logging.info(f"Saving volume with shape: {volume.shape}, dtype: {volume.dtype}, min: {np.min(volume)}, max: {np.max(volume)}")

    # Downgrade float64 to float32
    # This is necessary for the TIFF file to be saved correctly
    if volume.dtype == np.float64:
        logging.info(f"Downgrading float64 to float32")
        volume = volume.astype(np.float32)

    # Ensure proper data type
    if volume.dtype not in [np.int16, np.float32]:
        volume = volume.astype(np.int16)

    output_path = os.path.join(output_dir, f"{name}.tif")
    
    # Save using BigTIFF for large volumes, ImageJ-compatible if needed
    tiff.imwrite(
        output_path,
        volume,
        dtype=volume.dtype,
        compression='zlib',
        bigtiff=True
    )

def compute_bounding_box(volume: np.ndarray) -> tuple[slice, slice, slice]:
    logging.info("Computing bounding box")
    nonzero = np.argwhere(volume)
    if nonzero.size == 0:
        raise ValueError("Volume is empty or not segmented.")
    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0) + 1
    return min_coords, max_coords, tuple(slice(minc, maxc) for minc, maxc in zip(min_coords, max_coords))

def plot_graph_with_centerline(G, coords, path_idx):
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
    fig.write_html("graph_output.html")

def smooth_centerline(centerline: np.ndarray, smoothing_factor=5.0, n_points=100) -> np.ndarray:
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
    smoothed_centerline = np.stack([smoothed_points[2], smoothed_points[1], smoothed_points[0]], axis=1)
    return smoothed_centerline

def compute_centerline(mask: np.ndarray) -> np.ndarray:
    """
    Extract the centerline of a 3D incisor volume.

    Parameters
    ----------
    mask : ndarray, shape (Z,Y,X)
        Binary mask of the incisor.

    Returns
    -------
    centerline : ndarray, shape (M,3)
        Ordered (z,y,x) coordinates along the longest path.
    """
    global IS_VISUALIZE

    logging.info("Skeletonizing...")
    skel = skeletonize(mask)

    # 3. Build a weighted graph on skeleton voxels
    logging.info("Building graph...")
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

    # 4. Two‐pass Dijkstra to find graph diameter endpoints
    # Pass 1: from an arbitrary node (0) find farthest node u
    logging.info("Finding farthest node u...")
    dist1 = nx.single_source_dijkstra_path_length(G, source=0, weight='weight')
    u = max(dist1, key=dist1.get)

    # Pass 2: from u find farthest node v
    logging.info("Finding farthest node v...")
    dist2 = nx.single_source_dijkstra_path_length(G, source=u, weight='weight')
    v = max(dist2, key=dist2.get)

    # Extract the diameter path (list of node indices)
    logging.info("Finding diameter path...")
    path_idx = nx.dijkstra_path(G, source=u, target=v, weight='weight')

    # Plot the path in 3D
    if IS_VISUALIZE:
        plot_graph_with_centerline(G, coords, path_idx)

    # Convert node indices back to (z,y,x) coordinates
    logging.info("Converting node indices to coordinates...")
    centerline = coords[path_idx]

    logging.info("Smoothing centerline...")

    # Determine the number of points as the number of slices
    n_points = len(centerline)  # or fewer for heavier subsampling
    smoothing_factor = 25 * len(centerline)  # 10× to 50× stronger than default
    smoothed_centerline = smooth_centerline(centerline, smoothing_factor=smoothing_factor, n_points=n_points)

    # Fill in the centerline voxels
    centerline_mask = np.zeros_like(mask, dtype=np.uint8)
    for zf, yf, xf in smoothed_centerline:
        z, y, x = map(int, np.round([zf, yf, xf]))
        if 0 <= z < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            centerline_mask[z, y, x] = 1

    return smoothed_centerline, centerline_mask

def compute_local_frames(centerline):
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

def resample_slice(volume, origin, n, b, size=64, spacing=1.0):
    grid_u = np.linspace(-size/2, size/2, size) * spacing
    grid_v = np.linspace(-size/2, size/2, size) * spacing
    uu, vv = np.meshgrid(grid_u, grid_v)
    coords = origin + uu[..., None] * n + vv[..., None] * b
    coords = coords.reshape(-1, 3).T
    return map_coordinates(volume, coords, order=1, mode='constant', cval=0).reshape(size, size)

def cmpr_pipeline(path, output_dir, slice_size=64, spacing=0.5):
    global IS_VISUALIZE
    volume = load_volume(path)

    logging.info(f"Cropping volume to bounding box")
    binary_mask = (volume > 0).astype(np.uint8)
    binary_mask = mca.segmentation.morphological_closing(binary_mask, ball_radius=2)
    _, _, bbox = compute_bounding_box(binary_mask)

    cropped_volume = volume[bbox]
    cropped_mask = binary_mask[bbox]

    logging.info("Perform a morphological closing to fill holes in the binary mask")
    for z in tqdm(range(binary_mask.shape[0]), desc="Filling holes in mask"):
        binary_mask[z] = mca.segmentation.fill_holes(binary_mask[z])

    logging.info("Determining centerline of the incisor...")
    smoothed_centerline, centerline_mask = compute_centerline(cropped_mask)

    logging.info("Computing local frames...")
    frames = compute_local_frames(smoothed_centerline)

    logging.info("Generating slices...")
    slices = []
    for i in tqdm(range(len(smoothed_centerline)), desc="Generating slices", total=len(smoothed_centerline), unit=" slices"):
        origin = smoothed_centerline[i]
        n_vec, b_vec = frames[i]
        sl = resample_slice(cropped_volume, origin, n_vec, b_vec, size=slice_size, spacing=spacing)
        slices.append(sl)
        if (i + 1) % 10 == 0:
            logging.debug(f"Generated {i + 1}/{len(smoothed_centerline)} slices")

    slices_volume = np.stack(slices, axis=0)
    save_ct_volume_as_tiff(slices_volume, output_dir, "resliced_volume")
    logging.info(f"Saved slices to {output_dir}")   

    if IS_VISUALIZE:
        mca.visualization.create_3d_visualization(cropped_volume, None, {"slices": (slices_volume, "yellow"), "smoothed_centerline": (centerline_mask, "green")})

    return slices, smoothed_centerline, frames, cropped_volume

def parse_arguments():
    parser = argparse.ArgumentParser(description="Curved Multi-Planar Reconstruction for Incisor")
    parser.add_argument("--log", "-l", action="store_true", help="Enable logging")
    parser.add_argument("--slice_size", type=int, default=64, help="Slice width/height")
    parser.add_argument("--spacing", type=float, default=1.0, help="Spacing between pixels")
    parser.add_argument("--visualize", "-v", action="store_true", help="Enable interactive 3D visualization")
    return parser.parse_args()

def main():
    global IS_VISUALIZE
    args = parse_arguments()
    if args.log:
        setup_logging()
    if args.visualize:
        IS_VISUALIZE = True

    logging.info("CMPR Pipeline Starting")
    
    output_dir = "./results"
    if not os.path.exists(output_dir):
        logging.info("Creating output directory")
        os.makedirs(output_dir)

    # Find all of the folders in the input path (each folder is a stack of files)
    input_path = "../segment_mandible/segmentation_results/"
    input_folders = [
        name for name in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, name)) and name != "Compressed"
    ]

    for dir in input_folders:
        logging.info(f"Processing {dir}")
        cmpr_pipeline(
            path=os.path.join(input_path, dir, "incisor_volume.tif"),
            output_dir=os.path.join(output_dir, dir),
            slice_size=args.slice_size,
            spacing=args.spacing
        )

    logging.info("CMPR Pipeline Completed Successfully")

if __name__ == "__main__":
    main()
