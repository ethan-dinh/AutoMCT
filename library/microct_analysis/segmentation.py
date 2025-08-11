"""
Segmentation utilities for microCT data analysis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, binary_erosion, binary_dilation, binary_fill_holes
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu, threshold_local, gaussian, threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, ball, disk, skeletonize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from typing import Dict, Optional
import logging
from skimage import exposure
from skimage.exposure import equalize_adapthist
from tqdm import tqdm

logger = logging.getLogger(__name__)

def segment_slice(slice_2d: np.ndarray, 
                  method: str = 'otsu',
                  min_area: int = 100,
                  **kwargs) -> np.ndarray:
    """
    Segment a 2D slice using various methods.
    
    Args:
        slice_2d: 2D numpy array (grayscale image)
        method: Segmentation method ('otsu', 'local', 'watershed', 'manual')
        min_area: Minimum area for regions to keep
        **kwargs: Additional parameters for specific methods
    
    Returns:
        Labeled image where each region has a unique integer label
    """
    
    # Preprocessing
    if kwargs.get('gaussian_sigma', 0) > 0:
        slice_2d = gaussian(slice_2d, sigma=kwargs['gaussian_sigma'])
    
    if method == 'otsu':
        return _segment_otsu(slice_2d, min_area, **kwargs)
    elif method == 'local':
        return _segment_local_threshold(slice_2d, min_area, **kwargs)
    elif method == 'watershed':
        return _segment_watershed(slice_2d, min_area, **kwargs)
    elif method == 'manual':
        return _segment_manual_threshold(slice_2d, min_area, **kwargs)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

def _segment_otsu(slice_2d: np.ndarray, 
                  min_area: int = 100,
                  **kwargs) -> np.ndarray:
    
    """Otsu thresholding segmentation."""
    thresh = threshold_otsu(slice_2d)
    binary = slice_2d > thresh
    
    # Morphological operations
    if kwargs.get('morphology', True):
        binary = binary_opening(binary)
        binary = binary_closing(binary)
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=min_area)
    
    # Label connected components
    labeled = label(binary)
    
    return labeled

def _segment_local_threshold(slice_2d: np.ndarray,
                           min_area: int = 100,
                           **kwargs) -> np.ndarray:
    """Local adaptive thresholding segmentation."""
    block_size = kwargs.get('block_size', 35)
    thresh = threshold_local(slice_2d, block_size=block_size)
    binary = slice_2d > thresh
    
    # Morphological operations
    if kwargs.get('morphology', True):
        binary = binary_opening(binary)
        binary = binary_closing(binary)
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=min_area)
    
    # Label connected components
    labeled = label(binary)
    
    return labeled

def _segment_watershed(slice_2d: np.ndarray,
                      min_area: int = 100,
                      **kwargs) -> np.ndarray:
    """Watershed segmentation."""
    # Create distance map
    thresh = threshold_otsu(slice_2d)
    binary = slice_2d > thresh
    
    from scipy.ndimage import distance_transform_edt
    distance = distance_transform_edt(binary)
    
    # Find peaks
    min_distance = kwargs.get('min_distance', 10)
    peaks = peak_local_max(distance, min_distance=min_distance)
    
    # Create markers
    markers = np.zeros_like(slice_2d, dtype=bool)
    markers[peaks[:, 0], peaks[:, 1]] = True
    markers = label(markers)
    
    # Watershed segmentation
    labeled = watershed(-distance, markers, mask=binary)
    
    # Remove small regions
    for region in regionprops(labeled):
        if region.area < min_area:
            labeled[labeled == region.label] = 0
    
    return labeled

def _segment_manual_threshold(slice_2d: np.ndarray,
                            min_area: int = 100,
                            **kwargs) -> np.ndarray:
    """Manual thresholding segmentation."""
    threshold = kwargs.get('threshold', 128)
    binary = slice_2d > threshold
    
    # Morphological operations
    if kwargs.get('morphology', True):
        binary = binary_opening(binary)
        binary = binary_closing(binary)
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=min_area)
    
    # Label connected components
    labeled = label(binary)
    
    return labeled

def segment_volume(volume: np.ndarray,
                  method: str = 'otsu',
                  min_area: int = 100,
                  **kwargs) -> np.ndarray:
    """
    Segment entire 3D volume slice by slice.
    
    Args:
        volume: 3D numpy array
        method: Segmentation method
        min_area: Minimum area for regions
        **kwargs: Additional parameters
    
    Returns:
        3D labeled volume
    """
    from tqdm import tqdm
    
    depth = volume.shape[0]
    labeled_volume = np.zeros_like(volume, dtype=np.int32)
    
    logger.info(f"Segmenting volume with {depth} slices using {method} method")
    
    for i in tqdm(range(depth), desc="Segmenting slices"):
        labeled_volume[i] = segment_slice(volume[i], method, min_area, **kwargs)
    
    return labeled_volume

def get_segmentation_stats(labeled_slice: np.ndarray) -> Dict:
    """
    Get statistics about segmented regions.
    
    Args:
        labeled_slice: Labeled 2D array
    
    Returns:
        Dictionary with segmentation statistics
    """
    regions = regionprops(labeled_slice)
    
    if not regions:
        return {
            'num_regions': 0,
            'total_area': 0,
            'mean_area': 0,
            'max_area': 0,
            'min_area': 0
        }
    
    areas = [region.area for region in regions]
    
    return {
        'num_regions': len(regions),
        'total_area': sum(areas),
        'mean_area': np.mean(areas),
        'max_area': max(areas),
        'min_area': min(areas),
        'std_area': np.std(areas)
    }

def filter_regions_by_area(labeled_slice: np.ndarray,
                          min_area: int = 100,
                          max_area: Optional[int] = None) -> np.ndarray:
    """
    Filter regions by area constraints.
    
    Args:
        labeled_slice: Labeled 2D array
        min_area: Minimum area to keep
        max_area: Maximum area to keep (None for no limit)
    
    Returns:
        Filtered labeled array
    """
    filtered = labeled_slice.copy()
    
    for region in regionprops(labeled_slice):
        if region.area < min_area or (max_area and region.area > max_area):
            filtered[filtered == region.label] = 0
    
    return filtered

def get_largest_region(labeled_item: np.ndarray) -> np.ndarray:
    """
    Get the largest region in a labeled slice or volume.
    """
    regions = regionprops(labeled_item)
    largest_region = max(regions, key=lambda x: x.area)
    return largest_region.label

def get_region_intensity(slice_2d: np.ndarray,
                         labeled_slice: np.ndarray,
                         region_label: int) -> float:
    """
    Get the average intensity of a specific region in a labeled slice.

    Args:
        slice_2d: The original grayscale slice (intensity image).
        labeled_slice: The labeled image with integer region labels.
        region_label: The label value of the region to measure.

    Returns:
        Mean intensity of the specified region, or 0.0 if not found.
    """
    props = regionprops(labeled_slice, intensity_image=slice_2d)
    for region in props:
        if region.label == region_label:
            return float(region.mean_intensity)
    return 0.0  # Return 0 if region not found

def morphological_closing(volume: np.ndarray, ball_radius: int = 1) -> np.ndarray:
    """
    Apply morphological operations to a volume.
    """
    volume = binary_closing(volume, ball(ball_radius))
    return volume

def morphological_opening(volume: np.ndarray, ball_radius: int = 0, disk_radius: int = 0) -> np.ndarray:
    """
    Apply morphological operations to a volume.
    """
    if ball_radius > 0:
        volume = binary_opening(volume, ball(ball_radius))
    if disk_radius > 0:
        volume = binary_opening(volume, disk(disk_radius))
    return volume

def label_3d_volume(binary_volume: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Label a 3D binary volume.
    """
    return label(binary_volume, connectivity=connectivity)

def label_slice(slice_2d: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Label a 2D slice.
    """
    return label(slice_2d, connectivity=connectivity)

def convert_to_binary(volume: np.ndarray, fill_holes: bool = False) -> np.ndarray:
    """
    Convert a volume to a binary volume.

    Parameters:
        volume (np.ndarray): The volume to convert to a binary volume.
        fill_holes (bool): Whether to fill holes in the volume.

    Returns:
        binary_volume (np.ndarray): The binary volume.
    """
    if fill_holes:
        volume = binary_closing(volume)
    return volume > 0

def dilate_mask(mask: np.ndarray, radius: int = 1, if_2d: bool = False) -> np.ndarray:
    """
    Dilate a mask.

    Parameters:
        mask (np.ndarray): The mask to dilate.
        radius (int): The radius of the ball to use for dilation.

    Returns:
        dilated_mask (np.ndarray): The dilated mask.
    """
    if if_2d:
        return binary_dilation(mask, disk(radius))
    else:
        return binary_dilation(mask, ball(radius))

def multi_otsu_threshold(volume: np.ndarray, num_thresholds: int = 3, nbins: int = 256) -> np.ndarray:
    """
    Apply multi-level OTSU thresholding to a volume.

    Parameters:
        volume (np.ndarray): The volume to threshold.
        num_thresholds (int): The number of thresholds to use.

    Returns:
        thresholds (np.ndarray): The thresholds.
    """
    thresholds = threshold_multiotsu(volume, classes=num_thresholds, nbins=nbins)
    return thresholds

def adjust_gamma(volume: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Increase the gamma of a volume.
    """
    return exposure.adjust_gamma(volume, gamma=gamma)

def find_histogram_valley_threshold(volume: np.ndarray, nbins: int = 512, sigma: float = 1.0) -> float:
    """
    Find the deepest valley in the histogram between two peaks.
    
    Parameters:
        volume: np.ndarray
            3D array of the image/volume.
        nbins: int
            Number of histogram bins.
        sigma: float
            Gaussian filter sigma to smooth the histogram.
    
    Returns:
        threshold: float
            Intensity value corresponding to the deepest valley between peaks.
    """
    # Step 1: Normalize nonzero values
    foreground = volume[volume > 0]
    p1, p99 = np.percentile(foreground, [1, 99])
    norm = np.clip((foreground - p1) / (p99 - p1 + 1e-6), 0, 1)

    # Step 2: Compute histogram
    hist, bin_edges = np.histogram(norm, bins=nbins, range=(0, 1))
    smoothed_hist = gaussian_filter(hist.astype(float), sigma=sigma)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Step 3: Find peaks
    peaks, _ = find_peaks(smoothed_hist, distance=nbins // 10)

    if len(peaks) < 2:
        return 0.0

    # Sort peaks by height
    peak_indices = peaks[np.argsort(smoothed_hist[peaks])[::-1][:2]]
    peak_indices = np.sort(peak_indices)

    # Step 4: Find valley (minimum) between peaks
    valley_region = smoothed_hist[peak_indices[0]:peak_indices[1]+1]
    valley_index = np.argmin(valley_region) + peak_indices[0]
    threshold = bin_centers[valley_index]

    return threshold

def enhance_edges(volume: np.ndarray, sigma: float = 1.0, equalize: bool = True) -> np.ndarray:
    """
    Enhance enamel–dentin edges slice-by-slice using 2D Sobel filtering and optional adaptive histogram equalization.

    Parameters:
    - volume (np.ndarray): 3D volume (Z, Y, X)
    - sigma (float): Gaussian smoothing sigma
    - equalize (bool): Whether to apply adaptive histogram equalization to boost contrast

    Returns:
    - np.ndarray: 3D edge-enhanced volume (Z, Y, X) with values in [0, 1]
    """

    logger.info(f"Applying Gaussian smoothing with sigma={sigma}")
    smoothed = gaussian_filter(volume, sigma=sigma)

    edge_volume = np.zeros_like(smoothed, dtype=np.float32)

    logger.info("Enhancing edges slice-by-slice...")
    for z in tqdm(range(smoothed.shape[0]), desc="Enhancing edges", total=smoothed.shape[0], unit="slice"):
        slice_2d = smoothed[z]

        # Compute 2D Sobel gradient magnitude
        gx = sobel(slice_2d, axis=1)  # x-direction
        gy = sobel(slice_2d, axis=0)  # y-direction
        grad_mag = np.hypot(gx, gy)

        # Normalize slice
        grad_mag -= grad_mag.min()
        if grad_mag.max() > 0:
            grad_mag /= grad_mag.max()

        # Optional contrast enhancement
        if equalize:
            grad_mag = equalize_adapthist(grad_mag, clip_limit=0.03)

        edge_volume[z] = grad_mag

    logger.info("Edge enhancement complete.")
    return edge_volume

def skeletonize_volume(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize a volume.
    """
    return skeletonize(mask)

def erode_mask(mask: np.ndarray, radius: int = 1, if_2d: bool = False) -> np.ndarray:
    """
    Erode a mask.
    """
    if if_2d:
        return binary_erosion(mask, disk(radius))
    else:
        return binary_erosion(mask, ball(radius))

def difference_of_gaussians(image, low_sigma=1.0, high_sigma=5.0):
    low_pass = gaussian_filter(image, sigma=high_sigma)
    high_pass = gaussian_filter(image, sigma=low_sigma)
    dog = high_pass - low_pass
    return dog

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a mask.
    """
    return binary_fill_holes(mask)