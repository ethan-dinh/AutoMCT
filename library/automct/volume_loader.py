"""
Volume loading utilities for microCT data from BMP files.
"""

import os
import numpy as np
from PIL import Image
import logging
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

def load_bmp_stack(directory: str, 
                   file_pattern: str = "*.bmp",
                   exclude_pattern: str = None,
                   sort_by: str = "name") -> np.ndarray:
    """
    Load a stack of BMP files into a 3D volume.
    
    Args:
        directory: Path to directory containing BMP files
        file_pattern: Glob pattern for file matching
        exclude_pattern: Glob pattern for files to exclude
        sort_by: Sorting method ('name', 'number', 'date')
    
    Returns:
        3D numpy array with shape (depth, height, width)
    
    Raises:
        FileNotFoundError: If no BMP files found
        ValueError: If images have inconsistent dimensions
    """
    import glob
    
    # Find all BMP files
    pattern = os.path.join(directory, file_pattern)
    file_list = glob.glob(pattern)

    if exclude_pattern:
        # Get absolute paths of files to exclude
        full_exclude_pattern = os.path.join(directory, exclude_pattern)
        excluded_files = set(os.path.join(directory, os.path.basename(path)) for path in glob.glob(full_exclude_pattern))

        # Log excluded files
        logger.info(f"Excluding files: {excluded_files}")
        
        # Filter out excluded files
        file_list = [f for f in file_list if f not in excluded_files]
    
    if not file_list:
        raise FileNotFoundError(f"No BMP files found in {directory}")
    
    # Sort files
    if sort_by == "name":
        file_list.sort()
    elif sort_by == "number":
        # Extract numbers from filenames for natural sorting
        file_list.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    elif sort_by == "date":
        file_list.sort(key=lambda x: os.path.getmtime(x))
    
    logger.info(f"Loading {len(file_list)} BMP files from {directory}")
    
    # Load first image to get dimensions
    first_img = Image.open(file_list[0]).convert('L')
    width, height = first_img.size
    
    # Pre-allocate volume array
    volume = np.zeros((len(file_list), height, width), dtype=np.uint8)
    
    # Load all slices
    for i, file_path in enumerate(file_list):
        try:
            img = Image.open(file_path).convert('L')
            img_array = np.array(img)
            
            # Check dimensions
            if img_array.shape != (height, width):
                raise ValueError(f"Image {file_path} has inconsistent dimensions: "
                              f"expected {(height, width)}, got {img_array.shape}")
            
            volume[i] = img_array
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    logger.info(f"Volume loaded successfully: {volume.shape}")
    return volume

def load_single_bmp(file_path: str) -> np.ndarray:
    """
    Load a single BMP file.
    
    Args:
        file_path: Path to BMP file
    
    Returns:
        2D numpy array
    """
    img = Image.open(file_path).convert('L')
    return np.array(img)

def get_volume_info(volume: np.ndarray) -> dict:
    """
    Get information about the loaded volume.
    
    Args:
        volume: 3D numpy array
    
    Returns:
        Dictionary with volume information
    """
    return {
        'shape': volume.shape,
        'dtype': volume.dtype,
        'min_value': volume.min(),
        'max_value': volume.max(),
        'mean_value': volume.mean(),
        'std_value': volume.std()
    }

def background_subtraction(volume: np.ndarray) -> np.ndarray:
    """
    Subtract the background from the volume.
    """
    return volume - volume.min()

def normalize_volume(volume: np.ndarray, 
                    method: str = 'minmax') -> np.ndarray:
    """
    Normalize volume values.
    
    Args:
        volume: 3D numpy array
        method: Normalization method ('minmax', 'zscore', 'histogram')
    
    Returns:
        Normalized volume
    """
    if method == 'minmax':
        vmin, vmax = volume.min(), volume.max()
        return (volume - vmin) / (vmax - vmin)
    elif method == 'zscore':
        mean, std = volume.mean(), volume.std()
        return (volume - mean) / std
    elif method == 'histogram':
        # Histogram equalization
        from skimage import exposure
        return exposure.equalize_hist(volume)
    else:
        raise ValueError(f"Unknown normalization method: {method}") 

def gaussian_filter_volume(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply a Gaussian filter to the volume.
    """
    logger.info(f"Applying Gaussian filter with sigma = {sigma}")
    return gaussian_filter(volume, sigma=sigma)

