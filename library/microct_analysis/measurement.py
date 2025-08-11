"""
Measurement utilities for quantifying regions in microCT data.
"""

import numpy as np
from skimage.measure import regionprops
from typing import Callable, Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

def measure_regions(slice_2d: np.ndarray,
                    labeled: np.ndarray,
                    properties: Optional[List[str]] = None,
                    extra_properties: Optional[List[Callable]] = None) -> List[Dict]:
    """
    Measure specified region properties in a labeled 2D image.

    Args:
        slice_2d: Original grayscale image (2D array).
        labeled: Labeled image where each region has a unique integer label.
        properties: List of standard property names to extract (e.g., 'area', 'centroid').
        extra_properties: List of custom functions (regionmask[, intensity]) -> scalar.

    Returns:
        List of dictionaries containing region measurements.
    """
    props = regionprops(labeled, intensity_image=slice_2d, extra_properties=extra_properties)

    if properties is None:
        properties = ["label", "area", "centroid", "mean_intensity"]

    results = []
    for region in props:
        region_dict = {}
        for prop in properties:
            try:
                value = getattr(region, prop)
                # Call the value if it's a method
                if callable(value):
                    value = value()
                # Convert to JSON-safe format
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, np.generic):
                    value = value.item()
                region_dict[prop] = value
            except AttributeError:
                region_dict[prop] = None
        results.append(region_dict)

    return results

def compare_two_regions(results: List[Dict],
                       region_indices: List[int] = None) -> Optional[Dict]:
    """
    Compare two regions from measurement results.
    
    Args:
        results: List of region measurement dictionaries
        region_indices: Indices of regions to compare (None for first two)
    
    Returns:
        Dictionary with comparison metrics or None if insufficient regions
    """
    if len(results) < 2:
        logger.warning("Need at least 2 regions for comparison")
        return None
    
    if region_indices is None:
        region_indices = [0, 1]
    
    if len(region_indices) != 2:
        raise ValueError("Must specify exactly 2 region indices")
    
    r1_idx, r2_idx = region_indices
    
    if r1_idx >= len(results) or r2_idx >= len(results):
        raise ValueError("Region indices out of range")
    
    r1, r2 = results[r1_idx], results[r2_idx]
    
    comparison = {
        'region1_index': r1_idx,
        'region2_index': r2_idx,
        'region1_label': r1.get('label'),
        'region2_label': r2.get('label'),
        'intensity_comparison': {
            'r1_mean': r1.get('mean_intensity'),
            'r2_mean': r2.get('mean_intensity'),
            'difference': r1.get('mean_intensity', 0) - r2.get('mean_intensity', 0),
            'ratio': (r1.get('mean_intensity', 1) / r2.get('mean_intensity', 1)) 
                    if r2.get('mean_intensity', 0) != 0 else float('inf')
        },
        'area_comparison': {
            'r1_area': r1.get('area'),
            'r2_area': r2.get('area'),
            'area_difference': r1.get('area', 0) - r2.get('area', 0),
            'area_ratio': (r1.get('area', 1) / r2.get('area', 1)) 
                         if r2.get('area', 0) != 0 else float('inf')
        },
        'shape_comparison': {
            'r1_eccentricity': r1.get('eccentricity'),
            'r2_eccentricity': r2.get('eccentricity'),
            'r1_solidity': r1.get('solidity'),
            'r2_solidity': r2.get('solidity')
        }
    }
    
    return comparison

def analyze_slice_regions(slice_2d: np.ndarray,
                         labeled: np.ndarray,
                         top_n: int = 5) -> Dict:
    """
    Comprehensive analysis of regions in a slice.
    
    Args:
        slice_2d: Original grayscale image
        labeled: Labeled image
        top_n: Number of top regions to analyze in detail
    
    Returns:
        Dictionary with comprehensive analysis
    """
    # Get all region measurements
    regions = measure_regions(slice_2d, labeled)
    
    if not regions:
        return {
            'num_regions': 0,
            'summary_stats': {},
            'top_regions': [],
            'comparisons': []
        }
    
    # Calculate summary statistics
    areas = [r.get('area', 0) for r in regions]
    intensities = [r.get('mean_intensity', 0) for r in regions]
    
    summary_stats = {
        'num_regions': len(regions),
        'total_area': sum(areas),
        'mean_area': np.mean(areas),
        'std_area': np.std(areas),
        'mean_intensity': np.mean(intensities),
        'std_intensity': np.std(intensities),
        'max_area': max(areas),
        'min_area': min(areas),
        'max_intensity': max(intensities),
        'min_intensity': min(intensities)
    }
    
    # Get top regions by area
    sorted_by_area = sorted(regions, key=lambda x: x.get('area', 0), reverse=True)
    top_regions = sorted_by_area[:top_n]
    
    # Compare top regions
    comparisons = []
    for i in range(min(len(top_regions), 2)):
        for j in range(i + 1, min(len(top_regions), 2)):
            comp = compare_two_regions(top_regions, [i, j])
            if comp:
                comparisons.append(comp)
    
    return {
        'num_regions': len(regions),
        'summary_stats': summary_stats,
        'top_regions': top_regions,
        'comparisons': comparisons,
        'all_regions': regions
    }

def calculate_region_statistics(volume: np.ndarray,
                              labeled_volume: np.ndarray,
                              slice_range: Tuple[int, int] = None) -> Dict:
    """
    Calculate statistics across multiple slices.
    
    Args:
        volume: Original 3D volume
        labeled_volume: 3D labeled volume
        slice_range: Range of slices to analyze (start, end)
    
    Returns:
        Dictionary with cross-slice statistics
    """
    if slice_range is None:
        slice_range = (0, volume.shape[0])
    
    start_slice, end_slice = slice_range
    
    all_analyses = []
    for i in range(start_slice, end_slice):
        analysis = analyze_slice_regions(volume[i], labeled_volume[i])
        analysis['slice_index'] = i
        all_analyses.append(analysis)
    
    # Aggregate statistics across slices
    total_regions = sum(a['num_regions'] for a in all_analyses)
    all_areas = []
    all_intensities = []
    
    for analysis in all_analyses:
        for region in analysis.get('all_regions', []):
            all_areas.append(region.get('area', 0))
            all_intensities.append(region.get('mean_intensity', 0))
    
    cross_slice_stats = {
        'total_slices_analyzed': len(all_analyses),
        'total_regions': total_regions,
        'mean_regions_per_slice': total_regions / len(all_analyses) if all_analyses else 0,
        'area_stats': {
            'mean': np.mean(all_areas),
            'std': np.std(all_areas),
            'min': min(all_areas) if all_areas else 0,
            'max': max(all_areas) if all_areas else 0
        },
        'intensity_stats': {
            'mean': np.mean(all_intensities),
            'std': np.std(all_intensities),
            'min': min(all_intensities) if all_intensities else 0,
            'max': max(all_intensities) if all_intensities else 0
        }
    }
    
    return {
        'cross_slice_stats': cross_slice_stats,
        'slice_analyses': all_analyses
    } 