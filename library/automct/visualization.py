"""
Visualization utilities for microCT analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def plot_slice_with_regions(slice_2d: np.ndarray,
                           labeled: np.ndarray,
                           title: str = "Slice with Segmented Regions",
                           figsize: Tuple[int, int] = (12, 4)) -> None:
    """
    Plot original slice with overlaid segmented regions.
    
    Args:
        slice_2d: Original grayscale slice
        labeled: Labeled regions
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original slice
    axes[0].imshow(slice_2d, cmap='gray')
    axes[0].set_title('Original Slice')
    axes[0].axis('off')
    
    # Segmented regions
    axes[1].imshow(labeled, cmap='tab20')
    axes[1].set_title('Segmented Regions')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(slice_2d, cmap='gray')
    axes[2].imshow(labeled, cmap='tab20', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_region_statistics(analysis_results: Dict,
                          save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive region statistics.
    
    Args:
        analysis_results: Results from region analysis
        save_path: Optional path to save the plot
    """
    if 'region_analysis' not in analysis_results:
        logger.warning("No region analysis results found")
        return
    
    region_stats = analysis_results['region_analysis']
    slice_analyses = region_stats['slice_analyses']
    
    if not slice_analyses:
        logger.warning("No slice analyses found")
        return
    
    # Extract data
    slice_indices = [a['slice_index'] for a in slice_analyses]
    num_regions = [a['num_regions'] for a in slice_analyses]
    
    # Get area and intensity data
    all_areas = []
    all_intensities = []
    for analysis in slice_analyses:
        for region in analysis.get('all_regions', []):
            all_areas.append(region.get('area', 0))
            all_intensities.append(region.get('mean_intensity', 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Number of regions per slice
    axes[0, 0].plot(slice_indices, num_regions, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Slice Index')
    axes[0, 0].set_ylabel('Number of Regions')
    axes[0, 0].set_title('Regions per Slice')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Area distribution
    if all_areas:
        axes[0, 1].hist(all_areas, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Region Area')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Region Area Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Intensity distribution
    if all_intensities:
        axes[1, 0].hist(all_intensities, bins=50, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Mean Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Region Intensity Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Area vs Intensity scatter
    if all_areas and all_intensities:
        axes[1, 1].scatter(all_areas, all_intensities, alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Region Area')
        axes[1, 1].set_ylabel('Mean Intensity')
        axes[1, 1].set_title('Area vs Intensity')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_segmentation_statistics(analysis_results: Dict,
                               save_path: Optional[str] = None) -> None:
    """
    Plot segmentation statistics.
    
    Args:
        analysis_results: Results from analysis
        save_path: Optional path to save the plot
    """
    if 'segmentation_stats' not in analysis_results:
        logger.warning("No segmentation statistics found")
        return
    
    seg_stats = analysis_results['segmentation_stats']
    
    if not seg_stats:
        logger.warning("No segmentation statistics found")
        return
    
    # Extract data
    slice_indices = [s['slice_index'] for s in seg_stats]
    num_regions = [s['num_regions'] for s in seg_stats]
    total_areas = [s['total_area'] for s in seg_stats]
    mean_areas = [s['mean_area'] for s in seg_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Number of regions per slice
    axes[0, 0].plot(slice_indices, num_regions, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Slice Index')
    axes[0, 0].set_ylabel('Number of Regions')
    axes[0, 0].set_title('Regions per Slice')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Total area per slice
    axes[0, 1].plot(slice_indices, total_areas, 'g-o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Slice Index')
    axes[0, 1].set_ylabel('Total Area')
    axes[0, 1].set_title('Total Area per Slice')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean area per slice
    axes[1, 0].plot(slice_indices, mean_areas, 'r-o', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Slice Index')
    axes[1, 0].set_ylabel('Mean Area')
    axes[1, 0].set_title('Mean Area per Slice')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Histogram of number of regions
    axes[1, 1].hist(num_regions, bins=20, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Number of Regions')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Regions per Slice')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def create_3d_visualization(volume: np.ndarray,
                           labeled_volume: np.ndarray = None,
                           additional_volumes: Dict[str, np.ndarray] = None,
                           slice_range: Tuple[int, int] = None,
                           ndisplay: int = 2,
                           show_bounding_box: bool = False) -> None:
    """
    Create 3D visualization using napari.
    """

    # Attempt to import napari
    try:
        import napari

    except ImportError as e:
        logger.warning("napari not available. Install with: pip install napari")
        return

    # Create 3D visualization using napari
    try:
        if slice_range:
            start, end = slice_range
            volume = volume[start:end]
            if labeled_volume is not None:
                labeled_volume = labeled_volume[start:end]

        viewer = napari.Viewer(ndisplay=ndisplay)
        viewer.add_image(volume, name='Original Volume', colormap='gray')

        if additional_volumes is not None:
            for name, (volume, color) in additional_volumes.items():
                viewer.add_image(volume, name=name, colormap=color, blending='additive')

        if labeled_volume is not None:
            viewer.add_labels(labeled_volume, name='Segmented Regions', opacity=0.35)

        if show_bounding_box:
            for layer in viewer.layers:
                if layer.name == 'Original Volume':
                    layer.bounding_box.visible = True
                    break

        logger.info("3D visualization opened in napari viewer")
        napari.run()

    except Exception as e:
        logger.error(f"Error creating 3D visualization: {e}")


def plot_comparison_results(analysis_results: Dict,
                          save_path: Optional[str] = None) -> None:
    """
    Plot region comparison results.
    
    Args:
        analysis_results: Results from analysis
        save_path: Optional path to save the plot
    """
    if 'region_analysis' not in analysis_results:
        logger.warning("No region analysis results found")
        return
    
    region_stats = analysis_results['region_analysis']
    slice_analyses = region_stats['slice_analyses']
    
    # Extract comparison data
    slice_indices = []
    intensity_diffs = []
    area_diffs = []
    intensity_ratios = []
    area_ratios = []
    
    for analysis in slice_analyses:
        for comparison in analysis.get('comparisons', []):
            slice_indices.append(analysis['slice_index'])
            
            intensity_comp = comparison.get('intensity_comparison', {})
            area_comp = comparison.get('area_comparison', {})
            
            intensity_diffs.append(intensity_comp.get('difference', 0))
            area_diffs.append(area_comp.get('area_difference', 0))
            
            ratio = intensity_comp.get('ratio', 1)
            if ratio != float('inf'):
                intensity_ratios.append(ratio)
            else:
                intensity_ratios.append(1)
            
            area_ratio = area_comp.get('area_ratio', 1)
            if area_ratio != float('inf'):
                area_ratios.append(area_ratio)
            else:
                area_ratios.append(1)
    
    if not slice_indices:
        logger.warning("No comparison data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Intensity differences
    axes[0, 0].scatter(slice_indices, intensity_diffs, alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Slice Index')
    axes[0, 0].set_ylabel('Intensity Difference (R1 - R2)')
    axes[0, 0].set_title('Intensity Differences')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Area differences
    axes[0, 1].scatter(slice_indices, area_diffs, alpha=0.6, s=30, color='green')
    axes[0, 1].set_xlabel('Slice Index')
    axes[0, 1].set_ylabel('Area Difference (R1 - R2)')
    axes[0, 1].set_title('Area Differences')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Intensity ratios
    axes[1, 0].scatter(slice_indices, intensity_ratios, alpha=0.6, s=30, color='red')
    axes[1, 0].set_xlabel('Slice Index')
    axes[1, 0].set_ylabel('Intensity Ratio (R1/R2)')
    axes[1, 0].set_title('Intensity Ratios')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Area ratios
    axes[1, 1].scatter(slice_indices, area_ratios, alpha=0.6, s=30, color='purple')
    axes[1, 1].set_xlabel('Slice Index')
    axes[1, 1].set_ylabel('Area Ratio (R1/R2)')
    axes[1, 1].set_title('Area Ratios')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def visualize_results(analysis_results: Dict,
                    save_dir: Optional[str] = None) -> None:
    """
    Create comprehensive visualization of analysis results.
    
    Args:
        analysis_results: Complete analysis results
        save_dir: Optional directory to save plots
    """
    logger.info("Creating comprehensive visualizations")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create plots
    plot_region_statistics(analysis_results, 
                          save_path=f"{save_dir}/region_statistics.png" if save_dir else None)
    
    plot_segmentation_statistics(analysis_results,
                                save_path=f"{save_dir}/segmentation_statistics.png" if save_dir else None)
    
    plot_comparison_results(analysis_results,
                           save_path=f"{save_dir}/comparison_results.png" if save_dir else None)
    
    logger.info("Visualization completed") 