"""
Main analysis module for automated microCT data processing.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

from .volume_loader import load_bmp_stack, get_volume_info, normalize_volume
from .segmentation import segment_volume, get_segmentation_stats
from .measurement import calculate_region_statistics, analyze_slice_regions

logger = logging.getLogger(__name__)

class MicroCTAnalyzer:
    """
    Main class for automated microCT analysis.
    """
    
    def __init__(self, 
                 segmentation_method: str = 'otsu',
                 min_area: int = 100,
                 normalize: bool = True,
                 normalization_method: str = 'minmax',
                 volume: np.ndarray = None
                 ):
        """
        Initialize the analyzer.
        
        Args:
            segmentation_method: Method for segmentation
            min_area: Minimum area for regions
            normalize: Whether to normalize volume
            normalization_method: Normalization method
        """
        self.segmentation_method = segmentation_method
        self.min_area = min_area
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.volume = volume
        self.labeled_volume = None
        self.results = {}
        
    def load_data(self, directory: str, **kwargs) -> None:
        """
        Load BMP stack from directory.
        
        Args:
            directory: Path to directory containing BMP files
            **kwargs: Additional arguments for load_bmp_stack
        """
        logger.info(f"Loading data from {directory}")
        self.volume = load_bmp_stack(directory, **kwargs)
        
        if self.normalize:
            logger.info("Normalizing volume")
            self.volume = normalize_volume(self.volume, self.normalization_method)
        
        volume_info = get_volume_info(self.volume)
        logger.info(f"Volume loaded: {volume_info}")
        self.results['volume_info'] = volume_info
        
    def segment_volume(self, **kwargs) -> None:
        """
        Segment the loaded volume.
        
        Args:
            **kwargs: Additional arguments for segmentation
        """
        if self.volume is None:
            raise ValueError("No volume loaded. Call load_data() first.")
        
        logger.info("Starting volume segmentation")
        self.labeled_volume = segment_volume(
            self.volume, 
            method=self.segmentation_method,
            min_area=self.min_area,
            **kwargs
        )
        
        # Calculate segmentation statistics
        segmentation_stats = []
        for i in tqdm(range(self.volume.shape[0]), desc="Analyzing segments"):
            stats = get_segmentation_stats(self.labeled_volume[i])
            stats['slice_index'] = i
            segmentation_stats.append(stats)
        
        self.results['segmentation_stats'] = segmentation_stats
        
    def analyze_regions(self, slice_range: Tuple[int, int] = None) -> None:
        """
        Analyze regions in the segmented volume.
        
        Args:
            slice_range: Range of slices to analyze (start, end)
        """
        if self.labeled_volume is None:
            raise ValueError("No segmented volume. Call segment_volume() first.")
        
        logger.info("Starting region analysis")
        region_stats = calculate_region_statistics(
            self.volume, 
            self.labeled_volume,
            slice_range=slice_range
        )
        
        self.results['region_analysis'] = region_stats
        
    def run_complete_analysis(self, 
                             directory: str,
                             slice_range: Tuple[int, int] = None,
                             **kwargs) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            directory: Path to directory containing BMP files
            slice_range: Range of slices to analyze
            **kwargs: Additional arguments for loading and segmentation
        
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("Starting complete microCT analysis")
        
        # Load data
        self.load_data(directory, **kwargs)
        
        # Segment volume
        self.segment_volume(**kwargs)
        
        # Analyze regions
        self.analyze_regions(slice_range)
        
        # Add analysis metadata
        self.results['analysis_metadata'] = {
            'segmentation_method': self.segmentation_method,
            'min_area': self.min_area,
            'normalize': self.normalize,
            'normalization_method': self.normalization_method,
            'slice_range': slice_range
        }
        
        logger.info("Analysis completed successfully")
        return self.results
    
    def save_results(self, output_path: str, format: str = 'json') -> None:
        """
        Save analysis results to file.
        
        Args:
            output_path: Path to output file
            format: Output format ('json', 'csv', 'excel')
        """
        if not self.results:
            raise ValueError("No results to save. Run analysis first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        elif format == 'csv':
            # Save summary statistics as CSV
            if 'region_analysis' in self.results:
                summary_df = pd.DataFrame([self.results['region_analysis']['cross_slice_stats']])
                summary_df.to_csv(output_path, index=False)
        elif format == 'excel':
            # Save detailed results as Excel
            with pd.ExcelWriter(output_path) as writer:
                # Summary statistics
                if 'region_analysis' in self.results:
                    summary_df = pd.DataFrame([self.results['region_analysis']['cross_slice_stats']])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Segmentation statistics
                if 'segmentation_stats' in self.results:
                    seg_df = pd.DataFrame(self.results['segmentation_stats'])
                    seg_df.to_excel(writer, sheet_name='Segmentation', index=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_summary_report(self) -> str:
        """
        Generate a text summary report.
        
        Returns:
            Formatted summary report
        """
        if not self.results:
            return "No analysis results available."
        
        report = []
        report.append("=" * 50)
        report.append("MICROCT ANALYSIS SUMMARY REPORT")
        report.append("=" * 50)
        
        # Volume information
        if 'volume_info' in self.results:
            vol_info = self.results['volume_info']
            report.append(f"\nVolume Information:")
            report.append(f"  Shape: {vol_info['shape']}")
            report.append(f"  Data type: {vol_info['dtype']}")
            report.append(f"  Value range: {vol_info['min_value']} - {vol_info['max_value']}")
            report.append(f"  Mean value: {vol_info['mean_value']:.2f}")
        
        # Segmentation statistics
        if 'segmentation_stats' in self.results:
            seg_stats = self.results['segmentation_stats']
            total_regions = sum(s['num_regions'] for s in seg_stats)
            avg_regions = total_regions / len(seg_stats) if seg_stats else 0
            report.append(f"\nSegmentation Statistics:")
            report.append(f"  Total regions found: {total_regions}")
            report.append(f"  Average regions per slice: {avg_regions:.1f}")
        
        # Region analysis
        if 'region_analysis' in self.results:
            region_stats = self.results['region_analysis']['cross_slice_stats']
            report.append(f"\nRegion Analysis:")
            report.append(f"  Slices analyzed: {region_stats['total_slices_analyzed']}")
            report.append(f"  Total regions: {region_stats['total_regions']}")
            report.append(f"  Mean regions per slice: {region_stats['mean_regions_per_slice']:.1f}")
            
            if 'area_stats' in region_stats:
                area_stats = region_stats['area_stats']
                report.append(f"  Area statistics:")
                report.append(f"    Mean: {area_stats['mean']:.1f}")
                report.append(f"    Std: {area_stats['std']:.1f}")
                report.append(f"    Range: {area_stats['min']:.1f} - {area_stats['max']:.1f}")
        
        report.append("\n" + "=" * 50)
        return "\n".join(report)

def analyze_volume(volume: np.ndarray,
                  segmentation_method: str = 'otsu',
                  min_area: int = 100,
                  slice_range: Tuple[int, int] = None) -> Dict:
    """
    Convenience function for quick volume analysis.
    
    Args:
        volume: 3D numpy array
        segmentation_method: Segmentation method
        min_area: Minimum area for regions
        slice_range: Range of slices to analyze
    
    Returns:
        Analysis results dictionary
    """
    analyzer = MicroCTAnalyzer(
        segmentation_method=segmentation_method,
        min_area=min_area
    )
    
    analyzer.volume = volume
    analyzer.segment_volume()
    analyzer.analyze_regions(slice_range)
    
    return analyzer.results 