"""
MicroCT Analysis Package

Automated layer-by-layer analysis of microCT data from BMP files.
"""

__version__ = "1.0.0"
__author__ = "Ethan Dinh DDS/PhD Student, UCSF"

from .volume_loader import load_bmp_stack
from .segmentation import segment_slice, segment_volume
from .cmpr import cmpr_pipeline, analyze_incisor_volume, compute_centerline
from .measurement import measure_regions, compare_two_regions
from .analysis import analyze_volume, MicroCTAnalyzer
from .visualization import visualize_results

__all__ = [
    'load_bmp_stack',
    'segment_slice', 
    'segment_volume',
    'cmpr_pipeline',
    'analyze_incisor_volume',
    'compute_centerline',
    'measure_regions',
    'compare_two_regions',
    'analyze_volume',
    'MicroCTAnalyzer',
    'visualize_results'
] 