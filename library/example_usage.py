#!/usr/bin/env python3
"""
Example usage of the microCT analysis package.

This script demonstrates how to use the package for automated
layer-by-layer analysis of microCT data from BMP files.
"""

import os
import numpy as np
import logging

# Import the analysis package
from microct_analysis import (
    MicroCTAnalyzer, 
    load_bmp_stack, 
    segment_slice, 
    measure_regions,
    compare_two_regions,
    visualize_results
)

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def example_basic_workflow():
    """
    Example 1: Basic workflow using the MicroCTAnalyzer class.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Workflow")
    print("=" * 60)
    
    # Create analyzer
    analyzer = MicroCTAnalyzer(
        segmentation_method='otsu',
        min_area=100,
        normalize=True
    )
    
    # Example directory (replace with your actual path)
    data_directory = "/path/to/your/bmp/files"
    
    if os.path.exists(data_directory):
        # Run complete analysis
        results = analyzer.run_complete_analysis(
            directory=data_directory,
            slice_range=(0, 10)  # Analyze first 10 slices
        )
        
        # Print summary report
        print(analyzer.get_summary_report())
        
        # Save results
        analyzer.save_results("results.json", format='json')
        
        # Create visualizations
        visualize_results(results, save_dir="plots/")
    else:
        print(f"Data directory not found: {data_directory}")
        print("Please update the data_directory path in the script.")

def example_step_by_step():
    """
    Example 2: Step-by-step workflow for more control.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Step-by-Step Workflow")
    print("=" * 60)
    
    # Example directory (replace with your actual path)
    data_directory = "/path/to/your/bmp/files"
    
    if os.path.exists(data_directory):
        # Step 1: Load BMP stack
        print("Loading BMP stack...")
        volume = load_bmp_stack(data_directory)
        print(f"Volume shape: {volume.shape}")
        
        # Step 2: Segment a single slice
        print("\nSegmenting slice 0...")
        slice_0 = volume[0]
        labeled_slice = segment_slice(slice_0, method='otsu', min_area=100)
        print(f"Found {labeled_slice.max()} regions in slice 0")
        
        # Step 3: Measure regions
        print("\nMeasuring regions...")
        regions = measure_regions(slice_0, labeled_slice)
        print(f"Measured {len(regions)} regions")
        
        # Step 4: Compare regions
        if len(regions) >= 2:
            comparison = compare_two_regions(regions)
            if comparison:
                print("\nRegion comparison:")
                print(f"  Intensity difference: {comparison['intensity_comparison']['difference']:.2f}")
                print(f"  Area difference: {comparison['area_comparison']['area_difference']:.2f}")
        
        # Step 5: Analyze multiple slices
        print("\nAnalyzing multiple slices...")
        for i in range(min(5, volume.shape[0])):
            slice_i = volume[i]
            labeled_i = segment_slice(slice_i, method='otsu', min_area=100)
            regions_i = measure_regions(slice_i, labeled_i)
            print(f"Slice {i}: {len(regions_i)} regions")
    else:
        print(f"Data directory not found: {data_directory}")
        print("Please update the data_directory path in the script.")

def example_different_segmentation_methods():
    """
    Example 3: Comparing different segmentation methods.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Different Segmentation Methods")
    print("=" * 60)
    
    # Example directory (replace with your actual path)
    data_directory = "/path/to/your/bmp/files"
    
    if os.path.exists(data_directory):
        # Load a single slice for comparison
        volume = load_bmp_stack(data_directory)
        slice_0 = volume[0]
        
        # Test different segmentation methods
        methods = ['otsu', 'local', 'watershed']
        
        for method in methods:
            print(f"\nTesting {method} segmentation:")
            labeled = segment_slice(slice_0, method=method, min_area=100)
            regions = measure_regions(slice_0, labeled)
            print(f"  Found {len(regions)} regions")
            if regions:
                areas = [r['area'] for r in regions]
                print(f"  Mean area: {np.mean(areas):.1f}")
                print(f"  Total area: {sum(areas)}")
    else:
        print(f"Data directory not found: {data_directory}")
        print("Please update the data_directory path in the script.")

def example_custom_analysis():
    """
    Example 4: Custom analysis with specific parameters.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Analysis")
    print("=" * 60)
    
    # Create analyzer with custom parameters
    analyzer = MicroCTAnalyzer(
        segmentation_method='watershed',
        min_area=200,
        normalize=True,
        normalization_method='histogram'
    )
    
    # Example directory (replace with your actual path)
    data_directory = "/path/to/your/bmp/files"
    
    if os.path.exists(data_directory):
        # Run analysis with custom parameters
        results = analyzer.run_complete_analysis(
            directory=data_directory,
            slice_range=(5, 15)  # Analyze slices 5-15
        )
        
        # Extract specific statistics
        if 'region_analysis' in results:
            region_stats = results['region_analysis']['cross_slice_stats']
            print(f"Total regions found: {region_stats['total_regions']}")
            print(f"Mean regions per slice: {region_stats['mean_regions_per_slice']:.1f}")
            
            if 'area_stats' in region_stats:
                area_stats = region_stats['area_stats']
                print(f"Mean region area: {area_stats['mean']:.1f}")
                print(f"Area standard deviation: {area_stats['std']:.1f}")
        
        # Save results in different formats
        analyzer.save_results("results_custom.json", format='json')
        analyzer.save_results("results_custom.xlsx", format='excel')
        
    else:
        print(f"Data directory not found: {data_directory}")
        print("Please update the data_directory path in the script.")

def create_sample_data():
    """
    Create sample BMP files for testing (if no real data is available).
    """
    print("\n" + "=" * 60)
    print("Creating Sample Data")
    print("=" * 60)
    
    # Create sample directory
    sample_dir = "sample_data"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample BMP files
    for i in range(10):
        # Create a simple test image with some regions
        img = np.zeros((100, 100), dtype=np.uint8)
        
        # Add some circular regions
        y, x = np.ogrid[:100, :100]
        
        # Region 1: Circle in top-left
        mask1 = (x - 25)**2 + (y - 25)**2 <= 15**2
        img[mask1] = 150
        
        # Region 2: Circle in bottom-right
        mask2 = (x - 75)**2 + (y - 75)**2 <= 20**2
        img[mask2] = 200
        
        # Region 3: Rectangle in center
        img[40:60, 40:60] = 100
        
        # Save as BMP
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(sample_dir, f"slice_{i:03d}.bmp"))
    
    print(f"Created {sample_dir} directory with sample BMP files")
    return sample_dir

def main():
    """Run all examples."""
    setup_logging()
    
    print("MicroCT Analysis Package - Example Usage")
    print("=" * 60)
    
    # Create sample data if no real data is available
    sample_dir = create_sample_data()
    
    # Update the data directory to use sample data
    data_directory = sample_dir
    
    # Run examples
    example_basic_workflow()
    example_step_by_step()
    example_different_segmentation_methods()
    example_custom_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nTo use with your own data:")
    print("1. Replace the data_directory path in the examples")
    print("2. Run: python example_usage.py")
    print("3. Or use the CLI: python cli.py /path/to/bmp/files --output results.json")

if __name__ == '__main__':
    main() 