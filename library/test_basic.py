#!/usr/bin/env python3
"""
Basic test script for the microCT analysis package.
"""

import os
import numpy as np
import tempfile
import shutil
from PIL import Image

def create_test_data():
    """Create test BMP files."""
    test_dir = tempfile.mkdtemp(prefix="microct_test_")
    
    # Create test images
    for i in range(5):
        # Create a simple test image
        img = np.zeros((64, 64), dtype=np.uint8)
        
        # Add some regions
        y, x = np.ogrid[:64, :64]
        
        # Circle 1
        mask1 = (x - 20)**2 + (y - 20)**2 <= 10**2
        img[mask1] = 150
        
        # Circle 2
        mask2 = (x - 44)**2 + (y - 44)**2 <= 12**2
        img[mask2] = 200
        
        # Rectangle
        img[25:35, 25:35] = 100
        
        # Save as BMP
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(test_dir, f"test_slice_{i:03d}.bmp"))
    
    return test_dir

def test_basic_functionality():
    """Test basic package functionality."""
    print("Testing basic functionality...")
    
    # Create test data
    print("Creating test data...")
    test_dir = create_test_data()
    
    try:
        # Import the package
        from automct import (
            load_bmp_stack,
            segment_slice,
            measure_regions,
            MicroCTAnalyzer
        )
        
        # Test 1: Load BMP stack
        print("  ✓ Testing BMP stack loading...")
        volume = load_bmp_stack(test_dir)
        assert volume.shape[0] == 5, f"Expected 5 slices, got {volume.shape[0]}"
        print(f"    Loaded volume with shape: {volume.shape}")
        
        # Test 2: Segment a slice
        print("  ✓ Testing segmentation...")
        slice_0 = volume[0]
        labeled = segment_slice(slice_0, method='otsu', min_area=10)
        print(f"    Found {labeled.max()} regions in slice 0")
        
        # Test 3: Measure regions
        print("  ✓ Testing region measurement...")
        regions = measure_regions(slice_0, labeled)
        print(f"    Measured {len(regions)} regions")
        
        # Test 4: Analyzer class
        print("  ✓ Testing analyzer class...")
        analyzer = MicroCTAnalyzer(
            segmentation_method='otsu',
            min_area=10,
            normalize=False
        )
        
        results = analyzer.run_complete_analysis(
            directory=test_dir,
            slice_range=(0, 3)  # Test with 3 slices
        )
        
        assert 'volume_info' in results, "Missing volume_info in results"
        assert 'segmentation_stats' in results, "Missing segmentation_stats in results"
        assert 'region_analysis' in results, "Missing region_analysis in results"
        
        print("    Analyzer completed successfully")
        
        # Test 5: Save results
        print("  ✓ Testing result saving...")
        analyzer.save_results("./test_results.json", format='json')
        assert os.path.exists("test_results.json"), "Results file not created"
        print("    Results saved successfully")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        if os.path.exists("test_results.json"):
            os.remove("test_results.json")

def test_cli():
    """Test command line interface."""
    print("\nTesting CLI...")
    
    # Create test data
    test_dir = create_test_data()
    
    try:
        import subprocess
        import sys
        
        # Test CLI with test data
        cmd = [
            sys.executable, "-m", "automct.cli",
            test_dir,
            "--output", "./cli_test_results.json",
            "--slice-range", "0", "3",
            "--summary-report"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ CLI test passed!")
            print("  Output:", result.stdout)
            return True
        else:
            print("  ❌ CLI test failed!")
            print("  Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        if os.path.exists("cli_test_results.json"):
            os.remove("cli_test_results.json")

def main():
    """Run all tests."""
    print("MicroCT Analysis Package - Basic Tests")
    print("=" * 50)
    
    # Test basic functionality
    basic_success = test_basic_functionality()
    
    # Test CLI
    cli_success = test_cli()

    
    print("\n" + "=" * 50)
    if basic_success and cli_success:
        print("🎉 All tests passed! The package is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    print("\nTo run the full example:")
    print("python example_usage.py")

if __name__ == '__main__':
    main() 