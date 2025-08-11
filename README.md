# MicroCT Analysis Package

Automated layer-by-layer analysis of microCT data from BMP files using Python and open-source tools.

## 🚀 Features

- **Volume Reconstruction**: Convert BMP file stacks into 3D volumes
- **Multiple Segmentation Methods**: Otsu, local thresholding, watershed, and manual thresholding
- **Region Analysis**: Comprehensive measurement of region properties (area, intensity, shape, etc.)
- **Region Comparison**: Compare two regions per slice with various metrics
- **Visualization**: Generate plots and 3D visualizations
- **Command Line Interface**: Easy-to-use CLI for batch processing
- **Export Results**: Save results in JSON, CSV, or Excel formats

## 📦 Installation

### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate microct-analysis
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv microct-env
source microct-env/bin/activate  # On Windows: microct-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from microct_analysis import MicroCTAnalyzer

# Create analyzer
analyzer = MicroCTAnalyzer(
    segmentation_method='otsu',
    min_area=100,
    normalize=True
)

# Run complete analysis
results = analyzer.run_complete_analysis(
    directory="/path/to/bmp/files",
    slice_range=(0, 50)  # Analyze first 50 slices
)

# Save results
analyzer.save_results("results.json", format='json')

# Print summary
print(analyzer.get_summary_report())
```

### Command Line Interface

```bash
# Basic analysis
python cli.py /path/to/bmp/files --output results.json

# Analysis with custom parameters
python cli.py /path/to/bmp/files \
    --segmentation-method watershed \
    --min-area 200 \
    --output results.json

# Analysis with visualization
python cli.py /path/to/bmp/files \
    --output results.json \
    --visualize \
    --save-plots plots/

# Analysis of specific slice range
python cli.py /path/to/bmp/files \
    --slice-range 10 50 \
    --output results.json
```

## 📚 Detailed Usage

### Step-by-Step Workflow

```python
from microct_analysis import (
    load_bmp_stack, 
    segment_slice, 
    measure_regions,
    compare_two_regions
)

# 1. Load BMP stack
volume = load_bmp_stack("/path/to/bmp/files")
print(f"Volume shape: {volume.shape}")

# 2. Segment a slice
slice_0 = volume[0]
labeled_slice = segment_slice(slice_0, method='otsu', min_area=100)

# 3. Measure regions
regions = measure_regions(slice_0, labeled_slice)
print(f"Found {len(regions)} regions")

# 4. Compare regions
if len(regions) >= 2:
    comparison = compare_two_regions(regions)
    print(f"Intensity difference: {comparison['intensity_comparison']['difference']}")
```

### Segmentation Methods

The package supports multiple segmentation methods:

- **Otsu**: Automatic thresholding based on image histogram
- **Local**: Adaptive thresholding for varying illumination
- **Watershed**: Watershed segmentation for complex regions
- **Manual**: User-defined threshold value

```python
# Different segmentation methods
labeled_otsu = segment_slice(slice_2d, method='otsu')
labeled_local = segment_slice(slice_2d, method='local', block_size=35)
labeled_watershed = segment_slice(slice_2d, method='watershed', min_distance=10)
labeled_manual = segment_slice(slice_2d, method='manual', threshold=128)
```

### Region Measurements

Each region is measured for various properties:

- **Area**: Number of pixels in the region
- **Intensity**: Mean, min, max, and standard deviation
- **Shape**: Eccentricity, solidity, perimeter
- **Position**: Centroid coordinates, bounding box
- **Orientation**: Major/minor axis length, orientation angle

### Visualization

```python
from microct_analysis import visualize_results

# Create comprehensive visualizations
visualize_results(results, save_dir="plots/")

# 3D visualization (requires napari)
from microct_analysis import create_3d_visualization
create_3d_visualization(volume, labeled_volume)
```

## 📊 Output Formats

### JSON Output
```json
{
  "volume_info": {
    "shape": [100, 512, 512],
    "dtype": "uint8",
    "min_value": 0,
    "max_value": 255
  },
  "segmentation_stats": [...],
  "region_analysis": {
    "cross_slice_stats": {...},
    "slice_analyses": [...]
  }
}
```

### Excel Output
- **Summary Sheet**: Cross-slice statistics
- **Segmentation Sheet**: Per-slice segmentation statistics
- **Regions Sheet**: Detailed region measurements

## 🔧 Configuration

### Analysis Parameters

```python
analyzer = MicroCTAnalyzer(
    segmentation_method='otsu',      # Segmentation method
    min_area=100,                   # Minimum region area
    normalize=True,                  # Normalize volume
    normalization_method='minmax'    # Normalization method
)
```

### CLI Options

```bash
python cli.py --help
```

Available options:
- `--segmentation-method`: otsu, local, watershed, manual
- `--min-area`: Minimum area for regions
- `--slice-range`: Range of slices to analyze
- `--normalize`: Normalize volume
- `--visualize`: Create visualizations
- `--save-plots`: Directory to save plots
- `--verbose`: Verbose output

## 📁 Project Structure

```
MicroCT-Analysis/
├── microct_analysis/          # Main package
│   ├── __init__.py
│   ├── volume_loader.py       # BMP loading utilities
│   ├── segmentation.py        # Segmentation methods
│   ├── measurement.py         # Region measurement
│   ├── analysis.py           # Main analysis class
│   └── visualization.py      # Plotting utilities
├── cli.py                    # Command-line interface
├── example_usage.py          # Usage examples
├── environment.yml           # Conda environment
├── requirements.txt          # pip requirements
└── README.md               # This file
```

## 🧪 Examples

Run the example script to see the package in action:

```bash
python example_usage.py
```

This will:
1. Create sample BMP files
2. Demonstrate basic workflow
3. Show step-by-step analysis
4. Compare segmentation methods
5. Run custom analysis

## 🔬 Scientific Applications

This package is designed for:

- **Tissue Analysis**: Quantify tissue regions in medical images
- **Material Science**: Analyze material structure and porosity
- **Biological Research**: Study cellular structures and distributions
- **Quality Control**: Automated inspection of manufactured parts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **scikit-image**: Image processing and segmentation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **napari**: 3D visualization
- **Pillow**: Image I/O

## 📞 Support

For questions or issues:
1. Check the examples in `example_usage.py`
2. Review the CLI help: `python cli.py --help`
3. Open an issue on GitHub

## 🔄 Version History

- **v1.0.0**: Initial release with basic functionality
  - BMP stack loading
  - Multiple segmentation methods
  - Region measurement and comparison
  - Visualization tools
  - Command-line interface 