# Installation Guide

This guide provides detailed instructions for setting up the MicroCT Analysis package.

## Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip
- Git (for cloning the repository)

## Installation Methods

### Method 1: Conda Environment (Recommended)

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd MicroCT-Analysis
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate automct
   ```

3. **Verify installation**:
   ```bash
   python test_basic.py
   ```

### Method 2: pip Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv microct-env
   source microct-env/bin/activate  # On Windows: microct-env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python test_basic.py
   ```

### Method 3: Development Installation

1. **Clone and install in development mode**:
   ```bash
   git clone <repository-url>
   cd MicroCT-Analysis
   pip install -e .
   ```

2. **Install development dependencies** (optional):
   ```bash
   pip install -e ".[dev]"
   ```

## Environment Setup

### Required Dependencies

The package requires the following key dependencies:

- **numpy**: Numerical computations
- **scikit-image**: Image processing and segmentation
- **Pillow**: Image I/O
- **matplotlib/seaborn**: Visualization
- **pandas**: Data manipulation
- **napari**: 3D visualization (optional)
- **tqdm**: Progress bars

### Optional Dependencies

- **opencv-python**: Additional image processing
- **plotly**: Interactive plots
- **ipywidgets**: Jupyter widgets

## Platform-Specific Notes

### macOS

- Conda is the recommended installation method
- All dependencies should work out of the box

### Linux

- May need to install system-level dependencies:
  ```bash
  sudo apt-get install libgl1-mesa-glx libglib2.0-0
  ```

### Windows

- Use conda for best compatibility
- If using pip, ensure you have the Microsoft Visual C++ Build Tools

## Verification

After installation, run the test script to verify everything works:

```bash
python test_basic.py
```

You should see output like:
```
MicroCT Analysis Package - Basic Tests
==================================================
Testing basic functionality...
  ✓ Testing BMP stack loading...
    Loaded volume with shape: (5, 64, 64)
  ✓ Testing segmentation...
    Found 3 regions in slice 0
  ✓ Testing region measurement...
    Measured 3 regions
  ✓ Testing analyzer class...
    Analyzer completed successfully
  ✓ Testing result saving...
    Results saved successfully

✅ All basic tests passed!

Testing CLI...
  ✅ CLI test passed!

==================================================
🎉 All tests passed! The package is working correctly.
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the conda environment is activated
2. **Missing dependencies**: Reinstall with `pip install -r requirements.txt`
3. **Permission errors**: Use `sudo` for system-wide installation or use virtual environments
4. **napari issues**: Install separately with `pip install napari`

### Getting Help

1. Check the test output for specific error messages
2. Verify your Python version: `python --version`
3. Check installed packages: `pip list`
4. Try reinstalling the environment from scratch

## Next Steps

After successful installation:

1. **Run the example script**:
   ```bash
   python example_usage.py
   ```

2. **Try the CLI**:
   ```bash
   python cli.py --help
   ```

3. **Read the documentation** in `README.md`

4. **Start analyzing your data**:
   ```bash
   python cli.py /path/to/your/bmp/files --output results.json
   ``` 