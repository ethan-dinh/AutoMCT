#!/bin/bash

# MicroCT Analysis Package - Distribution Script
# This script builds and packages the microCT analysis package for distribution

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if we're in a virtual environment
check_environment() {
    if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
        return 0
    else
        return 1
    fi
}

# Function to clean previous builds
clean_build() {
    print_status "Cleaning previous builds..."
    
    # Remove build artifacts
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf __pycache__/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Build cleaned"
}

# Function to run tests before building
run_tests() {
    print_status "Running tests before distribution..."
    
    if [ -f "test_basic.py" ]; then
        python test_basic.py
        print_success "Tests passed"
    else
        print_warning "No test file found, skipping tests"
    fi
}

# Function to build source distribution
build_sdist() {
    print_status "Building source distribution..."
    
    python setup.py sdist
    
    # Check if build was successful
    if [ -d "dist" ] && [ "$(ls -A dist/*.tar.gz 2>/dev/null)" ]; then
        print_success "Source distribution created"
        ls -la dist/*.tar.gz
    else
        print_error "Source distribution build failed"
        return 1
    fi
}

# Function to build wheel distribution
build_wheel() {
    print_status "Building wheel distribution..."
    
    # Check if wheel is available
    if ! python -c "import wheel" 2>/dev/null; then
        print_status "Installing wheel..."
        pip install wheel
    fi
    
    python setup.py bdist_wheel
    
    # Check if build was successful
    if [ -d "dist" ] && [ "$(ls -A dist/*.whl 2>/dev/null)" ]; then
        print_success "Wheel distribution created"
        ls -la dist/*.whl
    else
        print_error "Wheel distribution build failed"
        return 1
    fi
}

# Function to check distribution files
check_distribution() {
    print_status "Checking distribution files..."
    
    if [ ! -d "dist" ]; then
        print_error "No dist directory found"
        return 1
    fi
    
    local files=($(ls dist/))
    if [ ${#files[@]} -eq 0 ]; then
        print_error "No distribution files found"
        return 1
    fi
    
    print_success "Distribution files found:"
    for file in "${files[@]}"; do
        echo "  - $file ($(du -h "dist/$file" | cut -f1))"
    done
}

# Function to install and test distribution
test_distribution() {
    print_status "Testing distribution installation..."
    
    # Create temporary directory for testing
    local test_dir=$(mktemp -d)
    cd "$test_dir"
    
    # Copy distribution files
    cp -r "$OLDPWD/dist" .
    
    # Create virtual environment for testing
    python -m venv test_env
    source test_env/bin/activate
    
    # Install from distribution
    pip install dist/*.whl || pip install dist/*.tar.gz
    
    # Test import
    python -c "import microct_analysis; print('Import successful')"
    
    # Test CLI
    if command_exists microct-analyze; then
        microct-analyze --help > /dev/null
        print_success "CLI test passed"
    else
        print_warning "CLI not found in PATH"
    fi
    
    # Cleanup
    deactivate
    cd "$OLDPWD"
    rm -rf "$test_dir"
    
    print_success "Distribution test completed"
}

# Function to create release notes
create_release_notes() {
    print_status "Creating release notes..."
    
    local version=$(python setup.py --version 2>/dev/null || echo "1.0.0")
    local date=$(date +"%Y-%m-%d")
    
    cat > "RELEASE_NOTES.md" << EOF
# MicroCT Analysis Package v$version

Release Date: $date

## What's New

- Automated layer-by-layer analysis of microCT data
- Multiple segmentation methods (Otsu, local, watershed, manual)
- Comprehensive region measurement and comparison
- 3D visualization capabilities
- Command-line interface for batch processing
- Export results in JSON, CSV, and Excel formats

## Installation

### From Source
\`\`\`bash
pip install dist/microct_analysis-*.whl
\`\`\`

### From PyPI (when available)
\`\`\`bash
pip install microct-analysis
\`\`\`

## Usage

\`\`\`python
from microct_analysis import MicroCTAnalyzer

analyzer = MicroCTAnalyzer()
results = analyzer.run_complete_analysis("/path/to/bmp/files")
\`\`\`

## Command Line

\`\`\`bash
microct-analyze /path/to/bmp/files --output results.json --visualize
\`\`\`

## Requirements

- Python 3.8+
- See requirements.txt for full dependency list

## Documentation

See README.md for detailed documentation and examples.
EOF
    
    print_success "Release notes created: RELEASE_NOTES.md"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean previous builds only"
    echo "  -t, --test          Run tests only"
    echo "  -s, --sdist         Build source distribution only"
    echo "  -w, --wheel         Build wheel distribution only"
    echo "  -a, --all           Build all distributions (default)"
    echo "  -i, --install-test  Test installation from distribution"
    echo "  -r, --release       Create release notes"
    echo "  -v, --verbose       Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build all distributions"
    echo "  $0 --clean          # Clean previous builds"
    echo "  $0 --test           # Run tests only"
    echo "  $0 --install-test   # Test distribution installation"
}

# Main distribution function
main() {
    local clean_only=false
    local test_only=false
    local sdist_only=false
    local wheel_only=false
    local all_distributions=true
    local install_test=false
    local create_release=false
    local verbose=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -c|--clean)
                clean_only=true
                shift
                ;;
            -t|--test)
                test_only=true
                shift
                ;;
            -s|--sdist)
                sdist_only=true
                all_distributions=false
                shift
                ;;
            -w|--wheel)
                wheel_only=true
                all_distributions=false
                shift
                ;;
            -a|--all)
                all_distributions=true
                shift
                ;;
            -i|--install-test)
                install_test=true
                shift
                ;;
            -r|--release)
                create_release=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set verbose mode
    if [ "$verbose" = true ]; then
        set -x
    fi
    
    print_status "MicroCT Analysis Package - Distribution Script"
    echo "=================================================="
    
    # Check if we're in the right directory
    if [ ! -f "setup.py" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check if we're in a virtual environment
    if ! check_environment; then
        print_warning "Not in a virtual environment. Consider activating one."
    fi
    
    # Clean builds if requested
    if [ "$clean_only" = true ]; then
        clean_build
        exit 0
    fi
    
    # Run tests if requested
    if [ "$test_only" = true ]; then
        run_tests
        exit 0
    fi
    
    # Clean previous builds
    clean_build
    
    # Run tests before building
    run_tests
    
    # Build distributions
    if [ "$all_distributions" = true ] || [ "$sdist_only" = true ]; then
        build_sdist
    fi
    
    if [ "$all_distributions" = true ] || [ "$wheel_only" = true ]; then
        build_wheel
    fi
    
    # Check distribution files
    check_distribution
    
    # Test installation if requested
    if [ "$install_test" = true ]; then
        test_distribution
    fi
    
    # Create release notes if requested
    if [ "$create_release" = true ]; then
        create_release_notes
    fi
    
    print_success "Distribution build completed successfully!"
    echo ""
    echo "Distribution files created in: dist/"
    echo ""
    echo "Next steps:"
    echo "1. Test the distribution: $0 --install-test"
    echo "2. Upload to PyPI (when ready): twine upload dist/*"
    echo "3. Create a release on GitHub"
    echo ""
    print_success "Distribution ready! 📦"
}

# Run main function with all arguments
main "$@" 