#!/bin/bash

# MicroCT Analysis Package - Installation Script
# This script automates the installation and testing of the microCT analysis package

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "\n${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "\n${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "\n${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "\n${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    local python_cmd="$1"
    local version=$($python_cmd --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)
    
    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
        return 0
    else
        return 1
    fi
}

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "linux";;
        Darwin*)    echo "macos";;
        CYGWIN*)    echo "windows";;
        MINGW*)     echo "windows";;
        *)          echo "unknown";;
    esac
}

# Function to install conda if not present
install_conda() {
    local os=$(detect_os)
    
    print_status "Installing Miniconda..."
    
    if [ "$os" = "linux" ]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    elif [ "$os" = "macos" ]; then
        if [ "$(uname -m)" = "arm64" ]; then
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O miniconda.sh
        else
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
        fi
    else
        print_error "Unsupported OS for automatic conda installation"
        return 1
    fi
    
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Add conda to PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    
    print_success "Miniconda installed successfully"
}

# Function to create conda environment
create_conda_env() {
    print_status "Creating conda environment..."
    
    if conda env list | grep -q "automct"; then
    print_warning "Environment 'automct' already exists."
        echo -n "Do you want to update the existing environment? (y/n): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            print_status "Updating existing environment..."
            conda env update -f environment.yml
            print_success "Conda environment updated successfully"
        else
            print_status "Skipping environment update. Using existing environment."
        fi
    else
        conda env create -f environment.yml
        print_success "Conda environment created successfully"
    fi
}

# Function to install with pip
install_with_pip() {
    print_status "Installing with pip..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "microct-env" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv microct-env
    fi
    
    # Activate virtual environment
    source microct-env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    # Install package in development mode
    pip install -e .
    
    print_success "Pip installation completed successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Check if we're in a conda environment
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        print_status "Running tests in conda environment: $CONDA_DEFAULT_ENV"
        python test_basic.py
    else
        # Try to activate virtual environment
        if [ -d "microct-env" ]; then
            source microct-env/bin/activate
            print_status "Running tests in virtual environment"
            python test_basic.py
        else
            print_error "No environment found. Please run installation first."
            return 1
        fi
    fi
    
    print_success "Tests completed successfully"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --conda         Use conda installation (default)"
    echo "  -p, --pip           Use pip installation"
    echo "  -t, --test-only     Run tests only (skip installation)"
    echo "  -f, --force         Force reinstallation"
    echo "  -v, --verbose       Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                  # Install with conda and run tests"
    echo "  $0 --pip            # Install with pip and run tests"
    echo "  $0 --test-only      # Run tests only"
}

# Main installation function
main() {
    local install_method="conda"
    local test_only=false
    local force=false
    local verbose=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -c|--conda)
                install_method="conda"
                shift
                ;;
            -p|--pip)
                install_method="pip"
                shift
                ;;
            -t|--test-only)
                test_only=true
                shift
                ;;
            -f|--force)
                force=true
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
    
    print_status "MicroCT Analysis Package - Installation Script"
    echo "=================================================="
    
    # Check if we're in the right directory
    if [ ! -f "setup.py" ] || [ ! -f "requirements.txt" ]; then
        print_error "Please run this script from the installation directory (installation/)"
        exit 1
    fi
    
    if [ "$test_only" = true ]; then
        run_tests
        exit 0
    fi
    
    # Check Python availability
    if ! command_exists python3; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    if ! check_python_version python3; then
        print_error "Python 3.8 or higher is required"
        exit 1
    fi
    
    print_success "Python version check passed"
    
    # Installation based on method
    if [ "$install_method" = "conda" ]; then
        # Check if conda is available
        if ! command_exists conda; then
            print_warning "Conda not found. Installing Miniconda..."
            install_conda
        fi
        
        print_status "Using conda installation method"
        create_conda_env
        
        # Activate environment
        print_status "Activating conda environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate automct
        
        # Install package in development mode
        pip install -e .
        
    else
        print_status "Using pip installation method"
        install_with_pip
    fi
    
    # Run tests
    run_tests
    
    print_success "Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment:"
    if [ "$install_method" = "conda" ]; then
        echo "   conda activate automct"
    else
        echo "   source microct-env/bin/activate"
    fi
    echo "2. Run the example: python example_usage.py"
    echo "3. Use the CLI: python cli.py --help"
    echo ""
    print_success "Happy analyzing! 🎉"
}

# Run main function with all arguments
main "$@" 