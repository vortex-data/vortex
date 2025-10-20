#!/bin/bash
# Setup script for vortex-gpu Python benchmarks

set -e

echo "Setting up vortex-gpu Python benchmark environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ conda found"
    echo ""
    echo "Recommended setup using conda:"
    echo "  conda create -n vortex-gpu python=3.11"
    echo "  conda activate vortex-gpu"
    echo "  conda install -c rapidsai -c conda-forge -c nvidia cudf=24.10 python=3.11 cuda-version=12.0"
    echo "  pip install pyarrow"
    echo ""
    read -p "Would you like to create the conda environment now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda create -n vortex-gpu python=3.11 -y
        echo ""
        echo "Environment created! Activate it with:"
        echo "  conda activate vortex-gpu"
        echo ""
        echo "Then install cuDF:"
        echo "  conda install -c rapidsai -c conda-forge -c nvidia cudf=24.10 python=3.11 cuda-version=12.0"
    fi
else
    echo "⚠ conda not found"
    echo ""
    echo "Setting up with pip (requires CUDA toolkit already installed)..."
    echo ""

    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi

    echo "Activating virtual environment..."
    source .venv/bin/activate

    echo "Installing dependencies..."
    pip install -e .

    echo ""
    echo "✓ Setup complete!"
    echo ""
    echo "Activate the environment with:"
    echo "  source .venv/bin/activate"
fi

echo ""
echo "Once setup is complete, run the benchmark with:"
echo "  python cudf_benchmark.py <path_to_parquet_file>"
