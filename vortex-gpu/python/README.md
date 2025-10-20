# Vortex GPU Benchmarks

Python benchmarking scripts for evaluating GPU performance using cuDF.

## Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x (or adjust cudf version in pyproject.toml for your CUDA version)
- Python 3.9+
- conda (recommended for cuDF installation)

### Installation

#### Option 1: Using conda (Recommended)

cuDF is easiest to install via conda:

```bash
# Create a new conda environment
conda create -n vortex-gpu python=3.11

# Activate the environment
conda activate vortex-gpu

# Install cuDF and dependencies
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.10 python=3.11 cuda-version=12.0

# Install any additional dependencies
pip install pyarrow
```

#### Option 2: Using pip with CUDA 12.x

```bash
cd vortex-gpu/python

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

**Note:** pip installation of cuDF requires matching CUDA toolkit version. See [RAPIDS installation guide](https://docs.rapids.ai/install) for details.

## Usage

### cuDF Benchmark Script

Reads a Parquet file into a GPU-backed cuDF DataFrame and runs a simple arithmetic query (`x + 10`).

```bash
# Basic usage (assumes column named 'x')
python cudf_benchmark.py /path/to/data.parquet

# Specify a different column
python cudf_benchmark.py /path/to/data.parquet --column my_column

# Run multiple iterations for better statistics
python cudf_benchmark.py /path/to/data.parquet --column x --iterations 10

# Show help
python cudf_benchmark.py --help
```

### Example Output

```
Reading Parquet file: data.parquet
Read time: 0.123456 seconds
DataFrame shape: (1000000, 3)
Columns: ['x', 'y', 'z']

Column 'x' dtype: int64
Column 'x' shape: (1000000,)

Query execution time (iteration 1): 0.000234 seconds

First 10 values of original column 'x':
0    1
1    2
2    3
...

First 10 values after adding 10:
0    11
1    12
2    13
...

============================================================
Total time (read + query): 0.123690 seconds
============================================================
```

## Troubleshooting

### CUDA Version Mismatch

If you get CUDA-related errors, ensure your cuDF version matches your CUDA toolkit:

- CUDA 11.x: Use `cudf-cu11`
- CUDA 12.x: Use `cudf-cu12`

Update the dependency in `pyproject.toml` accordingly.

### GPU Not Detected

Verify your GPU is accessible:

```python
import cudf
print(cudf.Series([1, 2, 3]))  # Should work without errors
```

### Memory Issues

For large Parquet files, you may need to:
- Use a GPU with more memory
- Read the file in chunks
- Filter columns during read: `cudf.read_parquet(file, columns=['x'])`
