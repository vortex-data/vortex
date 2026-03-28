#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run Python benchmarks with a release-profile build of the native extension.
#
# Usage:
#   ./run_benchmarks.sh                     # run all benchmarks
#   ./run_benchmarks.sh -k "test_scan"      # run benchmarks matching a pattern
#   ./run_benchmarks.sh --benchmark-only    # skip non-benchmark tests (if any)
#
# All arguments are forwarded to pytest.

set -ex -o pipefail

ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
PYTHON_DIR="$ROOT/vortex-python"

# Ensure all packages are synced (includes pytest-benchmark dev dependency).
uv sync --all-packages

source "$ROOT/.venv/bin/activate"

# Build the native extension in release mode so benchmarks reflect production performance.
maturin develop --release --manifest-path "$PYTHON_DIR/Cargo.toml"

# Run benchmarks. Extra args are forwarded to pytest.
pytest "$PYTHON_DIR/benchmark" "$@"
