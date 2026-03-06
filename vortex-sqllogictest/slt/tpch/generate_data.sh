#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -ex -o pipefail

if ! command -v uvx &> /dev/null; then
  echo "Error: uvx not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

mkdir -p "${DATA_DIR}"

# 1. Generate TPC-H data at scale factor 0.1
echo "Generating TPC-H data (SF=0.1)..."
uvx tpchgen-cli -s 0.1 --format=parquet --output-dir "${DATA_DIR}/" 

# 2. Convert each parquet file to Vortex format
for f in "${DATA_DIR}"/*.parquet; do
  echo "Converting $(basename "$f") to Vortex..."
  cargo run --release --package vortex-tui --bin vx -- convert "$f"
done

# 3. Remove parquet files
echo "Cleaning up parquet files..."
rm -f "${DATA_DIR}"/*.parquet

echo "Done."
