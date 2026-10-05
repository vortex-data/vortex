#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates TPC-H data at scale factor 0.1 as Parquet under slt/tpch/data/ and
# converts each table to Vortex.

set -e -o pipefail

if ! command -v uvx &> /dev/null; then
  echo "Error: uvx not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

mkdir -p "${DATA_DIR}"

echo "Generating TPC-H data (SF=0.1)..."
uvx tpchgen-cli -s 0.1 --format=parquet --parquet-compression='ZSTD(3)' --output-dir "${DATA_DIR}/"

for f in "${DATA_DIR}"/*.parquet; do
  echo "Converting $(basename "$f") to Vortex..."
  rm -f "${f%.parquet}.vortex"
  (cd "${CRATE_DIR}" && cargo run --release --package vortex-tui --bin vx -- convert "$f")
done
