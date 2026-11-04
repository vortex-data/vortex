#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Writes a 1% proportional sample of the Join Order Benchmark (IMDB) tables to
# slt/job/data/ with sample_data.py, which reads only the leading row groups of
# the ~1.8 GB of Parquet files that vortex-bench downloads. Converts every table
# to Vortex and checks that the Parquet and Vortex tables hold identical data
# before the Vortex files are used by the SLT suite. Set VORTEX_SLT_PROFILE to
# reuse an existing cargo profile for conversion and the parity test (CI uses
# `ci`).

set -e -o pipefail

if ! command -v uv &> /dev/null; then
  echo "Error: uv not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

echo "Sampling JOB data..."
uv run --no-project --with duckdb==1.4.1 python "${SCRIPT_DIR}/sample_data.py" "${DATA_DIR}"

for f in "${DATA_DIR}"/*.parquet; do
  echo "Converting $(basename "$f") to Vortex..."
  (cd "${CRATE_DIR}" && cargo run --profile "${VORTEX_SLT_PROFILE:-release}" --package vortex-tui --bin vx -- convert "$f")
done

echo "Checking that the JOB Parquet and Vortex tables match..."
(
  cd "${CRATE_DIR}"
  cargo test --profile "${VORTEX_SLT_PROFILE:-release}" -p vortex-sqllogictest --test sqllogictests \
    -- --exact 'slt::duckdb::job/duckdb/parity.slt'
)
