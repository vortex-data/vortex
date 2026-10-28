#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates TPC-DS at scale factor 0.1 with tpcgen-cli into slt/tpcds/data/,
# one Parquet file per table, converts every table to Vortex,
# and checks that the Parquet and Vortex tables hold identical data before the
# Vortex files are used by the SLT suite. Set VORTEX_SLT_PROFILE to reuse an
# existing cargo profile for conversion and the parity test (CI uses `ci`).

set -e -o pipefail

if ! command -v uvx &> /dev/null; then
  echo "Error: uvx not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SCALE_FACTOR="0.1"

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

echo "Generating TPC-DS data (SF=${SCALE_FACTOR})..."
# Keep this version equal to the `tpcdsgen` crate that vortex-bench uses.
uvx tpcgen-cli@4.0.0 tpcds parquet -s "${SCALE_FACTOR}" --compression='ZSTD(3)' --output-dir "${DATA_DIR}/"
# The generator metadata table is not part of the TPC-DS schema that the tests use.
rm -f "${DATA_DIR}"/dbgen_version.parquet

# The Parquet files are kept so the `parquet.slt` suites and the parity test
# can read them.
for f in "${DATA_DIR}"/*.parquet; do
  echo "Converting $(basename "$f") to Vortex..."
  (cd "${CRATE_DIR}" && cargo run --profile "${VORTEX_SLT_PROFILE:-release}" --package vortex-tui --bin vx -- convert "$f")
done

# The parity test reads every table in both formats through DuckDB and fails if
# any row differs, so a bad conversion is caught before the query result files
# are trusted.
echo "Checking that the TPC-DS Parquet and Vortex tables match..."
(
  cd "${CRATE_DIR}"
  cargo test --profile "${VORTEX_SLT_PROFILE:-release}" -p vortex-sqllogictest --test sqllogictests \
    -- --exact 'slt::duckdb::tpcds/duckdb/parity.slt'
)
