#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates TPC-DS at scale factor 0.1 with DuckDB's tpcds extension into
# slt/tpcds/data/, one Parquet file per table, converts every table to Vortex,
# and checks that the Parquet and Vortex tables hold identical data before the
# Vortex files are used by the SLT suite. Set VORTEX_SLT_PROFILE to reuse an
# existing cargo profile for the parity test (CI uses `ci`).

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
uvx --with duckdb python - "${DATA_DIR}" "${SCALE_FACTOR}" <<'PY'
import sys
import duckdb

data_dir, scale_factor = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute(f"CALL dsdgen(sf={scale_factor})")
con.execute(f"EXPORT DATABASE '{data_dir}' (FORMAT parquet)")
PY
# EXPORT DATABASE also writes load/schema scripts that the tests do not use.
rm -f "${DATA_DIR}"/load.sql "${DATA_DIR}"/schema.sql

# The Parquet files are kept so the `parquet.slt` suites and the parity test
# can read them.
for f in "${DATA_DIR}"/*.parquet; do
  echo "Converting $(basename "$f") to Vortex..."
  (cd "${CRATE_DIR}" && cargo run --release --package vortex-tui --bin vx -- convert "$f")
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
