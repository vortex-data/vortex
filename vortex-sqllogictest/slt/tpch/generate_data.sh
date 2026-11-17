#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates TPC-H data at scale factor 0.1 with the benchmarks' `data-gen`
# binary (vortex-bench), so the SLT fixtures and the benchmarks share one
# TPC-H generator, and copies the Parquet and Vortex tables into slt/tpch/data/.
# Set VORTEX_SLT_PROFILE to reuse an existing cargo profile (CI uses `ci`).

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_DIR="$(cd "${CRATE_DIR}/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SCALE_FACTOR="0.1"
# data-gen writes to vortex-bench/data/tpch/<scale factor>/<format>/.
BENCH_DATA_DIR="${WORKSPACE_DIR}/vortex-bench/data/tpch/${SCALE_FACTOR}"

echo "Generating TPC-H data (SF=${SCALE_FACTOR})..."
# data-gen keeps Vortex files that already exist, so remove them to write fresh
# ones with the current compressor.
rm -rf "${BENCH_DATA_DIR}/vortex-file-compressed"
(
  cd "${WORKSPACE_DIR}"
  cargo run --profile "${VORTEX_SLT_PROFILE:-release}" -p vortex-bench --bin data-gen -- \
    tpch --opt "scale-factor=${SCALE_FACTOR}" --formats parquet,vortex
)

# data-gen writes every table as numbered parts (`lineitem_0001.parquet`), while
# the tests read one file per table (`lineitem.parquet`). At this scale factor
# each table is a single part, so drop the part suffix. If the names don't line
# up, the test runner treats the data as absent and silently ignores the suite.
if compgen -G "${BENCH_DATA_DIR}/*/*_0002.*" > /dev/null; then
  echo "Error: data-gen split a TPC-H table into several parts, but the tests expect one file per table." >&2
  exit 1
fi

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"
for part in "${BENCH_DATA_DIR}"/parquet/*_0001.parquet "${BENCH_DATA_DIR}"/vortex-file-compressed/*_0001.vortex; do
  table_file="$(basename "${part}")"
  cp "${part}" "${DATA_DIR}/${table_file/_0001./.}"
done
