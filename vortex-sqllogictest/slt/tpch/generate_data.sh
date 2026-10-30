#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates TPC-H data at scale factor 0.1 with the benchmarks' `data-gen`
# binary (vortex-bench), so the SLT fixtures and the benchmarks share one
# TPC-H generator, and copies the Parquet and Vortex tables into slt/tpch/data/.
# Set VORTEX_SLT_PROFILE to reuse an existing cargo profile (CI uses `ci`), or
# VORTEX_SLT_DATA_GEN to use a prebuilt `data-gen` binary.

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_DIR="$(cd "${CRATE_DIR}/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SCALE_FACTOR="0.1"
# data-gen writes to vortex-bench/data/tpch/<scale factor>/<format>/.
BENCH_DATA_DIR="${WORKSPACE_DIR}/vortex-bench/data/tpch/${SCALE_FACTOR}"

echo "Generating TPC-H data (SF=${SCALE_FACTOR})..."
(
  cd "${WORKSPACE_DIR}"
  if [ -n "${VORTEX_SLT_DATA_GEN:-}" ]; then
    "${VORTEX_SLT_DATA_GEN}" tpch --opt "scale-factor=${SCALE_FACTOR}" --formats parquet,vortex
  else
    cargo run --profile "${VORTEX_SLT_PROFILE:-release}" -p vortex-bench --bin data-gen -- \
      tpch --opt "scale-factor=${SCALE_FACTOR}" --formats parquet,vortex
  fi
)

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"
cp "${BENCH_DATA_DIR}"/parquet/*.parquet "${BENCH_DATA_DIR}"/vortex-file-compressed/*.vortex "${DATA_DIR}/"
