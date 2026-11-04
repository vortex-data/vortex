#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates the StatPopGen fixture with the benchmarks' `data-gen` binary
# (vortex-bench): the first 1,000 variants of gnomAD v3.1.2 HGDP-1kG chr21,
# the smallest scale factor the benchmark supports. Copies the Parquet table
# into slt/statpopgen/data/, converts it to Vortex, and checks that the Parquet
# and Vortex tables hold identical data before the Vortex file is used by the
# SLT suite. Set VORTEX_SLT_PROFILE to reuse an existing cargo profile (CI uses
# `ci`).

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_DIR="$(cd "${CRATE_DIR}/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SCALE_FACTOR="1"
# data-gen writes to vortex-bench/data/statpopgen/<rows>/parquet/.
BENCH_PARQUET="${WORKSPACE_DIR}/vortex-bench/data/statpopgen/$((SCALE_FACTOR * 1000))/parquet/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.parquet"

echo "Generating StatPopGen data (scale factor ${SCALE_FACTOR})..."
(
  cd "${WORKSPACE_DIR}"
  cargo run --profile "${VORTEX_SLT_PROFILE:-release}" -p vortex-bench --bin data-gen -- \
    statpopgen --opt "scale-factor=${SCALE_FACTOR}" --formats parquet
)

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"
cp "${BENCH_PARQUET}" "${DATA_DIR}/statpopgen.parquet"

# The benchmark writes its Vortex files with an unstable edition; `vx convert`
# writes the default one, which the SLT runner's session reads.
echo "Converting statpopgen.parquet to Vortex..."
(cd "${CRATE_DIR}" && cargo run --profile "${VORTEX_SLT_PROFILE:-release}" --package vortex-tui --bin vx -- convert "${DATA_DIR}/statpopgen.parquet")

echo "Checking that the StatPopGen Parquet and Vortex tables match..."
(
  cd "${CRATE_DIR}"
  cargo test --profile "${VORTEX_SLT_PROFILE:-release}" -p vortex-sqllogictest --test sqllogictests \
    -- --exact 'slt::duckdb::statpopgen/duckdb/parity.slt'
)
