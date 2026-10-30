#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates the git-ignored fixtures used by the generated-data SLT suites with
# the benchmarks' `data-gen` binary (vortex-bench), so the SLT fixtures and the
# benchmarks share one generator and one Parquet-to-Vortex conversion. Each
# dataset's Parquet and Vortex tables are copied into slt/<dataset>/data/.
#
# Usage: generate_data.sh [--parallel] [DATASET...]
#
# Datasets:
#   tpch        TPC-H at scale factor 0.1.
#   tpcds       TPC-DS at scale factor 0.1.
#   clickbench  One shard (~1M rows) of the partitioned ClickBench `hits` table.
#
# With no arguments every dataset is generated. With --parallel the datasets are
# generated concurrently, each output line prefixed with its dataset name, and
# the script fails if any dataset fails.
#
# After generating TPC-DS and ClickBench, the script runs their
# duckdb/parity.slt, which fails if the Parquet and Vortex data differ, so a bad
# conversion is caught before the query result files are trusted.
#
# Environment variables:
#   VORTEX_SLT_PROFILE      cargo profile to build with (default: release; CI uses `ci`).
#   VORTEX_SLT_DATA_GEN     prebuilt `data-gen` binary to use instead of building one.
#   VORTEX_SLT_SKIP_PARITY  set to 1 to skip the parity checks, for callers that
#                           run the full SLT suite afterwards.

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${CRATE_DIR}/.." && pwd)"
BENCH_DATA_DIR="${WORKSPACE_DIR}/vortex-bench/data"
PROFILE="${VORTEX_SLT_PROFILE:-release}"
ALL_DATASETS=(tpch tpcds clickbench)

usage() {
  echo "Usage: $(basename "${BASH_SOURCE[0]}") [--parallel] [DATASET...]"
  echo
  echo "Datasets: ${ALL_DATASETS[*]} (default: all)"
}

parallel=false
datasets=()
for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      usage
      exit 0
      ;;
    --parallel)
      parallel=true
      ;;
    tpch|tpcds|clickbench)
      datasets+=("${arg}")
      ;;
    *)
      echo "Error: unknown dataset '${arg}'" >&2
      usage >&2
      exit 1
      ;;
  esac
done
if [ "${#datasets[@]}" -eq 0 ]; then
  datasets=("${ALL_DATASETS[@]}")
fi

DATA_GEN="${VORTEX_SLT_DATA_GEN:-}"
if [ -z "${DATA_GEN}" ]; then
  # Build `data-gen` once and run the binary directly, so the datasets can be
  # generated in parallel without contending for cargo's build lock.
  DATA_GEN="$(
    cd "${WORKSPACE_DIR}" &&
      cargo build --profile "${PROFILE}" -p vortex-bench --bin data-gen \
        --message-format=json-render-diagnostics |
      sed -n 's/.*"executable":"\([^"]*\)".*/\1/p'
  )"
  if [ -z "${DATA_GEN}" ]; then
    echo "Error: could not find the data-gen binary built by cargo" >&2
    exit 1
  fi
fi

# Runs `data-gen` for one benchmark and copies its Parquet and Vortex files into
# slt/<dataset>/data/, renaming each file with the given sed expression so the
# file names match the tables the SLT suites read.
#
# Usage: run_data_gen DATASET BENCH_SUBDIR RENAME_SED BENCHMARK [DATA_GEN_ARGS...]
run_data_gen() {
  local dataset="$1" bench_dir="${BENCH_DATA_DIR}/$2" rename="$3"
  shift 3
  local data_dir="${SCRIPT_DIR}/${dataset}/data"

  # data-gen skips existing Vortex files, so drop them to convert with the
  # current writer. The Parquet inputs are kept and reused.
  rm -rf "${bench_dir}/vortex-file-compressed"
  (cd "${WORKSPACE_DIR}" && "${DATA_GEN}" "$@" --formats parquet,vortex)

  rm -rf "${data_dir}"
  mkdir -p "${data_dir}"
  local f
  for f in "${bench_dir}"/parquet/*.parquet "${bench_dir}"/vortex-file-compressed/*.vortex; do
    cp "${f}" "${data_dir}/$(basename "${f}" | sed "${rename}")"
  done
}

parity() {
  if [ "${VORTEX_SLT_SKIP_PARITY:-}" = 1 ]; then
    return
  fi
  echo "Checking that the $1 Parquet and Vortex tables match..."
  (
    cd "${CRATE_DIR}"
    cargo test --profile "${PROFILE}" -p vortex-sqllogictest --test sqllogictests \
      -- --exact "slt::duckdb::$1/duckdb/parity.slt"
  )
}

generate() {
  case "$1" in
    tpch)
      echo "Generating TPC-H data (SF=0.1)..."
      run_data_gen tpch tpch/0.1 '' tpch --opt scale-factor=0.1
      ;;
    tpcds)
      # Every TPC-DS table fits in one `<table>_0001` file at this scale factor.
      echo "Generating TPC-DS data (SF=0.1)..."
      run_data_gen tpcds tpcds/0.1 's/_0001\././' tpcds --opt scale-factor=0.1
      parity tpcds
      ;;
    clickbench)
      echo "Downloading and converting ClickBench shard 0..."
      run_data_gen clickbench clickbench_partitioned_1 's/^hits_0\./hits./' \
        clickbench --opt files=1
      parity clickbench
      ;;
  esac
}

if [ "${parallel}" = true ]; then
  pids=()
  for dataset in "${datasets[@]}"; do
    echo "==> ${dataset} (in background)"
    (
      generate "${dataset}" 2>&1 | while IFS= read -r line; do
        printf '[%s] %s\n' "${dataset}" "${line}"
      done
    ) &
    pids+=("$!")
  done
  # `set -e` does not see background failures, so check every exit status.
  failed=()
  for i in "${!datasets[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed+=("${datasets[$i]}")
    fi
  done
  if [ "${#failed[@]}" -ne 0 ]; then
    echo "Error: data generation failed for: ${failed[*]}" >&2
    exit 1
  fi
else
  for dataset in "${datasets[@]}"; do
    echo "==> ${dataset}"
    generate "${dataset}"
  done
fi

echo "Done."
