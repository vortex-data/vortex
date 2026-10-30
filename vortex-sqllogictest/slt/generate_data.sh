#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates the git-ignored fixtures used by the generated-data SLT suites by
# running each dataset's own slt/<dataset>/generate_data.sh.
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
# Each dataset script honours these environment variables:
#   VORTEX_SLT_PROFILE      cargo profile to build with (default: release).
#   VORTEX_SLT_VX           prebuilt `vx` binary to use instead of `cargo run`.
#   VORTEX_SLT_DATA_GEN     prebuilt `data-gen` binary to use instead of `cargo run`.
#   VORTEX_SLT_SKIP_PARITY  set to 1 to skip the Parquet/Vortex parity checks,
#                           for callers that run the full SLT suite afterwards.

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

if [ "${parallel}" = true ]; then
  pids=()
  for dataset in "${datasets[@]}"; do
    echo "==> ${dataset} (in background)"
    (
      "${SCRIPT_DIR}/${dataset}/generate_data.sh" 2>&1 | while IFS= read -r line; do
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
    "${SCRIPT_DIR}/${dataset}/generate_data.sh"
  done
fi

echo "Done."
