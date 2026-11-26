#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generates the git-ignored fixtures used by the generated-data SLT suites by
# running each dataset's own slt/<dataset>/generate_data.sh.
#
# Usage: generate_data.sh [DATASET...]
#
# Datasets:
#   tpch        TPC-H at scale factor 0.1.
#   tpcds       TPC-DS at scale factor 0.1.
#   clickbench  One shard (~1M rows) of the partitioned ClickBench `hits` table.
#
# With no arguments every dataset is generated.

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_DATASETS=(tpch tpcds clickbench)

usage() {
  echo "Usage: $(basename "${BASH_SOURCE[0]}") [DATASET...]"
  echo
  echo "Datasets: ${ALL_DATASETS[*]} (default: all)"
}

datasets=()
for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      usage
      exit 0
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

# The datasets are generated concurrently. Their cargo builds still run one at a
# time, because cargo locks the target directory, but each dataset's download,
# generation, conversion and parity check overlaps with the other datasets'
# builds. Each dataset's output is buffered and printed once it finishes, so
# the logs do not interleave.
LOG_DIR="$(mktemp -d)"
trap 'rm -rf "${LOG_DIR}"' EXIT

pids=()
for dataset in "${datasets[@]}"; do
  "${SCRIPT_DIR}/${dataset}/generate_data.sh" > "${LOG_DIR}/${dataset}.log" 2>&1 &
  pids+=("$!")
done

failed=()
for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  status=0
  wait "${pids[$i]}" || status=$?
  echo "==> ${dataset}"
  cat "${LOG_DIR}/${dataset}.log"
  if [ "${status}" -ne 0 ]; then
    echo "Error: generating ${dataset} failed with exit status ${status}" >&2
    failed+=("${dataset}")
  fi
done

if [ "${#failed[@]}" -ne 0 ]; then
  echo "Error: failed datasets: ${failed[*]}" >&2
  exit 1
fi

echo "Done."
