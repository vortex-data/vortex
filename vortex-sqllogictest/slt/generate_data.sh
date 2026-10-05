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
#   tpcds       TPC-DS at scale factor 0.1 from DuckDB's tpcds extension.
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

for dataset in "${datasets[@]}"; do
  echo "==> ${dataset}"
  "${SCRIPT_DIR}/${dataset}/generate_data.sh"
done

echo "Done."
