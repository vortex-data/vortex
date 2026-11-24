#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Generate TPC-H Parquet locally with tpchgen-cli (zstd level 3).
# Usage: gen-tpch.sh <out-dir> [scale-factor]
set -euo pipefail
OUT=$1
SF=${2:-1}
if ! command -v tpchgen-cli >/dev/null; then
    echo "tpchgen-cli not found: install it with 'cargo install tpchgen-cli' or 'pip install tpchgen-cli'" >&2
    exit 1
fi
mkdir -p "$OUT"
[[ -f "$OUT/lineitem.parquet" ]] || tpchgen-cli parquet -s "$SF" --compression='ZSTD(3)' --output-dir="$OUT"
ls -la "$OUT"
