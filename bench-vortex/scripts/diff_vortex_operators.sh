#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run query_bench --check with VORTEX_OPERATORS=true and =false, then diff the output.
#
# Usage:
#   ./diff_vortex_operators.sh statpopgen --scale-factor 1 --targets duckdb:vortex -q 7 --check

set -euo pipefail

RUST_LOG="${RUST_LOG:-error}"

echo "Running with VORTEX_OPERATORS=true..."
output_true=$(VORTEX_OPERATORS=true RUST_LOG="$RUST_LOG" cargo run --release -p bench-vortex --bin query_bench -- "$@")

echo "Running with VORTEX_OPERATORS=false..."
output_false=$(VORTEX_OPERATORS=false RUST_LOG="$RUST_LOG" cargo run --release -p bench-vortex --bin query_bench -- "$@")

echo ""
echo "=== Diff (VORTEX_OPERATORS=true vs false) ==="
diff <(echo "$output_true") <(echo "$output_false") || true
