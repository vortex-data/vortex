#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# End to end: generate training data, fit one model per held-out source, run the model-driven
# compressor on every chunk with the model that never saw its source, and report.
#
# Usage: run_e2e.sh <parquet-dir> <work-dir>
set -euo pipefail

DATA=$1
WORK=$2
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BIN="$ROOT/target/release/compressor-feasibility"

cargo build --release -p compressor-feasibility --manifest-path "$ROOT/Cargo.toml"

"$BIN" --data-dir "$DATA" --tpch-sf 1 --max-chunks-per-column 16 --decode-reps 5 --compress-reps 3 \
    --variants default,model/runend+sparse,forced/ --out "$WORK/train"

uv run --no-project --with pandas --with scikit-learn python -W ignore "$HERE/train.py" \
    "$WORK/train" --export "$WORK/models"

"$BIN" --data-dir "$DATA" --tpch-sf 1 --max-chunks-per-column 16 --decode-reps 7 --compress-reps 3 \
    --variants default,model/runend+sparse --model-dir "$WORK/models" --gate 0.1 --exclude-candidates pco --out "$WORK/eval"

uv run --no-project --with pandas python "$HERE/e2e_report.py" "$WORK/eval"
