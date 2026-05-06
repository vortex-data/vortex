#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run the FSST-paper benchmark suite end-to-end from a clean state.
#
# Defaults: TPC-H SF=10, Vortex-only, 5 iterations, all queries, --release.
# Outputs raw JSONL, a results.csv, a markdown summary table, and a PNG plot
# to target/vortex-bench/fsst-runs/<timestamp>/.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run-fsst-benchmarks.sh [options]

Options:
  --scale-factor=SF       TPC-H scale factor (default: 10.0)
  --iterations=N          Iterations per query (default: 5)
  --queries=LIST          Comma-separated query indices (default: all)
  --exclude-queries=LIST  Comma-separated query indices to skip
  --output-dir=DIR        Override output directory
  --profile=PROFILE       Cargo profile (release|bench, default: release)
  --no-clean              Skip `cargo clean` (faster re-runs)
  --skip-plot             Skip CSV/plot generation
  -h, --help              Show this help

Outputs (under target/vortex-bench/fsst-runs/<timestamp>/ by default):
  raw.jsonl       gh-json output, one measurement per line
  results.csv     flat per-query results
  summary.md      markdown table
  plot.png        bar chart (requires matplotlib)
  run-info.txt    branch/commit/scale/profile metadata
EOF
}

SCALE_FACTOR="10.0"
ITERATIONS=5
QUERIES=""
EXCLUDE_QUERIES=""
OUTPUT_DIR=""
PROFILE="release"
NO_CLEAN=0
SKIP_PLOT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale-factor=*)     SCALE_FACTOR="${1#*=}";     shift ;;
        --iterations=*)       ITERATIONS="${1#*=}";       shift ;;
        --queries=*)          QUERIES="${1#*=}";          shift ;;
        --exclude-queries=*)  EXCLUDE_QUERIES="${1#*=}";  shift ;;
        --output-dir=*)       OUTPUT_DIR="${1#*=}";       shift ;;
        --profile=*)          PROFILE="${1#*=}";          shift ;;
        --no-clean)           NO_CLEAN=1;                 shift ;;
        --skip-plot)          SKIP_PLOT=1;                shift ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="target/vortex-bench/fsst-runs/$TIMESTAMP"
fi
mkdir -p "$OUTPUT_DIR"

echo "==> output: $OUTPUT_DIR"

# ----- Clean -----
if [[ $NO_CLEAN -eq 0 ]]; then
    echo "==> cargo clean (datafusion-bench, vortex-bench)"
    cargo clean -p datafusion-bench
    cargo clean -p vortex-bench
fi

# ----- Build -----
echo "==> cargo build --profile $PROFILE -p datafusion-bench"
cargo build --profile "$PROFILE" -p datafusion-bench

# Cargo's target subdir is "debug" for the dev profile and otherwise the profile name.
case "$PROFILE" in
    dev)  PROFILE_DIR="debug" ;;
    *)    PROFILE_DIR="$PROFILE" ;;
esac
BIN="target/$PROFILE_DIR/datafusion-bench"
if [[ ! -x "$BIN" ]]; then
    echo "error: built binary not found at $BIN" >&2
    exit 1
fi

# ----- Metadata -----
{
    echo "branch:        $(git rev-parse --abbrev-ref HEAD)"
    echo "commit:        $(git rev-parse HEAD)"
    echo "scale_factor:  $SCALE_FACTOR"
    echo "iterations:    $ITERATIONS"
    echo "queries:       ${QUERIES:-all}"
    echo "exclude:       ${EXCLUDE_QUERIES:-none}"
    echo "profile:       $PROFILE"
    echo "started_utc:   $TIMESTAMP"
    echo "host:          $(uname -a)"
} > "$OUTPUT_DIR/run-info.txt"

# ----- Run -----
RAW_JSONL="$OUTPUT_DIR/raw.jsonl"
echo "==> running TPC-H @ SF=$SCALE_FACTOR (vortex only, $ITERATIONS iters/query)"
echo "    raw output -> $RAW_JSONL"

CMD=(
    "$BIN" tpch
    --formats vortex
    --display-format gh-json
    --opt "scale-factor=$SCALE_FACTOR"
    --iterations "$ITERATIONS"
    --hide-progress-bar
    -o "$RAW_JSONL"
)
if [[ -n "$QUERIES" ]]; then
    CMD+=(--queries "$QUERIES")
fi
if [[ -n "$EXCLUDE_QUERIES" ]]; then
    CMD+=(--exclude-queries "$EXCLUDE_QUERIES")
fi

"${CMD[@]}"

# ----- CSV + plot -----
if [[ $SKIP_PLOT -eq 0 ]]; then
    echo "==> generating CSV / summary / plot"
    python3 "$REPO_ROOT/scripts/plot-fsst-results.py" \
        --input "$RAW_JSONL" \
        --output-dir "$OUTPUT_DIR"
fi

echo "==> done"
ls -la "$OUTPUT_DIR"
