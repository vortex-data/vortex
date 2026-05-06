#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run every supported datafusion-bench dataset end-to-end from a clean state,
# Vortex format only, serially. If a benchmark's data cannot be obtained or
# the run fails for any other reason, log it and continue with the rest.
#
# Outputs go under target/vortex-bench/fsst-runs/<timestamp>/, with one
# subdirectory per benchmark plus aggregated results.csv / summary.md / plot.png
# at the top level.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run-fsst-benchmarks.sh [options]

Options:
  --tpc-scale-factor=SF  Scale factor for tpch / tpcds (default: 10.0)
  --iterations=N         Iterations per query (default: 5)
  --output-dir=DIR       Override output directory
  --profile=PROFILE      Cargo profile (release|bench, default: release)
  --only=B1,B2,...       Run only the listed benchmarks (default: all)
  --skip=B1,B2,...       Skip the listed benchmarks
  --no-clean             Skip `cargo clean` (faster re-runs)
  --skip-plot            Skip CSV / summary / plot generation
  -h, --help             Show this help

Benchmarks attempted by default:
  tpch tpcds statpopgen polarsignals clickbench fineweb gharchive

Outputs (under target/vortex-bench/fsst-runs/<timestamp>/ by default):
  run-info.txt           branch / commit / scale / profile metadata
  skipped.txt            benchmarks that failed, with reasons
  results.csv            aggregated CSV across all successful benchmarks
  summary.md             aggregated markdown table
  plot.png               aggregated bar chart (one bar per query, colored by dataset)
  <bench>/raw.jsonl      per-benchmark gh-json output
  <bench>/stdout.log     per-benchmark stdout / stderr
  <bench>/results.csv    per-benchmark CSV
  <bench>/summary.md     per-benchmark markdown
  <bench>/plot.png       per-benchmark bar chart
EOF
}

TPC_SCALE_FACTOR="10.0"
ITERATIONS=5
OUTPUT_DIR=""
PROFILE="release"
ONLY=""
SKIP=""
NO_CLEAN=0
SKIP_PLOT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tpc-scale-factor=*) TPC_SCALE_FACTOR="${1#*=}"; shift ;;
        --iterations=*)       ITERATIONS="${1#*=}";       shift ;;
        --output-dir=*)       OUTPUT_DIR="${1#*=}";       shift ;;
        --profile=*)          PROFILE="${1#*=}";          shift ;;
        --only=*)             ONLY="${1#*=}";             shift ;;
        --skip=*)             SKIP="${1#*=}";             shift ;;
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
SKIPPED_FILE="$OUTPUT_DIR/skipped.txt"
: > "$SKIPPED_FILE"

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
    echo "branch:           $(git rev-parse --abbrev-ref HEAD)"
    echo "commit:           $(git rev-parse HEAD)"
    echo "tpc_scale_factor: $TPC_SCALE_FACTOR"
    echo "iterations:       $ITERATIONS"
    echo "profile:          $PROFILE"
    echo "started_utc:      $TIMESTAMP"
    echo "host:             $(uname -a)"
} > "$OUTPUT_DIR/run-info.txt"

# Per-benchmark extra `--opt` arguments. Empty for benchmarks that take no
# meaningful "SF=10" argument.
opts_for() {
    case "$1" in
        tpch|tpcds) echo "--opt scale-factor=$TPC_SCALE_FACTOR" ;;
        *)          echo "" ;;
    esac
}

ALL_BENCHMARKS=(tpch tpcds statpopgen polarsignals clickbench fineweb gharchive)

# Filter according to --only / --skip.
in_csv() {
    # in_csv ITEM CSV
    local item="$1" csv="$2"
    [[ -z "$csv" ]] && return 1
    local IFS=','
    # shellcheck disable=SC2206
    local arr=($csv)
    for x in "${arr[@]}"; do
        [[ "$x" == "$item" ]] && return 0
    done
    return 1
}

BENCHMARKS=()
for b in "${ALL_BENCHMARKS[@]}"; do
    if [[ -n "$ONLY" ]] && ! in_csv "$b" "$ONLY"; then
        continue
    fi
    if [[ -n "$SKIP" ]] && in_csv "$b" "$SKIP"; then
        continue
    fi
    BENCHMARKS+=("$b")
done

if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    echo "no benchmarks selected" >&2
    exit 1
fi

echo "==> will attempt: ${BENCHMARKS[*]}"

# ----- Run each benchmark -----
SUCCEEDED=()
for bench in "${BENCHMARKS[@]}"; do
    echo
    echo "==> [$bench] starting"
    bench_dir="$OUTPUT_DIR/$bench"
    mkdir -p "$bench_dir"
    raw="$bench_dir/raw.jsonl"
    log="$bench_dir/stdout.log"

    # shellcheck disable=SC2206
    extra=( $(opts_for "$bench") )

    cmd=(
        "$BIN" "$bench"
        --formats vortex
        --display-format gh-json
        --iterations "$ITERATIONS"
        --hide-progress-bar
        -o "$raw"
        "${extra[@]}"
    )

    set +e
    "${cmd[@]}" > "$log" 2>&1
    rc=$?
    set -e

    if [[ $rc -eq 0 && -s "$raw" ]]; then
        echo "==> [$bench] ok"
        SUCCEEDED+=("$bench")
    else
        reason="exit=$rc"
        if [[ ! -s "$raw" ]]; then
            reason="$reason (no output)"
        fi
        # Pull the last few error-ish lines from the log to make the reason useful.
        tail_msg="$(grep -iE 'error|failed|panic' "$log" 2>/dev/null | tail -3 | tr '\n' '|' || true)"
        echo "==> [$bench] SKIPPED: $reason"
        [[ -n "$tail_msg" ]] && echo "    $tail_msg"
        printf '%s\t%s\t%s\n' "$bench" "$reason" "$tail_msg" >> "$SKIPPED_FILE"
    fi
done

echo
echo "==> succeeded: ${SUCCEEDED[*]:-<none>}"
if [[ -s "$SKIPPED_FILE" ]]; then
    echo "==> skipped: see $SKIPPED_FILE"
fi

# ----- Aggregate CSV / plots -----
if [[ $SKIP_PLOT -eq 0 && ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo "==> generating per-benchmark and aggregate CSV / summary / plot"
    python3 "$REPO_ROOT/scripts/plot-fsst-results.py" \
        --input-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR"
fi

echo "==> done"
ls -la "$OUTPUT_DIR"
