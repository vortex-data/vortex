#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Runs SQL benchmarks (datafusion-bench, duckdb-bench, lance-bench) for the given targets.
# This script is used by the sql-benchmarks.yml workflow.
#
# Usage:
#   run-sql-bench.sh <subcommand> <targets> [options]
#
# Arguments:
#   subcommand   The benchmark subcommand (e.g., tpch, clickbench, tpcds)
#   targets      Comma-separated list of engine:format pairs
#                (e.g., "datafusion:parquet,datafusion:vortex,duckdb:parquet")
#
# Options:
#   --scale-factor <sf>       Scale factor for the benchmark (e.g., 1.0, 10.0)
#   --iterations <n>          Number of iterations to pass to each benchmark binary
#   --remote-storage <url>    Remote storage URL (e.g., s3://bucket/path/)
#                             If provided, runs in remote mode (no lance support).
#   --benchmark-id <id>       Benchmark ID for error messages (e.g., tpch-s3)
#   --max-retries <n>         Max retries for noisy results (default: 2, env: BENCH_MAX_RETRIES)
#
# Environment variables:
#   BENCH_MAX_RETRIES         Max retry attempts for noisy runs (default: 2)
#   BENCH_COV_THRESHOLD       Per-benchmark CoV threshold for noise (default: 0.15)
#   BENCH_NOISY_FRACTION      Fraction of benchmarks that must be noisy to trigger rerun (default: 0.25)

set -Eeu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

subcommand="$1"
targets="$2"
shift 2

scale_factor=""
iterations=""
remote_storage=""
benchmark_id=""
max_retries="${BENCH_MAX_RETRIES:-2}"
cov_threshold="${BENCH_COV_THRESHOLD:-0.15}"
noisy_fraction="${BENCH_NOISY_FRACTION:-0.25}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale-factor)
            scale_factor="$2"
            shift 2
            ;;
        --iterations)
            iterations="$2"
            shift 2
            ;;
        --remote-storage)
            remote_storage="$2"
            shift 2
            ;;
        --benchmark-id)
            benchmark_id="$2"
            shift 2
            ;;
        --max-retries)
            max_retries="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

is_remote=false
if [[ -n "$remote_storage" ]]; then
    is_remote=true
fi

# Lance on remote storage is not supported. The infrastructure to generate and upload lance files
# to S3 does not exist. If you need lance on S3, you must first implement:
#   1. Lance data generation in data-gen (or a separate step)
#   2. Lance data upload to S3 before this step runs
if $is_remote && echo "$targets" | grep -q 'lance'; then
    echo "ERROR: Lance format is not supported for remote storage benchmarks."
    echo "Remove 'datafusion:lance' from targets for benchmark '${benchmark_id:-unknown}'."
    exit 1
fi

# Extract formats for each engine from the targets string.
# Example input: "datafusion:parquet,datafusion:vortex,datafusion:lance,duckdb:parquet"
#
# Pipeline: split by comma -> filter by engine prefix -> remove prefix -> rejoin with commas
#
# Lance is filtered out of df_formats because it uses a separate binary (lance-bench).
#
# The `|| true` is needed because some benchmarks don't use all engines (e.g., statpopgen only has
# duckdb targets). grep returns exit code 1 when no matches are found. Both greps must be in the
# subshell so that `|| true` covers the case where grep -v receives empty input.
df_formats=$(echo "$targets" | tr ',' '\n' | (grep '^datafusion:' | grep -v ':lance$' || true) | sed 's/datafusion://' | tr '\n' ',' | sed 's/,$//')
ddb_formats=$(echo "$targets" | tr ',' '\n' | (grep '^duckdb:' || true) | sed 's/duckdb://' | tr '\n' ',' | sed 's/,$//')
has_lance=$(echo "$targets" | grep -q 'datafusion:lance' && echo "true" || echo "false")

# Build options string.
opts=""
if $is_remote; then
    opts="--opt remote-data-dir=$remote_storage"
fi
if [[ -n "$scale_factor" ]]; then
    if [[ -n "$opts" ]]; then
        opts="--opt scale-factor=$scale_factor $opts"
    else
        opts="--opt scale-factor=$scale_factor"
    fi
fi
if [[ -n "$iterations" ]]; then
    opts="-i $iterations $opts"
fi

# Run a benchmark engine and retry if the results are too noisy.
#
# Arguments:
#   $1 - engine label (for logging)
#   $2 - output json filename
#   $3... - the benchmark command and arguments
run_with_retry() {
    local label="$1"
    local output_file="$2"
    shift 2

    for attempt in $(seq 0 "$max_retries"); do
        if [[ $attempt -gt 0 ]]; then
            echo "run-sql-bench: retrying $label (attempt $((attempt + 1))/$((max_retries + 1)))"
        fi

        # shellcheck disable=SC2086
        "$@" -o "$output_file"

        # Check noise levels. If the check script is missing, skip the check.
        if [[ ! -f "$REPO_ROOT/scripts/check-bench-noise.py" ]]; then
            break
        fi

        if python3 "$REPO_ROOT/scripts/check-bench-noise.py" "$output_file" \
            --cov-threshold "$cov_threshold" --noisy-fraction "$noisy_fraction"; then
            break
        fi

        if [[ $attempt -eq $max_retries ]]; then
            echo "run-sql-bench: $label still noisy after $((max_retries + 1)) attempts, using last results"
        fi
    done
}

touch results.json

if [[ -n "$df_formats" ]]; then
    run_with_retry "datafusion" df-results.json \
        target/release_debug/datafusion-bench "$subcommand" \
        -d gh-json \
        --formats "$df_formats" \
        $opts

    cat df-results.json >> results.json
fi

if [[ -n "$ddb_formats" ]]; then
    run_with_retry "duckdb" ddb-results.json \
        target/release_debug/duckdb-bench "$subcommand" \
        -d gh-json \
        --formats "$ddb_formats" \
        $opts \
        --delete-duckdb-database

    cat ddb-results.json >> results.json
fi

# Lance-bench only runs for local benchmarks.
if ! $is_remote && [[ "$has_lance" == "true" ]] && [[ -f "target/release_debug/lance-bench" ]]; then
    run_with_retry "lance" lance-results.json \
        target/release_debug/lance-bench "$subcommand" \
        -d gh-json \
        $opts

    cat lance-results.json >> results.json
fi
