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
#   --remote-storage <url>    Remote storage URL (e.g., s3://bucket/path/)
#                             If provided, runs in remote mode (no lance support).
#   --benchmark-id <id>       Benchmark ID for error messages (e.g., tpch-s3)

set -Eeu -o pipefail

subcommand="$1"
targets="$2"
shift 2

scale_factor=""
remote_storage=""
benchmark_id=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale-factor)
            scale_factor="$2"
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

touch results.json

if [[ -n "$df_formats" ]]; then
    # shellcheck disable=SC2086
    target/release_debug/datafusion-bench "$subcommand" \
        -d gh-json \
        --formats "$df_formats" \
        $opts \
        -o df-results.json

    cat df-results.json >> results.json
fi

if [[ -n "$ddb_formats" ]]; then
    # shellcheck disable=SC2086
    target/release_debug/duckdb-bench "$subcommand" \
        -d gh-json \
        --formats "$ddb_formats" \
        $opts \
        --delete-duckdb-database \
        -o ddb-results.json

    cat ddb-results.json >> results.json
fi

# Lance-bench only runs for local benchmarks.
if ! $is_remote && [[ "$has_lance" == "true" ]] && [[ -f "target/release_debug/lance-bench" ]]; then
    # shellcheck disable=SC2086
    target/release_debug/lance-bench "$subcommand" \
        -d gh-json \
        $opts \
        -o lance-results.json

    cat lance-results.json >> results.json
fi
