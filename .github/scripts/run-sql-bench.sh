#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Runs SQL benchmarks (datafusion-bench, duckdb-bench, lance-bench) for the given targets.
# This script is used by the sql-benchmarks.yml workflow.
#
# Delegates to scripts/run-bench.py which handles noise detection and automatic retries.
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

exec python3 "$REPO_ROOT/scripts/run-bench.py" "$@"
