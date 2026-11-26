#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run an ephyra random search (test TEST in a fuzz/tests/*_ephyra.rs binary) in parallel workers,
# one per core, each with its own seed.
#
# Usage: run-ephyra-fuzz.sh BINARY TEST BUDGET_MS WORKERS CRASH_DIR
#
# Workers save the smallest failing plan per failure signature to CRASH_DIR. Each writes its JSON
# report and log to ephyra-work/N, and the reports are concatenated into ephyra-report.jsonl.
# Worker output also streams to stdout, prefixed `[wN]` and cut to 400 characters a line. The
# script fails when any worker fails.

set -uo pipefail

BINARY=$1
TEST=$2
BUDGET_MS=$3
WORKERS=$4
CRASH_DIR=$(realpath -m "$5")
mkdir -p "$CRASH_DIR"

pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
  dir="ephyra-work/$worker"
  mkdir -p "$dir"
  (
    cd "$dir" || exit 1
    EPHYRA_TESTS=1000000000 \
    EPHYRA_BUDGET_MS="$BUDGET_MS" \
    EPHYRA_REGRESSIONS=0 \
    EPHYRA_REPORT_JSON="$PWD/report.jsonl" \
    VORTEX_FUZZ_CRASH_DIR="$CRASH_DIR" \
    RUST_BACKTRACE=1 \
      "$BINARY" --exact "$TEST" --nocapture --test-threads=1 2>&1 \
      | tee output.log \
      | sed -u -E "s/^(.{400}).+/\\1 [...]/; s/^/[w$worker] /"
  ) &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "worker $i passed"
  else
    echo "::error::ephyra worker $i failed; see ephyra-work/$i/output.log"
    status=1
  fi
done

cat ephyra-work/*/report.jsonl > ephyra-report.jsonl 2> /dev/null || true
echo "crash plans: $(find "$CRASH_DIR" -type f -name '*.json' | wc -l)"
exit $status
