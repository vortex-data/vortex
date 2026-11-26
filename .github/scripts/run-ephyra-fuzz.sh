#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run the ephyra array_ops property in parallel workers, one per core, each with its own seed.
#
# Usage: run-ephyra-fuzz.sh BINARY BUDGET_MS WORKERS
#
# Worker 0 replays and saves regressions in ephyra-work/0/.ephyra/regressions; the others search
# without them so that workers never write the same file. Each worker writes its own JSON report
# and log under ephyra-work/N, and the reports are concatenated into ephyra-report.jsonl. The
# script fails when any worker fails.

set -uo pipefail

BINARY=$1
BUDGET_MS=$2
WORKERS=$3

pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
  dir="ephyra-work/$worker"
  mkdir -p "$dir"
  regressions=0
  [ "$worker" = 0 ] && regressions=1
  (
    cd "$dir" || exit 1
    EPHYRA_TESTS=1000000000 \
    EPHYRA_BUDGET_MS="$BUDGET_MS" \
    EPHYRA_REGRESSIONS="$regressions" \
    EPHYRA_REPORT_JSON="$PWD/report.jsonl" \
      "$BINARY" --nocapture > output.log 2>&1
  ) &
  pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "worker $i passed"
  else
    echo "::error::ephyra worker $i failed; see ephyra-work/$i/output.log"
    tail -n 80 "ephyra-work/$i/output.log"
    status=1
  fi
done

cat ephyra-work/*/report.jsonl > ephyra-report.jsonl 2> /dev/null || true
exit $status
