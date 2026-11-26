#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Run an ephyra corpus search (`<target>_corpus` in fuzz/tests/<target>_ephyra.rs) in parallel
# workers, one per core.
#
# Usage: run-ephyra-fuzz.sh BINARY TEST BUDGET_MS WORKERS CORPUS_DIR CRASH_DIR [FILTER...]
#
# Every worker replays and extends the shared corpus in CORPUS_DIR and saves failing plans to
# CRASH_DIR. Worker N tracks the coverage of crate FILTER[N % count], so the workers together look
# for new code in several crates. Each worker writes its JSON report, log and, for an instrumented
# build, coverage page to ephyra-work/N, and the reports are concatenated into ephyra-report.jsonl.
# When CORPUS_KEY and CORPUS_ETAG_FILE are set, the corpus is checkpointed to R2 every 15 minutes
# while the workers run. Worker output also streams to stdout, prefixed `[wN]` and cut to 400
# characters a line; the full log stays in ephyra-work/N/output.log. The script fails when any
# worker fails.

set -uo pipefail

BINARY=$1
TEST=$2
BUDGET_MS=$3
WORKERS=$4
CORPUS_DIR=$(realpath -m "$5")
CRASH_DIR=$(realpath -m "$6")
shift 6
FILTERS=("$@")
mkdir -p "$CORPUS_DIR" "$CRASH_DIR"

checkpoint_pid=
if [ -n "${CORPUS_KEY:-}" ] && [ -n "${CORPUS_ETAG_FILE:-}" ]; then
  (
    last=$(find "$CORPUS_DIR" -type f -name '*.json' | wc -l)
    while sleep 900; do
      current=$(find "$CORPUS_DIR" -type f -name '*.json' | wc -l)
      if [ "$current" -ne "$last" ]; then
        echo "Checkpointing $current corpus entries"
        if .github/scripts/checkpoint-fuzz-corpus.sh "$CORPUS_DIR" "$CORPUS_KEY" "$CORPUS_ETAG_FILE"; then
          last=$current
        else
          echo "::warning::Unable to checkpoint the corpus"
        fi
      fi
    done
  ) &
  checkpoint_pid=$!
fi

pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
  dir="ephyra-work/$worker"
  mkdir -p "$dir"
  filter=""
  if [ "${#FILTERS[@]}" -gt 0 ]; then
    filter=${FILTERS[$((worker % ${#FILTERS[@]}))]}
  fi
  echo "worker $worker: coverage of ${filter:-the test crate}"
  (
    cd "$dir" || exit 1
    EPHYRA_TESTS=1000000000 \
    EPHYRA_BUDGET_MS="$BUDGET_MS" \
    EPHYRA_CORPUS_DIR="$CORPUS_DIR" \
    EPHYRA_CRASH_DIR="$CRASH_DIR" \
    EPHYRA_COVERAGE_FILTER="$filter" \
    EPHYRA_REPORT_JSON="$PWD/report.jsonl" \
    EPHYRA_COVERAGE_HTML="$PWD/coverage" \
    RUST_BACKTRACE=1 \
      "$BINARY" --ignored --exact "$TEST" --nocapture --test-threads=1 2>&1 \
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
  grep -h "^ephyra: \(replaying\|added\)" "ephyra-work/$i/output.log" | sed "s/^/worker $i: /"
done

if [ -n "$checkpoint_pid" ]; then
  kill "$checkpoint_pid" 2> /dev/null
  wait "$checkpoint_pid" 2> /dev/null
fi

cat ephyra-work/*/report.jsonl > ephyra-report.jsonl 2> /dev/null || true
echo "corpus: $(find "$CORPUS_DIR" -type f -name '*.json' | wc -l) entries; crashes: $(find "$CRASH_DIR" -type f -name '*.json' | wc -l)"
exit $status
