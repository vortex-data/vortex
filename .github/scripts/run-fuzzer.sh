#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

# The workflow accepts space-separated KEY=VALUE pairs.
read -r -a FUZZ_ENV <<< "$EXTRA_ENV"

CORPUS_DIR="fuzz/corpus/${FUZZ_NAME}"
CORPUS_KEY="${FUZZ_NAME}_corpus.tar.zst"
CORPUS_ETAG_FILE="$RUNNER_TEMP/${FUZZ_NAME}.etag"
mkdir -p "fuzz/artifacts/${FUZZ_NAME}"

LAST_CHECKPOINT_COUNT=$(find "$CORPUS_DIR" -type f | wc -l)
checkpoint_loop() {
  local current_count
  while sleep 900; do
    current_count=$(find "$CORPUS_DIR" -type f | wc -l)
    if [ "$current_count" -ne "$LAST_CHECKPOINT_COUNT" ]; then
      echo "Checkpointing $current_count corpus entries for $FUZZ_NAME"
      if .github/scripts/checkpoint-fuzz-corpus.sh \
        "$CORPUS_DIR" "$CORPUS_KEY" "$CORPUS_ETAG_FILE"; then
        LAST_CHECKPOINT_COUNT=$current_count
      else
        echo "::warning::Unable to checkpoint the $FUZZ_NAME corpus"
      fi
    fi
  done
}
checkpoint_loop &
CHECKPOINT_PID=$!
stop_checkpoint_loop() {
  kill "$CHECKPOINT_PID" 2>/dev/null || true
  wait "$CHECKPOINT_PID" 2>/dev/null || true
}
trap stop_checkpoint_loop EXIT

# A scheduled run must finish before the next cron slot, however late GitHub
# started it, so cap the budget at the time left until 10 minutes before that slot.
# libFuzzer treats -max_total_time=0 as unlimited, so the budget stays at least 60.
if [ "${CYCLE_HOURS:-0}" -gt 0 ]; then
  CYCLE_SECS=$((CYCLE_HOURS * 3600))
  NOW=$(date -u +%s)
  NEXT_SLOT=$(((NOW / CYCLE_SECS + 1) * CYCLE_SECS))
  REMAINING=$((NEXT_SLOT - NOW - 600))
  if [ "$REMAINING" -lt "$MAX_TIME" ]; then
    MAX_TIME=$((REMAINING > 60 ? REMAINING : 60))
    echo "Capping fuzz time at ${MAX_TIME}s to finish before the next scheduled run"
  fi
fi

# Fork mode replays the saved corpus to establish coverage, then replaces failed
# workers without ending exploration. Artifacts still fail the job after the budget.
FORK_FLAGS=("-fork=$FUZZ_JOBS" -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1)

set +e
env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 \
  "$GITHUB_WORKSPACE/fuzz-binaries/$FUZZ_TARGET" "$CORPUS_DIR" \
  "${FORK_FLAGS[@]}" "-max_total_time=$MAX_TIME" "-rss_limit_mb=${RSS_LIMIT_MB:-2048}" \
  -print_final_stats=1 \
  -artifact_prefix="fuzz/artifacts/${FUZZ_NAME}/" \
  2>&1 | tee fuzz_output.log
FUZZ_STATUS=${PIPESTATUS[0]}
set -e

stop_checkpoint_loop
trap - EXIT
exit "$FUZZ_STATUS"
