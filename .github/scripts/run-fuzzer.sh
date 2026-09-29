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

# Fork mode replays the saved corpus to establish coverage, then replaces failed
# workers without ending exploration. Artifacts still fail the job after the budget.
FORK_FLAGS=("-fork=$FUZZ_JOBS" -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1)

set +e
env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 \
  "$GITHUB_WORKSPACE/fuzz-binaries/$FUZZ_TARGET" "$CORPUS_DIR" \
  "${FORK_FLAGS[@]}" "-max_total_time=$MAX_TIME" -rss_limit_mb=0 -print_final_stats=1 \
  -artifact_prefix="fuzz/artifacts/${FUZZ_NAME}/" \
  2>&1 | tee fuzz_output.log
FUZZ_STATUS=${PIPESTATUS[0]}
set -e

stop_checkpoint_loop
trap - EXIT
exit "$FUZZ_STATUS"
