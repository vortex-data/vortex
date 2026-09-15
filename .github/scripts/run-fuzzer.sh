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

FEATURES_FLAG=()
if [ -n "$EXTRA_FEATURES" ]; then
  FEATURES_FLAG=(--features "$EXTRA_FEATURES")
fi
FORK_FLAG=()
if [ "$FUZZ_JOBS" -gt 1 ]; then
  # Skip fork mode's initial full-corpus merge before starting workers.
  FORK_FLAG=("-fork=$FUZZ_JOBS" "-keep_seed=1")
fi

set +e
if [ -n "$FUZZER_ARTIFACT" ]; then
  env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 \
    "$GITHUB_WORKSPACE/fuzz-binaries/$FUZZ_NAME" "$CORPUS_DIR" \
    "${FORK_FLAG[@]}" "-max_total_time=$MAX_TIME" -rss_limit_mb=0 \
    -artifact_prefix="fuzz/artifacts/${FUZZ_NAME}/" \
    2>&1 | tee fuzz_output.log
else
  env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 \
    cargo "+$NIGHTLY_TOOLCHAIN" fuzz run --release --debug-assertions \
    "${FEATURES_FLAG[@]}" \
    "$FUZZ_TARGET" "$CORPUS_DIR" -- \
    "${FORK_FLAG[@]}" "-max_total_time=$MAX_TIME" -rss_limit_mb=0 \
    2>&1 | tee fuzz_output.log
fi
FUZZ_STATUS=${PIPESTATUS[0]}
set -e

stop_checkpoint_loop
trap - EXIT
exit "$FUZZ_STATUS"
