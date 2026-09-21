#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

# The workflow accepts space-separated KEY=VALUE pairs.
read -r -a FUZZ_ENV <<< "$EXTRA_ENV"

FEATURES_FLAG=()
if [ -n "$EXTRA_FEATURES" ]; then
  FEATURES_FLAG=(--features "$EXTRA_FEATURES")
fi
CORPUS_DIR="fuzz/corpus/${FUZZ_NAME}"
MINIMIZED_DIR="${CORPUS_DIR}_minimized"
mkdir -p "$MINIMIZED_DIR"
ORIGINAL_COUNT=$(find "$CORPUS_DIR" -type f | wc -l)

cargo "+$NIGHTLY_TOOLCHAIN" fuzz build --release --debug-assertions \
  "${FEATURES_FLAG[@]}" "$FUZZ_TARGET"
FUZZ_BINARY=$(find target -type f \
  -path "*/release/$FUZZ_TARGET" -perm -u+x -print -quit)
if [ -z "$FUZZ_BINARY" ]; then
  echo "::error::Unable to find compiled fuzzer $FUZZ_TARGET"
  exit 1
fi

set +e
env "${FUZZ_ENV[@]}" "$FUZZ_BINARY" \
  -merge=1 "$MINIMIZED_DIR" "$CORPUS_DIR" -rss_limit_mb=0 \
  2>&1 | tee fuzz-minimize.log
MERGE_STATUS=${PIPESTATUS[0]}
set -e
MINIMIZED_COUNT=$(find "$MINIMIZED_DIR" -type f | wc -l)

if [ "$MERGE_STATUS" -ne 0 ] || grep -Fq "caused a failure" fuzz-minimize.log; then
  echo "::error::Corpus minimization encountered failing inputs"
  exit 1
fi
if [ "$ORIGINAL_COUNT" -gt 0 ] && [ "$MINIMIZED_COUNT" -eq 0 ]; then
  echo "::error::Refusing to replace a non-empty corpus with an empty corpus"
  exit 1
fi

echo "Minimized $ORIGINAL_COUNT inputs to $MINIMIZED_COUNT"
rm -rf "$CORPUS_DIR"
mv "$MINIMIZED_DIR" "$CORPUS_DIR"
