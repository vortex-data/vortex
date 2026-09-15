#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

mode=${1:?Usage: restore-fuzz-corpus.sh <explore|minimize>}
CORPUS_KEY="${FUZZ_NAME}_corpus.tar.zst"
CORPUS_DIR="fuzz/corpus/${FUZZ_NAME}"
CORPUS_ETAG_FILE="$RUNNER_TEMP/${FUZZ_NAME}.etag"

if python3 scripts/s3-download.py \
  "s3://vortex-fuzz-corpus/$CORPUS_KEY" "$CORPUS_KEY" \
  --etag-output "$CORPUS_ETAG_FILE"; then
  echo "Downloaded corpus successfully"
  tar -xf "$CORPUS_KEY"
  echo "found=true" >> "$GITHUB_OUTPUT"
else
  mkdir -p "$CORPUS_DIR"
  echo "found=false" >> "$GITHUB_OUTPUT"
  if [ "$mode" = explore ]; then
    echo "Corpus unavailable; creating an empty local corpus with create-only protection"
    printf 'CREATE_ONLY\n' > "$CORPUS_ETAG_FILE"
  else
    echo "No existing corpus found, nothing to minimize"
  fi
fi
