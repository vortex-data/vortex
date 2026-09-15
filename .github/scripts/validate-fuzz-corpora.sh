#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

validation_failed=false

validate_corpus() {
  local fuzz_name=$1
  local corpus_dir="fuzz/corpus/$fuzz_name"
  local artifact_dir="fuzz/validation-artifacts/$fuzz_name"
  local binary="$GITHUB_WORKSPACE/fuzz-binaries/$fuzz_name"
  local status seed_count

  mkdir -p "$artifact_dir" fuzz/validation-logs
  seed_count=$(find "$corpus_dir" -type f | wc -l)
  echo "Validating $seed_count seeds for $fuzz_name"
  if [ "$seed_count" -eq 0 ]; then
    return
  fi

  set +e
  find "$corpus_dir" -type f -print0 | \
    xargs -0 -r -n 1024 -P 4 \
      "$binary" -rss_limit_mb=0 -artifact_prefix="$artifact_dir/" \
      2>&1 | tee "fuzz/validation-logs/${fuzz_name}.log"
  status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    echo "::error::Existing seeds failed for $fuzz_name"
    validation_failed=true
  fi
}

for fuzz_name in \
  array_ops \
  compress_roundtrip \
  file_io \
  fsst_like \
  row_encode; do
  validate_corpus "$fuzz_name"
done

if [ "$validation_failed" = true ]; then
  exit 1
fi
