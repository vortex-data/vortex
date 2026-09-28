#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- array_ops compress_roundtrip file_io fsst_like row_encode
fi

for fuzz_name in "$@"; do
  corpus_key="${fuzz_name}_corpus.tar.zst"
  corpus_dir="fuzz/corpus/${fuzz_name}"
  mkdir -p "$corpus_dir"
  if python3 scripts/s3-download.py \
    "s3://vortex-fuzz-corpus/$corpus_key" "$corpus_key"; then
    tar -xf "$corpus_key"
  else
    echo "No existing corpus found for $fuzz_name"
  fi
  rm -f "$corpus_key"
  du -sh "$corpus_dir"
done
