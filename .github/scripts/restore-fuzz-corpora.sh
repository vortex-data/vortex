#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

for fuzz_name in \
  array_ops \
  compress_roundtrip \
  file_io \
  fsst_like \
  row_encode; do
  corpus_key="${fuzz_name}_corpus.tar.zst"
  corpus_dir="fuzz/corpus/${fuzz_name}"
  mkdir -p "$corpus_dir"
  if python3 scripts/s3-download.py \
    "s3://vortex-fuzz-corpus/$corpus_key" "$corpus_key"; then
    tar -xf "$corpus_key"
  else
    echo "No existing corpus found for $fuzz_name"
  fi
done
