#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

mkdir -p fuzz-binaries

cargo "+$NIGHTLY_TOOLCHAIN" fuzz build --release --debug-assertions

copy_fuzzer() {
  local target=$1
  local output_name=${2:-$target}
  local binary
  binary=$(find target -type f -path "*/release/$target" -perm -u+x -print -quit)
  if [ -z "$binary" ]; then
    echo "::error::Unable to find compiled fuzzer $target"
    exit 1
  fi
  install -m 755 "$binary" "fuzz-binaries/$output_name"
}

for target in array_ops compress_roundtrip file_io fsst_like row_encode; do
  copy_fuzzer "$target"
done

tar -acf cpu-fuzzers.tar.zst fuzz-binaries
