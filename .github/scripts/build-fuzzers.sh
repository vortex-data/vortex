#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

mkdir -p fuzz-binaries

if [[ "$(uname -s)" == Linux ]]; then
  # Use mold to handle long-range calls in the instrumented ARM64 binaries.
  export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-fuse-ld=mold"
fi

FEATURES_FLAG=()
if [ -n "${EXTRA_FEATURES:-}" ]; then
  FEATURES_FLAG=(--features "$EXTRA_FEATURES")
fi
TARGET_FLAG=()
if [ -n "${FUZZ_TARGET:-}" ]; then
  TARGET_FLAG=("$FUZZ_TARGET")
fi
BUILD_START=$SECONDS
cargo "+$NIGHTLY_TOOLCHAIN" fuzz build --release --debug-assertions \
  "${FEATURES_FLAG[@]}" "${TARGET_FLAG[@]}"
echo "Fuzzer compilation took $((SECONDS - BUILD_START))s"

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

TARGETS=(array_ops compress_roundtrip file_io fsst_like row_encode)
if [ -n "${FUZZ_TARGET:-}" ]; then
  TARGETS=("$FUZZ_TARGET")
fi
for target in "${TARGETS[@]}"; do
  copy_fuzzer "$target"
done
