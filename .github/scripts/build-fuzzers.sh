#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

mkdir -p fuzz-binaries

FEATURES_FLAG=()
if [ -n "${EXTRA_FEATURES:-}" ]; then
  FEATURES_FLAG=(--features "$EXTRA_FEATURES")
fi
TARGET_FLAG=(--bins)
if [ -n "${FUZZ_TARGET:-}" ]; then
  TARGET_FLAG=(--bin "$FUZZ_TARGET")
fi

# This uses the flags of `cargo fuzz build --release --debug-assertions`, but it puts the
# SanitizerCoverage flags only on workspace crates. libFuzzer scans every coverage counter after
# each input, and dependencies hold about 40% of the counters. Dropping them makes each input
# about 30% faster.
HOST=$(rustc "+$NIGHTLY_TOOLCHAIN" -vV | sed -n 's/^host: //p')
SANCOV=(
  -Cpasses=sancov-module
  -Cllvm-args=-sanitizer-coverage-level=4
  -Cllvm-args=-sanitizer-coverage-inline-8bit-counters
  -Cllvm-args=-sanitizer-coverage-pc-table
  -Cllvm-args=-sanitizer-coverage-trace-compares
)
if [[ "$HOST" == *-linux-* ]]; then
  SANCOV+=(-Cllvm-args=-sanitizer-coverage-stack-depth)
fi
SANCOV_TOML=$(printf '"%s",' "${SANCOV[@]}")

BUILD_START=$SECONDS
RUSTFLAGS="--cfg fuzzing -Cllvm-args=-simplifycfg-branch-fold-threshold=0 \
-Zsanitizer=address -Cdebug-assertions -Ccodegen-units=1 ${RUSTFLAGS:-}" \
  cargo "+$NIGHTLY_TOOLCHAIN" build -Zprofile-rustflags --release \
  --manifest-path fuzz/Cargo.toml --target "$HOST" \
  --config 'profile.release.debug="line-tables-only"' \
  --config "profile.release.rustflags=[${SANCOV_TOML%,}]" \
  --config 'profile.release.package."*".rustflags=[]' \
  --config 'profile.release.build-override.rustflags=[]' \
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

TARGETS=(array_ops arrow_roundtrip compress_roundtrip file_corruption file_io file_io_v2 fsst_like row_encode)
if [ -n "${FUZZ_TARGET:-}" ]; then
  TARGETS=("$FUZZ_TARGET")
fi
for target in "${TARGETS[@]}"; do
  copy_fuzzer "$target"
done
