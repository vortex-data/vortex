#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

# The workflow accepts space-separated KEY=VALUE pairs.
read -r -a FUZZ_ENV <<< "$EXTRA_ENV"

if [ -n "$FUZZER_ARTIFACT" ]; then
  env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 \
    "$GITHUB_WORKSPACE/fuzz-binaries/$FUZZ_NAME" \
    "$FIRST_CRASH" \
    2>&1 | tee fuzz_output.log || true
else
  FEATURES_FLAG=()
  if [ -n "$EXTRA_FEATURES" ]; then
    FEATURES_FLAG=(--features "$EXTRA_FEATURES")
  fi
  env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 \
    cargo "+$NIGHTLY_TOOLCHAIN" fuzz run --release --debug-assertions \
    "${FEATURES_FLAG[@]}" \
    "$FUZZ_TARGET" \
    "$FIRST_CRASH" \
    2>&1 | tee fuzz_output.log || true
fi
