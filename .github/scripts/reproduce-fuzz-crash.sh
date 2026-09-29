#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

# The workflow accepts space-separated KEY=VALUE pairs.
read -r -a FUZZ_ENV <<< "$EXTRA_ENV"

env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 \
  "$GITHUB_WORKSPACE/fuzz-binaries/$FUZZ_TARGET" "$FIRST_CRASH" \
  2>&1 | tee -a fuzz_output.log || true
