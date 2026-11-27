#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

# The workflow accepts space-separated KEY=VALUE pairs.
read -r -a FUZZ_ENV <<< "$EXTRA_ENV"

mkdir -p fuzz_repro
for crash in "fuzz/artifacts/$FUZZ_NAME"/*; do
  env "${FUZZ_ENV[@]}" RUST_BACKTRACE=1 timeout --kill-after=10 600 \
    "$GITHUB_WORKSPACE/fuzz-binaries/$FUZZ_TARGET" "$crash" "-rss_limit_mb=${RSS_LIMIT_MB:-2048}" \
    2>&1 | tee "fuzz_repro/$(basename "$crash").log" || true
done
