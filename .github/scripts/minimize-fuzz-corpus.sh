#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

python3 -u scripts/minimize_fuzz_corpus.py \
  --binary "fuzz-binaries/$FUZZ_TARGET" \
  --corpus "fuzz/corpus/$FUZZ_NAME" \
  --work-dir "fuzz/minimize/$FUZZ_NAME" \
  --artifacts "fuzz/artifacts/$FUZZ_NAME" \
  --shards "${MINIMIZE_SHARDS:-8}" \
  2>&1 | tee fuzz-minimize.log
