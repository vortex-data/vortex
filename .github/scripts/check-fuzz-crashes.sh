#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

# Find actual crash files, not just the directory structure
FIRST_CRASH=$(find fuzz/artifacts -type f \( -name "crash-*" -o -name "leak-*" -o -name "timeout-*" -o -name "oom-*" \) 2>/dev/null | head -1 || true)

if [ -n "$FIRST_CRASH" ]; then
  echo "crashes_found=true" >> "$GITHUB_OUTPUT"
  echo "first_crash=$FIRST_CRASH" >> "$GITHUB_OUTPUT"
  echo "first_crash_name=$(basename "$FIRST_CRASH")" >> "$GITHUB_OUTPUT"

  # Count all crashes for reporting
  CRASH_COUNT=$(find fuzz/artifacts -type f \( -name "crash-*" -o -name "leak-*" -o -name "timeout-*" -o -name "oom-*" \) | wc -l)
  echo "crash_count=$CRASH_COUNT" >> "$GITHUB_OUTPUT"
  echo "Found $CRASH_COUNT crash(es), will process first: $(basename "$FIRST_CRASH")"
else
  echo "crashes_found=false" >> "$GITHUB_OUTPUT"
  echo "crash_count=0" >> "$GITHUB_OUTPUT"
  echo "No crashes found"
fi
