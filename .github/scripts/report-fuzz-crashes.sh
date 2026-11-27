#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

shopt -s nullglob
status=0
for log in logs/fuzz_repro/*.log; do
  crash=$(basename "$log" .log)
  python3 -m fuzz_report extract "$log" \
    --crash-dir crash_artifacts --crash-name "$crash" -o crash_info.json
  jq -e '.error_variant != "unknown" or .panic_message != "unknown"' crash_info.json \
    > /dev/null || continue
  gh issue list --repo "$REPO" --label fuzzer --state open \
    --json number,title,body,url --limit 100 > fuzzer_issues.json
  GITHUB_OUTPUT=/dev/null python3 -m fuzz_report check-duplicate \
    crash_info.json fuzzer_issues.json -o dedup_result.json
  : > report_output
  GITHUB_OUTPUT=report_output python3 -m fuzz_report report crash_info.json \
    --repo "$REPO" --dedup-result dedup_result.json \
    -v "FUZZ_TARGET=$FUZZ_TARGET" -v "FUZZ_NAME=$FUZZ_NAME" -v "CRASH_FILE=$crash" \
    -v "BRANCH=$BRANCH" -v "COMMIT=$COMMIT" -v "ARTIFACT_URL=$ARTIFACT_URL" || status=1
  if [ ! -f new_crash_info.json ] && grep -q "^issue_created=true" report_output; then
    cp crash_info.json new_crash_info.json
    cat report_output >> "$GITHUB_OUTPUT"
  fi
  grep -E "^(validation_failed|missing_variables)=" report_output >> "$GITHUB_OUTPUT" || true
done
exit "$status"
