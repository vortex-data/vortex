#!/bin/bash
# Main deduplication check - chains all individual checks
# Usage: check-duplicate.sh <crash_info_json> <issues_json>
#
# Runs checks in order of confidence:
#   1. Seed hash (exact match)
#   2. Panic location (same file:line)
#   3. Stack trace hash (same call path)
#   4. Error pattern (normalized message)
#
# If ANY check matches, returns duplicate=true
# Outputs combined result JSON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRASH_INFO="${1:-}"
ISSUES_JSON="${2:-fuzzer_issues.json}"

if [[ -z "$CRASH_INFO" || ! -f "$CRASH_INFO" ]]; then
    echo "Usage: $0 <crash_info_json> [issues_json]" >&2
    exit 1
fi

# Load crash info
seed_hash=$(jq -r '.seed_hash' "$CRASH_INFO")
panic_location=$(jq -r '.panic_location' "$CRASH_INFO")
stack_hash=$(jq -r '.stack_trace_hash' "$CRASH_INFO")
message_hash=$(jq -r '.message_hash' "$CRASH_INFO")
error_variant=$(jq -r '.error_variant' "$CRASH_INFO")

# Array to collect all check results
checks=()

# Check 1: Seed hash (exact duplicate)
result=$("$SCRIPT_DIR/check-seed-hash.sh" "$seed_hash" "$ISSUES_JSON")
checks+=("$result")
if [[ $(echo "$result" | jq -r '.match') == "true" ]]; then
    echo "$result" | jq '. + {duplicate: true, check_order: 1}'
    exit 0
fi

# Check 2: Panic location (same crash site)
result=$("$SCRIPT_DIR/check-panic-location.sh" "$panic_location" "$ISSUES_JSON")
checks+=("$result")
if [[ $(echo "$result" | jq -r '.match') == "true" ]]; then
    echo "$result" | jq '. + {duplicate: true, check_order: 2}'
    exit 0
fi

# Check 3: Stack trace hash (same call path)
result=$("$SCRIPT_DIR/check-stack-trace.sh" "$stack_hash" "$ISSUES_JSON")
checks+=("$result")
if [[ $(echo "$result" | jq -r '.match') == "true" ]]; then
    echo "$result" | jq '. + {duplicate: true, check_order: 3}'
    exit 0
fi

# Check 4: Error pattern (normalized message)
result=$("$SCRIPT_DIR/check-error-pattern.sh" "$message_hash" "$error_variant" "$ISSUES_JSON")
checks+=("$result")
if [[ $(echo "$result" | jq -r '.match') == "true" ]]; then
    echo "$result" | jq '. + {duplicate: true, check_order: 4}'
    exit 0
fi

# No matches found
cat << EOF
{
  "duplicate": false,
  "checks_run": 4,
  "reason": "No duplicate detected by any check"
}
EOF
