#!/bin/bash
# Check if a panic location already exists in open fuzzer issues
# Usage: check-panic-location.sh <panic_location> <issues_json>
#
# Outputs:
#   - match: true/false
#   - issue_number: matching issue number (if found)

set -euo pipefail

PANIC_LOCATION="${1:-}"
ISSUES_JSON="${2:-fuzzer_issues.json}"

if [[ -z "$PANIC_LOCATION" || "$PANIC_LOCATION" == "unknown" ]]; then
    echo '{"match": false, "check": "panic_location", "reason": "no panic location provided"}'
    exit 0
fi

if [[ ! -f "$ISSUES_JSON" ]]; then
    echo '{"match": false, "check": "panic_location", "reason": "no issues file"}'
    exit 0
fi

# Extract file:line pattern
# e.g., "vortex-array/src/compute/slice.rs:142"
file_pattern=$(echo "$PANIC_LOCATION" | grep -oP '[^/]+\.rs:\d+' || echo "$PANIC_LOCATION")

# Search for this pattern in issue bodies
match=$(jq -r --arg loc "$file_pattern" '
    .[] | select(.body | test($loc; "i")) |
    {number: .number, url: .url, title: .title}
' "$ISSUES_JSON" 2>/dev/null | head -1)

if [[ -n "$match" && "$match" != "null" ]]; then
    issue_number=$(echo "$match" | jq -r '.number')
    issue_url=$(echo "$match" | jq -r '.url')
    issue_title=$(echo "$match" | jq -r '.title')

    cat << EOF
{
  "match": true,
  "check": "panic_location",
  "confidence": "high",
  "issue_number": $issue_number,
  "issue_url": $(echo "$issue_url" | jq -R .),
  "issue_title": $(echo "$issue_title" | jq -R .),
  "panic_location": $(echo "$PANIC_LOCATION" | jq -R .),
  "reason": "Same panic location (file:line)"
}
EOF
else
    cat << EOF
{
  "match": false,
  "check": "panic_location",
  "panic_location": $(echo "$PANIC_LOCATION" | jq -R .),
  "reason": "No matching panic location found"
}
EOF
fi
