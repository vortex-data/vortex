#!/bin/bash
# Check if a stack trace hash already exists in open fuzzer issues
# Usage: check-stack-trace.sh <stack_hash> <issues_json>
#
# Outputs:
#   - match: true/false
#   - issue_number: matching issue number (if found)

set -euo pipefail

STACK_HASH="${1:-}"
ISSUES_JSON="${2:-fuzzer_issues.json}"

if [[ -z "$STACK_HASH" || "$STACK_HASH" == "unknown" ]]; then
    echo '{"match": false, "check": "stack_trace", "reason": "no stack hash provided"}'
    exit 0
fi

if [[ ! -f "$ISSUES_JSON" ]]; then
    echo '{"match": false, "check": "stack_trace", "reason": "no issues file"}'
    exit 0
fi

# Search for stack hash in issue bodies
# The stack hash appears in the "Stack Hash" field of our template
# Use -c for compact output (one JSON per line)
match=$(jq -c --arg hash "$STACK_HASH" '
    .[] | select(.body | contains($hash)) |
    {number: .number, url: .url, title: .title}
' "$ISSUES_JSON" 2>/dev/null | head -1)

if [[ -n "$match" && "$match" != "null" ]]; then
    issue_number=$(echo "$match" | jq -r '.number')
    issue_url=$(echo "$match" | jq -r '.url')
    issue_title=$(echo "$match" | jq -r '.title')

    cat << EOF
{
  "match": true,
  "check": "stack_trace",
  "confidence": "high",
  "issue_number": $issue_number,
  "issue_url": $(echo "$issue_url" | jq -R .),
  "issue_title": $(echo "$issue_title" | jq -R .),
  "reason": "Same stack trace (top 5 frames match)"
}
EOF
else
    cat << EOF
{
  "match": false,
  "check": "stack_trace",
  "reason": "No matching stack trace hash found"
}
EOF
fi
