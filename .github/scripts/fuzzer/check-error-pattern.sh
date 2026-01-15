#!/bin/bash
# Check if a normalized error pattern already exists in open fuzzer issues
# Usage: check-error-pattern.sh <message_hash> <error_variant> <issues_json>
#
# Outputs:
#   - match: true/false
#   - issue_number: matching issue number (if found)

set -euo pipefail

MESSAGE_HASH="${1:-}"
ERROR_VARIANT="${2:-}"
ISSUES_JSON="${3:-fuzzer_issues.json}"

if [[ -z "$MESSAGE_HASH" ]]; then
    echo '{"match": false, "check": "error_pattern", "reason": "no message hash provided"}'
    exit 0
fi

if [[ ! -f "$ISSUES_JSON" ]]; then
    echo '{"match": false, "check": "error_pattern", "reason": "no issues file"}'
    exit 0
fi

# First try: exact message hash match
match=$(jq -r --arg hash "$MESSAGE_HASH" '
    .[] | select(.body | contains($hash)) |
    {number: .number, url: .url, title: .title}
' "$ISSUES_JSON" | head -1)

if [[ -n "$match" && "$match" != "null" ]]; then
    issue_number=$(echo "$match" | jq -r '.number')
    issue_url=$(echo "$match" | jq -r '.url')
    issue_title=$(echo "$match" | jq -r '.title')

    cat << EOF
{
  "match": true,
  "check": "error_pattern",
  "confidence": "high",
  "issue_number": $issue_number,
  "issue_url": $(echo "$issue_url" | jq -R .),
  "issue_title": $(echo "$issue_title" | jq -R .),
  "reason": "Same error pattern (normalized message match)"
}
EOF
    exit 0
fi

# Second try: same error variant (lower confidence)
if [[ -n "$ERROR_VARIANT" && "$ERROR_VARIANT" != "unknown" ]]; then
    match=$(jq -r --arg variant "$ERROR_VARIANT" '
        .[] | select(.body | contains($variant)) |
        {number: .number, url: .url, title: .title}
    ' "$ISSUES_JSON" | head -1)

    if [[ -n "$match" && "$match" != "null" ]]; then
        issue_number=$(echo "$match" | jq -r '.number')
        issue_url=$(echo "$match" | jq -r '.url')
        issue_title=$(echo "$match" | jq -r '.title')

        cat << EOF
{
  "match": true,
  "check": "error_pattern",
  "confidence": "medium",
  "issue_number": $issue_number,
  "issue_url": $(echo "$issue_url" | jq -R .),
  "issue_title": $(echo "$issue_title" | jq -R .),
  "error_variant": $(echo "$ERROR_VARIANT" | jq -R .),
  "reason": "Same error variant type"
}
EOF
        exit 0
    fi
fi

cat << EOF
{
  "match": false,
  "check": "error_pattern",
  "reason": "No matching error pattern found"
}
EOF
