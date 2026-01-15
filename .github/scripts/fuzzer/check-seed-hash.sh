#!/bin/bash
# Check if a crash seed hash already exists in open fuzzer issues
# Usage: check-seed-hash.sh <seed_hash> <issues_json>
#
# Outputs:
#   - match: true/false
#   - issue_number: matching issue number (if found)
#   - issue_url: matching issue URL (if found)

set -euo pipefail

SEED_HASH="${1:-}"
ISSUES_JSON="${2:-fuzzer_issues.json}"

if [[ -z "$SEED_HASH" ]]; then
    echo "Usage: $0 <seed_hash> [issues_json]" >&2
    exit 1
fi

if [[ ! -f "$ISSUES_JSON" ]]; then
    echo '{"match": false, "reason": "no issues file"}'
    exit 0
fi

# Search for seed hash in issue bodies
# The seed hash appears in the "Seed Hash" field of our template
match=$(jq -r --arg hash "$SEED_HASH" '
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
  "check": "seed_hash",
  "confidence": "exact",
  "issue_number": $issue_number,
  "issue_url": $(echo "$issue_url" | jq -R .),
  "issue_title": $(echo "$issue_title" | jq -R .),
  "reason": "Exact seed hash match - same crash input"
}
EOF
else
    cat << EOF
{
  "match": false,
  "check": "seed_hash",
  "reason": "No matching seed hash found"
}
EOF
fi
