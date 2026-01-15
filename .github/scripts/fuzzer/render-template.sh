#!/bin/bash
# Render a template file by substituting {{VAR}} placeholders with environment variables
# Usage: render-template.sh <template_file> [output_file]
#
# All {{VAR_NAME}} patterns are replaced with $VAR_NAME from environment
# If a variable is not set, it's replaced with "(not set)"

set -euo pipefail

TEMPLATE_FILE="${1:-}"
OUTPUT_FILE="${2:-/dev/stdout}"

if [[ -z "$TEMPLATE_FILE" || ! -f "$TEMPLATE_FILE" ]]; then
    echo "Usage: $0 <template_file> [output_file]" >&2
    exit 1
fi

# Read template
content=$(cat "$TEMPLATE_FILE")

# Find all {{VAR}} patterns and substitute them
while [[ "$content" =~ \{\{([A-Z_][A-Z0-9_]*)\}\} ]]; do
    var_name="${BASH_REMATCH[1]}"
    var_value="${!var_name:-(not set)}"

    # Escape special characters in value for sed
    escaped_value=$(printf '%s\n' "$var_value" | sed -e 's/[&\\/]/\\&/g; s/$/\\/' -e '$s/\\$//')

    # Replace the placeholder
    content=$(echo "$content" | sed "s|{{${var_name}}}|${escaped_value}|g")
done

# Write output
echo "$content" > "$OUTPUT_FILE"
