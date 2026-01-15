#!/bin/bash
# Render a template file by substituting {{VAR}} placeholders with environment variables
# Usage: render-template.sh <template_file> [output_file]
#
# All {{VAR_NAME}} patterns are replaced with $VAR_NAME from environment
# If a variable is not set, it's replaced with "(not set)"
# Handles multiline values properly

set -euo pipefail

TEMPLATE_FILE="${1:-}"
OUTPUT_FILE="${2:-/dev/stdout}"

if [[ -z "$TEMPLATE_FILE" || ! -f "$TEMPLATE_FILE" ]]; then
    echo "Usage: $0 <template_file> [output_file]" >&2
    exit 1
fi

# Use awk for more robust multiline handling
awk '
{
    line = $0
    # Find all {{VAR}} patterns in the line
    while (match(line, /\{\{[A-Z_][A-Z0-9_]*\}\}/)) {
        # Extract the variable name (without braces)
        var_with_braces = substr(line, RSTART, RLENGTH)
        var_name = substr(var_with_braces, 3, length(var_with_braces) - 4)

        # Get value from environment
        var_value = ENVIRON[var_name]
        if (var_value == "") {
            var_value = "(not set)"
        }

        # Replace the placeholder
        line = substr(line, 1, RSTART - 1) var_value substr(line, RSTART + RLENGTH)
    }
    print line
}
' "$TEMPLATE_FILE" > "$OUTPUT_FILE"
