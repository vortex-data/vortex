#!/bin/bash
# Extract crash information from fuzzer output log
# Usage: extract-crash-info.sh <log_file> <crash_file> <output_json>
#
# Outputs JSON with:
#   - panic_location: file:line where panic occurred
#   - panic_message: the panic/error message
#   - error_variant: VortexFuzzError variant if applicable
#   - stack_trace: array of top stack frames
#   - stack_trace_hash: sha256 of normalized stack trace
#   - crash_type: crash/leak/timeout/oom
#   - seed_hash: sha256 of the crash seed file

set -euo pipefail

LOG_FILE="${1:-}"
CRASH_FILE="${2:-}"
OUTPUT_JSON="${3:-/dev/stdout}"

if [[ -z "$LOG_FILE" ]]; then
    echo "Usage: $0 <log_file> <crash_file> [output_json]" >&2
    exit 1
fi

if [[ ! -f "$LOG_FILE" ]]; then
    echo "Error: Log file not found: $LOG_FILE" >&2
    exit 1
fi

# Extract panic location (file:line)
extract_panic_location() {
    # Look for "panicked at" pattern
    grep -oP "panicked at [^,]+, \K[^:]+:\d+" "$LOG_FILE" 2>/dev/null | head -1 || \
    # Or look for assertion failures
    grep -oP "assertion.*failed.*at \K[^:]+:\d+" "$LOG_FILE" 2>/dev/null | head -1 || \
    # Or extract from stack trace (first vortex_ frame)
    grep -oP "vortex[^/]+/src/[^:]+:\d+" "$LOG_FILE" 2>/dev/null | head -1 || \
    echo "unknown"
}

# Extract panic/error message
extract_panic_message() {
    # Look for Rust panic format: "panicked at file:line:\nmessage"
    # The message is on the line after "panicked at"
    grep -A1 "panicked at" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' | head -1 || \
    # Or look for "panicked at 'message'" format
    grep -oP "panicked at '\K[^']+(?=')" "$LOG_FILE" 2>/dev/null | head -1 || \
    # Or error message
    grep -oP "ERROR: \K.*" "$LOG_FILE" 2>/dev/null | head -1 || \
    # Or assertion message
    grep -oP "assertion \`?failed\`?: \K.*" "$LOG_FILE" 2>/dev/null | head -1 || \
    echo "unknown"
}

# Extract VortexFuzzError variant or panic type
extract_error_variant() {
    # Look for VortexFuzzError enum variants
    local variant
    variant=$(grep -oP "(ScalarMismatch|SearchSortedError|MinMaxMismatch|ArrayNotEqual|DTypeMismatch|LengthMismatch|VortexError)" "$LOG_FILE" 2>/dev/null | head -1)
    if [[ -n "$variant" ]]; then
        echo "$variant"
        return
    fi

    # Detect common panic types from message
    if grep -q "index out of bounds" "$LOG_FILE" 2>/dev/null; then
        echo "IndexOutOfBounds"
    elif grep -q "assertion.*failed" "$LOG_FILE" 2>/dev/null; then
        echo "AssertionFailed"
    elif grep -q "unwrap.*None" "$LOG_FILE" 2>/dev/null; then
        echo "UnwrapNone"
    elif grep -q "overflow" "$LOG_FILE" 2>/dev/null; then
        echo "Overflow"
    elif grep -q "out of memory\|OOM" "$LOG_FILE" 2>/dev/null; then
        echo "OutOfMemory"
    elif grep -q "timeout" "$LOG_FILE" 2>/dev/null; then
        echo "Timeout"
    elif grep -q "SEGV\|segfault" "$LOG_FILE" 2>/dev/null; then
        echo "Segfault"
    else
        echo "unknown"
    fi
}

# Extract stack trace frames (function names only, normalized)
extract_stack_frames() {
    # Extract frames like "#0 0x... in function_name"
    # Keep only function names, strip addresses
    grep -oP '#\d+\s+0x[a-f0-9]+\s+in\s+\K[^\s(]+' "$LOG_FILE" 2>/dev/null | \
    # Filter to vortex frames and std frames
    grep -E '^(vortex|std|core|alloc)' | \
    head -10 || \
    # Fallback: try rust backtrace format
    grep -oP '^\s+\d+:\s+\K[^\s]+' "$LOG_FILE" 2>/dev/null | \
    grep -E '^(vortex|std|core|alloc)' | \
    head -10 || \
    echo "unknown"
}

# Compute hash of stack trace (for deduplication)
compute_stack_hash() {
    local frames="$1"
    # Take top 5 frames, hash them
    echo "$frames" | head -5 | sha256sum | cut -d' ' -f1
}

# Determine crash type from filename
get_crash_type() {
    local crash_name="${1:-unknown}"
    if [[ "$crash_name" == crash-* ]]; then
        echo "crash"
    elif [[ "$crash_name" == leak-* ]]; then
        echo "leak"
    elif [[ "$crash_name" == timeout-* ]]; then
        echo "timeout"
    elif [[ "$crash_name" == oom-* ]]; then
        echo "oom"
    else
        echo "unknown"
    fi
}

# Compute seed hash
compute_seed_hash() {
    local crash_file="$1"
    if [[ -f "$crash_file" ]]; then
        sha256sum "$crash_file" | cut -d' ' -f1
    else
        echo "unknown"
    fi
}

# Main extraction
panic_location=$(extract_panic_location)
panic_message=$(extract_panic_message)
error_variant=$(extract_error_variant)
stack_frames=$(extract_stack_frames)
stack_hash=$(compute_stack_hash "$stack_frames")
crash_type=$(get_crash_type "$(basename "${CRASH_FILE:-}")")
seed_hash=$(compute_seed_hash "$CRASH_FILE")

# Convert stack frames to JSON array
stack_json=$(echo "$stack_frames" | jq -R -s 'split("\n") | map(select(length > 0))')

# Normalize panic message for pattern matching (replace numbers with placeholders)
normalized_message=$(echo "$panic_message" | sed -E 's/[0-9]+/N/g')
message_hash=$(echo "$normalized_message" | sha256sum | cut -d' ' -f1)

# Output JSON
cat > "$OUTPUT_JSON" << EOF
{
  "panic_location": $(echo "$panic_location" | jq -R .),
  "panic_message": $(echo "$panic_message" | jq -R .),
  "error_variant": $(echo "$error_variant" | jq -R .),
  "stack_frames": $stack_json,
  "stack_trace_hash": "$stack_hash",
  "normalized_message": $(echo "$normalized_message" | jq -R .),
  "message_hash": "$message_hash",
  "crash_type": "$crash_type",
  "seed_hash": "$seed_hash"
}
EOF
