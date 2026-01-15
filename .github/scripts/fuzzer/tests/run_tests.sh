#!/bin/bash
# Test runner for fuzzer crash reporting scripts
# Usage: ./run_tests.sh [test_name]
#
# Run all tests: ./run_tests.sh
# Run specific test: ./run_tests.sh test_extract_index_bounds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUZZER_DIR="$(dirname "$SCRIPT_DIR")"
FIXTURES_DIR="$SCRIPT_DIR/fixtures"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0
SKIPPED=0

# Helper functions
pass() {
    echo -e "${GREEN}PASS${NC}: $1"
    ((PASSED++)) || true
}

fail() {
    echo -e "${RED}FAIL${NC}: $1"
    echo "  Expected: $2"
    echo "  Got: $3"
    ((FAILED++)) || true
}

skip() {
    echo -e "${YELLOW}SKIP${NC}: $1 - $2"
    ((SKIPPED++)) || true
}

assert_eq() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        pass "$name"
    else
        fail "$name" "$expected" "$actual"
    fi
}

assert_contains() {
    local name="$1"
    local haystack="$2"
    local needle="$3"
    if [[ "$haystack" == *"$needle"* ]]; then
        pass "$name"
    else
        fail "$name" "contains '$needle'" "'$haystack'"
    fi
}

assert_json_field() {
    local name="$1"
    local json="$2"
    local field="$3"
    local expected="$4"
    local actual
    actual=$(echo "$json" | jq -r ".$field")
    assert_eq "$name" "$expected" "$actual"
}

# ============================================================================
# TESTS: extract-crash-info.sh
# ============================================================================

# Helper to run extract and get output
run_extract() {
    local log="$1"
    local crash="$2"
    local tmpfile
    tmpfile=$(mktemp)
    "$FUZZER_DIR/extract-crash-info.sh" "$log" "$crash" "$tmpfile"
    cat "$tmpfile"
    rm "$tmpfile"
}

test_extract_index_bounds() {
    echo "--- Testing extract-crash-info.sh with index out of bounds ---"
    local output
    output=$(run_extract "$FIXTURES_DIR/crash_index_bounds.log" "$FIXTURES_DIR/crash-abc123")

    assert_json_field "panic_location" "$output" "panic_location" "vortex-array/src/compute/slice.rs:142"
    assert_json_field "error_variant" "$output" "error_variant" "IndexOutOfBounds"
    assert_json_field "crash_type" "$output" "crash_type" "crash"
    assert_contains "panic_message contains bounds" "$(echo "$output" | jq -r '.panic_message')" "index out of bounds"
    assert_contains "has seed_hash" "$output" "seed_hash"
    assert_contains "has stack_trace_hash" "$output" "stack_trace_hash"
}

test_extract_scalar_mismatch() {
    echo "--- Testing extract-crash-info.sh with ScalarMismatch ---"
    local output
    output=$(run_extract "$FIXTURES_DIR/crash_scalar_mismatch.log" "$FIXTURES_DIR/crash-def456")

    assert_json_field "error_variant" "$output" "error_variant" "ScalarMismatch"
    assert_json_field "crash_type" "$output" "crash_type" "crash"
    assert_contains "panic_message contains mismatch" "$(echo "$output" | jq -r '.panic_message')" "mismatch"
}

test_extract_length_mismatch() {
    echo "--- Testing extract-crash-info.sh with LengthMismatch ---"
    local output
    output=$(run_extract "$FIXTURES_DIR/crash_length_mismatch.log" "$FIXTURES_DIR/crash-ghi789")

    assert_json_field "error_variant" "$output" "error_variant" "LengthMismatch"
    # Verify panic_message contains the length error info
    assert_contains "panic_message contains len" "$(echo "$output" | jq -r '.panic_message')" "len"
}

test_extract_timeout() {
    echo "--- Testing extract-crash-info.sh with timeout ---"
    local output
    output=$(run_extract "$FIXTURES_DIR/timeout.log" "$FIXTURES_DIR/timeout-jkl012")

    assert_json_field "crash_type" "$output" "crash_type" "timeout"
    assert_json_field "error_variant" "$output" "error_variant" "Timeout"
}

# ============================================================================
# TESTS: check-seed-hash.sh
# ============================================================================

test_seed_hash_match() {
    echo "--- Testing check-seed-hash.sh with matching hash ---"
    # Get the actual hash of an existing seed in the issues
    local output
    output=$("$FUZZER_DIR/check-seed-hash.sh" \
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "seed_hash match found" "$output" "match" "true"
    assert_json_field "seed_hash correct issue" "$output" "issue_number" "100"
}

test_seed_hash_no_match() {
    echo "--- Testing check-seed-hash.sh with no match ---"
    local output
    output=$("$FUZZER_DIR/check-seed-hash.sh" \
        "0000000000000000000000000000000000000000000000000000000000000000" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "seed_hash no match" "$output" "match" "false"
}

# ============================================================================
# TESTS: check-panic-location.sh
# ============================================================================

test_panic_location_match() {
    echo "--- Testing check-panic-location.sh with matching location ---"
    local output
    output=$("$FUZZER_DIR/check-panic-location.sh" \
        "vortex-array/src/compute/slice.rs:142" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "panic_location match found" "$output" "match" "true"
    assert_json_field "panic_location correct issue" "$output" "issue_number" "100"
}

test_panic_location_no_match() {
    echo "--- Testing check-panic-location.sh with no match ---"
    local output
    output=$("$FUZZER_DIR/check-panic-location.sh" \
        "some/other/file.rs:999" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "panic_location no match" "$output" "match" "false"
}

# ============================================================================
# TESTS: check-stack-trace.sh
# ============================================================================

test_stack_trace_match() {
    echo "--- Testing check-stack-trace.sh with matching hash ---"
    local output
    output=$("$FUZZER_DIR/check-stack-trace.sh" \
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "stack_trace match found" "$output" "match" "true"
    assert_json_field "stack_trace correct issue" "$output" "issue_number" "100"
}

test_stack_trace_no_match() {
    echo "--- Testing check-stack-trace.sh with no match ---"
    local output
    output=$("$FUZZER_DIR/check-stack-trace.sh" \
        "0000000000000000000000000000000000000000000000000000000000000000" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "stack_trace no match" "$output" "match" "false"
}

# ============================================================================
# TESTS: check-error-pattern.sh
# ============================================================================

test_error_pattern_match() {
    echo "--- Testing check-error-pattern.sh with matching hash ---"
    local output
    output=$("$FUZZER_DIR/check-error-pattern.sh" \
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" \
        "IndexOutOfBounds" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "error_pattern match found" "$output" "match" "true"
}

test_error_pattern_variant_match() {
    echo "--- Testing check-error-pattern.sh with variant match ---"
    local output
    output=$("$FUZZER_DIR/check-error-pattern.sh" \
        "nomatchhash0000000000000000000000000000000000000000000000000000" \
        "ScalarMismatch" \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "error_pattern variant match" "$output" "match" "true"
    assert_json_field "error_pattern variant issue" "$output" "issue_number" "101"
}

# ============================================================================
# TESTS: check-duplicate.sh (integration)
# ============================================================================

test_duplicate_chain_seed_match() {
    echo "--- Testing check-duplicate.sh chain with seed match ---"
    # Create a crash info that matches by seed hash
    local crash_info='{"seed_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","panic_location":"other.rs:1","stack_trace_hash":"xxx","message_hash":"yyy","error_variant":"Other"}'
    echo "$crash_info" > /tmp/test_crash_info.json

    local output
    output=$("$FUZZER_DIR/check-duplicate.sh" \
        /tmp/test_crash_info.json \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "duplicate chain seed match" "$output" "duplicate" "true"
    assert_json_field "duplicate chain check type" "$output" "check" "seed_hash"
    assert_json_field "duplicate chain check order" "$output" "check_order" "1"

    rm /tmp/test_crash_info.json
}

test_duplicate_chain_panic_location_match() {
    echo "--- Testing check-duplicate.sh chain with panic location match ---"
    # Create a crash info that matches by panic location (not seed)
    local crash_info='{"seed_hash":"nomatch0000000000000000000000000000000000000000000000000000000","panic_location":"vortex-array/src/compute/slice.rs:142","stack_trace_hash":"xxx","message_hash":"yyy","error_variant":"Other"}'
    echo "$crash_info" > /tmp/test_crash_info.json

    local output
    output=$("$FUZZER_DIR/check-duplicate.sh" \
        /tmp/test_crash_info.json \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "duplicate chain panic match" "$output" "duplicate" "true"
    assert_json_field "duplicate chain check type" "$output" "check" "panic_location"
    assert_json_field "duplicate chain check order" "$output" "check_order" "2"

    rm /tmp/test_crash_info.json
}

test_duplicate_chain_no_match() {
    echo "--- Testing check-duplicate.sh chain with no match ---"
    # Create a crash info that doesn't match anything
    local crash_info='{"seed_hash":"nomatch0000000000000000000000000000000000000000000000000000000","panic_location":"brand/new/file.rs:999","stack_trace_hash":"nomatch000000000000000000000000000000000000000000000000000000","message_hash":"nomatch000000000000000000000000000000000000000000000000000000","error_variant":"BrandNewError"}'
    echo "$crash_info" > /tmp/test_crash_info.json

    local output
    output=$("$FUZZER_DIR/check-duplicate.sh" \
        /tmp/test_crash_info.json \
        "$FIXTURES_DIR/existing_issues.json")

    assert_json_field "duplicate chain no match" "$output" "duplicate" "false"

    rm /tmp/test_crash_info.json
}

# ============================================================================
# TESTS: render-template.sh
# ============================================================================

test_render_template() {
    echo "--- Testing render-template.sh ---"
    # Create a simple template
    cat > /tmp/test_template.md << 'EOF'
# {{TITLE}}

Target: {{TARGET}}
Value: {{VALUE}}
EOF

    export TITLE="Test Title"
    export TARGET="file_io"
    export VALUE="42"

    local tmpout
    tmpout=$(mktemp)
    "$FUZZER_DIR/render-template.sh" /tmp/test_template.md "$tmpout"
    local output
    output=$(cat "$tmpout")
    rm "$tmpout"

    assert_contains "render title" "$output" "Test Title"
    assert_contains "render target" "$output" "file_io"
    assert_contains "render value" "$output" "42"

    rm /tmp/test_template.md
    unset TITLE TARGET VALUE
}

test_render_template_missing_var() {
    echo "--- Testing render-template.sh with missing variable ---"
    cat > /tmp/test_template.md << 'EOF'
Value: {{MISSING_VAR}}
EOF

    local tmpout
    tmpout=$(mktemp)
    "$FUZZER_DIR/render-template.sh" /tmp/test_template.md "$tmpout"
    local output
    output=$(cat "$tmpout")
    rm "$tmpout"

    assert_contains "render missing var" "$output" "(not set)"

    rm /tmp/test_template.md
}

# ============================================================================
# MAIN
# ============================================================================

run_all_tests() {
    echo "========================================"
    echo "Running fuzzer script tests"
    echo "========================================"
    echo ""

    # Extract tests
    test_extract_index_bounds
    test_extract_scalar_mismatch
    test_extract_length_mismatch
    test_extract_timeout

    echo ""

    # Seed hash tests
    test_seed_hash_match
    test_seed_hash_no_match

    echo ""

    # Panic location tests
    test_panic_location_match
    test_panic_location_no_match

    echo ""

    # Stack trace tests
    test_stack_trace_match
    test_stack_trace_no_match

    echo ""

    # Error pattern tests
    test_error_pattern_match
    test_error_pattern_variant_match

    echo ""

    # Duplicate chain tests
    test_duplicate_chain_seed_match
    test_duplicate_chain_panic_location_match
    test_duplicate_chain_no_match

    echo ""

    # Template tests
    test_render_template
    test_render_template_missing_var

    echo ""
    echo "========================================"
    echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}, ${YELLOW}$SKIPPED skipped${NC}"
    echo "========================================"

    if [[ $FAILED -gt 0 ]]; then
        exit 1
    fi
}

# Run specific test or all tests
if [[ $# -gt 0 ]]; then
    # Run specific test
    test_name="$1"
    if declare -f "$test_name" > /dev/null; then
        "$test_name"
        echo ""
        echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
        if [[ $FAILED -gt 0 ]]; then
            exit 1
        fi
    else
        echo "Unknown test: $test_name"
        echo "Available tests:"
        declare -F | grep "test_" | awk '{print "  " $3}'
        exit 1
    fi
else
    run_all_tests
fi
