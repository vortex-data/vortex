#!/bin/bash

# Script to mark slow tests for miri exclusion based on patterns
# Usage: ./mark-slow-miri-tests.sh <crate-name>

set -e

CRATE=$1

if [ -z "$CRATE" ]; then
    echo "Usage: $0 <crate-name>"
    echo "Example: $0 vortex-scalar"
    exit 1
fi

echo "Analyzing $CRATE for potential slow miri tests..."

# Find test files
TEST_FILES=$(find $CRATE -name "*.rs" -path "*/src/*" -o -path "*/tests/*" | grep -E "test|mod\.rs|lib\.rs")

echo "Found test files:"
echo "$TEST_FILES"

echo ""
echo "Searching for potential slow tests..."
echo "Patterns: large, big, many, stress, bench, perf, conformance, consistency"
echo ""

for file in $TEST_FILES; do
    # Look for test functions that might be slow
    grep -n "#\[test\]\|#\[rstest\]\|fn test_" "$file" 2>/dev/null | while read -r line; do
        # Check if the test name contains keywords suggesting it might be slow
        if echo "$line" | grep -iE "large|big|many|stress|bench|perf|conformance|consistency|slow|heavy" > /dev/null; then
            echo "Potential slow test in $file:"
            echo "  $line"
            
            # Check if already marked with cfg_attr(miri, ignore)
            line_num=$(echo "$line" | cut -d: -f1)
            if [ "$line_num" -gt 1 ]; then
                prev_line=$((line_num - 1))
                if sed -n "${prev_line}p" "$file" | grep -q "cfg_attr(miri, ignore)"; then
                    echo "  -> Already marked for miri exclusion"
                else
                    echo "  -> Consider adding: #[cfg_attr(miri, ignore)]"
                fi
            fi
            echo ""
        fi
    done
done

echo ""
echo "To mark a test for exclusion, add this attribute above the test:"
echo "  #[cfg_attr(miri, ignore)] // Too slow for miri in CI"
echo ""
echo "For rstest cases, you can mark individual cases or the entire test."