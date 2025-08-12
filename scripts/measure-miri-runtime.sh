#!/usr/bin/env bash
set -euo pipefail

# Script to measure miri test runtime for each crate
# This helps identify which crates/tests are slow and need optimization

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Output file for results
RESULTS_FILE="miri-runtime-results.txt"

# List of crates to test (from our CI configuration)
CRATES=(
    "vortex"
    "vortex-alp"
    "vortex-array"
    "vortex-btrblocks"
    "vortex-buffer"
    "vortex-bytebool"
    "vortex-datafusion"
    "vortex-dict"
    "vortex-dtype"
    "vortex-expr"
    "vortex-fastlanes"
    "vortex-ffi"
    "vortex-file"
    "vortex-flatbuffers"
    "vortex-fsst"
    "vortex-io"
    "vortex-ipc"
    "vortex-layout"
    "vortex-mask"
    "vortex-pco"
    "vortex-runend"
    "vortex-scalar"
    "vortex-zstd"
)

echo "Measuring miri test runtime for each crate..."
echo "============================================="
echo "" | tee "$RESULTS_FILE"

# Track total time
TOTAL_START=$(date +%s)

# Test each crate individually
for crate in "${CRATES[@]}"; do
    echo -e "${BLUE}Testing $crate...${NC}"
    echo "Testing $crate..." >> "$RESULTS_FILE"
    
    # Record start time
    START=$(date +%s)
    
    # Run miri tests for this crate, capturing output
    # Using timeout to prevent hanging on particularly slow tests
    # 5 minute timeout per crate
    if timeout 300 cargo +nightly miri nextest run -p "$crate" --no-fail-fast 2>&1 | tee "miri-$crate.log" | grep -E "(PASS|FAIL|TIMEOUT|SLOW)" | tail -20; then
        END=$(date +%s)
        DURATION=$((END - START))
        
        # Count passed and failed tests
        PASSED=$(grep -c "PASS" "miri-$crate.log" || echo "0")
        FAILED=$(grep -c "FAIL" "miri-$crate.log" || echo "0")
        
        echo -e "${GREEN}✓ $crate completed in ${DURATION}s (Passed: $PASSED, Failed: $FAILED)${NC}"
        echo "  Duration: ${DURATION}s, Passed: $PASSED, Failed: $FAILED" >> "$RESULTS_FILE"
        
        # Find slow tests (tests taking more than 5 seconds)
        echo "  Slow tests (>5s):" >> "$RESULTS_FILE"
        grep -E "PASS \[[[:space:]]*[5-9]\.|PASS \[[[:space:]]*[0-9][0-9]+\." "miri-$crate.log" | head -10 >> "$RESULTS_FILE" || echo "    None" >> "$RESULTS_FILE"
        
    else
        EXIT_CODE=$?
        END=$(date +%s)
        DURATION=$((END - START))
        
        if [ $EXIT_CODE -eq 124 ]; then
            echo -e "${RED}✗ $crate TIMEOUT after ${DURATION}s${NC}"
            echo "  TIMEOUT after ${DURATION}s" >> "$RESULTS_FILE"
        else
            echo -e "${YELLOW}⚠ $crate failed/skipped after ${DURATION}s${NC}"
            echo "  Failed/skipped after ${DURATION}s" >> "$RESULTS_FILE"
        fi
    fi
    
    echo "" >> "$RESULTS_FILE"
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "============================================="
echo -e "${BLUE}Total time: ${TOTAL_DURATION}s${NC}"
echo "Total time: ${TOTAL_DURATION}s" >> "$RESULTS_FILE"

# Summary of crates by runtime
echo ""
echo "Summary by runtime:" | tee -a "$RESULTS_FILE"
echo "==================" | tee -a "$RESULTS_FILE"

# Parse results and sort by duration
grep "Duration:" "$RESULTS_FILE" | sed 's/.*Duration: \([0-9]*\)s.*/\1/' | sort -rn | head -10 | while read duration; do
    crate_info=$(grep -B1 "Duration: ${duration}s" "$RESULTS_FILE" | head -1)
    echo "  ${duration}s - ${crate_info}" | tee -a "$RESULTS_FILE"
done

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Individual logs saved as: miri-<crate>.log"