#!/usr/bin/env bash
set -euo pipefail

# Script to find all crates in the workspace that use the 'unsafe' keyword
# This is used to determine which crates should be tested with miri

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find all Cargo.toml files in the workspace (excluding target directory)
CARGO_TOMLS=$(find . -name "Cargo.toml" -not -path "./target/*" -not -path "./.git/*" | sort)

# Array to store crates with unsafe code
declare -a UNSAFE_CRATES=()

echo "Scanning for crates that use 'unsafe' keyword..."
echo "================================================"

for cargo_toml in $CARGO_TOMLS; do
    # Get the directory containing the Cargo.toml
    crate_dir=$(dirname "$cargo_toml")
    
    # Skip the root Cargo.toml
    if [ "$crate_dir" = "." ]; then
        continue
    fi
    
    # Extract the crate name from Cargo.toml
    if grep -q '^\[package\]' "$cargo_toml" 2>/dev/null; then
        crate_name=$(grep '^name = ' "$cargo_toml" | head -1 | sed 's/name = "\(.*\)"/\1/')
        
        if [ -z "$crate_name" ]; then
            continue
        fi
        
        # Search for 'unsafe' keyword in all Rust files in this crate
        # Exclude test files if we only want to check production code
        if find "$crate_dir" -name "*.rs" -type f -not -path "*/target/*" -exec grep -l '\bunsafe\b' {} \; 2>/dev/null | head -1 | grep -q .; then
            echo -e "${YELLOW}Found unsafe code in:${NC} $crate_name (at $crate_dir)"
            UNSAFE_CRATES+=("$crate_name")
        fi
    fi
done

echo ""
echo "================================================"
echo -e "${GREEN}Summary:${NC}"
echo "Found ${#UNSAFE_CRATES[@]} crates using 'unsafe' keyword:"
echo ""

# Output the list of crates for easy copying
for crate in "${UNSAFE_CRATES[@]}"; do
    echo "  - $crate"
done

# Also output as a space-separated list for use in CI
echo ""
echo "As space-separated list (for CI):"
echo -n "${UNSAFE_CRATES[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '
echo ""

# Output as cargo package flags for miri
echo ""
echo "As cargo package flags for miri:"
for crate in "${UNSAFE_CRATES[@]}"; do
    echo -n " -p $crate"
done
echo ""