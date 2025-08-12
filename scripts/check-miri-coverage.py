#!/usr/bin/env python3
"""
Check that all crates using 'unsafe' are being tested with miri.

This script:
1. Finds all crates that use the 'unsafe' keyword
2. Checks which crates are allowlisted (can skip miri tests)
3. Verifies that all remaining unsafe crates are tested with miri
4. Exits with error if any unsafe crate is missing miri coverage
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Set, List

# Allowlist of crates that can skip miri tests
# These crates may have issues with miri due to:
# - FFI/JNI bindings that miri cannot handle
# - C++ interop
# - Platform-specific code
# - Known miri limitations (e.g., f16 support)
MIRI_ALLOWLIST: Set[str] = {
    "vortex-jni",         # JNI/Java FFI
    "vortex-cxx",         # C++ interop  
    "vortex-duckdb",      # DuckDB integration with FFI
    "vortex-fuzz",        # Fuzzing harness may not work with miri
    "vortex-datafusion",  # DataFusion integration - complex and not critical for unsafe validation
    # Note: vortex-ffi is partially tested with miri (some tests work)
    # Note: vortex-scalar and vortex-array are now tested in groups 8 and 9
}

def find_crates_with_unsafe() -> Set[str]:
    """Find all crates in the workspace that use the 'unsafe' keyword."""
    unsafe_crates = set()
    
    # Find all Cargo.toml files
    for cargo_toml in Path(".").rglob("Cargo.toml"):
        # Skip target directory and root Cargo.toml
        if "target" in cargo_toml.parts or cargo_toml.parent == Path("."):
            continue
        
        # Read the Cargo.toml to get the crate name
        try:
            with open(cargo_toml, 'r') as f:
                content = f.read()
                match = re.search(r'^name = "([^"]+)"', content, re.MULTILINE)
                if not match:
                    continue
                crate_name = match.group(1)
                
                # Check if this crate has any unsafe code
                crate_dir = cargo_toml.parent
                has_unsafe = False
                
                for rs_file in crate_dir.rglob("*.rs"):
                    if "target" in rs_file.parts:
                        continue
                    
                    try:
                        with open(rs_file, 'r') as f:
                            if re.search(r'\bunsafe\b', f.read()):
                                has_unsafe = True
                                break
                    except Exception:
                        continue
                
                if has_unsafe:
                    unsafe_crates.add(crate_name)
                    
        except Exception as e:
            print(f"Error processing {cargo_toml}: {e}", file=sys.stderr)
            continue
    
    return unsafe_crates

def get_miri_tested_crates() -> Set[str]:
    """Extract crates being tested with miri from CI configuration."""
    miri_crates = set()
    
    ci_file = Path(".github/workflows/ci.yml")
    if not ci_file.exists():
        print(f"CI file not found: {ci_file}", file=sys.stderr)
        return miri_crates
    
    with open(ci_file, 'r') as f:
        content = f.read()
        
        # Find the miri job section with matrix strategy
        # Look for the matrix groups and extract all package names
        miri_job_pattern = r'miri:.*?matrix:.*?group:(.*?)(?=\n\s{0,4}\w+:|\Z)'
        miri_job_match = re.search(miri_job_pattern, content, re.MULTILINE | re.DOTALL)
        
        if miri_job_match:
            matrix_section = miri_job_match.group(1)
            # Extract all package names from -p flags in the crates field
            package_pattern = r'-p\s+([\w-]+)'
            for package_match in re.finditer(package_pattern, matrix_section):
                miri_crates.add(package_match.group(1))
    
    return miri_crates

def main():
    """Main function to check miri coverage."""
    print("Checking miri test coverage for unsafe crates...")
    print("=" * 50)
    
    # Find all crates with unsafe code
    unsafe_crates = find_crates_with_unsafe()
    print(f"Found {len(unsafe_crates)} crates using 'unsafe' keyword:")
    for crate in sorted(unsafe_crates):
        print(f"  - {crate}")
    
    # Get crates currently tested with miri
    miri_tested = get_miri_tested_crates()
    print(f"\nCurrently testing {len(miri_tested)} crates with miri:")
    for crate in sorted(miri_tested):
        print(f"  - {crate}")
    
    # Remove allowlisted crates from the check
    required_miri_crates = unsafe_crates - MIRI_ALLOWLIST
    
    if MIRI_ALLOWLIST:
        print(f"\nAllowlisted crates (skipping miri):")
        for crate in sorted(MIRI_ALLOWLIST):
            print(f"  - {crate}")
    
    # Find crates that need miri but don't have it
    missing_miri = required_miri_crates - miri_tested
    
    if missing_miri:
        print(f"\n❌ ERROR: {len(missing_miri)} crates use 'unsafe' but are not tested with miri:")
        for crate in sorted(missing_miri):
            print(f"  - {crate}")
        
        print("\nTo fix this, update the miri job in .github/workflows/ci.yml to include these packages.")
        print("\nOr, if a crate should be exempt from miri testing, add it to MIRI_ALLOWLIST in this script.")
        sys.exit(1)
    else:
        print("\n✅ All unsafe crates are tested with miri!")
        
    # Check for unnecessary miri tests
    unnecessary_miri = miri_tested - unsafe_crates
    if unnecessary_miri:
        print(f"\n⚠️  Warning: {len(unnecessary_miri)} crates are tested with miri but don't use 'unsafe':")
        for crate in sorted(unnecessary_miri):
            print(f"  - {crate}")
        print("Consider removing these from miri tests to save CI time.")

if __name__ == "__main__":
    main()