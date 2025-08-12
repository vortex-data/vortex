# Miri Test Optimization Guide

## Overview

Miri testing is essential for detecting undefined behavior in unsafe code, but it runs significantly slower than regular tests. This guide explains how to optimize miri test performance.

## Current Configuration

### Matrix Strategy
The CI uses a matrix strategy to run miri tests in parallel across 8 balanced groups:
- **small-crates** (~104 tests): vortex-buffer, vortex-bytebool, vortex-fsst, vortex-pco, vortex-zstd
- **core-and-io** (~100 tests): vortex, vortex-btrblocks, vortex-ipc, vortex-io
- **mask-and-layout** (~166 tests): vortex-mask, vortex-layout
- **dtype-and-dict** (~148 tests): vortex-dtype, vortex-dict, vortex-ffi
- **expr-and-alp** (~209 tests): vortex-expr, vortex-alp
- **fastlanes-and-runend** (~191 tests): vortex-fastlanes, vortex-runend
- **file** (~50 tests): vortex-file, vortex-flatbuffers
- **array** (747 tests, conformance disabled): vortex-array with `--cfg skip_conformance_tests`

### Excluded Crates
Some crates are excluded from miri testing:

**FFI/Interop** (cannot work with miri):
- vortex-jni (JNI/Java FFI)
- vortex-cxx (C++ interop)
- vortex-duckdb (DuckDB FFI)
- vortex-fuzz (fuzzing harness)

**Performance** (too many tests):
- vortex-scalar (459 tests, ~78s)
- vortex-array (747 tests, times out)

## Marking Tests for Miri Exclusion

For crates with many tests, you can exclude slow tests from miri:

```rust
#[test]
#[cfg_attr(miri, ignore)]
fn expensive_test() {
    // This test will be skipped when running under miri
}
```

Or for entire modules:
```rust
#[cfg(not(miri))]
mod expensive_tests {
    // All tests in this module skip miri
}
```

## Known Issues

### f16 Support
Tests using f16 (half-precision float) may fail under miri due to:
- Inline assembly not supported on ARM64
- Limited f16 operations in miri

Affected tests:
- `vortex-dtype::ptype::tests::max_value_u64`
- `vortex-dtype::ptype::tests::to_bytes_rt`

### Configuration for vortex-array

The vortex-array crate uses a custom cfg flag to skip conformance tests during miri runs:

- **`skip_conformance_tests`**: When set, conformance test functions become no-ops
- This is automatically set in CI for miri runs via `RUSTFLAGS="--cfg skip_conformance_tests"`
- Conformance tests are comprehensive but slow, making them unsuitable for miri

### Recommendations for vortex-scalar

This crate needs selective miri testing:

1. **Identify critical unsafe code** that must be tested with miri
2. **Mark non-critical tests** with `#[cfg_attr(miri, ignore)]`
3. **Create miri-specific test suites** that focus on unsafe operations

Example approach:
```rust
// In vortex-scalar/src/lib.rs or test modules
#[cfg(test)]
mod tests {
    // Critical unsafe test - always run with miri
    #[test]
    fn test_unsafe_buffer_access() {
        // Test unsafe memory operations
    }
    
    // Performance/integration test - skip miri
    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_large_dataset_processing() {
        // Test with large data that would be slow in miri
    }
}
```

## Running Miri Locally

```bash
# Install miri
rustup component add --toolchain nightly miri

# Run miri on a specific crate
cargo +nightly miri nextest run -p vortex-buffer

# Run miri with specific test filter
cargo +nightly miri test -p vortex-dtype --lib -- ptype::tests

# Check which tests would run
cargo test -p vortex-array --lib -- --list | grep test_name
```

## Monitoring Performance

Use the `scripts/measure-miri-runtime.sh` script to measure test times:
```bash
./scripts/measure-miri-runtime.sh
```

Target: Each matrix group should complete within 4 minutes.