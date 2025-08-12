# Miri Testing for Unsafe Code

This directory contains scripts to ensure that all crates using `unsafe` code are tested with miri, Rust's interpreter for detecting undefined behavior.

## Scripts

### `find-unsafe-crates.sh`
Bash script that identifies all crates in the workspace that use the `unsafe` keyword.

**Usage:**
```bash
./scripts/find-unsafe-crates.sh
```

**Output:**
- Lists all crates containing unsafe code
- Provides output in multiple formats (list, space-separated, cargo package flags)

### `check-miri-coverage.py`
Python script that verifies all unsafe-using crates are tested with miri in CI.

**Usage:**
```bash
python3 scripts/check-miri-coverage.py
```

**Features:**
- Finds all crates with unsafe code
- Checks which crates are configured for miri testing in CI
- Validates that all unsafe crates (minus allowlisted ones) are tested
- Exits with error if coverage is incomplete

## CI Integration

The miri job in `.github/workflows/ci.yml`:
1. Uses a matrix strategy with 10 parallel groups for better performance
2. The `meta` group runs `check-miri-coverage.py` to ensure all unsafe crates are covered
3. Groups 1-9 execute miri tests on different sets of crates
4. Each group is balanced by test count for optimal parallelization

## Allowlist

Some crates are allowlisted from miri testing due to:
- **FFI/JNI bindings**: `vortex-jni`, `vortex-cxx`, `vortex-duckdb`
- **Fuzzing harness**: `vortex-fuzz`
- **Complex integration**: `vortex-datafusion`

To add a crate to the allowlist, edit `MIRI_ALLOWLIST` in `check-miri-coverage.py`.

## Adding New Crates

When adding a new crate that uses `unsafe`:
1. The CI check will automatically detect it
2. Add it to the miri test list in `.github/workflows/ci.yml`
3. Or add it to `MIRI_ALLOWLIST` if it cannot be tested with miri

## Troubleshooting

### Miri failures with f16
Some tests using f16 (half-precision float) may fail under miri due to limited support.

### FFI/JNI issues
Crates with foreign function interfaces typically cannot be fully tested with miri.

### Large test suites
For crates with many tests (like vortex-array with 747 tests or vortex-scalar with 459 tests), 
we use `#[cfg_attr(miri, ignore)]` on slow tests to reduce miri runtime.

Tests marked for exclusion:
- Tests with "large" in the name (typically 100+ seconds)
- Tests with "many_small_chunks" (typically 50-125+ seconds)  
- Other tests taking >50 seconds under miri

See `vortex-array-miri-optimization.md` for detailed analysis.

### Running miri locally
```bash
# Install miri
rustup component add --toolchain nightly miri

# Run miri on a specific crate
cargo +nightly miri test -p vortex-buffer

# Run with nextest (as CI does)
cargo +nightly miri nextest run -p vortex-buffer
```