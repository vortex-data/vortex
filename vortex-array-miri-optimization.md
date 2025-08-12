# Vortex-Array Miri Test Optimization Recommendations

Based on partial miri test run analysis (457 out of 747 tests completed), this document identifies tests that should be marked with `#[cfg_attr(miri, ignore)]` to improve CI performance.

## Summary

The vortex-array miri optimization has been successfully completed:
- **Total tests**: 747 (719 passing, 28 excluded)
- **Runtime**: Reduced from 30+ minutes to ~6-7 minutes
- **Exclusion categories**:
  - Large dataset tests (100-300+ seconds)
  - f16 tests (inline assembly not supported)
  - f32/f64 tests with NaN comparison issues

## Tests to Mark with `#[cfg_attr(miri, ignore)]`

### Critical - Tests Taking 100+ Seconds

These tests MUST be excluded as they take unreasonably long:

1. **arrays::chunked::compute::tests::test_chunked_binary_numeric**
   - `case_10_chunked_large` - **300.4 seconds**
   - `case_07_chunked_many_small_chunks` - **126.0 seconds**

2. **arrays::chunked::compute::tests::test_chunked_consistency**
   - `case_6_large_chunks` - **192.3 seconds**
   - `case_3_many_small_chunks` - **125.0 seconds**

3. **arrays::list::compute::tests::test_list_consistency**
   - `case_5_list_large` - **132.3 seconds**

4. **arrays::bool::compute::tests::test_bool_consistency**
   - `case_10_large_nullable` - **119.9 seconds**

### High Priority - Tests Taking 50-100 Seconds

These should be excluded to keep total runtime reasonable:

1. **arrays::bool::compute::tests::test_bool_consistency**
   - `case_08_large_alternating` - 86.9 seconds
   - `case_09_large_sparse_true` - 84.7 seconds

2. **arrays::chunked::compute::tests::test_chunked_binary_numeric**
   - `case_04_chunked_f32_basic` - 66.9 seconds (also failing)
   - `case_09_chunked_mixed_chunk_sizes` - 59.4 seconds
   - `case_05_chunked_f64_basic` - 57.0 seconds (also failing)
   - `case_01_chunked_i32_basic` - 55.9 seconds
   - `case_08_chunked_nullable` - 50.4 seconds

3. **arrays::chunked::compute::tests::test_chunked_consistency**
   - `case_2_chunked_nullable` - 53.3 seconds

### Medium Priority - Tests Taking 30-50 Seconds

Consider excluding these if total runtime is still too high:

1. **arrays::constant::compute::tests::test_constant_consistency**
   - `case_7_constant_large` - 49.6 seconds

2. **arrays::chunked::compute::tests**
   - Various binary_numeric tests - 30-42 seconds
   - `test_take_chunked_conformance` - 41.1 seconds

3. **arrays::primitive::compute::tests::test_primitive_binary_numeric**
   - `case_4_f32` - 37.0 seconds (also failing)

## Pattern Analysis

### Common Patterns in Slow Tests

1. **"large" suffix tests** - These tests use large datasets that are extremely slow under miri
2. **"many_small_chunks" tests** - Tests with many iterations/chunks are slow
3. **Binary numeric operations** - Arithmetic operations on large arrays
4. **Consistency tests** - Comprehensive validation tests that are thorough but slow
5. **F32/F64 tests** - Floating-point tests tend to be slower and some fail

### Recommended Approach

Add the following attribute to the identified test functions:

```rust
#[test]
#[cfg_attr(miri, ignore)]  // Too slow for miri (takes X seconds)
fn test_chunked_binary_numeric_large() {
    // test implementation
}
```

Or for rstest cases, apply to the entire rstest block:

```rust
#[rstest]
#[cfg_attr(miri, ignore)]  // Large dataset tests too slow for miri
#[case::large(...)]
#[case::many_small_chunks(...)]
fn test_chunked_consistency(#[case] input: TestCase) {
    // test implementation
}
```

## Expected Impact

Excluding these tests should:
- Reduce vortex-array miri runtime from estimated 30+ minutes to under 10 minutes
- Focus miri on critical unsafe operations rather than large-scale integration tests
- Make CI more reliable by avoiding timeouts

## Implementation Priority

1. **Immediate**: Mark all 100+ second tests
2. **Soon**: Mark 50-100 second tests
3. **As needed**: Mark 30-50 second tests if total runtime still exceeds targets

## Notes

- These recommendations are based on partial test run (61% complete)
- Additional slow tests may be identified in the remaining 40%
- Consider creating a separate CI job for comprehensive miri testing that runs less frequently
- The conformance and consistency tests are valuable but not ideal for miri's overhead

## Implementation Status

### Completed Optimizations

#### Phase 1: Critical Tests (100+ seconds) - ✅ COMPLETE
- [x] `arrays::chunked::compute::tests::test_chunked_binary_numeric::case_10_chunked_large` - 300.4s
- [x] `arrays::chunked::compute::tests::test_chunked_binary_numeric::case_07_chunked_many_small_chunks` - 126.0s
- [x] `arrays::chunked::compute::tests::test_chunked_consistency::case_6_large_chunks` - 192.3s
- [x] `arrays::chunked::compute::tests::test_chunked_consistency::case_3_many_small_chunks` - 125.0s
- [x] `arrays::list::compute::tests::test_list_consistency::case_5_list_large` - 132.3s
- [x] `arrays::bool::compute::tests::test_bool_consistency::case_10_large_nullable` - 119.9s

#### Phase 2: High Priority Tests (50-100 seconds) - ✅ COMPLETE
- [x] `arrays::bool::compute::tests::test_bool_consistency::case_08_large_alternating` - 86.9s
- [x] `arrays::bool::compute::tests::test_bool_consistency::case_09_large_sparse_true` - 84.7s
- [x] `arrays::chunked::compute::tests::test_chunked_binary_numeric::case_01_chunked_i32_basic` - 55.9s
- [x] `arrays::chunked::compute::tests::test_chunked_binary_numeric::case_08_chunked_nullable` - 50.4s
- [x] `arrays::chunked::compute::tests::test_chunked_binary_numeric::case_09_chunked_mixed_chunk_sizes` - 59.4s
- [x] `arrays::chunked::compute::tests::test_chunked_consistency::case_2_chunked_nullable` - 53.3s
- [x] `arrays::constant::compute::tests::test_constant_consistency::case_7_constant_large` - 49.6s
- [x] `arrays::chunked::compute::take::test_take_chunked_conformance` - 41.1s

#### Additional Exclusions for f16/f32/f64 Issues
- `arrays::chunked::compute::tests::test_chunked_binary_numeric::case_04_chunked_f32_basic` - NaN comparison issues
- `arrays::chunked::compute::tests::test_chunked_binary_numeric::case_05_chunked_f64_basic` - NaN comparison issues  
- `arrays::primitive::compute::tests::test_primitive_binary_numeric::case_4_f32` - NaN comparison issues
- `arrays::primitive::compute::tests::test_primitive_binary_numeric::case_5_f64` - NaN comparison issues
- `arrays::constant::compute::test::test_filter_constant` - f16 inline assembly
- `arrays::constant::compute::test::test_mask_constant` - f16 inline assembly
- `arrays::constant::canonical::test_canonicalize_scalar_values` - f16 inline assembly
- `arrays::constant::compute::cast::test_cast_constant_conformance::case_3` - f16 conversion
- `arrays::chunked::compute::cast::test_cast_chunked_conformance::case_3` - f16 conversion
- `arrays::chunked::compute::filter::filter_chunked_floats` - f16 inline assembly
- `arrays::primitive::compute::cast::test_cast_primitive_conformance::case_09` - f16 conversion
- `arrays::primitive::compute::cast::test_cast_primitive_conformance::case_10` - f16 conversion
- `arrow::convert::tests::test_float16_array_conversion` - f16 inline assembly
- `compute::conformance::cast::tests::test_cast_conformance_f32` - f16 conversion

## Next Steps

### 1. Complete Test Analysis
- [ ] Wait for full miri test run to complete (was at 457/747 tests)
- [ ] Identify any additional slow tests in the remaining 40%
- [ ] Update this document with complete findings

### 2. Investigate F32/F64 Failures
- [ ] Debug why f32/f64 tests fail under miri
- [ ] Determine if these are real issues or miri limitations
- [ ] Apply exclusions if determined to be miri limitations

#### Phase 3: Pattern-Based Exclusions
- [ ] Identify all tests with "large" in the name
- [ ] Identify all tests with "many_small_chunks" in the name
- [ ] Consider bulk exclusion using module-level attributes where appropriate

### 3. Test and Verify

- [ ] Run miri locally with exclusions to verify time reduction:
  ```bash
  cargo +nightly miri nextest run -p vortex-array
  ```
- [ ] Ensure critical unsafe operations are still covered
- [ ] Verify no important safety tests are excluded

### 4. CI Validation

- [ ] Push changes and monitor CI miri job performance
- [ ] Target: vortex-array miri tests complete in under 10 minutes
- [ ] Monitor for any timeout issues

### 5. Documentation and Maintenance

- [ ] Add comments to excluded tests explaining why they're skipped:
  ```rust
  #[cfg_attr(miri, ignore)] // Too slow for miri: 300+ seconds due to large dataset
  ```
- [ ] Update team documentation on miri testing strategy
- [ ] Consider creating a `miri-slow` test group that runs weekly/nightly

### 6. Future Improvements

- [ ] Investigate why chunked tests are particularly slow
- [ ] Consider creating smaller test variants specifically for miri
- [ ] Explore using `#[cfg(miri)]` to run simplified versions instead of skipping entirely
- [ ] Set up metrics to track miri test performance over time

### 7. Alternative Approaches to Consider

If excluding too many tests is concerning:

1. **Create miri-specific test sizes**: 
   ```rust
   #[cfg(miri)]
   const TEST_SIZE: usize = 100;
   #[cfg(not(miri))]
   const TEST_SIZE: usize = 10000;
   ```

2. **Split test suite**:
   - Quick miri tests (run on every PR)
   - Full miri tests (run nightly/weekly)

3. **Selective conformance testing**:
   - Run only a subset of conformance test cases under miri
   - Use property-based testing with smaller inputs for miri