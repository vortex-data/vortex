# Miri Test Group Analysis

## Current Groupings vs Test Counts

### Final Configuration (as of latest CI update):

0. **meta** (0 tests)
   - Runs miri coverage check only
   - No crate tests

1. **group-1** (104 tests total)
   - vortex-buffer: 21 tests
   - vortex-bytebool: 25 tests
   - vortex-fsst: 19 tests
   - vortex-pco: 19 tests
   - vortex-zstd: 20 tests

2. **group-2** (100 tests total)
   - vortex: 4 tests
   - vortex-btrblocks: 18 tests
   - vortex-ipc: 4 tests
   - vortex-io: 74 tests

3. **group-3** (166 tests total)
   - vortex-mask: 105 tests
   - vortex-layout: 61 tests

4. **group-4** (148 tests total)
   - vortex-dtype: 94 tests
   - vortex-dict: 51 tests
   - vortex-ffi: 3 tests

5. **group-5** (209 tests total)
   - vortex-expr: 108 tests
   - vortex-alp: 101 tests

6. **group-6** (191 tests total)
   - vortex-fastlanes: 142 tests
   - vortex-runend: 49 tests

7. **group-7** (~50 tests total)
   - vortex-file: ~50 tests
   - vortex-flatbuffers: 0 tests

8. **group-8** (747 tests)
   - vortex-array: 747 tests

9. **group-9** (459 tests)
   - vortex-scalar: 459 tests

## Benefits of Current Grouping:

- **Balanced test counts**: Most groups have 100-200 tests (except the large crates in groups 8-9)
- **Parallel execution**: 10 groups run concurrently, reducing total CI time
- **Logical organization**: Related crates grouped together where possible
- **Special handling for large crates**: vortex-array and vortex-scalar get dedicated groups
- **Meta group**: Separates the coverage check from actual test execution

## Notes:

- Groups 1-7 are well-balanced with 50-200 tests each
- Group 8 (vortex-array) has 747 tests including conformance tests
- Group 9 (vortex-scalar) has 459 tests
- Both groups 8 and 9 may benefit from `#[cfg_attr(miri, ignore)]` on slow tests to reduce runtime
- The meta group ensures coverage checking happens once, not in every parallel job