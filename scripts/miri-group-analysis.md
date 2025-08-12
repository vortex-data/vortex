# Miri Test Group Analysis

## Current Groupings vs Test Counts

### Current Groups:
1. **fast** (132 tests total)
   - vortex-buffer: 21 tests
   - vortex-mask: 105 tests
   - vortex-flatbuffers: 0 tests
   - vortex-ffi: 3 tests

2. **medium** (337 tests total)
   - vortex-dtype: 94 tests
   - vortex-expr: 108 tests
   - vortex-io: 74 tests
   - vortex-layout: 61 tests

3. **encodings-1** (177 tests total)
   - vortex-alp: 101 tests
   - vortex-bytebool: 25 tests
   - vortex-dict: 51 tests

4. **encodings-2** (161 tests total)
   - vortex-fastlanes: 142 tests
   - vortex-fsst: 19 tests

5. **encodings-3** (88 tests total)
   - vortex-pco: 19 tests
   - vortex-runend: 49 tests
   - vortex-zstd: 20 tests

6. **core** (26 tests total)
   - vortex: 4 tests
   - vortex-btrblocks: 18 tests
   - vortex-ipc: 4 tests

7. **integration** (unknown, likely ~50)
   - vortex-file: ? tests

8. **array** (747 tests with conformance disabled)
   - vortex-array: 747 tests total

## Issues with Current Grouping:
- **Imbalanced**: medium (337) vs encodings-3 (88) vs core (26)
- **encodings-2** is dominated by fastlanes (142 tests)
- **core** group is very light (26 tests)

## Optimized Grouping Proposal:

### Group 1: "small-crates" (~100 tests)
- vortex-buffer: 21 tests
- vortex-bytebool: 25 tests
- vortex-fsst: 19 tests
- vortex-pco: 19 tests
- vortex-zstd: 20 tests
- Total: 104 tests

### Group 2: "core-and-io" (~100 tests)
- vortex: 4 tests
- vortex-btrblocks: 18 tests
- vortex-ipc: 4 tests
- vortex-io: 74 tests
- Total: 100 tests

### Group 3: "mask-and-layout" (~166 tests)
- vortex-mask: 105 tests
- vortex-layout: 61 tests
- Total: 166 tests

### Group 4: "dtype-and-dict" (~148 tests)
- vortex-dtype: 94 tests
- vortex-dict: 51 tests
- vortex-ffi: 3 tests
- Total: 148 tests

### Group 5: "expr-and-alp" (~209 tests)
- vortex-expr: 108 tests
- vortex-alp: 101 tests
- Total: 209 tests

### Group 6: "fastlanes-and-runend" (~191 tests)
- vortex-fastlanes: 142 tests
- vortex-runend: 49 tests
- Total: 191 tests

### Group 7: "file" (~50 tests estimated)
- vortex-file: ~50 tests
- vortex-flatbuffers: 0 tests
- Total: ~50 tests

### Group 8: "array" (conformance disabled)
- vortex-array: 747 tests (but conformance disabled)

## Benefits of New Grouping:
- More balanced test counts (100-200 per group except array)
- Logical grouping by functionality where possible
- No single crate dominates a group (except vortex-array which is special)
- Better parallelization potential