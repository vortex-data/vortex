# Fuzzing Improvement TODO

This document tracks planned improvements to Vortex's fuzzing infrastructure.

## Priority 0 (Quick Wins)

### Dictionary Files
- [ ] Create `fuzz/dicts/vortex.dict` with common tokens
  - Encoding names (`vortex.dict`, `vortex.runend`, etc.)
  - DType identifiers (`u8`, `i32`, `utf8`, etc.)
  - Magic bytes and boundary values
  - Common field names from tests
- [ ] Add target-specific dictionaries if needed (`array_ops.dict`, `file_io.dict`)
- [ ] Update `run-fuzzer.yml` to pass `-dict=` flag to libFuzzer

### Seed Corpus
- [ ] Create `fuzz/seeds/` directory structure
  - `fuzz/seeds/array_ops/`
  - `fuzz/seeds/file_io/`
  - `fuzz/seeds/compress_roundtrip/`
  - `fuzz/seeds/compress_gpu/`
- [ ] Write seed generator script (`fuzz/src/bin/generate_seeds.rs`)
- [ ] Generate edge case seeds:
  - Empty arrays for each DType
  - Single-element arrays
  - Arrays with boundary values (MAX, MIN, 0, -1)
  - Each encoding type with minimal structure
  - Nested structs and lists at various depths
- [ ] Update CI to copy seeds into corpus before fuzzing

## Priority 1 (High Impact)

### Corpus Minimization Automation
- [ ] Update `minimize_fuzz_corpus.yml` to run on schedule (weekly)
- [ ] Add corpus size threshold trigger
- [ ] Log before/after metrics (input count, total size)
- [ ] Ensure minimized corpus is uploaded back to R2

### Crash Database & Regression Testing
- [ ] Create `s3://vortex-fuzz-crashes/` bucket (or R2 equivalent)
- [ ] Update `report-fuzz-crash.yml` to archive crashes permanently
  - Store crash input file
  - Store metadata (date, target, issue URL, stack trace hash)
- [ ] Create `fuzz-regression.yml` workflow
  - Run on every PR and push to main
  - Download all historical crashes
  - Verify none reproduce (regression test)
  - Fast-fail CI if regression detected
- [ ] Add logic to use fixed crashes as seeds for future fuzzing

### Expand Encoding Coverage
- [ ] Add more encodings to `compress_roundtrip`:
  - ALP (Adaptive Lossless floating Point)
  - BitPacked
  - Chunked
  - Delta
  - FastLanes
  - FoR (Frame of Reference)
  - Roaring
  - Sparse
  - Zigzag
- [ ] Expand GPU fuzzer beyond Dict encoding
- [ ] Add nested/combined encoding tests

### Coverage Tracking
- [ ] Add `fuzz-coverage.yml` workflow
  - Run `cargo fuzz coverage` for each target
  - Generate HTML coverage reports
  - Upload as artifacts
- [ ] Track coverage metrics over time
  - Store coverage percentage in a file/database
  - Alert on coverage regression
- [ ] Add coverage badge to README
- [ ] Consider coverage-guided corpus prioritization

## Priority 2 (Medium Impact)

### Differential Testing
- [ ] Create `fuzz/src/differential/` module
- [ ] Implement Arrow comparison oracle
  - Convert Vortex array to Arrow
  - Perform same operation in Arrow
  - Compare results
- [ ] Add differential fuzz target `differential_arrow.rs`
- [ ] Consider cross-version testing (if applicable)

### Deeper Expression Fuzzing
- [ ] Expand `arbitrary` impl for expressions
  - Nested boolean expressions (AND/OR trees)
  - Arithmetic expressions
  - Deeper projection paths
- [ ] Add expression-specific fuzz target
- [ ] Test invalid/malformed expressions for error handling

### Sanitizer Diversity
- [ ] Add periodic MSAN (Memory Sanitizer) runs
- [ ] Add periodic TSAN (Thread Sanitizer) runs for concurrent code
- [ ] Document sanitizer findings and fixes

### Structure-Aware File Format Fuzzing
- [ ] Create `file_format.rs` fuzz target
  - Mutate serialized file bytes directly
  - Fuzz file headers/footers independently
  - Test reader resilience to corruption
- [ ] Generate malformed file seeds

## Priority 3 (Nice to Have)

### Property-Based Unit Testing
- [ ] Evaluate adding `proptest` for faster iteration
- [ ] Identify good candidates for property tests
- [ ] Complement cargo-fuzz with proptest in CI

### Array Size Distribution
- [ ] Analyze current size distribution effectiveness
- [ ] Consider bimodal distribution (many small + some large)
- [ ] Add periodic large array testing (10K+ elements)
- [ ] Implement power-law distribution for edge cases

### Bounded Recursion
- [ ] Add explicit depth limits for nested List/Struct generation
- [ ] Implement probability decay for deeper nesting
- [ ] Prevent stack overflow from deeply nested structures

### Resource Limit Testing
- [ ] Add timeout testing for slow operations
- [ ] Test memory limits with large inputs
- [ ] Add OOM-specific seeds

### Mutation Strategies
- [ ] Investigate custom mutators for structure-aware mutation
- [ ] Combine arbitrary generation with corpus mutation
- [ ] Consider grammar-based fuzzing for expressions

---

## Completed

(Move items here as they're finished)

---

## References

- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html)
- [cargo-fuzz book](https://rust-fuzz.github.io/book/)
- [Google Fuzzing Guide](https://github.com/google/fuzzing/blob/master/docs/good-fuzz-target.md)
- [Differential Testing Paper](https://www.cs.purdue.edu/homes/xyzhang/fall07/Papers/diff-testing.pdf)
