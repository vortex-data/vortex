# Run-End Boolean Decoding Performance Notes

## Overview

This document captures the state of performance optimization work on `decompress_bool.rs` for run-end encoded boolean arrays.

## Problem Statement

The original benchmark comparison showed the new implementation was slower for the 1000 run length case (only 10 runs):

```
10000_1000_alternating_mostly_valid: develop 401 ns, new 714 ns, 0.56x slower
```

## Root Cause Analysis

### Benchmark Unfairness

The baseline benchmark (`decode_bool_nullable_develop`) and new implementation (`decode_bool_nullable`) measure different things:

**New implementation (what gets timed):**
```rust
bencher
    .with_inputs(|| (ends.clone(), values.clone()))  // Setup: just clone
    .bench_refs(|(ends, values)| {
        // TIMED: extraction + decode
        runend_decode_bools(ends.clone(), values.clone(), 0, total_length)
    });
```

Inside `runend_decode_bools` (all timed):
1. `values.validity_mask()?` - extract validity mask
2. `values.to_bit_buffer()` - extract bit buffer
3. `match_each_unsigned_integer_ptype!` - generic type dispatch
4. `trimmed_ends_iter()` - iterator with 3 chained `.map()` operations
5. Actual decode loop

**Baseline (what gets timed):**
```rust
bencher
    .with_inputs(|| {
        // NOT TIMED: all extraction done here
        let ends_slice: Vec<u32> = ends.as_slice::<u32>().to_vec();
        let values_buf = values.to_bit_buffer();
        let validity_buf = values.validity_mask().unwrap();
        let validity_bits = match validity_buf { ... };
        (ends_slice, values_buf, validity_bits)
    })
    .bench_refs(|(ends, values, validity)| {
        // TIMED: only the decode loop with pre-extracted data
        decode_bool_nullable_baseline(ends, values, validity, total_length)
    });
```

**Key insight:** The baseline excludes ~150ns of extraction overhead from timing.

### Overhead Sources for Few Runs

For 10 runs (1000 run length), the overhead dominates:

1. **`trimmed_ends_iter`** - 3 chained `.map()` per element:
   - `v - offset_e` (subtract offset)
   - `min(v, length_e)` (clamp to length)
   - `v.as_()` (convert to usize)

2. **Array method calls:**
   - `values.validity_mask()?`
   - `values.to_bit_buffer()`
   - `ends.as_slice::<E>()`

3. **Generic dispatch:** `match_each_unsigned_integer_ptype!` macro expansion

## Optimizations Implemented

### 1. Fast Path for Few Runs with No Offset

Added `decode_few_runs_no_offset<E>()` function that:
- Bypasses `trimmed_ends_iter` iterator chain
- Uses direct slice iteration: `for (i, &end) in ends.iter().enumerate()`
- Triggered when `offset == 0 && num_runs < PREFILL_RUN_THRESHOLD` (32)

```rust
// In runend_decode_bools():
if offset == 0 && num_runs < PREFILL_RUN_THRESHOLD {
    return Ok(match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        decode_few_runs_no_offset(
            ends.as_slice::<E>(),
            &values_buf,
            validity,
            nullability,
            length,
        )
    }));
}
```

### 2. Optimized Nullable Fast Path with fill_bits

For nullable decoding in the fast path, uses `fill_bits_true`/`fill_bits_false` instead of `append_n`:

```rust
Mask::Values(mask) => {
    let validity_buf = mask.bit_buffer();
    let mut decoded = BitBufferMut::new_unset(length);
    let mut decoded_validity = BitBufferMut::new_unset(length);
    let decoded_bytes = decoded.as_mut_slice();
    let validity_bytes = decoded_validity.as_mut_slice();
    let mut prev_end = 0usize;
    for (i, &end) in ends.iter().enumerate() {
        let end = end.as_().min(length);
        if end > prev_end {
            let is_valid = validity_buf.value(i);
            if is_valid {
                fill_bits_true(validity_bytes, prev_end, end);
                if values.value(i) {
                    fill_bits_true(decoded_bytes, prev_end, end);
                }
            }
        }
        prev_end = end;
    }
    BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
}
```

## Current Benchmark Results

### Nullable Cases

| Benchmark | New | Baseline | Speedup |
|-----------|-----|----------|---------|
| 10000_2_alternating_mostly_valid | 12.2 µs | 42.6 µs | **3.5x** |
| 10000_10_alternating_mostly_valid | 3.6 µs | 13.1 µs | **3.6x** |
| 10000_10_alternating_mostly_null | 2.8 µs | 12.1 µs | **4.3x** |
| 10000_10_mostly_true_mostly_valid | 3.0 µs | 11.8 µs | **3.9x** |
| 10000_100_alternating_mostly_valid | 0.90 µs | 2.27 µs | **2.5x** |
| 10000_1000_alternating_mostly_valid | 0.48 µs | 0.32 µs | **0.67x** (1.5x slower) |

### Non-Nullable Cases (1000 run length)

| Benchmark | Time |
|-----------|------|
| 10000_1000_all_false | ~191-200 ns |
| 10000_1000_all_true | ~191-202 ns |
| 10000_1000_alternating | ~194-201 ns |
| 10000_1000_mostly_false | ~192-199 ns |
| 10000_1000_mostly_true | ~192-201 ns |

Non-nullable fast path is very efficient.

## Progress

- **Before optimizations:** 0.56x (1.8x slower) for 1000 run length nullable
- **After optimizations:** 0.67x (1.5x slower) for 1000 run length nullable
- **Remaining gap:** ~150ns extraction overhead

## Remaining Work

### Option 1: Fix the Benchmark (Recommended)

Make the benchmark fair by including extraction in the baseline timing:

```rust
#[divan::bench(args = NULLABLE_BOOL_ARGS)]
fn decode_bool_nullable_develop_fair(bencher: Bencher, args: NullableBoolBenchArgs) {
    let (ends, values) = create_nullable_bool_test_data(...);
    bencher
        .with_inputs(|| (ends.clone(), values.clone()))
        .bench_refs(|(ends, values)| {
            // Now timing extraction too
            let ends_slice: Vec<u32> = ends.as_slice::<u32>().to_vec();
            let values_buf = values.to_bit_buffer();
            let validity_buf = values.validity_mask().unwrap();
            let validity_bits = match validity_buf {
                vortex_mask::Mask::Values(m) => m.bit_buffer().clone(),
                _ => BitBuffer::new_set(values.len()),
            };
            decode_bool_nullable_baseline(&ends_slice, &values_buf, &validity_bits, total_length)
        });
}
```

### Option 2: Lower-Level API

Add a public function that takes pre-extracted data for users who want maximum performance and are willing to manage extraction themselves:

```rust
pub fn runend_decode_bools_from_slices<E: IntegerPType>(
    ends: &[E],
    values: &BitBuffer,
    validity: &BitBuffer,  // or Option<&BitBuffer>
    length: usize,
) -> BoolArray
```

### Option 3: Reduce Extraction Overhead

Investigate ways to make `validity_mask()` and `to_bit_buffer()` cheaper:
- Caching
- Avoiding allocations
- Direct field access if possible

## Files Changed

- `encodings/runend/src/decompress_bool.rs`:
  - Added `PREFILL_RUN_THRESHOLD` constant at module level
  - Added `decode_few_runs_no_offset<E>()` function
  - Modified `runend_decode_bools()` to use fast path
  - Added tests: `decode_bools_nullable`, `decode_bools_nullable_few_runs`

## Tests

All tests pass:
```
running 8 tests
test decompress_bool::tests::decode_bools_all_false_single_run ... ok
test decompress_bool::tests::decode_bools_all_true_single_run ... ok
test decompress_bool::tests::decode_bools_alternating ... ok
test decompress_bool::tests::decode_bools_mostly_false ... ok
test decompress_bool::tests::decode_bools_mostly_true ... ok
test decompress_bool::tests::decode_bools_nullable ... ok
test decompress_bool::tests::decode_bools_nullable_few_runs ... ok
test decompress_bool::tests::decode_bools_with_offset ... ok
```

## Code Locations

- Implementation: `encodings/runend/src/decompress_bool.rs`
- Benchmarks: `encodings/runend/benches/run_end_decode.rs`
- Iterator helper: `encodings/runend/src/iter.rs` (`trimmed_ends_iter`)

## Investigation: fill_bits Performance (2025-02-02)

### Hypothesis

The `fill_bits_true`/`fill_bits_false` functions might be slow and could benefit from using u64 instead of u8 for the middle byte fill.

### Benchmark Results

Added benchmarks comparing byte-level (u8) vs word-level (u64) fill implementations:

| Range (bits) | Offset | u8 `.fill()` | u64 manual | Winner |
|--------------|--------|--------------|------------|--------|
| 10 | 0 | ~2.1ns | ~2.6ns | **u8** |
| 10 | 3 | ~1.1ns | ~1.2ns | ~same |
| 100 | 0 | ~4.1ns | ~6.5ns | **u8** |
| 100 | 5 | ~3.9ns | ~8.5ns | **u8 (2x)** |
| 1000 | 0 | ~2.4ns | ~6.7ns | **u8 (3x)** |
| 1000 | 7 | ~3.0ns | ~11ns | **u8 (4x)** |
| 5000 | 0 | ~9.7ns | ~9.8ns | ~same |
| 5000 | 1 | ~10ns | ~13ns | **u8** |

### Conclusion

**The fill functions are NOT the bottleneck.** The `.fill()` method is already highly optimized by LLVM - it generates vectorized memset-like code internally. The manual u64 approach adds overhead from:
1. Alignment checking (`align_offset`)
2. Extra branches for prefix/suffix handling
3. Unsafe pointer casts

The fill operations only take ~2-10ns, while the full decode takes ~200-700ns. The overhead comes from elsewhere.

### What IS the bottleneck?

For the 1000 run length nullable case:
- Baseline (pre-extracted data): ~320ns
- New implementation (includes extraction): ~480ns
- Difference: ~160ns

The overhead sources are:
1. **Extraction calls** (~150ns):
   - `values.validity_mask()?`
   - `values.to_bit_buffer()`
   - `ends.as_slice::<E>()`

2. **Iterator chain** (for non-fast-path cases):
   - `trimmed_ends_iter` with 3 chained `.map()` operations

### Next Steps

1. **Profile the extraction methods** - understand what makes `validity_mask()` and `to_bit_buffer()` expensive
2. **Consider caching** - if these methods are called frequently, cache results
3. **Accept the tradeoff** - the extraction overhead is necessary for a clean API; users who need maximum performance can use the lower-level functions directly

## Optimization: validity_mask() Fast Path (2025-02-02)

### Change

Added a fast path in `validity_mask()` (in `vortex-array/src/compute/filter.rs`) to avoid the expensive `fill_null()` call when the validity array is already a non-nullable BoolArray.

### Extraction Benchmark Results (After)

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| `validity_mask()` | ~150-166ns | ~98-102ns | **~40% faster** |
| All combined | ~195-208ns | ~127-135ns | **~35% faster** |

### Full Decode Benchmark Results (After)

| Benchmark | New | Baseline | Speedup |
|-----------|-----|----------|---------|
| 10000_2_alternating_mostly_valid | 14.3 µs | 49.9 µs | **3.5x faster** |
| 10000_10_alternating_mostly_valid | 4.0 µs | 15.3 µs | **3.8x faster** |
| 10000_100_alternating_mostly_valid | 922 ns | 2.6 µs | **2.8x faster** |
| 10000_1000_alternating_mostly_valid | 446 ns | 376 ns | 1.2x slower |

### Summary

The new implementation is now:
- **2.8x-3.8x faster** for typical cases (many runs)
- **~1.2x slower** only for the edge case with very few runs (10 runs at 1000 run length)

The remaining ~70ns gap in the 1000 run length case comes from:
1. Remaining extraction overhead (~50ns for validity_mask)
2. Iterator/function call overhead

This is an acceptable tradeoff since:
1. The few-runs case is already very fast (~446ns)
2. The common case (many runs) is significantly faster
3. Further optimization would require invasive changes to the core API

## Experiment: u64 Fill in decompress_bool.rs (2025-02-02)

### Hypothesis

Using u64 writes instead of byte-level `.fill()` for the middle portion of `fill_bits_true`/`fill_bits_false` might improve performance.

### Implementation

Modified `fill_bits_true`/`fill_bits_false` to use a `fill_bytes_u64` helper that:
1. Handles unaligned prefix bytes
2. Writes aligned u64s for the middle
3. Handles suffix bytes

### Result

**No improvement.** The u64 approach was about the same speed or slightly slower:
- Nullable 1000 run: ~458-498ns (vs ~374-446ns with byte fill)

### Why

1. **LLVM already optimizes `.fill()`** - It generates vectorized SIMD code for slice fills
2. **Overhead** - Alignment checking and branching add overhead that outweighs any benefit
3. **Small runs** - For small byte ranges, the u64 approach has more overhead

### Conclusion

Keep the simple byte-level `.fill()` implementation. It's already optimal.

## Ablation Study: Which Optimizations Matter? (2025-02-02)

Tested three strategies:
1. **Sequential** - append_n for each run (no prefill)
2. **Prefill zeros** - prefill buffer with 0s, fill true runs
3. **Adaptive** - choose prefill value based on majority

### Results

| Scenario | Sequential | Prefill 0s | Adaptive | Best |
|----------|------------|------------|----------|------|
| 10 runs, alternating | 120ns | 77ns | 125ns | prefill |
| 10 runs, mostly_true | 121ns | 86ns | 106ns | prefill |
| 32 runs, alternating | 752ns | 187ns | 294ns | prefill |
| 32 runs, mostly_true | 492ns | 463ns | 159ns | **adaptive** |
| 100 runs, alternating | 1.06µs | 323ns | 484ns | prefill |
| 100 runs, mostly_true | 1.08µs | 948ns | 166ns | **adaptive** |
| 1000 runs, alternating | 6.3µs | 1.5µs | 1.4µs | ~same |
| 1000 runs, mostly_true | 5.8µs | 2.2µs | 828ns | **adaptive** |

### Conclusions

1. **Prefill vs Sequential**: Prefill is **always faster** for many runs
   - 10 runs: 1.5x faster
   - 100 runs: 3x faster
   - 1000 runs: **4x faster**

2. **Adaptive prefill**: Critical for **skewed distributions** (common in real data)
   - Alternating (50/50): prefill_zeros is same or slightly better
   - Mostly_true (90%): adaptive is **2-3x faster**

Both optimizations are justified and should be kept.

## Final Implementation Architecture

### Entry Point: `runend_decode_bools`

```rust
pub fn runend_decode_bools(
    ends: PrimitiveArray,
    values: BoolArray,
    offset: usize,
    length: usize,
) -> VortexResult<BoolArray>
```

### Decision Tree

```
runend_decode_bools
├── Extract: validity_mask(), to_bit_buffer()
├── IF offset == 0 && num_runs < 32:
│   └── decode_few_runs_no_offset  ← Fast path, no iterator
└── ELSE:
    └── runend_decode_typed_bool   ← Uses trimmed_ends_iter
        ├── Mask::AllTrue → decode_bool_non_nullable
        │   ├── IF num_runs < 32: sequential append_n
        │   └── ELSE: adaptive prefill
        │       ├── more true → prefill 1s, clear false runs
        │       └── more false → prefill 0s, fill true runs
        ├── Mask::AllFalse → return all-invalid array
        └── Mask::Values → decode_bool_nullable
            ├── IF num_runs < 32: sequential append
            └── ELSE: 4 variants based on majority:
                ├── (true, valid)  → prefill decoded=1, validity=1
                ├── (true, null)   → prefill decoded=1, validity=0
                ├── (false, valid) → prefill decoded=0, validity=1
                └── (false, null)  → prefill decoded=0, validity=0
```

### Key Difference: `decode_few_runs_no_offset` vs `runend_decode_typed_bool`

| Aspect | `decode_few_runs_no_offset` | `runend_decode_typed_bool` |
|--------|----------------------------|---------------------------|
| Offset handling | Assumes `offset == 0` | Handles any offset |
| Iterator | Direct slice: `for (i, &end) in ends.iter()` | `trimmed_ends_iter` with 3 `.map()` chains |
| Overhead | Minimal | ~20-30ns iterator overhead |
| When used | `offset == 0 && num_runs < 32` | All other cases |

### `trimmed_ends_iter` Details

```rust
run_ends.iter()
    .map(|v| v - offset_e)      // subtract offset (redundant when offset=0)
    .map(|v| min(v, length_e))  // clamp to length
    .map(|v| v.as_())           // convert to usize
```

For 10 runs, these 3 chained closures add measurable overhead. For 1000 runs, it's amortized.

### Threshold: PREFILL_RUN_THRESHOLD = 32

Below 32 runs:
- Iterator overhead dominates
- Sequential `append_n` is competitive with prefill
- Use direct slice access, avoid iterator chain

Above 32 runs:
- Prefill + fill_bits is 3-4x faster than sequential
- Adaptive selection matters for skewed data
- Iterator overhead is negligible

## `fill_bits_true` / `fill_bits_false` Implementation

```rust
fn fill_bits_true(slice: &mut [u8], start: usize, end: usize) {
    // Handle same-byte case
    if start_byte == end_byte {
        let mask = ((1u16 << (end_bit - start_bit)) - 1) as u8;
        slice[start_byte] |= mask << start_bit;
    } else {
        // First partial byte
        if start_bit != 0 {
            slice[start_byte] |= !((1u8 << start_bit) - 1);
        }
        // Middle bytes - LLVM optimizes to SIMD
        slice[fill_start..end_byte].fill(0xFF);
        // Last partial byte
        if end_bit != 0 {
            slice[end_byte] |= (1u8 << end_bit) - 1;
        }
    }
}
```

Key insight: `.fill()` is already vectorized by LLVM. Manual u64 approach adds overhead without benefit.

## External Optimization: `validity_mask()` Fast Path

In `vortex-array/src/compute/filter.rs`:

```rust
// Added fast path for non-nullable canonical bool arrays
if !self.dtype().is_nullable() && self.is_canonical() {
    return Ok(Mask::from_buffer(self.to_bool().to_bit_buffer()));
}
```

This avoids the expensive `fill_null()` call when the validity array is already a non-nullable BoolArray (common case).

## Final Performance Summary

### vs Baseline (pre-extracted data)

| Scenario | New Impl | Baseline | Result |
|----------|----------|----------|--------|
| Many runs (10-100) | 3-4 µs | 13-15 µs | **3-4x faster** |
| Medium runs (100) | 800-900 ns | 2.6 µs | **2.8x faster** |
| Few runs (10 @ 1000 len) | 380-450 ns | 320-376 ns | ~1.2x slower |

### Absolute Performance (non-nullable, 10K elements)

| Runs | Time | Throughput |
|------|------|------------|
| 10 | ~200 ns | 50M elements/sec |
| 100 | ~350 ns | 28M elements/sec |
| 1000 | ~1.4 µs | 7M elements/sec |

## Files Modified

1. **`encodings/runend/src/decompress_bool.rs`**
   - Full implementation with all optimizations
   - ~430 lines including tests

2. **`encodings/runend/benches/run_end_decode.rs`**
   - Added baseline comparison benchmark
   - ~435 lines

3. **`vortex-array/src/compute/filter.rs`**
   - Added 4-line fast path for `validity_mask()`

4. **`encodings/runend/PERF_NOTES.md`**
   - This file - full documentation of investigation
