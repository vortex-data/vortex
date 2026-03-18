# FSST Decompress Optimization Benchmark Results

All benchmarks run on the same machine with 3 runs each for stability.
Medians compared (best of 3 runs used for comparison).

## Optimization Attempts

### 00: Baseline (inlined `decompress_and_build_views`)
The starting point with the function inlined into `fsst_decode_views`.

### 01: Inlined `decompress_and_build_views`
Manually inlined the helper function body. **No measurable effect** — the
`#[inline]` attribute was already causing the compiler to inline.

### 02: Specialized `build_fsst_views_i32`
Wrote a fully specialized view builder for i32 lengths that avoids generic
dispatch, uses u32 offset arithmetic, and uses direct index loops.
**No measurable effect** — the compiler already optimizes the generic
`build_views` + `make_view` equally well.

### 03: Skip length sum via `max_decompression_capacity` + `split_off`
Skip the O(n) length summation pass by using the decompressor's upper-bound
capacity estimate (`8 * (compressed_len + 1)`), then split off unused
capacity after decompression. **2-5% improvement**, consistent across sizes.

| Case | Baseline (µs) | Skip-sum (µs) | Delta |
|------|---------------|---------------|-------|
| (10000, 4, 4) | 71.78 | 70.32 | -2.0% |
| (10000, 16, 4) | 60.62 | 57.79 | -4.7% |
| (10000, 64, 4) | 106.4 | 103.8 | -2.4% |
| (10000, 256, 4) | 493.3 | 479.0 | -2.9% |
| (1000, 16, 4) | 5.46 | 5.24 | -4.1% |

**Tradeoff**: Allocates ~4x more memory (8 bytes per compressed byte vs exact
decompressed size). The `split_off` + drop of unused portion uses shared
memory semantics, so the underlying allocation is not freed until the frozen
buffer is dropped. This is acceptable for transient scan workloads.

## Component Isolation (from fsst_decompress bench)

| Component | 10K×16B | 10K×64B | 10K×256B |
|-----------|---------|---------|----------|
| Raw decompress | 27.7 µs | 94.8 µs | 409 µs |
| View building | 10.1 µs | 10.5 µs | 41 µs |
| Full pipeline | 57-65 µs | 106-118 µs | 479-507 µs |

Decompression dominates for strings >16B. View building is ~10µs constant
for 10K elements. Remaining overhead is allocation + buffer management.
