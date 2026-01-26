# FastLanes 1024-bit Transpose Optimization Plan

## Summary

Optimized the FastLanes 1024-bit transpose operation from ~3700 cycles to ~13 cycles (287x speedup) using AVX-512 VBMI instructions.

## Performance Results

| Implementation | Cycles/Call | Speedup vs Baseline |
|----------------|-------------|---------------------|
| baseline       | 3734        | 1x                  |
| scalar         | 1374        | 2.7x                |
| scalar_fast    | 215         | 17x                 |
| bmi2           | 129         | 29x                 |
| avx2           | 450         | 8x                  |
| avx2_gfni      | 175         | 21x                 |
| avx512_gfni    | 149         | 25x                 |
| **avx512_vbmi**| **13**      | **287x**            |
| vbmi_dual      | 12.5        | 299x                |
| vbmi_quad      | 12.4        | 301x                |

## Key Optimizations

### 1. AVX-512 VBMI Vectorized Gather/Scatter
- `vpermi2b` (`_mm512_permutex2var_epi8`) for gathering bytes from stride-8 positions
- `vpermb` (`_mm512_permutexvar_epi8`) for scattering transposed bytes
- Replaces scalar loops that were the main bottleneck

### 2. XOR/Shift Butterfly for 8x8 Bit Transpose
- 3-step butterfly algorithm using masks `0x00AA...`, `0x0000CCCC...`, `0x00000000F0F0F0F0`
- Transposes 8x8 bit matrix within each u64 in ~9 instructions per step

### 3. Multi-block Processing for ILP
- Dual-block (`transpose_1024x2_vbmi`): ~5% improvement
- Quad-block (`transpose_1024x4_vbmi`): ~7% improvement over single
- Diminishing returns beyond 4 blocks

## Static Permutation Tables

```rust
// GATHER_FIRST: Collects bytes 0,8,16,24,32,40,48,56 from each group
static GATHER_FIRST: [u8; 64] = [
    0, 16, 32, 48, 64, 80, 96, 112,   // Group 0
    8, 24, 40, 56, 72, 88, 104, 120,  // Group 1
    // ... etc
];

// SCATTER_8X8: 8x8 byte transpose pattern
static SCATTER_8X8: [u8; 64] = [
    0,  8, 16, 24, 32, 40, 48, 56,    // byte 0 from each group
    1,  9, 17, 25, 33, 41, 49, 57,    // byte 1 from each group
    // ... etc
];
```

## Files Modified

- `encodings/fastlanes/src/transpose/mod.rs` - Main implementations
- `encodings/fastlanes/examples/perf_transpose.rs` - Benchmark

## CPU Feature Requirements

| Implementation | Required Features |
|----------------|-------------------|
| baseline/scalar| None              |
| bmi2           | BMI2              |
| avx2           | AVX2              |
| avx2_gfni      | AVX2 + GFNI       |
| avx512_gfni    | AVX-512F/BW + GFNI|
| avx512_vbmi    | AVX-512F/BW + VBMI|

## Recommendations

1. **Default**: Use `transpose_1024_vbmi` when VBMI is available (~13 cycles)
2. **Batch processing**: Use `transpose_1024x2_vbmi` or `transpose_1024x4_vbmi` for bulk operations
3. **Fallback chain**: VBMI → AVX-512+GFNI → BMI2 → scalar_fast → baseline

## Future Work

- ARM NEON implementation (currently has placeholder)
- Streaming stores for large array processing
- Integration with bitpacking encode/decode paths
