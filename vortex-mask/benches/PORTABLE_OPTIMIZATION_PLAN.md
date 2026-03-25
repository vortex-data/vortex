# Portable intersect_by_rank Optimization Plan (Apple M-series)

Current non-BMI2 path: LUT PDEP + flat mask + branchless extract (2.1x vs baseline).
Goal: push further on ARM/Apple Silicon where BMI2/PDEP are unavailable.

## Current bottlenecks (profiled on x86_64, structure applies to ARM)

1. **`pdep_lut`**: 8 table lookups per u64 chunk. The 64KB LUT is too large for L1d (typically 64KB on M-series, shared with other data). Each lookup has ~4 cycle latency + potential L1 miss.
2. **`extract_bits_portable`**: branchless but still 2 loads + 3 ALU ops per chunk.
3. **`count_ones()`**: ARM has `CNT` (byte popcount) + horizontal add. Compiler should emit this, but worth verifying.
4. **Memory bandwidth at 10M+**: self + mask_flat + output = 3 streams. In-place would help here too.

## Optimization ideas to try on M-series

### 1. Nibble LUT (4-bit, 256 bytes — fits in L1)

Replace the 64KB byte-level LUT with a 16×16 = 256-byte nibble LUT.
16 iterations per u64 instead of 8, but each lookup is guaranteed L1-hot.

```rust
struct PdepNibbleLut {
    table: [[u8; 16]; 16],  // 256 bytes total
    counts: [u8; 16],        // 16 bytes
}

fn pdep_nibble_lut(mut source: u64, mask: u64) -> u64 {
    let mut result = 0u64;
    for nib_idx in 0..16u32 {
        let mask_nib = ((mask >> (nib_idx * 4)) & 0xF) as usize;
        if mask_nib == 0 { continue; }
        let count = NIB_LUT.counts[mask_nib];
        let src_nib = (source & 0xF) as usize;
        result |= (NIB_LUT.table[mask_nib][src_nib] as u64) << (nib_idx * 4);
        source >>= count;
    }
    result
}
```

**Expected**: fewer L1 misses at large sizes, ~same throughput at small sizes.
Trade-off: 16 iterations vs 8, but each is cheaper (smaller index, no 64KB footprint).

### 2. CLMUL-based PDEP (ARM NEON PMULL)

ARM has `PMULL` (polynomial multiply) which can implement PDEP-like operations.
The idea: use carry-less multiply to scatter source bits into mask positions.

```
pdep(src, mask) = clmul_scatter(src, mask)
```

This is complex to implement correctly but can be ~2 instructions per PDEP on ARM.
Reference: Geoff Langdale's work on simdjson uses similar CLMUL tricks.

Algorithm sketch:
1. For each set bit in mask, compute its "rank" (position among set bits)
2. Use CLMUL to spread source bits to those positions
3. Mask with the original mask

**Expected**: 2-4x faster than LUT at all sizes if implementable.
**Risk**: complex, may not be faster due to PMULL latency on M-series (~3 cycles).

### 3. In-place portable path

Same win as BMI2 in-place: avoid output allocation, reduce memory streams from 3 to 2.
On M-series with 64KB L1d, this matters even more than x86_64.

```rust
// Take self by value, get mutable BitBuffer, write results back
pub fn intersect_by_rank_owned(self, mask: &Mask) -> Mask {
    let self_buf = self.into_bit_buffer();
    let mut self_mut = self_buf.try_into_mut().unwrap_or_else(|b| b.into_mut());
    // ... loop writes back into self_mut ...
}
```

**Expected**: 1.2-1.3x at 10M+ (same ratio as x86_64 in-place vs production).

### 4. NEON vectorized popcount + prefix sum

Use NEON `VCNT` + `VADDLV` for 128-bit popcount, batch 2 chunks at a time.
Precompute prefix sums to break the serial rank dependency.

```rust
// Process 2 chunks per iteration
let chunks = vld1q_u64(self_ptr);
let popcounts = vcntq_u8(vreinterpretq_u8_u64(chunks));
// horizontal sum each 64-bit lane
let pop0 = vaddlvq_u8(vget_low_u8(popcounts));  // not quite right, need per-lane
// ... extract + pdep for each chunk independently ...
```

**Expected**: modest gain from ILP. The serial rank dependency limits this.

### 5. Verify compiler output on ARM

Check that the compiler is generating optimal ARM code:
- `count_ones()` → `CNT` + `UADDLV` (not a call to `__popcountdi2`)
- Branchless extract → no conditional branch in the loop
- LUT access → no unnecessary zero-extends or sign-extends

```bash
cargo rustc -p vortex-mask --lib --target aarch64-apple-darwin -- --emit asm -C opt-level=3
```

## Benchmarking plan

Run on M-series Mac with:

```bash
# Build and benchmark
cargo bench -p vortex-mask --bench pdep_portable

# Compare specific variants
cargo bench -p vortex-mask --bench pdep_portable -- "best_non_bmi2"
cargo bench -p vortex-mask --bench pdep_portable -- "baseline_portable"

# Inspect assembly
cargo rustc -p vortex-mask --lib -- --emit asm -C opt-level=3
# grep for pdep_lut or extract_bits_portable in the .s file
```

## Priority order

1. **Nibble LUT** — easiest to implement, guaranteed L1-friendly, no platform risk
2. **In-place portable** — straightforward, known 1.2x win from x86_64 data
3. **Verify compiler output** — free, may reveal easy wins
4. **CLMUL PDEP** — highest potential but most complex
5. **NEON batch popcount** — limited by serial dependency, try last
