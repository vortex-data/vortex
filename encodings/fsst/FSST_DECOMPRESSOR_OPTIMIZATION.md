# FSST Decompressor Optimization Exploration

## Summary

The `OptimizedDecompressor` in `encodings/fsst/src/decompressor.rs` replaces the default
fsst-rs decompressor with a version tuned for throughput. After exhaustive exploration of
~15 different optimization strategies, the current implementation achieves **16-22% speedup
on low-escape data** and **3-16% speedup on high-escape data** over the fsst-rs baseline.

## Current Implementation (committed)

**Architecture: N=1 re-entry with SWAR escape detection + cold branch hints + runtime BMI1/BMI2 dispatch**

Key design decisions:
- **Separate symbol/length tables**: `symbols: [u64; 256]` (2KB) + `lengths: [u8; 256]` (256B) = 2.3KB total, fits in L1 cache
- **Pre-converted u64 symbols**: Avoids per-lookup `Symbol::to_u64()` conversion
- **3-tier processing**: 32-code escape-free fast path → 8-code blocks with escape handling → scalar tail
- **N=1 re-entry**: After handling one 8-code escape block, immediately re-enters the 32-code fast path (optimal for low-escape data which is the common case)
- **Cold branch hints**: `cold()` no-op calls in escape branches tell LLVM to optimize code layout for the hot (escape-free) path
- **Runtime BMI dispatch**: `is_x86_feature_detected!("bmi1")` dispatches to `#[target_feature(enable = "bmi1,bmi2,popcnt")]` for better `tzcnt` codegen
- **SWAR escape detection**: `escape_mask()` detects 0xFF bytes in a u64 using bitwise tricks, avoiding per-byte branches
- **Unrolled escape match**: 8-arm match statement for escape position (0-7) avoids loop overhead

## Benchmark Results (current)

### Raw decompress_into (µs, median)

| Workload | Baseline (fsst-rs) | Optimized | Speedup |
|---|---|---|---|
| Low escape (10k, 16) | 38.5 | 32.4 | **-16%** |
| Low escape (10k, 64) | 153.9 | 127.5 | **-17%** |
| Low escape (10k, 256) | 680.4 | 532.5 | **-22%** |
| Low escape (100k, 64) | 1646 | 1376 | **-16%** |
| High escape (10k, 16) | 122.7 | 103.4 | **-16%** |
| High escape (10k, 64) | 517.6 | 471.2 | **-9%** |
| High escape (10k, 256) | 2115 | 1948 | **-8%** |
| High escape (100k, 64) | 7116 | 6892 | **-3%** |

### End-to-end to_canonical (µs, median) — includes view building

| Workload | Median |
|---|---|
| Low escape (10k, 16) | 116.8 |
| Low escape (10k, 64) | 219.1 |
| Low escape (10k, 256) | 630.3 |
| Low escape (100k, 64) | 2266 |
| High escape (10k, 16) | 193.5 |
| High escape (10k, 64) | 577.5 |
| High escape (10k, 256) | 2073 |
| High escape (100k, 64) | 5810 |
| URLs (10k) | 154.0 |
| URLs (100k) | 1599 |

## Optimization Strategies Explored

### 1. Separate symbol/length tables (SHIPPED ✅)
**Idea**: Store symbols as `[u64; 256]` and lengths as `[u8; 256]` separately instead of a combined struct.
**Result**: 2.3KB fits in L1 cache. Combined 16-byte entries (4KB) had worse cache behavior.
**Impact**: Foundation of all subsequent optimizations.

### 2. 32-code block fast path (SHIPPED ✅)
**Idea**: Load 4×u64 (32 codes), check all for escapes with a single OR of escape masks. If no escapes, process all 32 codes in a tight loop.
**Result**: Dramatic improvement for low-escape data by amortizing loop overhead.
**Impact**: ~10-15% improvement on low-escape workloads.

### 3. Re-entry after escapes, N=4 (SHIPPED ✅)
**Idea**: After encountering escapes and dropping to the 8-code path, process up to N blocks then re-enter the 32-code fast path. Avoids permanently being stuck in the slow path.
**Result**: Best variant overall. N=4 chosen after testing N=2,4,8,16.
**Impact**: Additional 5-10% over the 32-code-only approach.

### 4. AVX-512 SIMD escape scanning (REJECTED ❌)
**Idea**: Use `vpcmpeqb` to scan 64 bytes at once for escape codes.
**Result**: CPU frequency throttling on heavy AVX-512 usage negated the throughput benefit. SWAR on regular u64 was faster in practice.
**Impact**: Slower than SWAR approach. Not worth the platform dependency.

### 5. Combined 16-byte table (REJECTED ❌)
**Idea**: Pack symbol (u64) + length (u8) + padding into a 16-byte struct, single load per symbol.
**Result**: Table grows to 4KB, slightly worse cache behavior. Marginally better for high-escape data but worse for low-escape. Not worth the complexity.
**Impact**: ~0-2% difference either way, workload-dependent.

### 6. Prefix-sum emit (REJECTED ❌)
**Idea**: For a block of 8 codes, load all 8 lengths, compute prefix sums to get output offsets, then write all 8 symbols at precomputed offsets. Breaks the serial `out_ptr += length` dependency chain.
**Result**: ~40% slower. The extra prefix-sum computation overhead exceeds the benefit. The CPU's out-of-order engine already overlaps symbol loads with the serial add chain effectively.
**Theoretical analysis**: The serial dependency is `out_ptr += length[code]`, which is ~5 cycles per symbol (4-cycle L1 load + 1-cycle add). OOO execution already overlaps the next symbol's load with the current add, so the effective throughput is close to 4 cycles/symbol. Prefix-sum adds ~20 extra instructions per block with no throughput benefit.

### 7. 64-code blocks (REJECTED ❌)
**Idea**: Process 8×u64 = 64 codes in the escape-free fast path instead of 4×u64 = 32.
**Result**: No measurable improvement. The 32-code path already has enough work to amortize loop overhead. Larger blocks just increase the chance of hitting an escape and wasting the loads.

### 8. Re-entry batch sizes N=2, N=8, N=16 (REJECTED ❌)
**Idea**: Vary the number of 8-code blocks processed before re-entering the 32-code path.
**Result**: N=2 and N=8 tied with N=4. N=16 slightly worse for high-escape data (too many blocks before re-entering fast path). N=4 chosen as the balanced default.

### 9. Interleaved 2×8 dual-cursor (REJECTED ❌)
**Idea**: Process two 8-code blocks simultaneously with independent output pointers, breaking the serial dependency by having two independent output streams.
**Result**: ~2× slower. The interleaving created write conflicts (A7's 8-byte write spills into B's region), and the extra bookkeeping + register pressure overwhelmed any dependency-chain benefit. Even after fixing correctness (writing all A symbols first, then B), the overhead was too high.

### 10. Runtime BMI1/BMI2/POPCNT target feature dispatch (SHIPPED ✅)
**Idea**: Use `is_x86_feature_detected!("bmi1")` at runtime to dispatch to a `#[target_feature(enable = "bmi1,bmi2,popcnt")]` code path. This gives the compiler access to `tzcnt` (true count trailing zeros) instead of `bsf` (bit scan forward, undefined for 0 input).
**Result**: Consistent 2-4% improvement across all workloads, especially high-escape where `trailing_zeros` is called more often. Zero cost on CPUs without BMI1 (falls back to generic path).
**Impact**: Free performance on virtually all modern x86-64 CPUs (BMI1 available since Haswell 2013).

### 11. N=1 re-entry (SHIPPED ✅)
**Idea**: After handling one escape block, immediately re-enter the 32-code fast path instead of processing 4 blocks first (N=4).
**Result**: 1-3% improvement on low-escape data (gets back to the fast path sooner), tied on high-escape. Since low-escape is the common case for real data, N=1 is the better default.
**Impact**: Small but consistent win for the common case.

### 12. Compact loop-based escape handling (REJECTED ❌)
**Idea**: Replace the 8-arm match statement with a compact `while shift < first_esc` loop to reduce instruction cache pressure.
**Result**: Competitive with the match-based version (within 1-2%), but not consistently better. The match compiles to a jump table which is well-predicted for uniform escape positions.
**Impact**: No improvement. Kept the match for consistency with baseline fsst-rs.

### 13. 8-code only with pre-converted symbols (MEASURED)
**Idea**: Same as baseline fsst-rs algorithm (8-code blocks only, no 32-code batching) but with pre-converted u64 symbols.
**Result**: 5-8% faster than baseline on low-escape, 3-7% on high-escape. This isolates the value of pre-converting symbols to u64 (avoiding `Symbol::to_u64()` per lookup).
**Impact**: Confirms that pre-converted symbols account for roughly half the total speedup, with the 32-code batching + re-entry providing the other half.

### 14. Software prefetching (REJECTED ❌)
**Idea**: Prefetch the next block of input data or upcoming symbol table entries.
**Result**: No measurable improvement. The symbol table (2.3KB) is permanently resident in L1. Input data is accessed sequentially and the hardware prefetcher handles it well.

### 15. Inline 32-code escape handling (REJECTED ❌)
**Idea**: When the 32-code batch detects an escape, instead of breaking to the outer loop, process each of the 4 sub-blocks inline — emit clean blocks directly (reusing already-loaded data), handle the first dirty block, then `continue 'outer` to re-enter the fast path.
**Result**: 2-4% better on high-escape data (avoids re-loading clean sub-blocks), but 7-10% worse on low-escape data. The inline escape handling adds code to the 32-code loop body, increasing instruction cache pressure even when the clean path is taken.
**Impact**: Not worth it since low-escape is the common case. The simple `break` from the 32-code path is better.

### 16. `#[cold]` escape handler function (REJECTED ❌)
**Idea**: Extract the entire escape match into a separate `#[cold] #[inline(never)]` method, physically moving it to a cold text section.
**Result**: 3-4% slower than the `cold()` hint approach. The function call overhead (passing 6 arguments, saving/restoring pointers) outweighs the icache benefit.
**Impact**: The `cold()` no-op hint is the better approach — it influences code layout without adding call overhead.

### 17. `cold()` branch hints on escape paths (SHIPPED ✅)
**Idea**: Call a `#[cold] #[inline(never)] fn cold() {}` no-op at the top of escape branches. This causes LLVM to treat the entire branch as unlikely, improving code layout for the hot (escape-free) path.
**Result**: 1-3% improvement on low-escape data (the common case). The biggest win is on the largest workload: (100k,64) 1386µs → 1348µs (-2.7%). High-escape data is tied or marginally better.
**Impact**: Free performance improvement, zero runtime cost on the hot path.

## Why the Current Implementation Is Near-Optimal

The fundamental bottleneck is the **serial dependency chain**: each symbol write depends on the previous symbol's length to compute the output offset (`out_ptr += length[code]`). This creates a minimum latency of ~5 cycles per symbol (L1 load + add).

The CPU's out-of-order engine already overlaps subsequent operations:
- While waiting for `length[code_N]` to load, it speculatively loads `symbol[code_N+1]` and `length[code_N+1]`
- The u64 symbol write is fire-and-forget (no dependency on its completion)
- Net effective throughput is close to the serial dependency limit

Attempts to break the dependency (prefix-sum, interleaving, dual-cursor) add more instruction overhead than they save, because:
1. The symbol table fits in L1 (2.3KB), so loads are fast (~4 cycles)
2. The OOO window is large enough to overlap 10+ symbols of work
3. Any prefix-sum scheme requires reading ALL lengths first, which is the same serial dependency

## Potential Future Directions

### 18. Inlined `build_views` in FSST canonicalize path (SHIPPED ✅)
**Idea**: Replace the general-purpose `build_views()` (which calls `#[inline(never)]` `BinaryView::make_view()` per string) with an FSST-specific version that inlines view construction via `u128` byte manipulation.
**Result**: **26-47% end-to-end speedup** for short/medium strings. The biggest single improvement in this entire optimization effort.
**Key insight**: `make_view()` is `#[inline(never)]` with a 13-arm match, causing a function call per string. For 10k strings of average length 16 bytes, view building was 72% of total end-to-end time. Inlining eliminates the function call overhead and enables the compiler to keep loop variables in registers.

| Workload | Before | After | End-to-end Speedup |
|---|---|---|---|
| Low escape (10k, 16) | 116.8µs | 61.7µs | **-47%** |
| Low escape (10k, 64) | 219.1µs | 161.4µs | **-26%** |
| Low escape (100k, 64) | 2266µs | 1799µs | **-21%** |
| URLs (10k) | 154.0µs | 93.3µs | **-39%** |
| URLs (100k) | 1599µs | 1084µs | **-32%** |

## Potential Future Directions

These were **not explored** and might yield additional improvements:

1. **Multi-threaded decompression**: Splitting the compressed stream by string boundaries and decompressing chunks in parallel. Requires knowing string boundaries in the compressed stream (from the VarBin offsets).

2. **ARM NEON intrinsics**: The current code is x86-focused. ARM NEON has different performance characteristics (e.g., `vceqq_u8` for escape detection, different OOO capabilities).

3. **Profile-guided optimization (PGO)**: The compiler doesn't know that `escape_mask == 0` is the hot path. PGO would optimize code layout accordingly. (The `cold()` hints partially address this, but PGO could further optimize the 32-code loop body layout.)

4. **Batch decompression with per-string offsets**: Instead of decompressing the entire string heap as one blob and then building views, decompress strings individually into their final positions, eliminating the separate view-building pass.

5. **Upstream `make_view` inlining**: The `#[inline(never)]` on `BinaryView::make_view()` in `vortex-array` hurts all callers, not just FSST. Making it `#[inline]` (or providing an `#[inline(always)]` variant) would benefit all VarBinView builders without requiring per-encoding workarounds.

## Files

| File | Purpose |
|---|---|
| `encodings/fsst/src/decompressor.rs` | OptimizedDecompressor implementation |
| `encodings/fsst/src/canonical.rs` | Production usage: bulk decompress → build views |
| `encodings/fsst/benches/fsst_decompress.rs` | Benchmarks (divan framework, `--features _test-harness`) |

## How to Run Benchmarks

```bash
cargo bench -p vortex-fsst --features _test-harness --bench fsst_decompress
```

## How to Run Tests

```bash
cargo test -p vortex-fsst --features _test-harness -- decompressor
```
