# FSST Decompressor Optimization Exploration

## Summary

The `OptimizedDecompressor` in `encodings/fsst/src/decompressor.rs` replaces the default
fsst-rs decompressor with a version tuned for throughput. After exhaustive exploration of
~10 different optimization strategies, the current implementation achieves **16-18% speedup
on low-escape data** and **6-8% speedup on high-escape data** over the fsst-rs baseline.

## Current Implementation (committed)

**Architecture: Re-entry N=4 with SWAR escape detection**

Key design decisions:
- **Separate symbol/length tables**: `symbols: [u64; 256]` (2KB) + `lengths: [u8; 256]` (256B) = 2.3KB total, fits in L1 cache
- **Pre-converted u64 symbols**: Avoids per-lookup `Symbol::to_u64()` conversion
- **3-tier processing**: 32-code escape-free fast path → 8-code blocks with escape handling → scalar tail
- **Re-entry pattern**: After handling up to 4×8-code blocks with escapes, re-enters the 32-code fast path
- **SWAR escape detection**: `escape_mask()` detects 0xFF bytes in a u64 using bitwise tricks, avoiding per-byte branches
- **Unrolled escape match**: 8-arm match statement for escape position (0-7) avoids loop overhead

## Benchmark Results (current)

### Raw decompress_into (µs, median)

| Workload | Baseline (fsst-rs) | Optimized | Speedup |
|---|---|---|---|
| Low escape (10k, 16) | 38.8 | 32.4 | **-16%** |
| Low escape (10k, 64) | 153.1 | 127.7 | **-17%** |
| Low escape (10k, 256) | 632.8 | 531.3 | **-16%** |
| Low escape (100k, 64) | 1629 | 1383 | **-15%** |
| High escape (10k, 16) | 120.4 | 103.8 | **-14%** |
| High escape (10k, 64) | 518.8 | 481.0 | **-7%** |
| High escape (10k, 256) | 2109 | 1951 | **-7%** |
| High escape (100k, 64) | 7062 | 6658 | **-6%** |

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

### 10. Software prefetching (REJECTED ❌)
**Idea**: Prefetch the next block of input data or upcoming symbol table entries.
**Result**: No measurable improvement. The symbol table (2.3KB) is permanently resident in L1. Input data is accessed sequentially and the hardware prefetcher handles it well.

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

These were **not explored** and might yield additional improvements:

1. **Optimizing the `build_views` path** (`canonical.rs`): The end-to-end `to_canonical` benchmarks include view building (`BinaryView` construction from decompressed bytes + uncompressed lengths). This is a significant portion of end-to-end time, especially for short strings where the decompression itself is fast.

2. **Multi-threaded decompression**: Splitting the compressed stream by string boundaries and decompressing chunks in parallel. Requires knowing string boundaries in the compressed stream (from the VarBin offsets).

3. **ARM NEON intrinsics**: The current code is x86-focused. ARM NEON has different performance characteristics (e.g., `vceqq_u8` for escape detection, different OOO capabilities).

4. **Compact loop-based escape handling**: Replace the 8-arm match statement with a compact loop. This reduces instruction cache pressure but may hurt branch prediction. Worth benchmarking on workloads with moderate escape rates.

5. **`#[cold]` escape path**: Move escape handling to a separate `#[cold]` function to improve instruction cache locality for the hot (escape-free) path.

6. **Profile-guided optimization (PGO)**: The compiler doesn't know that `escape_mask == 0` is the hot path. PGO would optimize code layout accordingly.

7. **Batch decompression with per-string offsets**: Instead of decompressing the entire string heap as one blob and then building views, decompress strings individually into their final positions, eliminating the separate view-building pass.

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
