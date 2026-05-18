# OnPair GPU decode — AOT dict-statistic optimization notes

Companion to `PERF_SEARCH.md` (search log), `PERF_RESEARCH.md` (algorithm
research), `PERF_OPT_RESEARCH.md` (literature review of AOT-stat-driven
specialization). This file is the as-built record of which AOT-known
dict-statistic optimizations worked, which didn't, and the per-architecture
forward outlook.

## Kernel inventory (after cleanup)

Four kernels ship in `vortex-cuda/kernels/src/`:

| File | Role | Throughput (A100) |
|---|---|---|
| `onpair.cu` | Reference — thread-per-row, mirrors the CPU decoder. Simple + correct. | 33 GiB/s |
| `onpair_shmem.cu` | Production baseline — GSST shared-mem staging + aligned drain. | 509 GiB/s synthetic, 200-260 real |
| `onpair_shmem_s8.cu` | Stride-8 specialization — picked when dict `max_len ≤ 8`. | +7-14% over baseline |
| `onpair_shmem_s4l1.cu` | Stride-4 specialization — picked when dict `max_len ≤ 4`. | +9-18% over baseline |

Twelve experimental kernels were removed after the optimization sweep:
`onpair_warp`, `onpair_warp_padded`, `onpair_flat`, `onpair_split`,
`onpair_padded_out`, `onpair_shmem_2ch`, `onpair_shmem_block`,
`onpair_shmem_combined`, `onpair_shmem_hotdict`, `onpair_shmem_sorted`,
`onpair_shmem_transpose`, `onpair_shmem_u64`. None beat the trio above on
ABI-compatible byte-packed output. Their behavior is logged in
`PERF_SEARCH.md` for future reference.

## TL;DR

Two new kernels ship: `onpair_shmem_s8` and `onpair_shmem_s4l1`. They are
**variant kernels selected by the host based on the dict's `max_len`** —
no kernel-side cost when not applicable, clean wins when applicable.

| Best variant by dict `max_len` | Kernel | Measured improvement over `onpair_shmem` baseline |
|---|---|---|
| `max_len ≤ 4` | `onpair_shmem_s4l1` | **+9 to +18 %** on ClickBench / TPC-H |
| `max_len ≤ 8` | `onpair_shmem_s8` | **+7 to +14 %** on ClickBench / TPC-H |
| `max_len ≤ 16` | `onpair_shmem` (existing) | baseline |

Aggregate impact on the two real datasets we benched:
- **TPC-H lineitem SF=10**: 236 → 268 GiB/s effective, **+14 %**.
- **ClickBench `hits.parquet`** (25 string columns, 17.67 GB raw): 201 → 213 GiB/s, **+6 %**.

Synthetic best-case data was already at the s16 ceiling (511 GiB/s) and is
unchanged.

## The committed variants

### `onpair_shmem_s8` (stride-8, L1 dict)
- Host packs the dict at 8 B per entry (vs 16 in the baseline).
- Kernel loads tokens as `uint64_t` instead of `uint4`.
- Phase-3 byte memcpy is `#pragma unroll`'d to 8 iterations explicitly,
  vs NVCC's implicit 16-deep ladder. ~half the per-token store-instruction
  count.
- ABI: same shape as `onpair_shmem`, except `dict_padded` is `dict_size × 8` B.
- Selection criterion: `lens.iter().max() ≤ 8`.

### `onpair_shmem_s4l1` (stride-4, L1 dict)
- Host packs the dict at 4 B per entry.
- Kernel loads `uint32_t`.
- 4-deep `#pragma unroll` memcpy ladder.
- L1 dict, **no shared-mem cache, no `__syncthreads`** — that's the
  `_l1` suffix.
- ABI: same shape as `onpair_shmem` with stride-4 `dict_padded`.
- Selection criterion: `lens.iter().max() ≤ 4`.

Host-side cost is negligible: scanning the lens table once (`O(dict_size)`)
and rebuilding a tight dict (also `O(dict_size × max_len_pad)`). Strictly
cheaper than the existing `onpair_shmem_sorted` host pre-sort.

## What we tried and rejected

### `onpair_shmem_s4` (stride-4 with shared-mem dict cache) — REGRESSED, removed
- Cooperative load of the 16 KB stride-4 dict into shared memory at block
  start, then `__syncthreads()`.
- Result: **−8 to −18 %** vs `onpair_shmem`. The `__syncthreads` cost
  (block-wide barrier) outweighs the shared-vs-L1 latency saving on the
  small columns this targets.
- Why it should have worked on paper: GSST (Vonk 2025) reports their
  symbol-table-in-shared-mem step as one of their largest single jumps.
  But GSST's symbol table is ≤2 KiB and the workload is amortized over
  much longer per-warp output — the sync cost amortizes there.
- Why it didn't here: small `max_len ≤ 4` OnPair columns produce 32-128 B
  of warp-chunk output. A block-wide barrier per chunk is heavy relative
  to the actual decode work.
- File removed: `kernels/src/onpair_shmem_s4.cu`.

### Host pre-sort (`onpair_shmem_sorted`) on `max_len = 16` real columns — mixed
- Pre-sorting the 32 codes within each chunk improves dict-load locality
  (32 lanes hit 1-3 cache lines instead of 5-20).
- Measured: **+8.6 % on Title, +6.6 % on URL, +6 % on `l_shipinstruct`**.
- But: **−9 % on Referer, −1 % on PageCharset, −2 % on OriginalURL,
  −1 % on `l_returnflag/l_linestatus`**. The byte-offset indirection
  costs more than the L1-sector savings when L1 wasn't the bottleneck.
- Kernel stays in tree (it's a `PERF_SEARCH.md` documented variant) but
  it's *not* a safe default — needs profile-guided selection.
- Host cost: per-batch ~150 ns / chunk, ~0.5-1 s on a 60M-row column.
  Borderline; we keep it disabled by default in the dispatcher.

### `onpair_shmem_transpose` (GPU-FSST column-major shared staging) — regressed, kept as documented negative result
- Implements GPU-FSST's column-major staging idea on the decode side.
- Result: **−23 %** vs `onpair_shmem`. Adds a uint4 stage store + a
  second `__syncwarp` that's not absorbed by any spare slack — the
  kernel is **not** LSU-instruction-queue throttled, which is the
  problem GPU-FSST's drain solves.
- See `PERF_SEARCH.md` § A13.

### `onpair_shmem_hotdict` (top-256 dict entries in shared) — regressed
- See `kernels/src/onpair_shmem_hotdict.cu`. Stages the first 256 dict
  entries in shared based on a power-law-frequency assumption.
- Result: regressed. Same root cause as `onpair_shmem_s4`: `__syncthreads`
  cost > L1-vs-shared latency benefit.

### `onpair_shmem_combined` (32 B dict+lens combined entry) — regressed
- Co-locates the 16-B token and 1-B len into a 32-B padded slot so one
  L1 sector serves both. See `kernels/src/onpair_shmem_combined.cu`.
- Result: regressed slightly. The combined entry doubles dict cache
  footprint (128 KB vs 64 KB), pushing it out of L1.

### Sub-warp grouping (S6 from `PERF_OPT_RESEARCH.md`) — not tried
- Would pack 4 fixed-length-4 tokens into one aligned 16-B store, removing
  the warp scan + `__syncwarp` for the common case.
- Only works for **fixed-length** dicts where every entry has the same
  length. OnPair's dicts have variable length even when `max_len` is small
  (e.g., `l_shipinstruct` has `max_len=16` but `mean=1.42`, with
  highly skewed entry sizes).
- A useful direction if we ever ship an OnPair variant with fixed-length
  entries, e.g., for low-cardinality enum columns. Probably worth ~2-3×
  on those specifically. Not on the critical path today.

### Constant-memory dict (S7) — not tried
- `__constant__` storage is broadcast-optimized — good when many lanes
  hit the *same* code, bad under uniform random access.
- OnPair access pattern is the latter on real text data, so this would
  thrash. Estimated regression rather than win. Not pursued.

### Length-distribution dispatch using the full histogram — not tried
- Build a kernel per `MAXLEN_LE{4, 8, 16}` and run the right one. This
  is what we did, just keyed on `max_len` directly rather than a full
  percentile table.
- Higher-resolution dispatch (e.g., choose stride based on p95 + spillover
  for outliers) hits the same correctness problem as truncated stride:
  the long-tail entries have to be decoded correctly. A two-tier dict
  with a fallback path would work but the cross-warp divergence eats
  the win for any chunk that includes one long-tail entry. Skipped.

## Why the big columns barely improve

The columns that dominate real-data aggregate throughput on ClickBench
(URL/Title/Referer/OriginalURL = ~75 % of bytes) all have `max_len = 16`
in their compressed dict. They get forced onto the s16 path; the AOT-stat
optimization stride-table doesn't apply.

This isn't an OnPair quirk — it's the workload. Real web URLs and English
text genuinely produce dict entries up to the 16-byte cap because OnPair's
pair-encoding pass merges frequent pairs of single-byte tokens, and English
prefixes ("http://www.", "https://www.", " the ", " and ") naturally
generate 8-16-byte dict entries.

We've checked the dist (in the dict-stats log):

```
Title:   4096 entries, max_len=16, mean=6.46, p50=6, p95=14, stride-fit: 16
URL:     4096 entries, max_len=16, mean=5.16, p50=4, p95=13, stride-fit: 16
Referer: 4096 entries, max_len=16, mean=4.72, p50=4, p95=12, stride-fit: 16
l_comment: 4096 entries, max_len=16, mean=8.19, p50=8, p95=15, stride-fit: 16
```

p95 ≥ 12 in every case. Even an aggressive stride-12 variant wouldn't
cover the long tail; we'd need a spillover dict, which introduces
warp divergence as soon as any of the 32 lanes hits a tail entry.

## Per-architecture forward outlook

This optimization landscape changes with each NVIDIA generation. What
doesn't work on A100 may work on H100/Blackwell with new primitives.

### A100 (CC 8.0) — measured
- L1 latency ~30 cycles, shared latency ~5-10. Shared-mem dict caching
  pays only when the dict fits *and* the working set per chunk is
  long enough to amortize the `__syncthreads`.
- The s4l1 / s8 wins reported here are the realistic max from
  stride-based specialization. No further AOT-stat win is likely.

### H100 (CC 9.0) — projected
- **TMA (`cp.async.bulk`)**: asynchronous global → shared loads with
  hardware-managed addressing. Could load the entire dict (up to 128 KB
  with 1 block/SM) to shared during the first warp's wait on its first
  output, completely hiding the cost. **Would make the `s4` (shared-mem
  dict) variant viable** — eliminating the `__syncthreads` stall that
  killed it on A100. **Estimated: would re-enable the s4 path with a
  +20-30 % win on `max_len ≤ 4` columns, beyond the current s4l1.**
- **DSMEM (distributed shared memory)**: a cluster of CTAs can read each
  other's shared memory. The dict could live in one CTA and be visible
  to all others in the cluster without re-loading. **Would also help
  big-dict columns** (where the dict doesn't fit per-block) — fits in
  the cluster aggregate. Speculative win: **+15-25 % on the
  `max_len = 16` URL/Title/Referer family** that doesn't improve on A100.
- **`__cluster_block_thread()` synchronization** + `wgmma` async wait:
  could let the warp scan / drain stages overlap with dict loads.
  Speculative: small additional win on top of the above.

### H200 (CC 9.0, same SMs as H100)
- Same set of features as H100. The HBM3e increase (4.8 TB/s vs 3.35)
  helps DRAM-bound stages but doesn't change the AOT-stat optimization
  picture. **No new architectural lever for AOT-stat-driven
  specialization** beyond what H100 unlocks.

### B100/B200 (CC 10.0, Blackwell)
- **5th-gen Tensor Memory, async tensor instructions**: not directly
  applicable (we're not doing matmul).
- **CTA-cluster shared memory** carries over from Hopper.
- **HW Decompression Engine**: 600 GB/s for LZ4/Snappy/Deflate/GDeflate.
  **Does NOT apply to OnPair** (dict-coded format, not LZ-family).
- **TMA enhancements**: same general direction as Hopper. Helps the same
  way.
- Aggregate: Blackwell helps via HBM bandwidth scaling (8 TB/s), not via
  novel AOT-stat-specific features. **No fundamentally new optimization
  axis opens on Blackwell** for this kernel.

### Speculative roadmap for max-out

| Arch | Optimization | Estimated GiB/s on real ClickBench aggregate | Effort |
|---|---|---|---|
| A100 (current) | s8 + s4l1 + selective sort | **213** | done |
| A100 | Warp-internal bitonic sort by code (no host cost) | est. ~230-260 | 100-150 LOC |
| A100 | per-token len-bucket dispatch (S3 fused) | est. ~225 | 80-120 LOC |
| H100 | TMA-based dict prefetch + s8/s4 | est. ~450-550 | 200+ LOC, untested |
| H100 | DSMEM dict cluster-resident | est. ~600+ | 300+ LOC, untested |
| B200 | same as H100, HBM-scaled | est. ~1000+ | same as H100 |

These are projections from the architectural feature set and current
ncu-observed bottleneck mix. None are measured.

## What we'd need to break past A100's real-data ceiling

The fundamental observation: the kernel is **not DRAM-bound** on real
data (~200-250 GiB/s of decoded output = ~14-18 % of A100 HBM peak). The
ceiling is in compute/sync. Three options to break it:

1. **Reduce per-token work below current ~10 cycles**: warp-internal
   sort, fused decode-into-compute (G-ALP pattern), or sub-warp grouping
   for fixed-len dicts. Sketched but not implemented.
2. **Use TMA on H100+**: shifts dict load from L1 to async-prefetched
   shared. Removes the 30 %-of-cycles scoreboard stall observed on A100.
3. **Change the algorithm**: a kernel that doesn't materialize byte-packed
   output but produces a stride-16 padded buffer (`onpair_padded_out` at
   753 GiB/s on synthetic). User has rejected this output contract.

For the GPU-resident analytics pipeline use case (where the 511 GiB/s
synthetic number is the right metric), the AOT-stat optimizations here
move the floor (worst-case real-data column) up by ~14 % and don't
change the ceiling. For the round-trip case (where PCIe dominates),
none of this matters — multi-core CPU still wins end-to-end.
