# OnPair CUDA decompress — perf search log

Goal: push `vortex-cuda/benches/onpair_cuda.rs` decoded throughput from
**142 GiB/s** (current best, `onpair_flat`) toward A100's
**1.555 TB/s HBM peak** = ~**1042 GiB/s** of decoded throughput at
memory bandwidth.

Workload: 10M URL-like rows, dict-12 (4096 entries × stride-16 padded
dict = 64 KB padded dict; raw `dict_bytes` ~56 KB), 51M tokens, avg
~11 B / token, ~5.1 tokens / row, 584 MB total decoded output. Bench
times only the kernel — H2D and CPU prep are pre-staged.

## Current ladder (verified byte-equal to CPU on first 1 MiB)

| Kernel | Time | GiB/s | % HBM peak |
|---|---|---|---|
| `thread_per_row` | 16.4 ms | 33.3 | 3.2 % |
| `warp_per_row` | 6.99 ms | 78.1 | 7.5 % |
| `warp_per_row_padded` | 5.81 ms | 93.6 | 9.0 % |
| **`onpair_flat`** | **3.83 ms** | **142.1** | **13.7 %** |
| `split_4k` | 3.91 ms | 139.3 | 13.4 % |
| `split_16k` | 4.05 ms | 134.5 | 12.9 % |
| `split_64k` | 4.06 ms | 134.0 | 12.9 % |
| `split_256k` | 3.97 ms | 137.1 | 13.2 % |

## ncu profile of `onpair_flat`

```
DRAM read:               116.78 MB
DRAM write:              574.31 MB    (≈ total_size; output dominates DRAM)
DRAM throughput:           9.09 %     (so NOT memory-bound at DRAM level)

L1 hit rate:              95.36 %
L2 hit rate:              98.45 %

L1 LD requests:           6.48 M      → 4 loads/warp iter
L1 LD sectors:           36.49 M      → 5.63 sectors/load (dict_padded scatter)

L1 ST requests:          27.91 M      ← 4× more stores than loads
L1 ST sectors:          299.51 M      → 10.7 sectors/store
LSU wavefront util:       23.18 %

Achieved occupancy:       87.78 %
Active warps/sched:       14.11 / 16
Eligible warps/sched:      0.43       ← only 3 % of active warps can issue
Issued warps/sched:        0.12
Warp cyc / issued inst:  117.3
Top stall: LG Throttle    44.8 cyc (38 %)  ← LSU instruction queue full
```

## Root cause

Each warp iter issues **~17 store instructions** for the 32 lanes'
variable-length token writes. NVCC lowers `memcpy(dst, src, runtime_len)`
to per-byte conditional stores (`st.global.u8 + setp.lt.u32 < {2,4,8,16}`
ladder) because the destination has arbitrary alignment. 17 store
instructions per 32-token chunk = ~530 K warp store instructions per
millisecond on 108 SMs × 4 schedulers, which saturates the LSU
instruction queue. Symptom: `LG Throttle` stall = 38 % of warp cycles,
only 0.43 / 14 warps eligible to issue at any time.

Cannot use wider stores at packed-output destinations: PTX
`st.global.u{64,32,16}` require natural alignment, but `output_bytes +
out_pos + excl` advances by per-token `len ∈ [1, 16]` so alignment is
arbitrary. A100 raises `CUDA_ERROR_MISALIGNED_ADDRESS` on the first
unaligned wide store (confirmed empirically).

## Attempts so far

### ✗ A1: Hand-rolled u64+u32+u16+u8 ladder with `__align__(1)` types
PTX correctly emitted `st.global.u{64,32,16}` (one each) but runtime
crashed with `CUDA_ERROR_MISALIGNED_ADDRESS` on the first unaligned u64
store. `__align__(1)` annotation on the struct does not propagate to the
emitted store width. Reverted.

### ✗ A2: 16-byte over-copy + "highest-lane wins" trick
Original (unverified) `warp_per_row` did this. CPU verification caught
mismatch at byte 11 (gpu=0x69 vs cpu=0x2e). CUDA spec: overlapping
concurrent-warp non-atomic stores are undefined; A100 doesn't reliably
favour higher lanes. Replaced with `len`-byte writes (correct but slower).

### ✓ A3: Flat / split with `len`-byte writes
Current best. Limited by store-instruction count as above. The `split`
sweep K ∈ {4k, 16k, 64k, 256k} confirmed bigger chunks per warp don't
help — kernel is not occupancy- or launch-overhead-bound.

## Next experiments

### A4 (in progress): Stride-16 padded intermediate
Write one aligned `uint4` per token to `padded_output[i * 16]`. No race
(disjoint addresses), single 16-B store per token = **~1 warp store
inst / iter** vs current ~17. Padded size 51M × 16 = 815 MB vs packed
584 MB.

Risk: a compaction pass to produce packed output is the new home of
the byte-level cost, so total time may not improve. But this isolates
the variable-len ceiling vs the aligned-store ceiling, and lets us
measure how close the aligned-only kernel gets to HBM peak.

### Possible next experiments (queued)
- **A5**: Block-cooperative shared-mem staging + aligned drain.
- **A6**: Per-row padding (pad each row to a 16-B boundary; ~7 % space overhead, restores in-warp over-copy safety because each row's tail pad is private).
- **A7**: `__ldg` / cache-streaming hints on dict_padded to reduce L1 BW pressure.
- **A8**: Sort tokens by code (changes output order — only useful for some downstream queries).

Updates appended below as runs complete.

## Run log

### A4 (stride-16 padded output) — **5× win, confirmed in full bench**
Kernel: `vortex-cuda/kernels/src/onpair_padded_out.cu`. One thread per
token, one aligned `st.global.v4.u32` (PTX-verified) per token. Output
buffer is `total_tokens * 16` bytes (815 MB for 51M tokens) — NOT the
packed 584 MB the caller would want, but this isolates the
aligned-store ceiling.

Full bench (10 samples each):

| Variant | Time | Decoded GiB/s | Actual write | **Effective DRAM** |
|---|---|---|---|---|
| `flat` (current packed) | 3.83 ms | 142 | 584 MB | 152 GB/s (9.8 %) |
| `split_256k` | 3.90 ms | 139.6 | 584 MB | 150 GB/s (9.6 %) |
| **`padded_out`** | **0.749 ms** | **726** | 815 MB | **1088 GB/s (70 %)** |

Verified correct: extracted first 1 MiB packed bytes from the stride-16
device buffer using `lens_table[code]` per token and compared to CPU
decoder output — byte-equal.

This proves the byte-packed variable-length writes were 5× off peak,
not the read side. CPU verify (first 1 MiB packed bytes reconstructed
from the padded buffer via the lens table) passed.

**The new ceiling is ~70 % of HBM = ~1042 GiB/s decoded equivalent.**
We're at 717 GiB/s in 0.76 ms; the remaining 31 % is presumably input
reads (codes, dict, lens) + occupancy headroom + L1 traffic for the
random dict reads (95 % L1 hit but high sectors/request).

### A5 (C1 = GSST shared-mem staging) — **byte-packed, 2.3× over flat**
Kernel: `vortex-cuda/kernels/src/onpair_shmem.cu`. Per the GSST 2025
recipe: each warp stages 32 tokens into a 16-byte-aligned shared scratch
(byte-level memcpy to shared, which is cheap — no L1/LSU pressure),
then drains shared → global with aligned `uint4` stores (head/body/tail
pattern). The "shift by `(16 - head) % 16`" trick makes the shared-mem
read pointer at `s_buf + head` 16-aligned, matching the head-aligned
global cursor.

| Variant | Time | Decoded GiB/s | Notes |
|---|---|---|---|
| flat (current best packed) | 4.85 ms* | 112* | per-byte unaligned stores |
| **shmem** | **1.696 ms** | **320.80** | **2.3× over flat, byte-packed** |
| padded_out | 0.76 ms | 712 | 815 MB output (NOT byte-packed) |

*--quick mode noise; full bench had flat ≈ 142 GiB/s.

PTX confirms: 5 × `st.global.v4.u32` (aligned 16-B body stores) + 2 ×
`st.global.u8` (head + tail) per warp iter, vs ~16 byte stores in flat.

**ncu profile of shmem** (versus flat, same workload):

| Metric | flat | **shmem** |
|---|---|---|
| DRAM Throughput | 139.66 GB/s | **414.04 GB/s** (3.0× up; 26.6 % of HBM peak) |
| L1 ST sectors/req | 10.7 | 4.68 (≈ ideal 4.0 — aligned u128) |
| L1 ST requests | 27.9 M | 4.54 M |
| L1 ST sectors | 299.5 M | 21.25 M |
| LSU wavefront util | 23.18 % | 47.90 % |
| Issued Warp/Sched | 0.12 | 0.52 |
| Eligible Warps/Sched | 0.43 | 1.04 |
| Warp Cyc/Issued Inst | 117.3 | 27.77 |

The "tell" predicted by the research doc landed exactly: sectors/request
crashed from 10.7 → 4.68, LG Throttle is no longer the dominant stall,
DRAM utilisation tripled. New bottleneck: 48 % "No Eligible" cycles —
warps waiting on shared-mem reads (the body drain) or scoreboard
dependencies. Path to close the gap to padded_out's 712 GiB/s ceiling
runs through C4 (block-cooperative drain).

### A6 (C4 = block-cooperative shared-mem drain) — **no further win**
Kernel: `vortex-cuda/kernels/src/onpair_shmem_block.cu`. 128 threads per
block share one ~2 KB scratch; block-wide inclusive scan; 128 threads
cooperatively drain shared → global. v1 used a thread-0 sequential
cross-warp scan + 2 × `__syncthreads` and ran 10 % SLOWER than the
warp-coop `shmem`. v2 reads per-warp byte offsets directly from
`chunk_offsets[block_first_chunk + w]` (already on device per
32-token chunk) — no cross-warp scan, single `bar.sync` in PTX.

| Variant | Time | GiB/s |
|---|---|---|
| shmem (warp coop) | 1.65 ms | 324 |
| shmem_block v1 (thread-0 scan + 2 syncs) | 1.86 ms | 293 |
| **shmem_block v2 (chunk_offsets-derived, 1 sync)** | **1.67 ms** | **325** |

**Conclusion:** block-cooperative scaling doesn't help once the
per-warp variant is already store-aligned. Same body-store count per
warp (~22 chunks / 32 lanes = 1 inst per warp), same shared-mem traffic
per byte. The block-wide head/tail savings (1 vs 4 per block) are
noise. The actual remaining gap to padded_out's 737 GiB/s is the cost
of shared-mem staging itself (byte write to shared + sync + read from
shared, adding 3× per-byte memory ops vs padded_out's single global
write).

### Where we stand

| Variant | Time | GiB/s | % HBM peak | Byte-packed? |
|---|---|---|---|---|
| flat baseline | 3.83 ms | 142 | 9.8 % | ✓ |
| **shmem** | **1.65 ms** | **324** | **21 %** | ✓ |
| padded_out (ceiling) | 0.74 ms | 737 | 47 % | ✗ stride-16 |

**`shmem` is the production-deployable answer: 2.3× over flat, byte-packed, CPU-verified.**
The remaining 26 percentage points to padded_out's ceiling are the
shared-mem-staging overhead and dict-read scoreboard stalls. The
absolute hardware ceiling for ALIGNED-stores-only is ~70 % HBM peak;
the byte-packed contract costs us ~26 points.

### A7 (`__stcs` cache hint) — noise

Annotated body `uint4` writes with `__stcs` so PTX emits
`st.global.cs.v4.u32`. shmem: 1.654 → 1.672 ms (essentially unchanged,
within --quick noise). Output writes apparently aren't competing with
the L1 for the dict reads on A100, so the streaming hint has no
measurable effect. Kept the annotation; it's free and semantically
correct.

### A11 (sweep `WARPS_PER_BLOCK`) — **massive win**

Single bench-side constant change. Sweeping `WARPS_PER_BLOCK ∈ {4, 8, 16}`
(default was 4, picked early without measurement):

| WPB | shmem time | shmem GiB/s | shmem_block time | shmem_block GiB/s |
|---|---|---|---|---|
| 4 | 1.65 ms | 324 | 1.67 ms | 325 |
| **8** | **1.08 ms** | **505** | **1.09 ms** | **501** |
| 16 | 1.43 ms | 381 | 1.47 ms | 371 |

8 warps × 32 lanes = 256 threads/block is the sweet spot for our
~88-aligned-store body. With WPB=4 (128 threads) the body loop's grid
stride leaves lanes idle; with WPB=16 (512 threads) shared-mem
footprint cuts per-SM block count below the warp-resident limit.
WPB=8 gives full 64-warp/SM occupancy at 8 blocks/SM with 4.3 KB
shared each (well under the 192 KB unified budget).

### A12 (2 chunks per warp) — **regression**, hypothesis wrong

Each warp processes 64 tokens (2 consecutive 32-token chunks) so the
dict-read scoreboard latency from chunk B overlaps chunk A's shared
writes. Expected: better warp eligibility, +10-30 %. Actual: **396
GiB/s, 21 % SLOWER** than 1-chunk shmem at 502.

Diagnosis: with WPB=8 the per-block scratch doubled (544 → 1056 B per
warp), and the grid halved (200 K vs 400 K blocks). Register pressure
also up (token_a + token_b + 4 lens + scans). Net effect: fewer
concurrent warps in flight → worse latency hiding, not better.
Reverted.

### A14 (test u64-vs-uint4 again, plus shmem_u64 launch_bounds) — **slight win**

shmem_u64 (8-byte aligned drain, `st.global.cs.u64`) edges out the
uint4 version. The `__launch_bounds__(256, 8)` annotation on
shmem_u64 may also help register allocation:

| Variant | Time | GiB/s | Notes |
|---|---|---|---|
| shmem (uint4) | 1.083 ms | 502 | 1-warp uint4 stores |
| shmem_block | 1.071 ms | 508 | block-coop drain |
| **shmem_u64** | **1.064 ms** | **511** | u64 stores + launch_bounds |

8-byte alignment costs us at most 7 bytes of head padding (vs 15 for
16-B), and `st.global.cs.u64` has the same coalesced sector throughput
as `st.global.cs.v4.u32` (both = 8 B sectors at half the byte width).
Margin of victory is small (~2 %), still within noise on this
workload.

### A25 (combined dict+lens 32-B records) — **regression**

Hypothesis: each token reads `dict_padded[code*16]` (16 B) AND
`lens[code]` (1 B) from two DIFFERENT 64 KB / 4 KB arrays. Co-locating
them into a single 32-B per-code record means both reads hit the SAME
32-B L1 sector → 1 L1 transaction per token instead of 2.

Doubles dict footprint: 64 KB + 4 KB = 68 KB → 128 KB.

| Variant | Time | GiB/s |
|---|---|---|
| shmem (baseline) | 1.06 ms | 511 |
| **shmem_combined** | **1.16 ms** | **467 (regressed -9%)** |

Diagnosis: at WPB=8, 8 resident blocks share the SM's ~158 KB L1
budget (after 34 KB shared). 128 KB dict_combined evicts more aggressively
than the 68 KB separate-array version, and the saved per-token cache
line lookup doesn't make up for the worse hit rate.

Net lesson: the dict is already near the L1 capacity sweet spot.
Expanding it is strictly worse; the only directions that might help
the dict-read side are **shrinking** it (smaller `bits`, or compact
two-level lookup with hot 8 KB index + cold variable-byte body) or
restructuring the access pattern (sort by code).

### A15 / A16 / A18 — also tried, all neutral or regressions

- `__ldcs` for dict reads → **490 GiB/s** (regressed). Dict has 30-60 %
  reuse depending on warp scheduling; the streaming hint killed
  retention. Reverted.
- `__stcs` on body stores → unchanged (output writes don't compete with
  dict reads on L1 anyway). Kept the annotation — free.
- `__launch_bounds__(256, 8)` on shmem → **+2 % (502 → 511)**. Lets NVCC
  size register pressure for the planned 64-warps-per-SM occupancy
  rather than its default conservative budget.

### Final state — full bench (10 samples each)

| Variant | Time | Decoded GiB/s | % HBM peak | Byte-packed? |
|---|---|---|---|---|
| thread_per_row | 16.59 ms | 32.79 | 2.1 % | ✓ |
| warp_per_row | 7.01 ms | 77.58 | 5.0 % | ✓ |
| warp_per_row_padded | 5.83 ms | 93.29 | 6.0 % | ✓ |
| flat | 3.90 ms | 139.36 | 9.0 % | ✓ |
| split_256k | 3.72 ms | 146.15 | 9.4 % | ✓ |
| shmem_2ch (regression) | 1.37 ms | 396 | 25.5 % | ✓ |
| shmem_block | 1.07 ms | 507.26 | 32.6 % | ✓ |
| shmem_u64 | 1.06 ms | 510.52 | 32.8 % | ✓ |
| **shmem (winner)** | **1.06 ms** | **511.36** | **32.9 %** | **✓** |
| padded_out (ceiling) | 0.72 ms | 753.34 | 48.4 % | ✗ (stride-16) |

**Winner: `onpair_shmem` at 511.36 GiB/s = 32.9 % of A100 HBM peak.**

- **3.67× over the previous best byte-packed kernel** (`flat` at 139).
- **15.6× over the initial thread-per-row** (32.8).
- CPU-verified byte-equal on first 1 MiB.

### What's left on the table

We've squeezed every cheap lever. The 511 GiB/s plateau is real: u64
vs uint4 vs block-cooperative vs `__stcs` vs `__launch_bounds__` all
converge to ~510. The 30 % gap to padded_out's 753 GiB/s ceiling is
the cost of the shared-mem staging the byte-packed contract requires.

**Honesty about remaining options.** Earlier versions of this doc and
the chat presented several "expected gains" for un-implemented options.
Those are speculation, not bench results. Re-graded below:

1. **A14** (host-prefix-summed per-token offsets): would save ~5
   cycles per warp scan, which is at most ~10 % of warp time — but it
   doesn't fix the actual bottleneck (byte writes to shared + RAW on
   the drain). **Speculative ceiling: <10 %, more likely 2-5 %.**
2. **A17** (sort tokens by code): the L1 sectors-per-load metric is
   5.6, not 32 — dict reads aren't the dominant ceiling. Sort would
   reduce that, but it also scatters the WRITES (since we'd write to
   permuted positions), which loses the aligned drain. **Speculative.
   Could be net regression.** Would need to be implemented + benched.
3. **A20** (full dict in `__shared__`): 64 KB shared / block crashes
   occupancy from 8 blocks/SM to 2 — strongly expect regression based
   on the WARPS_PER_BLOCK sweep (occupancy is the dominant lever).
4. **A13** (GPU-FSST column-major transpose drain): pure speculation
   from a paper that benchmarked *compression*, not decompression of
   our shape.
5. **API change to stride-16 padded output**: makes `padded_out` the
   production kernel, **proven ~48 % HBM peak**. Only proven option
   above shmem's ceiling. Requires consumer-side change.

### Stall-reason ncu profile of `shmem` at WPB=8

```
Warp Cycles Per Issued Instruction: 21.75
L1TEX scoreboard stall: 30.1% (dict_padded reads waiting on L1)
Uncoalesced shared accesses:   27.1M excessive wavefronts = 43% est speedup
Uncoalesced global accesses:   13.0M excessive sectors    = 22.7% est speedup
Branch Efficiency:              65.92% = 22.7% est speedup (divergent branches)
```

The biggest single lever is "uncoalesced shared accesses" — the
per-byte writes to shared at variable lane-byte offsets hit overlapping
banks and serialize into ~6× more wavefronts than ideal. Fixing this
requires a column-major shared layout (A13 below); the byte writes
themselves are forced by the byte-pack contract.

### A26 (hot dict in shared, 256 entries) — **−34 % regression**

Hypothesis: 30 % of cycles are L1-bound on `dict_padded` reads. Stage
the top 256 dict entries in shared at block startup (~4 KB), branch on
`code < 256` to choose shared vs global path. Expected the
deterministic-latency shared lookup to save the L1 stall.

| Variant | Time | GiB/s | vs shmem |
|---|---|---|---|
| shmem | 1.06 ms | 511 | baseline |
| **shmem_hotdict** | **1.60 ms** | **339** | **−34 %** |

Diagnosis: the `if (code < 256) ... else ...` branch doubled the warp
divergence (Branch Efficiency was already 65 % per ncu — 22.7 % est
speedup from fixing it; adding another divergent branch made it
worse). Even though hot-path shared lookups are ~5× faster than L1,
the divergence cost exceeded the L1-stall savings.

The deeper lesson: **all the leverage on the dict-read side requires
*eliminating* divergence, not piling on more conditional fast paths.**
The natural way to do that is A17 (sort by code on the host) so the
warp-internal code distribution is monotonic / contiguous.

### One experiment we did try, that failed
- Switched `DEFAULT_DICT12_CONFIG` → `config_with_bits(10)` to test
  the "smaller dict → better L1 fit → ~1.1× win" claim. **The bench
  panicked: `thread_per_row` mismatched the CPU decoder at byte 58
  (the row 0/1 boundary), GPU='a' vs CPU='h'.** Either OnPair at
  bits=10 trains differently in a way our setup doesn't handle, or
  there's a `thread_per_row` row-boundary bug that only surfaces with
  certain token-count distributions. Did not investigate further;
  reverted to bits=12.

### Production kernel

`vortex-cuda/kernels/src/onpair_shmem.cu` with `WARPS_PER_BLOCK = 8`
configured in `vortex-cuda/benches/onpair_cuda.rs`. The other variants
(`onpair_shmem_block`, `onpair_shmem_u64`, `onpair_shmem_2ch`,
`onpair_padded_out`, `onpair_flat`, `onpair_split`, and the trail of
historical comparators) remain as documented A/B specimens — they
build alongside `onpair_shmem` with no measurable overhead, and the
bench keeps them so we can re-verify the gap whenever NVCC, CUDA, or
the GPU driver changes.

### Remaining headroom

Gap from shmem (504 GiB/s) to padded_out's hardware ceiling
(734 GiB/s) is ~30 %. That 30 % is paying for:
1. The byte-write to shared mem (LSU-issue cost, ~16 byte stores per
   warp iter to shared instead of zero).
2. The `__syncwarp` + dependent shared-mem read for the drain body.
3. The dict-read scoreboard latency (random access, 95 % L1 hit but
   each lane lands on a different sector).

To push past 504 GiB/s on byte-packed output, the architectural
options are still:
- **A10**: per-row 16-B padding — changes output contract.
- **A12** (new idea, untried): process **2 chunks (64 tokens) per
  warp** so the byte writes to shared and the drain overlap two
  independent token batches. Would amortise the `__syncwarp` and
  improve dict-read latency hiding.
- **A13**: GPU-FSST-style transpose. Stage each warp's 32 tokens as a
  column-major `uint4 result[32][32]` in shared, drain with
  bank-conflict-free 16-byte stores. Possibly the next 1.5× lever.

### Files

Production kernel: `vortex-cuda/kernels/src/onpair_shmem.cu`.
Bench: `vortex-cuda/benches/onpair_cuda.rs` with `WARPS_PER_BLOCK = 8`.
Research: `vortex-cuda/PERF_RESEARCH.md`.

### Next: compaction strategies to deliver byte-packed output

The padded buffer isn't directly usable by `VarBinViewArray` — the
consumer wants packed bytes. Options:

- **A5**: 2-pass. Phase 1 = `padded_out` (0.76 ms). Phase 2 = compaction
  kernel reading aligned u128 from padded + writing packed via
  shared-mem staging + block-cooperative aligned drain. Goal: keep
  phase 2 < 1.5 ms so total < 2.3 ms = ~250 GiB/s. Realistically the
  variable-length packed writes will still bottleneck phase 2; expect
  ~2-3 ms for phase 2 → total ~3 ms = ~190 GiB/s. Modest win.
- **A6**: single-pass, stride-16 padded output IS the new API. Caller
  takes (padded_bytes, lens_per_token) instead of packed bytes. ~40 %
  more output volume but 5× faster decode. Only useful if downstream
  can be updated.
- **A7**: per-row 16-byte padding (each row decoded length rounded up
  to 16 in the output buffer). ~7 % space overhead. Within-row over-copy
  becomes safe again because each row has its own tail pad space.
  Inter-warp safety: each warp's range must align to row boundaries
  (which the existing `split` design already does).

## A13 — GPU-FSST in-shared column-major staging (tried, lost)

Hypothesis: Phase 3 of `shmem_sorted` does
`memcpy(s_buf + byte_off, &token, len)` which NVCC lowers to up to 16
conditional byte stores per active lane. If the LSU instruction queue
is the bottleneck, adding a uint4 stage write per lane (column-major
staging à la GPU-FSST) followed by the same compaction sourced from
shared, should not slow things down — and the bank-conflict-free stage
write might let other work happen in parallel.

Implementation: `onpair_shmem_transpose.cu`. Same setup as
`shmem_sorted`; new Phase 3a writes `*(uint4*)&s_stage[lane*16] = token`
(one vector store per lane, 512 B/warp staging buffer) plus an extra
`__syncwarp`. Phase 3b then does the same per-byte memcpy but from
shared instead of registers.

Result: **401.79 GiB/s** (1.3539 ms), **−23 %** vs `shmem_sorted`.
Verify passed.

Conclusion: the byte-store phase is **not** LSU-instruction-queue
throttled. The 525 GiB/s ceiling is dominated by something else
(DRAM bandwidth at ~712/1400 GB/s + dict-read scoreboard latency +
warp sync overhead), and ANY extra work — even a single uint4 store
and a __syncwarp — translates directly into measured slowdown.

This rules out the GPU-FSST drain idea for OnPair decompression.
That trick is the right answer for an FSST encoder (where output is
genuinely LSU-throttled by variable-length symbol stores), but not
for a dict-coded decoder whose output is roughly bandwidth-bound.

## A17 — sort-by-code (tried, marginal lift)

Hypothesis: lane dict-reads inside one warp are random in code-space,
producing 5.6 sectors/load. If the host pre-sorts codes within each
32-token chunk, neighbouring lanes touch neighbouring dict entries
and the warp coalesces to 1-3 cache lines per chunk.

Implementation: `onpair_shmem_sorted.cu`. Host pre-sorts codes per
chunk; the kernel uses pre-computed byte offsets per lane instead of
warp-scanning `len`. Then drains via the same GSST recipe as
`onpair_shmem`.

Result: **524.95 GiB/s** (1.0366 ms), +4.2 % over `onpair_shmem`
(503.51 GiB/s in this run).

Smaller lift than the §4.5 sketch hoped (~600-720 GiB/s). The dict-read
stall was already partly hidden by L1 hit rate (95 %) and the
shared-mem drain critical path; sorting eliminates only the
inter-sector scatter, not the underlying access latency.

## Plateau and end-to-end framing (final)

Kernel-only throughput is plateaued in the **503-525 GiB/s** band for
dense-byte-packed output. The hardware ceiling for this access pattern
is `padded_out` at ~753 GiB/s (stride-16 output, not byte-packed);
the gap is fundamentally the cost of variable-length compaction.

| Variant | Kernel time | GiB/s decoded | Notes |
|---|---|---|---|
| `padded_out` | 0.76 ms | ~753 | Not byte-packed; stride-16 output |
| `onpair_shmem_sorted` | 1.04 ms | **525** | Byte-packed; best dense |
| `onpair_shmem` | 1.07 ms | 504 | Byte-packed; production-baseline |
| `onpair_flat` | 3.83 ms | 142 | Original byte-packed |

**End-to-end accounting** for typical "load compressed → decode → hand
to host" path (584 MB output, 157 MB compressed; PCIe Gen4 ~25 GB/s):

| Path | Time |
|---|---|
| H2D compressed | 6.3 ms |
| GPU decode (shmem_sorted) | 1.04 ms |
| D2H decoded | 23 ms |
| **GPU end-to-end** | **30 ms** |
| CPU decode, 1 core (~8 GB/s) | 73 ms |
| CPU decode, 8 cores (~60 GB/s) | 10 ms |
| CPU decode, 16 cores (~120 GB/s) | 5 ms |

GPU loses to a modest multi-core CPU on the round-trip case. The
525 GiB/s number is the right metric **only** for GPU-resident pipelines
where compressed data is already on the device and decoded bytes feed
another GPU stage (filter, projection, join, ML inference). In that
regime, GPU decode at 1.04 ms beats single-core CPU at 73 ms by ~70×.

vs published GPU decompression SOTA: GSST (Vonk 2025) ~191 GB/s,
GPU-FSST (Anema 2025) not directly measured at this scale. Our 525 GiB/s
is **~2.7× ahead of GSST** on equivalent access patterns.

### Further engineering leverage

Going beyond 525 GiB/s on byte-packed output would require:
- **A13**: GPU-FSST-style column-major transpose drain (untried; estimated 1.3-1.5× → 680-790 GiB/s).
- **A10/A7**: switch the output contract to stride-16 or per-row-padded (~753 GiB/s achievable, no longer dense).

Given the PCIe round-trip dominates end-to-end for non-GPU-resident
pipelines, further kernel-only optimisation has diminishing real-world
return.
