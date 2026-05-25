# OnPair-rs performance: investigation log

A running, evidence-backed record of compression-speed optimization ideas for
`vortex-onpair-rs`, what was tested, and the result. Each idea is tagged
**PROVEN** (benchmarked win, kept), **DISPROVEN** (benchmarked neutral/regression,
not kept), or **UNPROVEN** (not conclusively tested — with the exact reason).

Reproduce any number with the `bench_tpch` example:

```bash
cargo run --release -p vortex-onpair-rs --example bench_tpch
# env: ONPAIR_BENCH_PARQUET=<file> ONPAIR_BENCH_COLUMN=<col>
#      ONPAIR_BENCH_MAX_BYTES=<n> ONPAIR_BENCH_ITERS=<n>
#      ONPAIR_BENCH_THRESHOLD=<sample_fraction>  ONPAIR_BENCH_CPP=1
```

For *real* TPC-H `lineitem.l_comment` (not the synthetic shape), generate a
single-column parquet of the desired size with the `gen_l_comment` example,
then point `bench_tpch` at it:

```bash
OUT=target/l_comment.parquet TARGET_BYTES=$((1124*1024*1024)) \
  cargo run --release -p vortex-onpair-rs --example gen_l_comment
ONPAIR_BENCH_PARQUET=target/l_comment.parquet ONPAIR_BENCH_COLUMN=l_comment \
  cargo run --release -p vortex-onpair-rs --example bench_tpch
```

## Context

OnPair is a BPE-style dictionary string compressor (port of
[`onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp)). Pipeline:

1. **train** — build a dictionary of ≤`2^bits` tokens (1–16 bytes each) by
   greedy pair-merging over a shuffled sample (`trainer.rs`).
2. **parse/encode** — greedy longest-match tokenize each string via
   `LongestPrefixMatcher::find_longest_match`, LSB bit-pack the codes
   (`parser.rs`, `bits.rs`).
3. **decode** — code → dict bytes with a 16-byte SIMD over-copy (`column.rs`).

`find_longest_match` is the shared hot path: a `short_map` (1–8 byte tokens,
hashmap) + a `long_map` (8-byte prefix → bucket of 9–16 byte suffixes; linear,
promoted to a trie above 128 entries).

**Benchmark:** TPC-H `lineitem.l_comment`, bit widths 12 & 16, at 10 MB / 100 MB
/ 1 GiB. **Machine:** Intel Xeon 2.8 GHz, AVX-512 (F/BW/CD/DQ/VL/VNNI),
**L1d 32 KB, L2 1 MB/core, L3 33 MB**.

### Methodology note (important)

Throughput here is **memory-latency bound**, so absolute numbers drift with
machine state. **Always A/B back-to-back** (`git stash` HEAD → measure → pop →
measure) on the same machine; do not compare against numbers from another
session/run. An earlier round of this work was misled by comparing against
baselines that were contaminated by a concurrent benchmark process.

## Baseline (current `HEAD`, clean, 1 GiB)

| | train | parse | total | ratio | C++ total† |
|---|---|---|---|---|---|
| bits=12 | 1.9 s | 6.8 s | 8.7 s | 2.909 | 18.7 s |
| bits=16 | 17.2 s | 9.2 s | 26.3 s | 2.915 | 40.7 s |

† In-repo C++ (`onpair-sys`) is boost-stripped to `std::unordered_map`, so it is
slower than upstream `onpair_cpp` with `boost::unordered_flat_map`. Rust is ~2×
this reference at every size with matching ratios.

Decode is ~1.4–1.5 GiB/s (10–20× faster than parse) — **not a bottleneck**.

## PROVEN: the wall reproduces on real `l_comment` and a 2nd machine

The numbers above use the synthetic corpus. Re-running on **real** TPC-H
`lineitem.l_comment` (1024 MiB, ~40.5 M rows, generated with `gen_l_comment`)
on a different host (Intel Xeon 2.1 GHz, **L2 2 MiB/core, L3 260 MiB** — a much
larger cache than the baseline machine) reproduces the same per-token cost:

| bits | tokens | parse | ns/token | decode | ratio |
|---|---|---|---|---|---|
| 12 | 138.4 M | 6.46 s (158.6 MiB/s) | 46.7 | 1.05 s | 2.904 |
| 16 | 102.7 M | 8.03 s (127.5 MiB/s) | 78.2 | 0.98 s | 2.914 |

The ~47 / ~78 ns/token match the baseline machine's ~44 / ~80 almost exactly,
**despite 2× the L2 and ~8× the L3**. The big L3 does not rescue bits=16: the
bottleneck is the *latency* of a hash-located access whose working set exceeds
L2, not L3 capacity. bits=12 (94 KB working set, L2-resident) is correspondingly
compute/L2-latency bound, not memory-capacity bound.

## PROVEN: the parse is at the floor (a lower-bound decomposition)

"Parse is at the wall" is measurable, not hand-waved. The `bench_find` example
decomposes parse on the real `l_comment` dictionary into three numbers (256 MiB,
this machine):

| bits | (1) full parse | (2) dependent find | (3) independent find | bit-pack | **MLP headroom** |
|---|---|---|---|---|---|
| 12 | 43.5 ns/tok | 38.5 ns/tok | 38.3 ns/tok | ~11 % | **1.01×** |
| 16 | 73.7 ns/tok | 64.0 ns/tok | 62.9 ns/tok | ~13 % | **1.01×** |

- **(2) ≈ (1):** stripping bit-packing and boundary bookkeeping leaves ~89 % of
  the time, so `find_longest_match` *is* the parse; everything around it is
  already free. The ~11–13 % residual is the bit-packer (sequential writes to a
  contiguous buffer — already near write bandwidth).
- **(3) ≈ (2) — the decisive result.** (3) re-issues the *identical* set of finds
  but from a pre-collected position array, so iterations are independent and the
  out-of-order core is free to overlap their memory accesses. It is **not faster**
  (headroom 1.01×). That means there is **~zero memory-level-parallelism slack**:
  the finds do not stall on overlappable misses — each find is bound by its own
  dependent probe/scan chain at L2/L3-hit latency plus execution throughput,
  which the core already pipelines to the limit.

This is a stronger statement than the individual disproven experiments. The
entire *class* of "expose MLP across strings" ideas — the AVX-512 multi-string
gather (B2), the scalar multi-string interleave — is chasing headroom that the
hardware says is **not there** (1.01×). That is *why* every such attempt landed
neutral-or-worse, independent of implementation quality. Combined with the
exhausted space of "make each find cheaper" attempts below (hashers, data
structures, tries, packing, memoization, frequency, presence filters/masks), the
only sliver of output-preserving room left is the ~11 % bit-packer, and nothing
in `find`.

**The bit-packer sliver, tested (`bench_pack`).** Monomorphizing the pack on
`const BITS` (folding the shift/mask/spill to literals, as the decode
`TokenCursor` already does) is **~2 % faster at bits=16 and neutral at bits=12**
(output byte-identical). So even the entire 11 % residual yields only ~2 %: most
of it is unavoidable output-write traffic and execution-port pressure that
already overlaps `find`, not foldable arithmetic. Not worth the 8× monomorphized
parse-loop code bloat for the bits=16-only gain; bits=12 (the default) sees
nothing. Single-core parse is at its floor.

Reproduce: `cargo run --release -p vortex-onpair-rs --example bench_find`
(decomposition) and `--example bench_pack` (the const-BITS pack A/B).

## PROVEN: parallelize across rows — the real wall-clock win

The single-core per-token floor is fixed, but parse over rows is embarrassingly
parallel and `Column::compress` runs it on **one thread**. Splitting the rows
across threads (each runs the `find` loop on its row range; results merge into a
**byte-identical** packed stream + boundaries) scales nearly linearly. Real
`l_comment`, 256 MiB, this 4-core host (`bench_parallel`):

| threads | bits=12 | bits=16 |
|---|---|---|
| 1 | 0.98× (≈ serial) | 0.98× |
| 2 | 1.76× | 1.71× |
| 4 | **3.0×** (150→465 MiB/s) | **3.1×** (114→377 MiB/s) |

`threads=1` ≈ serial confirms the chunk/merge adds no penalty; the win is pure.
bits=16 scales slightly better (find is a larger share, so the serial bit-pack
tail is relatively smaller). This is **orthogonal to the per-token wall** — it
multiplies throughput by core count and is output-identical (roundtrip-safe), so
it's the highest-value parse change available. Implemented as a prototype in
`bench_parallel`; integrating it into `Column::compress` (or confirming the
Vortex compressor already parallelizes column/chunk work above this layer) is the
follow-up. The serial bit-pack merge (~11 %) caps speedup at ~`1/(0.11+0.89/C)`;
a per-chunk bit-packer with a bit-aligned merge would lift the ceiling further.

## PROVEN: the L2 cliff governs parse speed

Parse cost per token vs bit width (100 MB, the working set grows with `2^bits`):

| bits | tokens | dict bytes | ns/token |
|---|---|---|---|
| 12 | 4 096 | 34 KB | 39 |
| 13 | 8 192 | 77 KB | 46 |
| 14 | 16 384 | 170 KB | 49 |
| 15 | 32 514 | 354 KB | 54 |
| **16** | 65 290 | 743 KB | **77** |

Smooth rise to bits=15, then a **jump at bits=16** — exactly where the lookup
working set crosses the 1 MB L2. The ASM confirms the hot loop is two hashbrown
swisstable probes (already SSE-vectorized: `pcmpeqb`/`pmovmskb`) plus a bucket
scan — every step a dependent memory fetch. **bits=16 parse is L3-latency-bound.**

## PROVEN: `sample_fraction` is a strong training lever (output-changing)

bits=16, 100 MB — lowering the dynamic-threshold sample fraction (the `threshold`
field of `OnPairTrainingConfig`) scans less input during training:

| sample_fraction | train | ratio | vs 0.5 |
|---|---|---|---|
| 0.5 (default) | 1.06 s | 2.836 | — |
| 0.3 | 0.68 s | 2.830 | **1.6× faster, −0.2 % ratio** |
| 0.2 | 0.45 s | 2.824 | 2.4× faster, −0.4 % |
| 0.1 | 0.21 s | 2.798 | 5× faster, −1.3 % |
| 0.05 | 0.12 s | 2.747 | 9× faster, −3.1 % |

bits=12 ratio is unchanged across all fractions (its 4 096-token dict fills
regardless). **Recommendation: 0.3 is a near-free 1.6× training speedup** for
bits=16; 0.1 trades ~1.3 % ratio for 5×. Not changed in code because it alters
output (the dictionary), and the default mirrors `onpair_cpp`.

## DISPROVEN: faster hash functions

The `find` maps are probed mostly with *missing* keys (the longest-match length
loop); foldhash's avalanche rejects misses fastest. 300 MB parse seconds
(lower = better):

| hasher | bits=12 | bits=16 |
|---|---|---|
| **foldhash (default, kept)** | **2.02** | **2.55** |
| ahash (AES) | 2.22 | 2.64 |
| rustc-hash / FxHash | 2.26 | 2.82 |
| gxhash (SIMD, `target-cpu=native`) | 2.40 | 2.75 |

The trainer's integer-keyed pair-frequency map *does* keep a multiply-mix FxHash
(small win there). hashbrown is itself the SIMD swisstable, so "a SIMD hashmap"
is already in use.

### Full hasher sweep on the real **1 GiB** `l_comment` (MiB/s, higher = better)

Reproduced the comparison at the headline 1 GiB size, building each candidate
with `RUSTFLAGS="-C target-cpu=native"` and a `hash-*` feature flag (see
`Cargo.toml`; the swap point is `crate::hash::MapHasher`, wired into the LPM
`short_map`/`long_map` in `lpm.rs`). Same corpus, `seed=42`, `threshold=0.2`,
3 timed iterations (best taken). The achieved ratio is **identical across every
hasher** (2.904× at bits=12, 2.914× at bits=16) — the hasher only affects probe
speed, never the output — confirming this is a pure throughput experiment.

| hasher (crate) | b12 parse | b12 total | b16 parse | b16 total |
|---|---|---|---|---|
| **foldhash — hashbrown default (kept)** | **170.0** | **149.8** | **119.0** | **73.2** |
| rapidhash (`rapidhash` 1.4) | 152.1 | 133.6 | 107.7 | 67.1 |
| wyhash (`wyhash` 0.5) | 151.7 | 133.7 | 112.2 | 69.0 |
| rustc-hash / FxHash (`rustc-hash` 2) | 147.8 | 130.3 | 109.6 | 68.6 |
| ahash (`ahash` 0.8, AES) | 147.1 | 130.1 | 108.4 | 67.3 |
| gxhash (`gxhash` 3, SIMD `target-cpu=native`) | 144.0 | 128.5 | 108.3 | 68.3 |

**foldhash wins parse and total at both bit widths** — every alternative is
2–15 % slower on parse. The newer "fastest hash" crates (rapidhash, wyhash) are
the closest challengers but still trail; gxhash's AES/SIMD throughput advantage
on *bulk* hashing does not help here because each probe hashes only an 8-byte
(short) or 8+suffix key and the loop is latency-bound on the dependent cache
miss, not hash compute. This matches the 300 MB result above and the L2-cliff
analysis: the hasher is not the bottleneck. **No swap shipped; the default
stays foldhash.** The `hash-*` features remain in `Cargo.toml` purely so this
table can be reproduced.

### Maps and perfect hashing (surveyed, not adopted)

- **hashbrown SwissTable** is already the best general-purpose Rust map (flat,
  SIMD group probe, the same design as Abseil/boost `unordered_flat_map`);
  it is what we use. `std::collections::HashMap` *is* hashbrown underneath.
- **Perfect hashing** (`phf`, `boomphf`, `quetzal`/FCH) only applies to the
  read-only `short_map`/`long_map` after training. It removes collision
  handling but **not the dependent load that dominates** (idea B1/C1 below): the
  256 single-byte + dictionary keys still land in a 1.7–2 MB table at bits=16,
  so the L2 miss per probe remains. A minimal perfect hash would also forfeit
  the *miss-rejection* speed that makes foldhash win, since most probes miss.
  Left as UNPROVEN under "shrink the bits=16 working set" rather than measured,
  because the analysis says it cannot beat the cache wall.

## DISPROVEN: alternative `find` data structures

All measured neutral-or-worse vs the contiguous linear bucket scan + hashmap.
bits=16 parse (300 MB unless noted):

| structure | result |
|---|---|
| **baseline contiguous `Vec<LongEntry>` scan (kept)** | **2.55 s** |
| SoA `firsts: Vec<u8>` + SIMD `memchr` filter | 3.69 s (worse) |
| SoA `firsts` + plain-loop filter | 2.93 s (worse) |
| `repr(packed)` 11-byte `LongEntry` alone | ~neutral |
| binary-search to first-byte group | 3.05 s (worse) |
| lower trie-promotion threshold (more tries) | 3.24 s (worse) |
| per-first-byte `max_short_len` | ~neutral |
| map capacity pre-reservation | neutral / slightly worse (over-alloc) |
| read-only contiguous bucket **arena** | neutral (see below) |

Reason they lose: the per-entry check is ~1 cycle and the bucket is contiguous
(prefetcher-friendly); any second array / indirection / pointer-chase adds cache
lines without removing the dominant cost, which is the **memory fetch**, not
compute.

## DISPROVEN (by analysis + supporting tests): trie / ART for `find`

Researched (ART paper; ARCD15 comparison): ART lookup is only *comparable* to a
hash table (its wins are over B-trees, for ordered/range scans). For our 7–9
byte tokens a trie walks **one node per byte** (~7–9 dependent accesses) vs the
current ~2–3; and a 65 K-token trie is still ~MB-sized → still spills L2. The
in-tree pool-trie path was measured slower than linear scan for the typical
12–28-entry buckets (that is why promotion is gated at 128). Decisive external
signal: **FSST — the state of the art for this exact problem, by the Vortex
authors — uses small hash + direct tables, not a trie.**

## UNPROVEN: shrink the bits=16 working set under 1 MB L2 (idea B1/C1)

**Hypothesis:** since the L2 cliff is proven, packing the three structures small
enough to fit bits=16 in L2 would recover ~54 ns/token (~30 % parse).

**Why not conclusively done:** the cache arithmetic says it cannot reach L2.
Approximate bits=16 working set:

| structure | HEAD | maximally packed |
|---|---|---|
| short_map (`(u64,u8)`→`u16`, cap 32 768) | ~0.79 MB | ~0.52 MB (u64 key) |
| long_map values (32 B `Bucket`, cap 16 384) | ~0.52 MB | ~0.20 MB (12 B Range/Trie) |
| bucket entry data (43 K × 16 B) | ~0.70 MB | ~0.48 MB (11 B packed) |
| **total** | **~2.0 MB** | **~1.2 MB** |

Even maximal packing lands at **~1.2 MB > 1 MB L2**, so full L2-residency is
unreachable while keeping a 65 536-token dictionary. Supporting evidence:
`repr(packed)` `LongEntry`, the arena, and pre-reservation were each measured
**neutral** — each shrinks one component but the total stays > L2. A coordinated
three-way packing was not implemented because the analysis predicts at best a
small (~5–10 %) L3-traffic gain, not the 30 % L2 win, at high implementation/
soundness risk (`repr(packed)`, terminator-bit key packing, two-map split).
**To revisit:** only worth it if a way is found to drop the *total* below 1 MB
(e.g. a minimal perfect hash for the read-only short_map, or fewer tokens).

## DISPROVEN (on AVX-512 Xeon hardware): multi-string gather encode (idea B2)

**Hypothesis:** the FSST AVX-512 kernel encodes many strings in parallel via
gather, keeping 8–16 independent lookups in flight to **overlap cache-miss
latency** (memory-level parallelism). This is the textbook use of AVX-512 for a
latency-bound loop — raise throughput without lowering per-op latency, single-core.

**De-risked before the rewrite.** Rather than build the full vectorized parser
(per-lane variable-length bucket handling + greedy advance + bit-packing — a very
large change), the `bench_gather` example isolates the one load-bearing
assumption: *does an AVX-512 `vpgatherqq` over a custom open-addressing table
overlap DRAM-miss latency better than serial scalar probes on this CPU?* It
compares (a) serial hashbrown, (b) serial custom flat table, (c) AVX-512
masked-gather flat table, on random keys in a table sized past L3 so nearly every
probe misses to DRAM. Result (Xeon 2.1 GHz, AVX-512F/DQ; ns per probe):

| table size | (a) hashbrown serial | (b) flat serial | (c) gather512 | (c) vs (a) |
|---|---|---|---|---|
| 64 MiB  | 20.6 | 27.5 | 30.2 | **0.68×** |
| 512 MiB | 32.8 | 47.4 | 66.2 | **0.50×** |
| 2 GiB   | 43.0 | 53.9 | 75.6 | **0.57×** |

The gather is **0.5–0.7× the speed of plain scalar** — a throughput *loss* at
every DRAM size, and even loses to serial scalar probing of the same flat table.
Two decisive reasons:

1. **Scalar already gets the MLP.** hashbrown-serial runs at **43 ns/probe at
   2 GiB — faster than a single DRAM access (~80–100 ns)** — because the
   out-of-order core already overlaps consecutive *independent* loop iterations.
   The MLP B2 hoped to add by hand is largely already present. (This also
   explains why the scalar multi-string interleave regressed.)
2. **`vpgatherqq` is not parallel here.** On this Xeon the gather decomposes into
   serialized per-lane loads in the load ports, so 8 lanes do **not** overlap;
   the instruction's overhead makes it slower than the scalar loop it replaces.

Caveat: this is one (cloud) Xeon microarchitecture. A part with a genuinely
parallel gather (or a different core) could differ — re-run `bench_gather` there.
But on the hardware Vortex parse is tuned for, **B2 does not pay**, and the custom
flat table needed to enable it is itself slower than hashbrown (matching the
earlier "custom flat table regressed hard" finding). The last high-ceiling
output-preserving idea is therefore closed on this hardware: parse throughput is
governed by per-access DRAM/L2 latency that the scalar OoO engine already hides
as well as the ISA allows.

## UNPROVEN: FSST-hybrid 2-lookup scheme (idea C3)

FSST is GB/s because `shortCodes[65536]` + one hash probe over tiny
cache-resident tables. OnPair's 16-bit codes / ≤16-byte tokens break FSST's
8-bit / ≤8-byte assumptions, so it is not a drop-in. A hybrid (e.g. a 2-byte
direct table for the hottest short codes) is conceivable but research-grade and
unbuilt.

## DISPROVEN: memoization cache for encode (frequency idea applied to `find`)

The frequency/repetition insight *can* be applied to encode as memoization: the
longest match at a position depends only on the next ≤16 bytes, so when ≥16
bytes remain the result is fully determined by that window. A direct-mapped
cache (16 384 slots, keyed by the 16-byte window → `(token, len)`, verified by
window comparison so it stays output-identical) was added in front of
`find_longest_match`.

Result (100 MB, A/B vs HEAD): **neutral-to-worse** — bits=12 parse 0.684 s vs
0.640 s, bits=16 0.794 s vs 0.792 s. Measured hit rate only **~10 %**
(700 k/6.94 M at bits=12; 520 k/5.34 M at bits=16). Break-even is ~7–12 %
depending on `find` cost, so 10 % is too low; the 384 KB cache also adds L2
pressure.

Root cause (fundamental): matches are **variable length**, but the cache key is
a **fixed 16-byte window**. Positions that produce the *same* short match (say 4
bytes) usually differ in bytes 5–16, so they miss the cache. Keying by a shorter
window would be incorrect for long-token matches (which determine 67 % of
bits=16 lookups). Fixed-window memoization therefore over-specifies and can't
capture match-granularity repetition. Not kept.

## DISPROVEN: frequency-clustering for encode (the freq idea applied to `find`)

Two ways to apply the "exploit token frequency" idea to the encode hot path
were built and benchmarked; both neutral-or-worse. The root cause is the same:
`find_longest_match`'s dominant cost is the **hash probe into `long_map`** (done
every call to locate the bucket), and a swisstable slot is **hash-located** — it
cannot be reordered by frequency. The only orderable structure (bucket entry
data) is downstream of that probe and is not the bottleneck.

- **Frequency(size)-ordered bucket arena** — pack all linear buckets into one
  contiguous arena, largest-first (bucket size as a hot-prefix proxy), so hot
  bucket data clusters at the front (cache-resident). Result (100 MB, A/B):
  **neutral** — bits=12 parse 0.586 s vs 0.579 s, bits=16 0.734 s vs 0.735 s.
  The scattered probe to *reach* the bucket dominates; clustering the scanned
  data after it doesn't move the needle (and a plain unordered arena was also
  neutral).
- **16-byte-window memoization cache** — see the dedicated section above
  (~10 % hit, neutral-to-worse).

Note: the encode path never reads the dictionary by token id — `find` returns
the id straight from the hash structures (long bucket entries store their
suffix bytes inline). So "freq-sort the dict + inverse code" has no encode
surface to act on; it only helps decode's `dict_table[code]` access.

## DISPROVEN: multi-string interleave for memory-level parallelism (scalar B2 proxy)

A single string's parse is latency-bound (each `find` position depends on the
previous match, so consecutive finds can't overlap). Processing G=8 independent
strings in lockstep — buffering tokens per lane, flushing in string order —
issues G independent finds per round so the out-of-order engine can overlap
their cache misses (the scalar analogue of the FSST AVX-512 gather kernel).

Result (100 MB, A/B): **regressed** — bits=12 parse 0.737 s vs 0.581 s, bits=16
0.821 s vs 0.716 s. The per-token buffering + lockstep bookkeeping cost more than
the MLP gained, and the OoO engine already overlaps finds across string
boundaries in the simple loop. Output identical (238 tests pass). The *true*
AVX-512 gather kernel could still win (no buffering, real parallel lanes), but
this scalar proxy regressing lowers confidence that the MLP is easily captured,
and the gather rewrite over variable-length/bucket lookups is very high effort.

## CHANGED DEFAULT: `sample_fraction` 0.5 → 0.2

Given the proven `sample_fraction` tradeoff and the partial-shuffle win, the
default dynamic-threshold sample fraction is now **0.2** (`DynamicThreshold::
default()` and `DEFAULT_DICT12_CONFIG`). At 100 MB l_comment this trains the
bits=16 dictionary ~3.6× faster (total compression 1.78 s → 1.00 s, **1.8×**)
for a **−0.4 %** ratio (2.836 → 2.824); bits=12 is faster with ~equal ratio.
Decode is unaffected. Override per call via `OnPairTrainingConfig.threshold`.

## PROVEN: partial-shuffle the training order (training speedup, ratio preserved)

Per-section timing of `train` across real columns (TPC-H lineitem, 100 MB)
revealed an overlooked cost: the training **row shuffle**. `train` does a full
Fisher-Yates `order.shuffle()` over **all `n` rows**, which is *memory-bound* (a
random swap per row across an `n*4`-byte buffer) and ∝ row count — so for
datasets with many short rows it dominates. Measured `[train.shuffle]` at 100 MB:

| column | rows | shuffle (full) | as % of train @ thr 0.1 |
|---|---|---|---|
| l_comment (free text) | 4.0 M | 0.025 s | small |
| l_shipinstruct (4 vals) | 8.7 M | 0.089 s | ~40 % |
| l_shipmode (7 short vals) | 24 M | **0.34–0.41 s** | **~56 %** |

With a dynamic threshold only the first ~`sample_fraction*n` rows are consumed
before the byte budget stops the scan, so shuffling all `n` is wasteful.
**Fix:** `order.partial_shuffle(rng, k)` with `k = min(n, 2*sample_fraction*n +
1024)` — shuffle only the prefix actually read. (A faster RNG was tried first
and was neutral, confirming the cost is memory, not the RNG.)

Clean A/B (thr 0.1, 100 MB), **ratio preserved**:

| column / bits | train: partial vs full | speedup | ratio partial vs full |
|---|---|---|---|
| l_shipmode / 12 | 0.248 s vs 0.818 s | **3.3×** | 0.779 / 0.779 |
| l_shipmode / 16 | 0.248 s vs 0.901 s | **3.6×** | 0.714 / 0.714 |
| l_comment / 16 | 0.155 s vs 0.273 s | **1.8×** | 2.798 / 2.798 |
| l_comment / 12 | 0.117 s vs 0.246 s | 2.1× | 2.895 / 2.899 |

At `threshold=0.5` (default) `k=n`, so it's a full shuffle (unchanged). 238 unit
tests + 19 cross-impl parity tests pass. **Kept.** The win scales with smaller
`sample_fraction` and with shorter rows.

## Compression section breakdown (for reference)

Timed sections (enable `--features lpm-stats`): `train.shuffle` (above) →
`train.scan` (BPE find+freq+merge — dominates for free text, controlled by
`sample_fraction`) → `train.sort+lpm` (lexicographic dict sort + LPM rebuild —
**always negligible, ≤0.022 s**) → `parse` (~92 % `find`, ~8 % bit-packing) →
`decode` (~1.1 GiB/s, index-based, flat across bit widths). Caveat: OnPair is a
poor fit for tiny low-cardinality columns (l_shipmode ratio 0.78× = expansion);
prefer dict/RLE there.

## DISPROVEN: changing the `long_map` probe (custom flat table; slim value)

The every-call `long_map` probe is the hottest memory access, so two ways to
"change the probe" were built and benchmarked (100 MB, A/B vs HEAD):

- **Custom inline flat open-addressing table** (`(u64 prefix, u32 desc)` slots,
  linear probing, one cache line/probe) replacing hashbrown for the read-only
  long map: **regressed hard** — bits=16 parse 1.21 s vs 0.72 s. A naive
  linear-probe table with a multiply-shift hash taking *low* bits clusters
  badly; hashbrown's SIMD swisstable is far better engineered. Lesson: don't
  hand-roll the hash table.
- **Slim descriptor value in hashbrown** — keep hashbrown's probing but store a
  4-byte packed descriptor (arena range or trie root) instead of the 32-byte
  `Bucket`, shrinking the probed value array ~671 KB→278 KB, with bucket
  entries moved to a contiguous arena. Output-identical (238 tests + roundtrip
  pass). Result: **neutral** — bits=16 parse 0.704 s vs 0.713 s (~1 %, within
  noise). The value-array size isn't the bottleneck: the long probe is already
  hashbrown-fast, and the working set stays > L2 (the 818 KB short_map + arena
  dominate), so shrinking one component doesn't change the cache regime.

Conclusion: the probe cannot be made meaningfully faster by shrinking or
re-implementing the long map. Its cost is the fundamental latency of a
hash-located access into a working set that exceeds L2 — only a *total* working
set < 1 MB (needs a non-pow2 / minimal-perfect-hash short_map; the hand-rolled
table lost to hashbrown) or fewer tokens (lower bits) changes that.

## DISPROVEN: skip/shrink the per-call probes (short-present mask; long filter)

Two output-identical attempts to cut the per-token probe work, both built,
roundtrip-verified (238 unit + 19 cross-impl tests pass), and A/B'd
back-to-back on real `l_comment` (saved binaries, alternated run-by-run to
remove thermal drift):

- **Per-2-byte short-present length mask** — the short-map fallback probes
  lengths `max_short_len..1`, one hashbrown lookup each. A `[u8; 65536]` bitset
  keyed by the first 2 input bytes (bit `L-2` set iff a length-`L` short token
  shares that prefix) lets the loop visit only present lengths, plus a direct
  `[Token; 256]` table for the always-present length-1 fallback. This is a
  stronger version of the disproven "per-first-byte `max_short_len`" — it skips
  *interior* absent lengths, not just caps the top. Result (300 MB, A/B):
  **neutral** (bits=12 1.938 vs 1.935 s; bits=16 2.373 vs 2.388 s). The missing
  short probes are already cheap — foldhash rejects misses fast — so removing
  them is lost in the noise; the dominant cost is the long probe, untouched.

- **L1-resident long-prefix presence filter** — a 16 KiB bitset (`2^17` bits,
  multiply-shift indexed) set for every long-token 8-byte prefix; a zero bit
  proves absence and *skips the `long_map` hash probe entirely* (false positives
  fall through to the exact probe, so output is identical). This targets the
  actual bottleneck — the every-call long probe — rather than the cheap short
  loop. Result (tight interleaved A/B): bits=12 ~2% faster at 300 MB but **flat
  at 100 MB** (same 4096-token dict, so a genuine win should be size-invariant —
  the 300 MB delta is within the run-to-run band), and bits=16 **~0.6–1.2 %
  slower** at both sizes (the filter's own access + 16 KB compete with the
  ~1.7 MB working set). Net **neutral-to-mild-regression**; not kept.

Both reconfirm the central finding: the cost is the *memory-latency* of the
long-map probe into a >L2 working set, and neither avoiding nor reorganizing the
cheaper surrounding probes changes the regime. Only fewer tokens (lower bits),
a total working set < L2 (unreachable at 65 536 tokens), or memory-level
parallelism (the AVX-512 multi-string gather, idea B2) can move bits=16.

## LOW VALUE: frequency-sort + inverse-code (idea D1)

Sorting the dictionary by frequency with an indirection (so emitted codes are
unchanged) clusters hot tokens for **index-based** access — i.e. it would speed
**decode** (`dict_table[code]` with a Zipfian code stream). But decode is already
~1.4–1.5 GiB/s (not a bottleneck), and it does **not** help encode, which is
**hash-based** (hashbrown scatters keys by hash regardless of token frequency).
Not pursued. (Applying the frequency idea to encode *as memoization* was tried
and disproven — see the section above.)

## Summary of what is settled vs open

- **Settled (kept):** bucket LPM, foldhash (maps) + FxHash (freq), `max_short_len`,
  bits=16 overflow fix. Rust ~2× the in-repo C++; ratios match.
- **Settled (rejected):** alternative hashers, SoA/first-byte/binary-search/
  packed/arena/pre-reserve `find` layouts, trie/ART for `find`, per-2-byte
  short-present length mask, L1 long-prefix presence filter (skip the probe).
  Confirmed on real `l_comment` and a larger-cache (2 MiB L2 / 260 MiB L3) host.
- **Settled (rejected, hardware-tested):** AVX-512 multi-string gather (B2) —
  `vpgatherqq` is 0.5–0.7× scalar at DRAM sizes on this Xeon (`bench_gather`); the
  scalar OoO engine already hides the latency the gather aimed to overlap.
- **Open, output-preserving:** none with a proven win — bits=16 parse is at the
  L2/L3 memory-latency wall; the only high-ceiling idea (B2) is now disproven on
  this hardware. A part with a genuinely parallel gather could reopen it.
- **Open, output-changing:** `sample_fraction` (proven training lever); for
  free-text like `l_comment`, bits=16 buys ~nothing over bits=12 in ratio
  (2.914 vs 2.904 at 1 GiB; sometimes worse) at ~1.7× the cost — prefer bits=12.
- **PROVEN, highest-value parse win:** parallelize parse across rows — ~3× on
  4 cores, output-identical (`bench_parallel`). Orthogonal to the single-core
  wall; awaits integration into `Column::compress`.
</content>
