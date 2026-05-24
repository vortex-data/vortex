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

## UNPROVEN: AVX-512 multi-string gather encode (idea B2)

**Hypothesis:** the FSST AVX-512 kernel encodes many strings in parallel via
gather, keeping 8–16 independent lookups in flight to **overlap cache-miss
latency** (memory-level parallelism). This is the *correct* use of AVX-512 for a
latency-bound loop — it raises throughput without lowering per-op latency, and is
single-core.

**Why not done:** large rewrite. The lookup must be vectorized across lanes
(per-lane gather into short/long maps, per-lane variable-length bucket handling,
per-lane greedy advance and bit-packing). OnPair's 16-bit codes, ≤16-byte
tokens, and bucket scan make this materially harder to vectorize than FSST's
fixed 8-bit / ≤8-byte / 2-lookup scheme. Highest performance ceiling of any
remaining idea; highest effort/risk. Not falsified — just unbuilt.

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
  packed/arena/pre-reserve `find` layouts, trie/ART for `find`.
- **Open, output-preserving:** none with a proven win — bits=16 parse is at the
  L2/L3 memory-latency wall.
- **Open, output-changing:** `sample_fraction` (proven training lever).
- **Open, high-effort/high-ceiling:** AVX-512 multi-string gather encode (B2).
</content>
