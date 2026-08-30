# Encoding performance investigation

This document records the encoding optimizations retained after comparing this
crate with the original paper Rust implementation and the Boost-based C++
implementation. It also records the complete benchmark matrix so that the
headline result is not separated from workloads where C++ remains faster.

## Retained changes

### Pair-frequency training table

Training packs two adjacent `u16` token identifiers into one `u32` key. The
frequency table is a raw `hashbrown::HashTable` with a specialized integer hash,
so the hash is computed once and passed directly to lookup, insertion, and
removal. Incrementing an existing counter returns its value from the same table
entry rather than performing a second indexed lookup. Counters are saturating
`u8` values, matching the maximum adaptive threshold.

The table is allocated up front from the dictionary budget and input size. The
training-time short-token and long-prefix maps are also reserved from the
dictionary budget. These changes remove growth and rehashing from the hot
training loop.

In focused 32 MiB comparisons after these changes, Rust training was 23–28%
faster than GCC on ClickBench URLs, Apache access logs, and a short-string
corpus at 12 bits. It was about 3% faster or effectively tied at 16 bits. That
result did not generalize to every corpus: over the broader 13-corpus matrix,
12-bit aggregate training throughput was 0.694 GB/s for Rust versus 0.777 GB/s
for GCC.

### Read-only short-token buckets

After training, dictionaries of at most 4,096 tokens can build a read-only
short-token index:

- a 65,536-entry directory selects candidates by their first two bytes;
- candidates with the same prefix are contiguous and longest-first;
- a compact length bitmap and boundary index jump directly over token lengths
  that cannot fit in the remaining input;
- keys, masks, token identifiers, and lengths use separate arrays;
- a 256-entry table handles the guaranteed one-byte fallback without hashing;
- masked packed comparisons replace repeated hash probes for lengths 2–8;
- AVX2 compares four `u64` candidates at once when a bucket is large enough;
- a scalar loop handles short buckets and the SIMD tail.

The short-prefix index is enabled only when the average row is longer than
eight bytes and the dictionary contains at most 4,096 tokens. The additional
length bitmap is enabled for average rows of at most 256 bytes, where short
row-tail windows are frequent. Larger dictionaries and short-row-heavy inputs
retain the hash-map path; long-row inputs retain the original length-skip loop.
These gates avoid regressions observed when the static indexes were applied
unconditionally.

Focused ablations reduced parsing time by approximately 10–22% on workloads
dominated by failed short-token hash probes. Adding the length bitmap reduced
median parse time by approximately 10% on TPC-H `l_comment`, 4% on news
headlines and MS MARCO queries, and 3% on Apache access logs. Across the full
13-corpus 32 MiB Rust A/B, aggregate parsing throughput improved 1.2% and
end-to-end throughput improved 0.8% before applying the long-row gate.

An exact per-length-range layout was also tested and rejected. It split mixed
length buckets into several smaller scans, reducing SIMD occupancy and adding
one pass per present length. Parsing regressed 13% on Apache logs, 55% on MS
MARCO URLs, and nearly 2x on FineWeb. The retained design uses the bitmap only
to choose the first viable candidate, then compares the remaining mixed-length
bucket four candidates at a time.

### Filtered long-prefix lookup

Parsing probes the first eight input bytes for a 9–16 byte token before trying
the short-token index. On most text corpora this long-prefix probe misses more
than 97% of the time. Completed dictionaries with at most 512 distinct long
prefixes now use an 8 KiB one-bit membership filter in front of a raw
`hashbrown::HashTable`. The same specialized integer hash selects the filter
bit and is passed directly to the table, so the common definite miss avoids a
SwissTable control-byte probe. Parser construction freezes this index once;
training continues to use the mutable map. The frozen/unfrozen choice is
hoisted outside the per-token parse loop.

The 512-prefix gate is important. The measured 12-bit dictionaries contained
158–856 distinct long prefixes and the selected text corpora had 2–11% prefix
hit rates; several larger/high-hit dictionaries had 37–65% hit rates. Those
larger dictionaries retain the original map because a filter adds work to a
frequent hit path.

A post-change Samply profile of five in-memory parses of the 128 MiB FineWeb
sample used 4.547 CPU seconds, down from 4.818 seconds (5.6%). A repeated
focused MS MARCO URL run reduced median parse time from 248.44 ms to 229.41 ms
(7.7%). In the full Rust A/B below, aggregate 32 MiB end-to-end throughput
improved 1.0%; individual medians remain noisy enough that small changes should
be treated as ties.

| Corpus | Previous ms | Filtered ms | Time change |
|---|---:|---:|---:|
| Amazon Book Titles | 328.29 | 315.79 | -3.81% |
| Apache access logs | 113.58 | 116.88 | +2.91% |
| ClickBench URLs | 126.79 | 127.06 | +0.21% |
| DBpedia abstracts | 221.59 | 226.48 | +2.21% |
| FineWeb | 263.20 | 263.26 | +0.02% |
| MS MARCO queries | 292.89 | 297.97 | +1.73% |
| MS MARCO URLs | 321.97 | 290.87 | -9.66% |
| ClickBench titles | 119.49 | 117.70 | -1.50% |
| Amazon Book Reviews | 243.14 | 244.65 | +0.62% |
| News Headlines | 303.29 | 299.15 | -1.36% |
| Tweets | 298.91 | 299.62 | +0.24% |
| Stack v3 | 195.16 | 193.48 | -0.86% |
| TPC-H `l_comment` | 192.05 | 198.18 | +3.19% |
| FineWeb shard 0, 128 MiB | 900.79 | 900.88 | +0.01% |
| FineWeb shard 1, 128 MiB | 907.34 | 901.81 | -0.61% |

Two broader replacements were rejected. A custom open-addressed
structure-of-arrays table regressed representative parsing by 8–19% because
its mixing and dependent slot/key/bucket loads cost more than SwissTable. A
prehashed raw table without the membership filter helped miss-heavy corpora but
regressed high-hit and 16-bit dictionaries by roughly 3–4%.

### Existing upstream advantages over the paper Rust implementation

These were already present before the changes above, but materially affect any
comparison with the original `onpair_rs` repository:

- dictionary budgets are configurable from 9 to 16 bits; the default is 12;
- dynamic training scans a byte budget, 15% by default, instead of always using
  a fixed merge threshold over the complete shuffled input;
- the merge threshold adapts to the observed dictionary fill rate;
- only the rows needed for the training sample are shuffled;
- sampling can use a deterministic seed;
- oversized long-prefix buckets are promoted to tries rather than rejecting
  further tokens;
- dictionary and row buffers use compact contiguous representations, including
  `u32` row offsets where selected by the caller;
- training and parsing are separate, so a trained `Parser` can be reused.

The 12-bit configuration is an algorithmic trade-off, not a free
micro-optimization: it trains a smaller dictionary and usually encodes faster,
but emits more codes and generally compresses less than a 16-bit dictionary.

## Benchmark protocol

- Input files were stored on an NVMe SSD.
- Each file was loaded and flattened into byte and offset vectors before timing.
- Timed regions include in-memory training, parsing, and their allocations, but
  exclude file I/O and decompression verification.
- One thread was pinned to CPU core 0.
- 32 MiB inputs used two warmups and the median of five iterations.
- 128 MiB inputs used one warmup and the median of three iterations.
- Rust used `-C target-cpu=native`, fat LTO, and one codegen unit.
- C++ used Boost 1.89, `-O3 -DNDEBUG -march=native`, and LTO.
- GCC was 11.5; Clang was 15.0.7.
- Every one of the 105 measured configurations passed a full round trip.

“Best” below means the faster 12- or 16-bit result for each implementation and
corpus. The original Rust implementation supports only its fixed 16-bit mode
with merge threshold 5.

## All 32 MiB wins

Throughput is decimal GB/s. The bit width selected as fastest appears in
parentheses.

| Corpus | Winner | Optimized Rust | Original Rust | C++ GCC | C++ Clang |
|---|---|---:|---:|---:|---:|
| Amazon Book Titles | GCC | 0.1053 (12) | 0.0843 (16) | **0.1071 (16)** | 0.1069 (16) |
| Apache access logs | Rust | **0.2907 (12)** | 0.1407 (16) | 0.2726 (16) | 0.2707 (16) |
| ClickBench URLs | Rust | **0.2621 (16)** | 0.1413 (16) | 0.2492 (16) | 0.2537 (16) |
| DBpedia abstracts | Rust | **0.1535 (12)** | 0.1058 (16) | 0.1441 (12) | 0.1375 (16) |
| FineWeb | Rust | **0.1267 (12)** | 0.0933 (16) | 0.1227 (12) | 0.1161 (16) |
| MS MARCO queries | GCC | 0.1152 (12) | 0.0812 (16) | **0.1198 (12)** | 0.1138 (12) |
| MS MARCO URLs | GCC | 0.1078 (12) | 0.0797 (16) | **0.1169 (12)** | 0.1104 (12) |
| ClickBench titles | Rust | **0.2778 (16)** | 0.1362 (16) | 0.2564 (16) | 0.2419 (16) |
| Amazon Book Reviews | Rust | **0.1391 (12)** | 0.0925 (16) | 0.1333 (12) | 0.1222 (12) |
| News Headlines | GCC | 0.1112 (12) | 0.0837 (16) | **0.1178 (12)** | 0.1155 (16) |
| Tweets | Rust | **0.1129 (12)** | 0.0787 (16) | 0.1101 (12) | 0.1058 (16) |
| Stack v3 | Rust | **0.1711 (12)** | 0.1276 (16) | 0.1514 (16) | 0.1462 (16) |
| TPC-H `l_comment` | Clang | 0.1710 (12) | 0.0807 (16) | 0.1803 (12) | **0.1852 (12)** |

Win count: optimized Rust 8, GCC 4, Clang 1, original Rust 0.

Aggregating input bytes divided by the sum of per-corpus median times gives:

| Implementation | Aggregate GB/s |
|---|---:|
| Optimized Rust | **0.1453** |
| C++ GCC | 0.1442 |
| C++ Clang | 0.1392 |
| Original Rust | 0.0969 |

Optimized Rust is 0.8% faster than GCC, 4.4% faster than Clang, and 49.9%
faster than the original Rust implementation under this best-per-corpus
selection.

## All 128 MiB wins

The two inputs are independent FineWeb Parquet shards, not repeated 32 MiB
data.

| Corpus | Winner | Optimized Rust | Original Rust | C++ GCC | C++ Clang |
|---|---|---:|---:|---:|---:|
| FineWeb shard 0 | Rust | **0.1487 (12)** | 0.1424 (16) | 0.1360 (12) | 0.1229 (12) |
| FineWeb shard 1 | Rust | **0.1459 (12)** | 0.1421 (16) | 0.1368 (12) | 0.1242 (12) |

Optimized Rust wins both samples. Aggregate throughput is 0.1473 GB/s for
optimized Rust, 0.1422 for original Rust, 0.1364 for GCC, and 0.1235 for
Clang.

## Strict 16-bit comparison

The best-setting table mixes dictionary budgets. When all implementations use
16-bit dictionaries, the result changes:

| Input group | Optimized Rust | Original Rust | C++ GCC | C++ Clang |
|---|---:|---:|---:|---:|
| 13 × 32 MiB | 0.1220 | 0.0969 | **0.1367** | 0.1363 |
| 2 × 128 MiB | 0.0980 | **0.1422** | 0.1196 | 0.1165 |

The original Rust implementation is therefore still substantially faster on
the two large FineWeb samples when both Rust implementations use 16-bit
dictionaries.

## Output-format caveat

The Rust parser currently writes a native `Vec<u16>` code stream at every
dictionary width. The 12-bit compression size used during analysis is a packed
equivalent; packing is not performed in the timed Rust parse. The C++ parser
does perform bit packing. Consequently, the narrow best-setting Rust win over
C++ is not yet an exactly equivalent output-format comparison. The comparison
with original Rust is unaffected because both Rust implementations write
native `u16` codes.

Experiments with unconditional direct tables, unconditional SIMD, alternate
long-bucket packing, and packing the Rust output inside the parse loop were
neutral or slower and are not retained in the implementation.
