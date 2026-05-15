# Chunk skip indexes for string columns — design notes & literature

This document collects the literature behind the per-chunk skip indexes
implemented in `vortex_onpair::skip` (`DictPresence`, `TrigramBloom`,
`SeamBloom`, `TokenPairBloom`) and the prior art they descend from.

## Problem statement

Given a columnar string store split into fixed-size **chunks** (1 K–16 K
rows), build a small per-chunk fingerprint that lets a SQL filter
(`col = x`, `col LIKE 'p%'`, `col LIKE '%s%'`) **rule out whole
chunks** without reading the chunk's compressed bytes. The fingerprint
must be **sound** (no false negatives — never prune a chunk that has a
match) but is allowed false positives.

Two axes:

* **Compression-aware vs codes-agnostic**: indexes built from the raw
  decoded bytes work on any encoding; indexes built from the
  compressor's dictionary or token codes exploit structure.
* **Equality / prefix / substring**: each predicate type wants a
  different filter shape.

## Foundational prior art

| year | structure | reference | how it applies here |
|------|-----------|-----------|---------------------|
| 1968 | PATRICIA trie | Morrison, *J. ACM* 15(4) | sorted-dict prefix-range lookup |
| 1970 | Bloom filter | Bloom, *CACM* 13(7) | every Bloom-based index below |
| 1973 | Suffix tree | Weiner, *SWAT* 14 | building block for FM-index |
| 1973 | Inverted index | Knuth, *Sorting & Searching* §6.5 | per-term postings list |
| 1984 | Signature file | Faloutsos & Christodoulakis, *TOIS* 2(4) | per-document hash, the direct ancestor of per-chunk Bloom |
| 1990 | Suffix array | Manber & Myers, *SODA* | substring index |
| 1994 | q-gram index | Burkhardt et al., *SIGMOD* | per-block q-gram filter for approximate string matching |
| 1995 | n-gram language model | (broad IR literature) | trigram statistics for text |
| 2000 | FM-index | Ferragina & Manzini, *FOCS* | self-indexed text; impractical at chunk scale but inspires SeamBloom |
| 2000s | zonemap / min-max | Netezza (Acceleron, 2002), Vertica/C-Store | range queries; weak for strings |
| 2007 | pg_trgm | Bartunov et al. (Postgres extension) | trigram GIN/Gist index — the direct ancestor of `TrigramBloom` |

## Modern hash-based filters

| year | filter | reference | trade vs Bloom |
|------|--------|-----------|----------------|
| 2012 | Quotient filter | Bender et al., *VLDB* | cache-friendlier, supports merges |
| 2014 | Cuckoo filter | Fan et al., *CoNEXT* | supports deletions, tighter at high load |
| 2017 | Counting QF | Pandey et al., *SIGMOD* | counts, mergeable, used in genomics |
| 2019 | Xor filter | Graf & Lemire, *JEA* | smaller than Bloom at same FPR, immutable |
| 2020 | Binary fuse filter | Graf & Lemire, follow-up | construction speed parity with Bloom |
| 2021 | Ribbon filter | Dillinger & Walzer, *SEA* | static; very tight, RocksDB ships it |
| 2018 | Learned Bloom filter | Mitzenmacher, *NeurIPS* | ML model + backup filter; data-dependent |

For chunk-skip purposes the FPR-at-given-bytes trade is what matters.
Replacing the Bloom in `TrigramBloom` with a Xor or ribbon filter
shrinks the index by ~30% at identical FPR; the implementation cost is
losing easy on-the-fly construction (these filters need batch builds).

## System-level skip indexes in production engines

### Postgres `pg_trgm` (2007–)

GIN/Gist inverted index on **all trigrams of all strings** in a column,
postings list per trigram pointing at row IDs. Far heavier than a
per-chunk Bloom (per-row granularity instead of per-chunk) but offers
exact substring answers. Postgres uses pg_trgm for `LIKE '%s%'`,
`ILIKE '%s%'`, regex prefix anchors, and similarity search.

### ClickHouse `ngrambf_v1` / `tokenbf_v1` (2018+)

Two skip-index types per `MergeTree` granule:

* `ngrambf_v1(n, size_of_bloom_filter_in_bytes, n_hash_functions, seed)`
  — Bloom of all overlapping n-grams of the granule's text.
  `n=3` is the canonical setting. This is **architecturally identical
  to our `TrigramBloom`**, parameterised the same way (bits per row +
  k hashes).
* `tokenbf_v1` — Bloom of whitespace-split tokens. Better for
  natural-language search, weaker for URL/code substring search.

ClickHouse exposes these via `INDEX … TYPE ngrambf_v1(...) GRANULARITY
N` on a `MergeTree` table.

References: ClickHouse docs § "Data Skipping Indexes", *Korotkov &
Milovidov, VLDB 2024*.

### Snowflake search optimization service (2019–)

Per-micropartition (typically 16 MB compressed, ~100 K–1 M rows)
fingerprint that supports `EQUALITY`, `SUBSTRING`, and `IN` predicates.
The substring fingerprint is essentially a trigram Bloom; details are
proprietary but described in the docs and engineering blog
(*"Search Optimization Service for Substring Search"*, 2022).

### BtrBlocks (CWI, 2023)

Kuschewski et al., *SIGMOD 2023*. Per-block dictionary-presence
bitmap ("TLB" — Table-Level Bitmap) for dict-encoded string columns:
one bit per global dict id × one per block, telling the scanner which
blocks **cannot** contain a literal. This is **architecturally
identical to our `DictPresence`** (we restrict the global dict to the
OnPair-trained dict).

### Parquet column statistics & Page Index (2016–)

Parquet stores per-row-group `min/max/null_count`, and the
Parquet 2.9 Page Index extends this to per-page. For strings these are
lex bounds — useful for range and prefix queries, useless for
substring. Mentioned here because it's the baseline every columnar
engine has and that the more sophisticated indexes above improve on.

### Quickwit / Tantivy (2019–)

Inverted index per segment with per-term postings; Tantivy adds
*positional* postings for phrase queries. Heavier than chunk Bloom but
sub-second substring on TB-scale text. Quickwit is the columnar
ClickHouse-style fork of Tantivy.

### DuckDB (2024) & DataFusion (2024)

Runtime min/max filters (a.k.a. *zone maps*) for joined columns;
*adaptive predicate pushdown* runtime filters. No general substring
skip index in 2024; community discussions reference porting
ClickHouse's `ngrambf_v1` shape.

## Compression-substrate context

These don't skip but they shape what a skip-index can exploit:

* **FSST** — Boncz, Neumann, Leis, *PVLDB* 13(11) 2020. Per-column
  256-entry frequent-substring dictionary; 1-byte codes; greedy
  longest-match parser. The `dfa::ContainsBloom` shape used per-row
  inside OnPair / FSST predicate evaluation comes from the FSST paper.
* **OnPair / OpPair** — Gargiulo (recent, 2024). Generalises FSST to
  9–16-bit codes, larger dict (up to 64 K), still lex-sorted with the
  same greedy-LPM property. Used as the substrate here.
* **Roaring bitmaps** — Chambi et al., *Software: Practice &
  Experience* 46(5) 2016. Replaces dense `Vec<u64>` bitmaps in
  `DictPresence` for sparse chunks; ~3–5× smaller in the common case.

## How our four indexes map to the literature

| ours | lineage | tracks | strong for |
|------|---------|--------|------------|
| `DictPresence` | BtrBlocks TLB (2023); signature files (1984) | per-chunk bitmap over a shared dict | `eq`, `LIKE 'p%'` |
| `TrigramBloom` | pg_trgm (2007), ClickHouse `ngrambf_v1` (2018) | per-chunk Bloom of byte trigrams | `LIKE '%s%'` |
| `SeamBloom` | (novel; combines FSST-style dict knowledge with q-gram Bloom) | per-chunk Bloom of trigrams that *cross a token seam*, interior trigrams recovered from `DictPresence` + dict | `LIKE '%s%'` at smaller byte budget |
| `TokenPairBloom` | (novel; the natural OnPair-codes-side filter) | per-chunk Bloom of consecutive `(c_i, c_{i+1})` code pairs | `eq` (strictly stronger than `DictPresence`); weaker than `TrigramBloom` for `LIKE '%s%'` because each trigram has many candidate dict pairs and the FPRs multiply |

## What the data collector says

Run `examples/skip_sweep.rs` with your column. The
`{variant, chunk_size, bits_per_row}` grid is a 2-3 D Pareto frontier;
the column you should sort the CSV by depends on what you're
optimising:

* **Min bytes / row at ≤ 5 pp above floor for `LIKE '%s%'`** → C
  (SeamBloom + DictPresence) at 32 bits / row is usually the answer
  on URL-shaped columns. ~4.5 B / row, +3.9 pp vs floor.
* **Min eval time for short needles** → B (TrigramBloom); each
  query is `|needle| - 2` Bloom probes, ~10–100 ns per chunk.
* **Highest substring recall regardless of size** → B at ≥ 64 bits /
  row (~8 B / row, +3.7 pp vs floor; diminishing returns above).
* **`eq` selectivity (only)** → A (DictPresence) at 0.5 B / row;
  D (TokenPairBloom) adds adjacency for stricter eq pruning but is
  rarely needed in practice.

## Open directions

1. **Roaring `DictPresence`** — drop-in size win on sparse chunks.
2. **Xor / ribbon-filter `TrigramBloom`** — ~30 % size reduction at the
   same FPR.
3. **Per-column tuning** — the sweep already exposes the data needed
   to pick `bits_per_row` per column; a small auto-tuner that
   maximises pruning subject to a byte-budget would close the loop.
4. **Skip-index pushdown into Vortex `scan`** — at the moment the
   skip checks are open-coded in tests and examples; wiring them
   into the scan planner would actually save I/O.
5. **Position-aware seam Bloom** — capture not just *which* token-pair
   seams occurred but at *which row position*, enabling
   `LIKE 'p%'` pruning that's currently only `DictPresence`-bound.
