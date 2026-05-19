# string-skip

Block-level skip indexes for string columns — range, prefix, substring, and
wildcard pruning over dictionary-coded data.

## What it does

Given a chunked string column, build per-chunk metadata that tells a query
planner "this chunk *might* contain a matching row" or "this chunk
*definitely cannot*". The latter case skips the chunk entirely, avoiding
decompression and scan I/O.

Sound: never returns "skip" for a chunk that truly matches. False positives
(extra scans) are allowed, false negatives (missed matches) are bugs and
covered by property tests.

## Predicates supported

| Class | Example | Mechanism |
|-------|---------|-----------|
| Equality | `col = 'x'` | min/max range + dict bitmap |
| Range | `col < 'x'`, `BETWEEN a AND b` | min/max only — exact on sorted data |
| Prefix | `LIKE 'p%'` | min/max as a range — exact on sorted data |
| Suffix | `LIKE '%s'` | code-bigram bloom |
| Substring | `LIKE '%x%'` | code-bigram bloom |
| Prefix + suffix | `LIKE 'a%b'` | min/max range + bloom |
| Single wildcard | `LIKE '%a_b%'` | bloom on both anchored parts |
| Multi-fragment | `LIKE '%a%b%c%'` | bloom AND across fragments |
| Length | `LENGTH(col) > k`, `BETWEEN` | min_len / max_len |
| Null | `IS NULL`, `IS NOT NULL` | null count |
| Set membership | `IN ('x', 'y', 'z')` | min/max + dict bitmap per value |

## Variants

- **`DictPresence`** — bitmap over dict ids. 0.05 B/row. Exact for equality
  and `LIKE 'p%'` (on top of min/max).
- **`HybridBloom`** — BitFunnel-style code-bigram bloom that skips
  ubiquitous bigrams. ~2-4 B/row. Tight for substring on URL-like data.
- **`TieredBloom`** — variable-k bloom (BitFunnel's full
  frequency-conscious idea). Larger metadata, tightest substring pruning
  on high-diversity columns.
- **`ChunkStats`** — min/max/length/null per chunk. A few bytes. Exact
  pruning for range/length/null on sorted data.

## How to plug in your dict encoding

Implement two minimal traits:

```rust
pub trait TokenDict {
    fn len(&self) -> usize;
    fn token_bytes(&self, id: u16) -> &[u8];
    // invariant: tokens are lex-sorted by their bytes
}
```

A code stream is just `(codes: &[u16], codes_offsets: &[u32])` where
`codes_offsets[i..i+1]` describes row `i`'s slice into `codes`. OnPair,
FSST, and any custom BPE-style encoding fit this shape.

## Usage

```rust
use string_skip::*;

// Per-column metadata (built once at write time)
let dict = MyDict::from(...);                       // your TokenDict impl
let index = DictIndex::build(&dict);
let ubiq = UbiquitousBigrams::build(&codes, &offsets, chunk_size, 75);

// Per-chunk metadata (built once at write time)
let stats    = ChunkStats::from_rows(&rows[lo..hi]);
let presence = DictPresence::build(&codes[tok_lo..tok_hi], dict.len());
let bloom    = HybridBloom::build(&codes, &offsets, lo, hi, 16, &ubiq);

// Query (at read time)
let pred = Pred::Contains(b"needle".to_vec());
let state = ChunkSkipState {
    stats: &stats, presence: &presence, bloom: Some(&bloom),
    tiered: None, ubiq: &ubiq, tiers: &BigramTiers::empty(),
    dict: &dict, index: &index,
};
if chunk_might_match(&pred, &state) {
    // scan this chunk
} else {
    // skip — provably no match
}
```

## Vortex integration

Vortex's chunked-layout reader implements `LayoutReader::pruning_evaluation`.
A `ChunkStatsLayoutReader` wraps the OnPair chunked reader, holds
deserialized per-chunk skip indexes, and calls `chunk_might_match` from
this crate to short-circuit pruning. Mirrors `FileStatsLayoutReader` in
the v2 file format. See `SKIP_DESIGN.md` for the full plan.

## Dependencies

Three runtime deps (all leaf or near-leaf):

| Crate | Why |
|-------|-----|
| `memchr` | SIMD-accelerated `memmem::find` for single-token substring checks |
| `bincode` + `serde` | Standard binary serialization |
| `thiserror` | Single-line error enum |

Dev-only: `proptest` (soundness), `criterion` (benches), `parquet` +
`arrow-array` (demo data loader).

## Soundness guarantees

Two layers of testing:

1. Unit tests (29 cases): each component (bloom, presence, ubiq, tiers,
   chunk_stats, pred) tested in isolation.
2. Property tests (`tests/soundness.rs`): generates random rows + random
   predicates and asserts the invariant "if any row truly matches, the
   chunk evaluator returns `true`". 50 cases × 20 predicates = 1000+
   soundness checks per `cargo test`.

## Benchmarks

`cargo bench -p string-skip` runs criterion microbenchmarks:

| Operation | bits=1024 | bits=16384 | bits=131072 |
|-----------|----------:|-----------:|------------:|
| `bloom.insert(h1, h2)` | 6 ns | 33 ns | 145 ns |
| `bloom.contains(h1, h2)` | 17 ns | 13 ns | 12 ns |
| `bloom.contains_k(h1, h2, k=3)` | 14 ns | — | — |
| `dict_presence.is_set(id)` | 0.65 ns | — | — |
| `ubiq.contains(a, b)` | 4 ns | — | — |

These are well within range of `fastbloom` / `sbbf-rs-safe` for our access
patterns (L1-resident blooms, 5-10 probes per query). The simpler
implementation here gives variable-k support for `TieredBloom` that fixed-k
libraries can't.

## Status

Standalone crate, no Vortex dependency. Used by `vortex-onpair` via a thin
adapter (TODO). Vortex layout/reader integration: TODO (see SKIP_DESIGN.md).

## License

Apache-2.0.
