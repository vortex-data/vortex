# OnPair

[![Crates.io Version](https://img.shields.io/crates/v/onpair)](https://crates.io/crates/onpair)
[![docs.rs](https://img.shields.io/docsrs/onpair)](https://docs.rs/onpair)
[![CI](https://img.shields.io/github/actions/workflow/status/spiraldb/onpair/ci.yml?branch=develop)](https://github.com/spiraldb/onpair/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/onpair)](LICENSE)
[![MSRV](https://img.shields.io/crates/msrv/onpair)](https://github.com/spiraldb/onpair/blob/develop/rust-toolchain.toml)

OnPair is a Rust codec for compressing string columns while keeping every
row independently accessible. It populates a dictionary of up to 2^16 patterns
recurring across the column, then encodes each row as a sequence of indices into
it. Decoding just looks up each index and copies out the bytes, so columns
decompress at high throughput, and any single row can be read or dropped without
touching the ones around it.

## Why OnPair

- **Fast decompression.** The decoder looks up each index and copies out the
  bytes, with no entropy-decoding stage in the hot path, so decoding sustains
  high throughput.
- **Random access.** Each row is a self-contained run of codes, so it decodes
  on its own, with no block to unpack and no neighbours to rebuild.
- **Efficient compression.** OnPair reaches ratios competitive with far heavier
  methods while keeping its encoding pass fast, so strong compression comes at a
  fraction of the usual cost.
- **Compressed-domain search.** Evaluate equality, prefix, and substring
  predicates without decoding rows back to bytes.

OnPair is designed for high-cardinality columns, whose many distinct values tend
to share substrings rather than repeat in full. On low-cardinality columns it is
better to deduplicate first: encode the column as references to the distinct
values, then run OnPair over that set of unique strings. Deduplication removes
the value-level repetition, and OnPair compresses the substring redundancy that
remains.

## Benchmarks

Benchmarks comparing OnPair against the main comparable string-compression
algorithms are in progress.

<!--
TODO: Add numbers and plots comparing OnPair against the main similar codecs,
across compression ratio, random-access latency, decompression throughput, and
compressed-domain search.
-->

## Quick start

```sh
cargo add onpair
```

```rust
use onpair::{compress, Config, DECODE_PADDING};

// Three values in the Arrow layout OnPair takes: one byte buffer plus offsets.
let values = b"catdogbird";
let offsets = [0u32, 3, 6, 10]; // "cat", "dog", "bird"

let column = compress(values, &offsets, Config::default()).unwrap();
let view = column.view();

// A value decodes to its original length, so a buffer of the longest value plus
// DECODE_PADDING (which absorbs the decoder's 16-byte tail write) fits any row.
let max_len = 4; // longest value in this column
let mut decoded = Vec::<u8>::with_capacity(max_len + DECODE_PADDING);
// SAFETY: capacity >= this row's decoded length + DECODE_PADDING.
let len = unsafe { view.decompress_row_into(1, decoded.spare_capacity_mut()) };
unsafe { decoded.set_len(len) }; // SAFETY: decode initialized `len` bytes.

assert_eq!(decoded, b"dog");
```

`Config::default()` caps the dictionary at 2^12 entries; raise `max_dict_bits`
for a larger one, up to 2^16.


## Further reading

- [In-memory interchange format](docs/interchange-format.md)
- [Encoding performance investigation](docs/encoding-performance.md)
- [OnPair: Short Strings Compression for Fast Random Access](https://arxiv.org/abs/2508.02280)
- [C++ reference implementation](https://github.com/gargiulofrancesco/onpair_cpp)

OnPair is licensed under [Apache-2.0](LICENSE).
