# Vortex

[![Build Status](https://github.com/fulcrum-so/vortex/actions/workflows/ci.yml/badge.svg)](https://github.com/spiraldb/vortex/actions)
[![Crates.io](https://img.shields.io/crates/v/vortex-array.svg)](https://crates.io/crates/vortex-array)
[![Documentation](https://docs.rs/vortex-array/badge.svg)](https://docs.rs/vortex-array)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vortex-array)](https://pypi.org/project/vortex-array/)

Vortex is an extensible, state-of-the-art columnar file format, with associated tools for working with compressed Apache Arrow arrays
in-memory, on-disk, and over-the-wire.

Vortex is an aspiring successor to Apache Parquet, with dramatically faster random access reads (100-200x faster) and scans (2-10x faster),
while preserving approximately the same compression ratio and write throughput as Parquet with zstd.
It is designed to support very wide tables (at least 10s of thousands of columns) and (eventually) on-device decompression on GPUs.

Vortex is intended to be to columnar file formats what Apache DataFusion is to query engines: highly extensible,
extremely fast, & batteries-included.

> [!CAUTION]
> This library is still under rapid development and is a work in progress!
>
> Some key features are not yet implemented, both the API and the serialized format are likely to change in breaking ways,
> and we cannot yet guarantee correctness in all cases.

The major features of Vortex are:

* **Logical Types** - a schema definition that makes no assertions about physical layout.
* **Zero-Copy to Arrow** - "canonicalized" (i.e., fully decompressed) Vortex arrays can be zero-copy converted to/from Apache Arrow arrays.
* **Extensible Encodings** - a pluggable set of physical layouts. In addition to the builtin set of Arrow-compatible encodings,
  the Vortex repository includes a number of state-of-the-art encodings (e.g., FastLanes, ALP, FSST, etc.) that are implemented
  as extensions. While arbitrary encodings can be implemented as extensions, we have intentionally chosen a small set
  of encodings that are highly data-parallel, which in turn allows for efficient vectorized decoding, random access reads,
  and (in the future) decompression on GPUs.
* **Cascading Compression** - data can be recursively compressed with multiple nested encodings.
* **Pluggable Compression Strategies** - the built-in Compressor is based on BtrBlocks, but other strategies can trivially be used instead.
* **Compute** - basic compute kernels that can operate over encoded data (e.g., for filter pushdown).
* **Statistics** - each array carries around lazily computed summary statistics, optionally populated at read-time.
  These are available to compute kernels as well as to the compressor.
* **Serialization** - Zero-copy serialization of arrays, both for IPC and for file formats.
* **Columnar File Format (in progress)** - A modern file format that uses the Vortex serde library to store compressed array data.
  Optimized for random access reads and extremely fast scans; an aspiring successor to Apache Parquet.

## Overview: Logical vs Physical

One of the core design principles in Vortex is strict separation of logical and physical concerns.

For example, a Vortex array is defined by a logical data type (i.e., the type of scalar elements) as well as a physical encoding
(the type of the array itself). Vortex ships with several built-in encodings, as well as several extension encodings.

The built-in encodings are primarily designed to model the Apache Arrow in-memory format, enabling us to construct
Vortex arrays with zero-copy from Arrow arrays. There are also several built-in encodings (e.g., `sparse` and
`chunked`) that are useful building blocks for other encodings. The included extension encodings are mostly designed
to model compressed in-memory arrays, such as run-length or dictionary encoding.

Analogously, `vortex-serde` is designed to handle the low-level physical details of reading and writing Vortex arrays. Choices
about which encodings to use or how to logically chunk data are left up to the `Compressor` implementation.

One of the unique attributes of the (in-progress) Vortex file format is that it encodes the physical layout of the data within the
file's footer. This allows the file format to be effectively self-describing and to evolve without breaking changes to
the file format specification.

For example, the Compressor implementation can choose to chunk data into a Parquet-like layout with
row groups and aligned pages (ChunkedArray of StructArray of ChunkedArrays with equal chunk sizes). Alternatively, it can choose
to chunk different columns differently based on their compressed size and data distributions (e.g., a column that is constant
across all rows can be a single chunk, whereas a large string column may be split arbitrarily many times).

In the same vein, the format is designed to support forward compatibility by optionally embedding WASM decoders directly into the files
themselves. This should help avoid the rapid calcification that has plagued other columnar file formats.

## Components

### Logical Types

The Vortex type-system is still in flux. The current set of logical types is:

* Null
* Bool
* Integer(8, 16, 32, 64)
* Float(16, b16, 32, 64)
* Binary
* UTF8
* Struct
* List (partially implemented)
* Date/Time/DateTime/Duration (implemented as an extension type)
* Decimal: TODO
* FixedList: TODO
* Tensor: TODO
* Union: TODO

### Canonical/Flat Encodings

Vortex includes a base set of "flat" encodings that are designed to be zero-copy with Apache Arrow. These are the
canonical representations of each of the logical data types. The canonical encodings currently supported are:

* Null
* Bool
* Primitive (Integer, Float)
* Struct
* VarBin (Binary, UTF8)
* VarBinView (Binary, UTF8)
* Extension
* ...with more to come

### Compressed Encodings

Vortex includes a set of highly data-parallel, vectorized encodings. These encodings each correspond to a compressed
in-memory array implementation, allowing us to defer decompression. Currently, these are:

* Adaptive Lossless Floating Point (ALP)
* BitPacked (FastLanes)
* Constant
* Chunked
* Delta (FastLanes)
* Dictionary
* Fast Static Symbol Table (FSST)
* Frame-of-Reference
* Run-end Encoding
* RoaringUInt
* RoaringBool
* Sparse
* ZigZag
* ...with more to come

### Compression

Vortex's default compression strategy is based on the
[BtrBlocks](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf) paper.

Roughly, for each chunk of data, a sample of at least ~1% of the data is taken. Compression is then attempted
(recursively) with a set of lightweight encodings. The best-performing combination of encodings is then chosen to encode
the entire chunk. This sounds like it would be very expensive, but given the logical types and basic statistics about a
chunk, it is possible to cheaply prune many encodings and ensure the search space does not explode in size.

### Compute

Vortex provides the ability for each encoding to specialize the implementation of a compute function to avoid
decompressing where possible. For example, filtering a dictionary-encoded UTF8 array can be more cheaply performed by
filtering the dictionary first.

Note--as mentioned above--that Vortex does not intend to become a full-fledged compute engine, but rather to implement
basic compute operations as may be required for efficient scanning & pushdown.

### Statistics

Vortex arrays carry lazily-computed summary statistics. Unlike other array libraries, these statistics can be populated
from disk formats such as Parquet and preserved all the way into a compute engine. Statistics are available to compute
kernels as well as to the compressor.

The current statistics are:

* BitWidthFreq
* TrailingZeroFreq
* IsConstant
* IsSorted
* IsStrictSorted
* Max
* Min
* RunCount
* TrueCount
* NullCount

### Serialization / Deserialization (Serde)

The goals of the `vortex-serde` implementation are:

* Support scanning (column projection + row filter) with zero-copy and zero heap allocation.
* Support random access in constant or near-constant time.
* Forward statistical information (such as sortedness) to consumers.
* Provide IPC format for sending arrays between processes.
* Provide an extensible, best-in-class file format for storing columnar data on disk or in object storage.

TODO: insert diagram here

## Integration with Apache Arrow

Apache Arrow is the de facto standard for interoperating on columnar array data. Naturally, Vortex is designed to
be maximally compatible with Apache Arrow. All Arrow arrays can be converted into Vortex arrays with zero-copy,
and a Vortex array constructed from an Arrow array can be converted back to Arrow, again with zero-copy.

It is important to note that Vortex and Arrow have different--albeit complementary--goals.

Vortex explicitly separates logical types from physical encodings, distinguishing it from Arrow. This allows
Vortex to model more complex arrays while still exposing a logical interface. For example, Vortex can model a UTF8
`ChunkedArray` where the first chunk is run-length encoded and the second chunk is dictionary encoded.
In Arrow, `RunLengthArray` and `DictionaryArray` are separate incompatible types, and so cannot be combined in this way.

## Usage

For best performance we recommend using [MiMalloc](https://github.com/microsoft/mimalloc) as the application's
allocator.

```rust
#[global_allocator]
static GLOBAL_ALLOC: MiMalloc = MiMalloc;
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Setup

### Mac

The project has several optional-but-recommended external dependencies:

```bash
# Required if you want to modify any of the .fbs or .proto files
brew install flatbuffers protobuf

# Required for benchmarks
brew install duckdb
```

You also need the Rust toolchain installed. If you haven't already, install [rustup](https://rustup.rs/)
with one of the following commands:

```bash
# option 1
brew install rustup

# option 2
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

This repo uses uv to manage the combined Rust/Python monorepo build. After installing uv, make sure to run:

```bash
# Install uv from https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

## License

Licensed under the Apache License, Version 2.0 (the "License").

## Governance

Vortex is and will remain an open-source project. Our intent is to model its governance structure after the
[Substrait project](https://substrait.io/governance/), which in turn is based on the model of the Apache Software Foundation.
Expect more details on this in Q4 2024.

## Acknowledgments 🏆

This project is inspired by and--in some cases--directly based upon the existing, excellent work of many researchers
and OSS developers.

In particular, the following academic papers have strongly influenced development:

* Maximilian Kuschewski, David Sauerwein, Adnan Alhomssi, and Viktor Leis.
  [BtrBlocks: Efficient Columnar Compression for Data Lakes](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf).
  Proc. ACM Manag. Data 1, 2, Article 118 (June 2023), 14 pages.
* Azim Afroozeh and Peter Boncz. [The FastLanes Compression Layout: Decoding >100 Billion Integers per Second with Scalar
  Code](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf). PVLDB, 16(9): 2132 - 2144, 2023.
* Peter Boncz, Thomas Neumann, and Viktor Leis. [FSST: Fast Random Access String
  Compression](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf).
  PVLDB, 13(11): 2649-2661, 2020.
* Azim Afroozeh, Leonardo X. Kuffo, and Peter Boncz. [ALP: Adaptive Lossless floating-Point
  Compression](https://ir.cwi.nl/pub/33334/33334.pdf). Proc. ACM Manag. Data 1, 4 (SIGMOD), Article 230
  (December 2023), 26 pages.
* Biswapesh Chattopadhyay, Priyam Dutta, Weiran Liu, Ott Tinn, Andrew Mccormick, Aniket Mokashi, Paul Harvey,
  Hector Gonzalez, David Lomax, Sagar Mittal, et al. [Procella: Unifying serving and analytical
  data at YouTube](https://dl.acm.org/citation.cfm?id=3360438). PVLDB, 12(12): 2022-2034, 2019.
* Dominik Durner, Viktor Leis, and Thomas Neumann. [Exploiting Cloud Object Storage for High-Performance
  Analytics](https://www.durner.dev/app/media/papers/anyblob-vldb23.pdf). PVLDB, 16(11): 2769-2782, 2023.

Additionally, we benefited greatly from:

* the existence, ideas, & implementations of both [Apache Arrow](https://arrow.apache.org) and
  [Apache DataFusion](https://github.com/apache/datafusion).
* the [parquet2](https://github.com/jorgecarleitao/parquet2) project by [Jorge Leitao](https://github.com/jorgecarleitao).
* the public discussions around choices of compression codecs, as well as the C++ implementations thereof,
  from [duckdb](https://github.com/duckdb/duckdb).
* the [Velox](https://github.com/facebookincubator/velox) and [Nimble](https://github.com/facebookincubator/nimble) projects,
  and discussions with their maintainers.

Thanks to all of the aforementioned for sharing their work and knowledge with the world! 🚀
