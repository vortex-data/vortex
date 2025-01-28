# Vortex: a State-of-the-Art Columnar File Format

Vortex is a fast & extensible columnar file format that is based around the latest research from the
database community. It is built around cascading compression with lightweight, vectorized encodings
(i.e., no block compression), allowing for both efficient random access and extremely fast
decompression.

Vortex includes an accompanying in-memory format for these (recursively) compressed arrays,
that is zero-copy compatible with Apache Arrow in uncompressed form. Taken together, the Vortex
library is a useful toolkit with compressed Arrow data in-memory, on-disk, & over-the-wire.

Vortex consolidates the metadata in a series of flatbuffers in the footer, in order to minimize
the number of reads (important when reading from object storage) & the deserialization overhead
(important for wide tables with many columns).

Vortex aspires to succeed Apache Parquet by pushing the Pareto frontier outwards: 1-2x faster
writes, 2-10x faster scans, and 100-200x faster random access reads, while preserving the same
approximate compression ratio as Parquet v2 with zstd.

Its features include:

- A zero-copy data layout for disk, memory, and the wire.
- Kernels for computing on, filtering, slicing, indexing, and projecting compressed arrays.
- Builtin state-of-the-art codecs including FastLanes (integer bit-packing), ALP (floating point),
  and FSST (strings).
- Support for custom user-implemented codecs.
- Support for, but no requirement for, row groups.
- A read sub-system supporting filter and projection pushdown.

Vortex's flexible layout empowers writers to choose the right layout for their setting: fast writes,
fast reads, small files, few columns, many columns, over-sized columns, etc.

## Concepts

Vortex is more like an ecosystem of building blocks than it is a specific format or specification. Almost everything
in Vortex is extensible, enabling it to be used for both general-purpose columnar data processing, and niche
embedded use-cases where specific encodings and performance characteristics are required.

This section of the documentation covers the core concepts of Vortex and how they piece together.

```{toctree}
---
maxdepth: 3
includehidden:
caption: Concepts
---

Arrays <concepts/arrays>
Layouts <concepts/layouts>
Data Types <concepts/dtypes>
Compute <concepts/compute>
```

## User Guides

Vortex is currently available for both Python and Rust. The user guides for each language provide a comprehensive
overview of the Vortex API and how to use it.

```{toctree}
---
maxdepth: 3
includehidden:
caption: User Guide
---

Python <python/index>
Rust <rust/index>
```

## Specifications

Vortex currently defines two serialization formats: a file format and an IPC format. The file format is designed for
random access to [Vortex Layouts](/concepts/layouts) on disk, while the IPC format is designed to efficiently send
possibly compressed [Vortex Arrays](/concepts/arrays) over the wire.

```{toctree}
---
maxdepth: 3
includehidden:
caption: Specifications
---

specs/file-format
specs/ipc-format
specs/dtype-format
```

```{toctree}
---
hidden:
caption: Project Links
---

Spiral <https://spiraldb.com>
GitHub <https://github.com/spiraldb/vortex>
PyPI <https://pypi.org/project/vortex-array>
Crates <https://crates.io/crates/vortex>
```
