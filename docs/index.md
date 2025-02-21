# Vortex: an extensible, state-of-the-art columnar format

Vortex is an in-memory, on-disk, and over-the-wire format for working with columnar data, built around
the latest research from the database community.

## In-memory

Vortex in-memory arrays support:

* Zero-copy interoperability with [Apache Arrow](https://arrow.apache.org).
* Cascading compression with lightweight, vectorized encodings such as
  [FastLanes](https://github.com/spiraldb/fastlanes),
  [FSST](https://github.com/spiraldb/fsst),
  and [ALP](https://github.com/spiraldb/alp).
* Fast random access to compressed data.
* Compute push-down over compressed data.
* Array statistics for efficient compute.

## On-disk

Vortex ships with an extensible file format supporting:

* Zero-allocation reads, deferring both deserialization and decompression.
* Zero-copy reads from memory-mapped files.
* FlatBuffer metadata to support ultra-wide schemas (>>100k columns) with O(1) column access.
* Fully customizable layouts and encodings (row-groups, column-groups -- the writer decides).
* Forwards compatibility by optionally embedding [WASM](https://webassembly.org/) decompression kernels for new encodings.

## Over-the-wire

Vortex defines a work-in-progress IPC format for sending possibly compressed arrays over the wire.

* Zero-copy serialization and deserialization.
* Support for both compressed and uncompressed data.
* Enables partial compute push-down to storage servers.
* Enables client-side browser decompression with Vortex WASM.

## Extensibility

Vortex is designed to be incredibly extensible. Almost all reader and writer logic is extensible at compile-time
by providing various implementations of Rust traits, and encodings and layouts are extensible at runtime with
dynamically loaded libraries or WebAssembly kernels.

Please reach out to us if you'd like to extend Vortex with your own encodings, layouts, or other functionality.

## Concepts

Almost everything in Vortex is extensible, enabling it to be used for both general-purpose columnar data processing
and for niche use-cases where specific encodings and performance characteristics are required. In some sense,
Vortex is really an ecosystem of building blocks (albeit with batteries-included sane defaults).

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

## Quickstarts

Vortex is currently available for both Python and Rust. If you want to get started right away, feel free to
jump right in with the language-specific quickstarts. For more detailed information, see the
[User Guides](#user-guides) instead.

```{toctree}
---
maxdepth: 1
includehidden:
caption: Quickstarts
---

Python <quickstart/python>
Rust <quickstart/rust>
```

## User Guides

The user guides for each language provide a comprehensive overview of the Vortex API and how to use it.

```{toctree}
---
maxdepth: 3
caption: User Guides
---

Python <python/index>
Rust <rust/index>
```

## Specifications

Vortex currently defines two serialization formats: a file format and an IPC format. The file format is designed for
efficient access to [Vortex Layouts](/concepts/layouts) on disk, while the IPC format is designed to send
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

references
Spiral <https://spiraldb.com>
GitHub <https://github.com/spiraldb/vortex>
PyPI <https://pypi.org/project/vortex-array>
Crates <https://crates.io/crates/vortex>
```
