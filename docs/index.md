# Vortex

Vortex is an extensible, state-of-the-art format for columnar data. It includes
specifications & tools for manipulating possibly-compressed arrays in-memory,
on-disk (file format), and over-the-wire (IPC format). Vortex is built around the
latest research from the database community.

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

It can be useful to view Vortex as an ecosystem of building blocks rather than a singular specific format. Almost
everything in Vortex is extensible, enabling it to be used for both general-purpose columnar data processing, and niche
embedded use-cases where specific encodings and performance characteristics are required.

This section of the documentation covers the core concepts of Vortex and how they fit together.

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

Vortex is currently available for both Python and Rust. To get started, we recommend the language-specific quickstarts.

```{toctree}
---
maxdepth: 1
includehidden:
caption: Quickstarts
---

Python <quickstart/python>
Rust <quickstart/rust>
```

## API Documentation

Vortex is primarily written in Rust, and the Rust API is the most complete API for Vortex.
The Vortex Python bindings provide a more usable Python interface to the Vortex Rust library.

```{toctree}
---
maxdepth: 2
caption: API Documentation
---

Python API <api/python/index>
Rust API <https://docs.rs/vortex>
```

## User Guides

These user guides provide end-to-end overviews of how to use or extend Vortex
in your own projects. We intend to extend this collection of guides to cover more
use-cases in the future.

```{toctree}
---
maxdepth: 1
caption: User Guides
---

guides/python-integrations
guides/writing-an-encoding
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
