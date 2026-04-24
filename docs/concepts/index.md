# Concepts

Vortex is a modular ecosystem for working with compressed columnar data: in-memory, on-disk,
over-the-wire, and integrated with query engines.

```{toctree}
---
maxdepth: 2
---

dtypes
arrays
expressions
layouts
file-format
scanning
```


## Core Concepts

**[DTypes](dtypes.md)** are Vortex's logical type system. Types like `UTF8` describe what data means
without dictating physical layout, allowing the same logical data to use different encodings.

**[Arrays](arrays.md)** are the in-memory representation. Unlike Arrow, Vortex arrays can be
*compressed*—an integer array might be bit-packed rather than stored as a flat buffer. Arrays
share the same representation on disk and over the wire, enabling zero-copy I/O.

**[Compute](expressions.md)** functions operate directly on compressed arrays where possible,
dispatching to encoding-specific kernels or falling back to canonical implementations.

## Storage & I/O

**[Layouts](layouts.md)** organize arrays into larger-than-memory datasets (e.g., chunked row groups)
and can read from any block storage: local disk, object stores, caches, etc.

**[File Format](../specs/file-format.md)** (`.vortex` files) serialize layouts to disk with efficient
segment retrieval, FlatBuffer metadata for O(1) schema access, and support for memory mapping.

**[IPC Format](../specs/ipc-format.md)** provides streaming transfer of compressed arrays.

## Integrations

**Language bindings:** [Rust](https://docs.rs/vortex), [Python](../api/python/index.rst),
[Java](../api/java/index.rst), [C](../api/c/index.rst), [C++](../api/cpp/index.rst)

**Query engines:** [DataFusion](../user-guide/datafusion.md), [DuckDB](../user-guide/duckdb.md),
[Spark](../user-guide/spark.md), [Polars](../user-guide/polars.md), [Ray](../user-guide/ray.md)
