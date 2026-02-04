# Ecosystem Overview

Vortex is more than a file format. It is a modular and highly extensible ecosystem for working with
compressed columnar data across the full lifecycle: in-memory computation, on-disk storage,
over-the-wire transfer, and integration with query engines.

This page provides a map of the major components and how they relate to each other.

## Component Map

```{mermaid}
graph TD
    subgraph types["Type System"]
        dtype["DTypes<br/><small>Null · Bool · Primitive · UTF8<br/>Binary · Struct · List · Extension</small>"]
    end

    subgraph core["In-Memory"]
        array["Arrays"]
        compute["Compute Kernels"]
        encodings["Encodings<br/><small>ALP · FastLanes · FSST · RunEnd<br/>Sparse · ZigZag · PCO · Zstd · ...</small>"]
    end

    subgraph storage["Storage & I/O"]
        layout["Layouts"]
        file["File Format (.vortex)"]
        ipc["IPC Format"]
        scan["Scan API"]
    end

    subgraph bindings["Language Bindings"]
        python["Python"]
        java["Java"]
        cffi["C / C++"]
    end

    subgraph engines["Query Engines"]
        datafusion["DataFusion"]
        duckdb["DuckDB"]
        spark["Spark"]
        trino["Trino"]
    end

    dtype --> array
    array --> compute
    array --> encodings
    encodings --> layout
    layout --> file
    layout --> ipc
    file --> scan
    ipc --> scan
    scan --> engines
    array --> bindings
    file --> bindings
```

## Type System

**[DTypes](dtypes.md)** define the logical types of data in Vortex. The type system is deliberately
logical rather than physical: a `UTF8` dtype says "this is a string" without dictating how the bytes
are laid out. This separation allows the same logical data to be stored in many different physical
encodings.

Vortex supports: Null, Bool, Primitive (i8 through f64), UTF8, Binary, Decimal, Struct, List, and
Extension types.

## In-Memory Representation

**[Arrays](arrays.md)** are Vortex's in-memory data representation. Unlike Arrow, a Vortex array
can be *compressed* — an array of integers might be stored as a FastLanes bit-packed array rather
than a flat buffer. Arrays form trees: a compressed array contains child arrays and buffers that
together define the encoding.

**[Compute](scalar_fns.md)** functions operate directly on compressed arrays where possible, avoiding
decompression. The compute layer dispatches to encoding-specific kernel implementations, falling
back to canonical (Arrow-compatible) implementations when no specialized kernel exists. Compute
functions are either scalar (element-wise) or aggregate (group-wise), and can be extended via
Vortex plugins.

## Serialized Representation

**[Arrays](arrays.md)** share the same serialized representation on disk, as in memory, as over the
wire. This zero-copy design means no serialization or deserialization overhead when reading from
disk or sending data between processes. And all while retaining the ability to operate directly on
compressed data.

**[Layouts](layouts.md)** organize arrays into larger-than-memory hierarchical datasets. For example
a chunked layout breaks a large table into what you may know as row groups. This structure allows for
pruning unused data when scanning a layout into a stream of Vortex arrays. Layouts store data in
abstract segment storage, meaning they can read from any form of block storage including local disk,
object stores, Redis, Postgres block storage, in-memory caches, and more.

**[File Format](../specs/file-format.md)** (`.vortex` files) serialize a layout tree to disk and provide
an efficient way to retrieve segments using coalescing and prefetching tuned to the underlying storage.
The file format uses FlatBuffer for metadata to allow for O(1) access to even ultra-wide schemas, and
is carefully laid out to support memory mapping and Direct I/O.

**[IPC Format](../specs/ipc-format.md)** provides a streaming message-oriented API for efficient transfer of
possibly-compressed Vortex arrays.

## Scan API

The **[Scan API](scan.md)** provides an abstract table scan interface that can sit between any storage backend
and query engine while supporting each engine's internal optimized data representations. For example, the
Vortex x DuckDB integration returns compressed `FSST` arrays directly to DuckDB without the unnecessary
decompression imposed by using Arrow as an intermediary.

The API is work-in-progress and designed to be easily implementable for other storage formats including Parquet,
Iceberg, PyArrow Datasets, and more.

## Language Bindings

Vortex provides bindings for multiple languages (with varying levels of support for the full Vortex API):

- **[Rust](https://docs.rs/vortex)** — Full-featured Rust API.
- **[Python](../api/python/index.rst)** — Full-featured bindings via PyO3, including
  Arrow/Pandas/Polars interop and the Dataset API.
- **[Java](../api/java/index.rst)** — JNI bindings with Apache Spark and Trino connectors.
- **[C](../api/c/index.rst)** — C FFI for embedding Vortex in any language with
  a C-compatible FFI.
- **[C++](../api/cpp/index.rst)** — C++ API for reading and writing Vortex files.

## Query Engine Integrations

Vortex integrates with several query engines:

- **[DataFusion](../user-guide/datafusion.md)** — Native Rust integration as a DataFusion TableProvider.
- **[DuckDB](../user-guide/duckdb.md)** — Integration via the Arrow Dataset API in Python.
- **[Spark](../user-guide/spark.md)** — Apache Spark DataSource V2 connector via JNI.
- **[Polars](../user-guide/polars.md)** — Trino connector via JNI.
- **[Ray](../user-guide/ray.md)** — Ray Data integration for distributed processing.

## Acceleration

**[GPU / CUDA](../developer-guide/embedding/gpu.md)** support is under active development, enabling
GPU-accelerated decompression and compute for Vortex arrays.
