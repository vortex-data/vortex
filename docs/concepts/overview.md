# Vortex Ecosystem Overview

Vortex is more than a file format. It is a modular ecosystem for working with compressed columnar
data across the full lifecycle: in-memory computation, on-disk storage, over-the-wire transfer, and
integration with query engines.

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

## In-Memory Core

**[Arrays](arrays.md)** are Vortex's in-memory data representation. Unlike Arrow, a Vortex array
can be *compressed* — an array of integers might be stored as a FastLanes bit-packed array rather
than a flat buffer. Arrays form trees: a compressed array contains child arrays and buffers that
together define the encoding.

**[Compute](compute.md)** functions operate directly on compressed arrays where possible, avoiding
decompression. The compute layer dispatches to encoding-specific kernel implementations, falling
back to canonical (Arrow-compatible) implementations when no specialized kernel exists.

**[Encodings](encodings.md)** define how arrays are physically stored. Vortex ships with canonical
encodings (zero-copy to Arrow) and compressed encodings drawn from recent database research:
ALP for floats, FastLanes for integers, FSST for strings, and many more. Encodings are
pluggable — you can register your own.

## Storage and I/O

**[Layouts](layouts.md)** represent larger-than-memory hierarchical columnar data. A layout defines
how data is organized into row groups, column chunks, and nested structures on disk. Layouts are
lazily fetched — only the data needed for a query is read.

The **[File Format](../specs/file-format.md)** (`.vortex` files) serializes layouts to disk with
zero-allocation reads, FlatBuffer metadata for O(1) column access, and optional WASM decompression
kernels for forward compatibility.

The **[IPC Format](../specs/ipc-format.md)** provides zero-copy serialization of possibly-compressed
arrays for inter-process communication.

The **[Scan API](scan.md)** provides an abstract table scan interface over Vortex data. It uses the
wire format for interchange and supports filter and projection pushdown. Query engines integrate
with Vortex through the Scan API.

## Language Bindings

Vortex provides bindings for multiple languages:

- **[Python](../getting-started/python.rst)** — Full-featured bindings via PyO3, including
  Arrow/Pandas/Polars interop and the Dataset API.
- **[Java](../getting-started/java.md)** — JNI bindings with Apache Spark and Trino connectors.
- **[C / C++](../developer-guide/embedding/ffi.md)** — C FFI for embedding Vortex in any language with
  a C-compatible FFI, plus a C++ wrapper.

## Query Engine Integrations

Vortex integrates with several query engines:

- **[DataFusion](../user-guide/datafusion.md)** — Native Rust integration as a DataFusion
  TableProvider.
- **[DuckDB](../user-guide/duckdb.md)** — Integration via the Arrow Dataset API in Python.
- **[Spark](../user-guide/spark.md)** — Apache Spark DataSource V2 connector via JNI.
- **[Trino](../user-guide/work-in-progress.md#trino)** — Trino connector via JNI.
- **[Ray](../user-guide/work-in-progress.md#ray-data)** — Ray Data integration for distributed processing.

## Acceleration

**[GPU / CUDA](../developer-guide/embedding/gpu.md)** support is under active development, enabling
GPU-accelerated decompression and compute for Vortex arrays.
