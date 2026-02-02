# Crate Architecture

The Vortex workspace is organized as a layered Rust monorepo. This page documents the crate
dependency structure and the role of each layer.

## Crate Dependency Graph

```{mermaid}
graph BT
    subgraph foundation["Foundation"]
        error[vortex-error]
        buffer[vortex-buffer]
        utils[vortex-utils]
        fb[vortex-flatbuffers]
        proto[vortex-proto]
    end

    subgraph typesys["Type System"]
        dtype[vortex-dtype]
        scalar[vortex-scalar]
        mask[vortex-mask]
        vector[vortex-vector]
    end

    subgraph core["Core"]
        session[vortex-session]
        metrics[vortex-metrics]
        compute[vortex-compute]
        array[vortex-array]
    end

    subgraph enc["Encodings (13 crates)"]
        encodings["alp · fastlanes · fsst · runend · sparse<br/>zigzag · pco · zstd · bytebool · btrblocks<br/>datetime-parts · decimal-byte-parts · sequence"]
    end

    subgraph stg["Storage & I/O"]
        io[vortex-io]
        layout[vortex-layout]
        ipc[vortex-ipc]
        file[vortex-file]
        scan[vortex-scan]
    end

    subgraph apps["Applications & Integrations"]
        vortex[vortex]
        datafusion[vortex-datafusion]
        ffi[vortex-ffi]
        tui[vortex-tui]
        cuda[vortex-cuda]
    end

    %% Foundation
    buffer --> error
    fb --> buffer

    %% Type System
    dtype --> buffer
    dtype --> fb
    dtype --> error
    scalar --> dtype
    scalar --> buffer
    mask --> buffer
    vector --> dtype
    vector --> mask

    %% Core
    session --> error
    metrics --> session
    compute --> vector
    compute --> dtype
    compute --> mask
    array --> compute
    array --> scalar
    array --> dtype
    array --> buffer
    array --> fb

    %% Encodings
    encodings --> array

    %% Storage
    io --> array
    io --> buffer
    layout --> array
    layout --> io
    ipc --> array
    file --> layout
    file --> io
    file --> encodings
    scan --> layout
    scan --> io

    %% Applications
    vortex --> file
    vortex --> scan
    vortex --> encodings
    datafusion --> vortex
    ffi --> vortex
    tui --> vortex
    tui --> datafusion
    cuda --> array
```

## Layer Descriptions

### Foundation

The bottom layer provides error handling, memory management, and serialization primitives.

| Crate | Role |
|-------|------|
| `vortex-error` | `VortexError` and `VortexResult` types, `vortex_err!` / `vortex_bail!` macros |
| `vortex-buffer` | Zero-copy aligned `Buffer<T>` and `BufferMut<T>`, guaranteed alignment to `T` |
| `vortex-utils` | Shared utilities (no Vortex-specific dependencies) |
| `vortex-flatbuffers` | FlatBuffer schema definitions for serialization |
| `vortex-proto` | Protocol Buffer definitions |

### Type System

Defines the logical type system and scalar values that the rest of Vortex operates on.

| Crate | Role |
|-------|------|
| `vortex-dtype` | `DType` enum: Null, Bool, Primitive, UTF8, Binary, Decimal, Struct, List, Extension |
| `vortex-scalar` | Single-value representations of each dtype |
| `vortex-mask` | Bitmask operations for validity/selection |
| `vortex-vector` | Vector abstraction over typed buffers |

### Core

The array trait, compute dispatch, and session management.

| Crate | Role |
|-------|------|
| `vortex-session` | Session object holding registries of encodings, layouts, extension types, compute functions |
| `vortex-metrics` | Metrics collection tied to sessions |
| `vortex-compute` | Compute function dispatch and kernel trait definitions |
| `vortex-array` | `Array` trait, canonical encodings, vtable system, statistics |

### Encodings

Each encoding lives in its own crate under `/encodings/`. All encodings depend on `vortex-array`
and implement the array vtable.

| Crate | Technique |
|-------|-----------|
| `vortex-alp` | Adaptive Lossless floating-Point compression |
| `vortex-fastlanes` | FastLanes bit-packing, delta, and frame-of-reference for integers |
| `vortex-fsst` | Fast Static Symbol Table compression for strings |
| `vortex-runend` | Run-end encoding for repetitive data |
| `vortex-sparse` | Sparse array encoding |
| `vortex-zigzag` | ZigZag encoding for signed integers |
| `vortex-pco` | Pco compression |
| `vortex-zstd` | Zstandard general-purpose compression |
| `vortex-bytebool` | Byte-per-boolean encoding |
| `vortex-btrblocks` | BtrBlocks-style compression |
| `vortex-datetime-parts` | DateTime field decomposition |
| `vortex-decimal-byte-parts` | Decimal byte decomposition |
| `vortex-sequence` | Sequence encoding |

### Storage and I/O

File format, IPC, layout system, and the Scan API.

| Crate | Role |
|-------|------|
| `vortex-io` | Async I/O abstraction (local, object store, HTTP) |
| `vortex-layout` | `LayoutReader` / `LayoutWriter` traits, built-in layouts (Flat, Struct, Chunked) |
| `vortex-ipc` | IPC format serialization and deserialization |
| `vortex-file` | `.vortex` file reading and writing, compression pipeline |
| `vortex-scan` | Abstract table scan over layouts with filter/projection pushdown |

### Applications and Integrations

Top-level crates that compose the ecosystem for end users.

| Crate | Role |
|-------|------|
| `vortex` | Umbrella crate re-exporting all encodings and core functionality |
| `vortex-datafusion` | Apache DataFusion `TableProvider` integration |
| `vortex-ffi` | C FFI bindings (generates `vortex.h` via cbindgen) |
| `vortex-tui` | Terminal UI for browsing and inspecting Vortex files |
| `vortex-cuda` | GPU-accelerated decompression and compute (Linux, CUDA) |

### Language Bindings (outside workspace)

| Directory | Role |
|-----------|------|
| `vortex-python/` | Python bindings via PyO3 / Maturin |
| `java/vortex-jni/` | Java JNI bindings |
| `java/vortex-spark/` | Apache Spark DataSource V2 connector |
| `java/vortex-trino/` | Trino connector |
| `vortex-cxx/` | C++ wrapper around the C FFI |
