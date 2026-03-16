# Crate Architecture

The Vortex workspace is organized as a Rust monorepo with four main groups: core crates,
encodings, language bindings, and query engine integrations.

## The `vortex` Crate

The `vortex` crate is the main entry point for all external consumers. It re-exports core
functionality and bundles the standard set of encodings. **All integrations and third-party
encodings should depend only on this crate** -- not on internal crates like `vortex-array` or
`vortex-file` directly.

This single-dependency design ensures:

- Stable API surface for external consumers.
- Freedom to refactor internal crate boundaries without breaking downstream code.
- Consistent versioning across the ecosystem.

Third-party encodings implement their vtables against types re-exported from `vortex`, and
query engine integrations build on the file reading and scan APIs exposed through it.

## Vortex Core

The core crates provide the foundation for the Vortex type system, array representation, file
format, and I/O.

| Crate                     | Role                                                                          |
| ------------------------- | ----------------------------------------------------------------------------- |
| `vortex-error`            | `VortexError` and `VortexResult` types, `vortex_err!` / `vortex_bail!` macros |
| `vortex-buffer`           | Zero-copy aligned `Buffer<T>` with guaranteed alignment                       |
| `vortex-array/src/dtype`  | `DType` enum: Null, Bool, Primitive, UTF8, Binary, Struct, List, Extension    |
| `vortex-array/src/scalar` | Single-value representations of each dtype                                    |
| `vortex-mask`             | Bitmask operations for validity and selection                                 |
| `vortex-session`          | Session object holding registries for encodings, layouts, and extension types |
| `vortex-array`            | `Array` trait, canonical encodings, vtable system, statistics                 |
| `vortex-io`               | Async I/O abstraction (local filesystem, object store, HTTP)                  |
| `vortex-layout`           | Layout traits and built-in layouts (Flat, Struct, Chunked)                    |
| `vortex-ipc`              | IPC format for inter-process communication                                    |
| `vortex-file`             | `.vortex` file reading and writing                                            |
| `vortex-scan`             | Table scan with filter and projection pushdown                                |
| `vortex-expr`             | Expression representation and optimization                                    |
| `vortex-flatbuffers`      | FlatBuffer schema definitions                                                 |

## Encodings

Encodings live in separate crates under `/encodings/`. Each encoding implements the array vtable
and registers itself with the session. The standard encodings are bundled into the `vortex` crate.

| Crate                       | Technique                                            |
| --------------------------- | ---------------------------------------------------- |
| `vortex-alp`                | Adaptive Lossless floating-Point compression         |
| `vortex-fastlanes`          | FastLanes bit-packing, delta, and frame-of-reference |
| `vortex-fsst`               | Fast Static Symbol Table compression for strings     |
| `vortex-runend`             | Run-end encoding for repetitive data                 |
| `vortex-sparse`             | Sparse array encoding                                |
| `vortex-zigzag`             | ZigZag encoding for signed integers                  |
| `vortex-roaring`            | Roaring bitmap encoding                              |
| `vortex-dict`               | Dictionary encoding                                  |
| `vortex-bytebool`           | Byte-per-boolean encoding                            |
| `vortex-datetime-parts`     | DateTime field decomposition                         |
| `vortex-decimal-byte-parts` | Decimal byte decomposition                           |
| `vortex-sequence`           | Arithmetic sequence encoding                         |

## Language Bindings

Language bindings expose Vortex to non-Rust environments.

| Directory          | Role                                  |
| ------------------ | ------------------------------------- |
| `vortex-python/`   | Python bindings via PyO3 and Maturin  |
| `java/vortex-jni/` | Java JNI bindings                     |
| `vortex-ffi/`      | C FFI bindings (generates `vortex.h`) |
| `vortex-cxx/`      | C++ wrapper around the C FFI          |

## Integrations

Query engine integrations allow Vortex files to be queried through existing analytics engines.

| Crate / Directory    | Engine     | Notes                                        |
| -------------------- | ---------- | -------------------------------------------- |
| `vortex-datafusion/` | DataFusion | `TableProvider` and `FileFormat` integration |
| `vortex-duckdb/`     | DuckDB     | Table function integration                   |
| `java/vortex-spark/` | Spark      | DataSource V2 connector via JNI              |
| `java/vortex-trino/` | Trino      | Trino connector (in development)             |

## Other Crates

| Crate          | Role                                                   |
| -------------- | ------------------------------------------------------ |
| `vortex-cuda`  | GPU-accelerated decompression and compute (Linux only) |
| `vortex-tui`   | Terminal UI for inspecting Vortex files                |
| `vortex-bench` | Benchmark harness and data generators                  |
