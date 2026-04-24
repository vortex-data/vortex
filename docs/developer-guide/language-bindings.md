# Language Bindings

Vortex provides bindings for multiple languages at varying levels of API depth. This page
documents the **tier model** that governs what each language binding exposes, the current state of
each binding, and the API surface available at each tier.

## Tier Model

Each language binding targets one of four tiers. Higher tiers are strict supersets of lower tiers.

### Tier 0: Arrow I/O

Read and write Arrow record batches to and from Vortex files. This is the minimum viable
integration point — any language with [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
support can reach this tier.

**Capabilities:** open files, write files, import/export Arrow streams.

### Tier 1: Scan API

Filter and projection pushdown via expressions. Expressions can be serialized as protobuf bytes or
constructed natively in the host language. Results are still returned as Arrow streams, making this
tier suitable for query engine integrations (e.g. DataFusion, DuckDB, Spark, Trino).

**Capabilities:** everything in Tier 0, plus scan builder with filter, projection, limit, and row
range pushdown, and expression construction.

### Tier 2: Native Arrays

Return Vortex array streams instead of (or in addition to) Arrow. At this tier, bindings can
inspect array trees (walk children, read encoding IDs and metadata), execute compute operations
over Vortex arrays, and export results to Arrow. This allows direct access to compressed
representations without requiring an upfront conversion to Arrow.

**Capabilities:** everything in Tier 1, plus Vortex array stream consumption, array tree
inspection, compute execution, and Arrow export.

### Tier 3: Plugins

Define custom encodings, compute functions, layouts, and extension DTypes as plugins and register
them into a Session. This is full extensibility — the host language participates in Vortex's
encoding and compute ecosystem.

**Capabilities:** everything in Tier 2, plus registration of array plugins, compute plugins,
layout plugins, and extension DTypes.

## Per-Tier API Surface

| Capability                    | Tier | Description                                                      |
|-------------------------------|------|------------------------------------------------------------------|
| Open a Vortex file            | 0    | Open a file from a path or byte source                           |
| Write a Vortex file           | 0    | Write Arrow data into a Vortex file                              |
| Export to Arrow stream        | 0    | Read an entire file as Arrow record batches                      |
| Import from Arrow stream      | 0    | Write Arrow record batches into a file                           |
| Scan with expressions         | 1    | Build a scan with filter, projection, limit, and row range       |
| Construct expressions         | 1    | Build filter/projection expressions (serialized or native)       |
| Consume scan results as Arrow | 1    | Execute a scan and receive Arrow record batches                  |
| Consume Vortex array stream   | 2    | Receive Vortex arrays instead of Arrow from a scan               |
| Inspect array trees           | 2    | Walk array children, read encoding IDs and metadata              |
| Execute compute over arrays   | 2    | Run compute functions (e.g. filter, take, cast) on Vortex arrays |
| Export arrays to Arrow        | 2    | Convert Vortex arrays to Arrow on demand                         |
| Access scalars                | 2    | Read individual scalar values from arrays                        |
| Register array plugin         | 3    | Define a custom encoding with its own array vtable               |
| Register compute plugin       | 3    | Define custom compute functions                                  |
| Register layout plugin        | 3    | Define a custom file layout                                      |
| Register extension DType      | 3    | Define a custom logical type                                     |

## Per-Language Status

| Language      | Current Tier | Target Tier | Technology | Notes                                         |
|---------------|--------------|-------------|------------|-----------------------------------------------|
| Rust          | 3            | 3           | Native     | Future: stable plugin API via C ABI           |
| Python        | ~2           | 3           | PyO3       | Already has native expressions + array access |
| C             | ~1           | 2           | cbindgen   | Foundation ABI for all non-Rust bindings      |
| C++           | ~1           | 2           | cxx -> C   | Migrate from cxx to wrap C API                |
| Java (JNI)    | ~1           | 1           | JNI        | Broad JDK compatibility, Arrow-based          |
| Java (Panama) | —            | 2           | Panama FFI | Direct C ABI access, requires JDK 22+         |

### Rust

Rust is the native implementation language and has full Tier 3 access. All array plugins, compute
plugins, layouts, and extension DTypes are defined in Rust. Future work may introduce a stable
plugin ABI for dynamically loading encoding crates.

### Python

Python bindings are implemented via [PyO3](https://pyo3.rs). They already provide native
expression construction and array access, placing them near Tier 2. The path to Tier 3 involves
formalizing which APIs are stable vs experimental and exposing plugin registration.

### C

The C API (generated via cbindgen) is the **foundation ABI** for non-Rust bindings. It currently
provides Tier ~1 capabilities (file I/O and basic scan). The C API is not yet ABI-stable — it
evolves with the project and should be statically linked. In the future, a subset of the API will
be flagged as stable for use via dynamic linking. The target is Tier 2, which requires array tree
inspection, compute execution, and stabilized error handling and memory ownership conventions.

### C++

C++ bindings currently use [cxx](https://cxx.rs) for Rust interop. The plan is to migrate to
wrapping the C API directly, providing RAII wrappers and CMake integration. Target is Tier 2.

### Java (JNI)

Java JNI bindings provide Tier ~1 capabilities today (Arrow I/O and basic scan). JNI will remain
at Tier 1 for broad JDK compatibility. This is the current integration point for Spark and Trino
connectors.

### Java (Panama)

A new binding layer using Java's Foreign Function & Memory API (Panama) to call the C API
directly. Panama enables native array access without JNI overhead, targeting Tier 2. Requires
JDK 22+. Trino already supports JDK 22 and can adopt Panama immediately. Spark targets older LTS
releases and will not support Panama for some time, so the JNI path remains essential for Spark
integration.
