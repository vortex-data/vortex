# Language Bindings Evolution

This document describes the evolution strategy for Vortex's language bindings. It covers the
current state, architectural principles, open questions, per-language migration plans, and a phased
roadmap. For the tier model and API surface definitions, see the
[Language Bindings](../developer-guide/language-bindings.md) developer guide page.

## Current State

| Language      | Technology | API Surface                                                                              |
|---------------|------------|------------------------------------------------------------------------------------------|
| Rust          | Native     | Full access (Tier 3). All array plugins, compute plugins, layouts, and extension DTypes. |
| Python        | PyO3       | Near Tier 2. Native expression construction, array access, file I/O.                     |
| C             | cbindgen   | Tier ~1. File I/O, basic scan, Arrow stream export.                                      |
| C++           | cxx        | Tier ~1. Wraps Rust via cxx bridge. File I/O and scan.                                   |
| Java (JNI)    | JNI        | Tier ~1. Arrow-based file I/O and basic scan via JNI.                                    |
| Java (Panama) | Panama FFI | Not yet implemented. Target Tier 2 via direct C ABI access. Requires JDK 22+.            |

## Architecture Principle

**The C API is the foundation ABI for non-Rust bindings.** The C API is not yet ABI-stable — it
evolves as fast as the rest of the project and should be **statically linked** to avoid breakage.
In the future, a subset of the API will be flagged as stable for use via dynamically linked
libraries.

Each language chooses the binding strategy that makes the most sense:

- **Python** uses PyO3 to call Rust directly and will continue to do so. PyO3 provides the best
  ergonomics and performance for Python.
- **Other languages** use Rust-specific bindings where it makes sense (e.g. cxx for C++), or wrap
  the C API otherwise (e.g. Java Panama, future C++ migration).
- New capabilities should be exposed through the C API so that any language can consume them, but
  languages are not required to go through C if a more direct path exists.
- The C API's error handling, memory ownership, and lifetime conventions must be documented
  regardless of stability guarantees, since all consumers depend on them.

## Open Questions

### Array Execution Model

The array execution model (lazy vs eager evaluation) is still being designed. This directly affects
what bindings can expose:

- **Eager execution** is simpler for bindings — arrays are materialized values that can be passed
  across the FFI boundary.
- **Lazy execution** requires bindings to handle expression graphs, materialization triggers, and
  potentially device placement (CPU vs GPU).

Until the execution model stabilizes, bindings should focus on the Scan API (Tier 1) where results
are always materialized Arrow streams, and avoid exposing internal array operations that may change.

### Stable Rust Plugin API

Should Rust plugins go through the C ABI for dynamic loading, or should they use a Rust-native
trait-based plugin interface?

- **C ABI approach:** Maximum portability, works across Rust compiler versions. Higher development
  cost per plugin.
- **Rust-native approach:** Ergonomic, zero-cost, but ties plugins to a specific Rust compiler
  version and Vortex ABI.

This is a future concern — the current priority is stabilizing the C API for non-Rust consumers.

### Panama Timeline for Java

Java's Foreign Function & Memory API (Panama) graduated to a stable API in JDK 22. Key
considerations:

- Trino already supports JDK 22 and can adopt Panama immediately.
- Spark targets older LTS releases and will not support Panama for some time.
- JNI must remain supported as the primary path for Spark and any other engine on older JDKs.
- Panama provides direct C ABI access, eliminating JNI overhead and enabling Tier 2 capabilities.

## Per-Language Migration Plan

### C FFI

Expand from Tier ~1 to Tier 2:

- **Array stream export:** Expose Vortex array streams (not just Arrow) through the C API.
- **Array tree inspection:** Provide functions to walk array children, read encoding IDs, and
  access array metadata.
- **Error handling:** Stabilize the error reporting convention (error codes, error message
  retrieval, thread-local error state).
- **Memory ownership:** Document and enforce ownership conventions (who allocates, who frees,
  reference counting semantics).

### C++

Migrate from cxx to wrapping the C API:

- **RAII wrappers:** Provide C++ classes that manage lifetime of C API objects (files, scanners,
  arrays, streams).
- **CMake integration:** Ship a CMake config so downstream projects can `find_package(Vortex)`.
- **Header generation:** Auto-generate C++ headers from the C API headers, adding type safety and
  namespace scoping.
- **Target:** Tier 2 (native array access through the C API).

### Python

Already near Tier 2-3. Python will continue to use PyO3 for direct Rust interop. The migration
plan focuses on formalization:

- **API stability:** Classify existing Python APIs as stable, experimental, or internal. Publish
  this classification in the Python API docs.
- **Plugin API:** Expose plugin registration (custom array plugins, compute plugins) to Python via
  PyO3. The registration model should mirror the C API's capabilities so that the same plugin
  concepts are available in both languages.

### Java (JNI)

Maintain current JNI bindings for Arrow I/O and basic scan. This track targets broad JDK
compatibility and remains the primary integration point for Spark, which targets older LTS
releases. JNI stays at Tier 1.

### Java (Panama)

Build a new binding layer using Panama's Foreign Function & Memory API to call the C API directly.
This enables native array access, lower overhead, and a path to Tier 2 capabilities. Panama
bindings are opt-in for environments running JDK 22+. Trino already supports JDK 22 and is the
likely first adopter. Spark will not support Panama until it moves to a compatible JDK. Connectors
should abstract over the binding layer so they can use JNI or Panama transparently.

### Rust

Remains native Tier 3. Future considerations:

- **Stable plugin ABI:** Investigate a C ABI-based plugin interface for dynamically loading
  encoding crates. This would allow third-party encodings to be distributed as shared libraries
  without requiring recompilation of Vortex.
- **Version compatibility:** A stable plugin ABI would need versioning and capability negotiation
  to handle evolution of the array and compute plugin interfaces.

## Phased Roadmap

### Phase 1: Complete C API at Tier 1

- Complete the C API for file I/O, scan with serialized expressions, and Arrow stream output.
- Document the API: function signatures, error handling conventions, memory ownership rules.
- Add integration tests that exercise the C API from C and C++.
- Ensure Python and Java (JNI) bindings are fully functional at Tier 1.

### Phase 2: Extend to Tier 2

- Add array stream export to the C API (Vortex arrays, not just Arrow).
- Add array tree inspection functions (walk children, read encoding IDs and metadata).
- Migrate C++ bindings from cxx to wrapping the C API with RAII wrappers.
- Prototype Java Panama bindings targeting Tier 2.
- Extend Python bindings to formalize Tier 2 APIs as stable.

### Phase 3: Plugin API Exploration

- Design the plugin registration interface in the C API (Tier 3 capabilities).
- Formalize the Python plugin API (array plugins, compute plugins, extension DTypes).
- Investigate Rust stable plugin ABI for dynamic loading of encoding crates.
- Evaluate Panama adoption timeline based on Spark/Trino JDK requirements.

## Expression Strategy

Expressions are the primary mechanism for pushing computation into Vortex (filters, projections,
computed columns). The bindings must support expression construction across all languages.

**Serialized expressions (protobuf)** are useful for portable expression interchange when the
expression can be fully bound to a known schema. The Rust expression tree stores dtype information
inside the tree, so serialized roots include the bound scope dtype.

**Native expression construction** is a convenience layer built per-language. Python, C, C++, and
Java expose user-facing builders that can be assembled before a scan exists; those bindings keep
that deferral local and materialize a Rust `BoundExpr` when the file, data source, or array dtype is
available.

However, serialized expressions have limitations that mean a mixed approach will likely remain
necessary:

- **Dynamic expressions** (e.g. UDFs, closures, expressions that reference runtime state) cannot
  be represented in serialized form. These require native expression handles passed through the
  binding layer.
- **Placeholders** (e.g. row index or row count) are native expression leaves and are not serialized
  in the current expression protobuf format.
- **Large literal values** embedded in expressions (e.g. large `IN` lists) can be a performance
  constraint when serialized, since the entire value must be copied through protobuf encoding and
  decoding.

The C API should therefore support both modes:

- Accept serialized expression bytes (`vortex_scan_set_filter_bytes`).
- Provide builder functions for common expressions (`vortex_expr_column`, `vortex_expr_eq`,
  `vortex_expr_and`, etc.) that return opaque expression handles.

This dual approach lets simple integrations pass pre-built protobuf while giving interactive users
and query engines with dynamic expressions an ergonomic builder API.
