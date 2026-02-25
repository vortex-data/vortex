# Language Bindings Evolution

This document describes the evolution strategy for Vortex's language bindings. It covers the
current state, architectural principles, open questions, per-language migration plans, and a phased
roadmap. For the tier model and API surface definitions, see the
[Language Bindings](../developer-guide/language-bindings.md) developer guide page.

## Current State

| Language | Technology | API Surface |
|---|---|---|
| Rust | Native | Full access (Tier 3). All array plugins, compute plugins, layouts, and extension DTypes. |
| Python | PyO3 | Near Tier 2. Native expression construction, array access, file I/O. |
| C | cbindgen | Tier ~1. File I/O, basic scan, Arrow stream export. |
| C++ | cxx | Tier ~1. Wraps Rust via cxx bridge. File I/O and scan. |
| Java (JNI) | JNI | Tier ~1. Arrow-based file I/O and basic scan via JNI. |
| Java (Panama) | Panama FFI | Not yet implemented. Target Tier 2 via direct C ABI access. Requires JDK 22+. |

## Architecture Principle

**The C ABI is the stable foundation.** All non-Rust language bindings ultimately wrap the C API.
The C API is the stability boundary — it defines the contract between Vortex's Rust core and the
outside world.

This principle has several implications:

- New capabilities must be exposed through the C API before they can be consumed by C++, Java
  (Panama), or any future language binding.
- The C API's error handling, memory ownership, and lifetime conventions must be documented and
  stable.
- Python is a partial exception: PyO3 bindings call Rust directly and may continue to do so for
  performance-sensitive paths. However, the plugin API surface should align with what the C API
  exposes, so that plugins written for one language are conceptually portable.

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

- Spark and Trino have different JDK version requirements and upgrade timelines.
- JNI must remain supported as a fallback for older JDK versions.
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

Already near Tier 2-3. The migration plan focuses on formalization:

- **API stability:** Classify existing Python APIs as stable, experimental, or internal. Publish
  this classification in the Python API docs.
- **Plugin API:** Expose plugin registration (custom array plugins, compute plugins) to Python.
  This may remain PyO3-native for ergonomics, but the registration model should mirror the C API's
  capabilities.
- **Long-term question:** Should the Python plugin API go through the C ABI (for portability) or
  remain PyO3-native (for performance and ergonomics)? A hybrid approach may be best — PyO3 for
  the hot path, C ABI for the plugin registration contract.

### Java (JNI)

Maintain current JNI bindings for Arrow I/O and basic scan. This track targets broad JDK
compatibility and is the integration point for Spark/Trino connectors today. JNI stays at Tier 1.

### Java (Panama)

Build a new binding layer using Panama's Foreign Function & Memory API to call the C API directly.
This enables native array access, lower overhead, and a path to Tier 2 capabilities. Panama
bindings are opt-in for environments running JDK 22+. Spark and Trino connectors should abstract
over the binding layer so they can use JNI or Panama transparently.

### Rust

Remains native Tier 3. Future considerations:

- **Stable plugin ABI:** Investigate a C ABI-based plugin interface for dynamically loading
  encoding crates. This would allow third-party encodings to be distributed as shared libraries
  without requiring recompilation of Vortex.
- **Version compatibility:** A stable plugin ABI would need versioning and capability negotiation
  to handle evolution of the array and compute plugin interfaces.

## Phased Roadmap

### Phase 1: Stabilize C API at Tier 1

- Stabilize the C API for file I/O, scan with serialized expressions, and Arrow stream output.
- Document the ABI: function signatures, error handling conventions, memory ownership rules.
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

**Serialized expressions (protobuf)** are the baseline for all languages. Any language that can
produce protobuf bytes can construct expressions — this is the lowest-common-denominator approach
and the one the C API should always accept.

**Native expression construction** is a convenience layer built per-language. Python already has
this via PyO3. C++ and Java should have builder APIs that produce the same protobuf under the hood.

The C API should support both modes:

- Accept serialized expression bytes (`vortex_scan_set_filter_bytes`).
- Provide builder functions for common expressions (`vortex_expr_column`, `vortex_expr_eq`,
  `vortex_expr_and`, etc.) that return opaque expression handles.

This dual approach lets simple integrations pass pre-built protobuf while giving interactive users
(e.g. Python REPL, C++ application code) an ergonomic builder API.
