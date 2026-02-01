# Vortex

Vortex is an extensible, state-of-the-art ecosystem for columnar data. It includes
specifications and tools for manipulating possibly-compressed arrays in-memory,
on-disk (file format), and over-the-wire (IPC and wire formats). Vortex is built
around the latest research from the database community.

Vortex can be understood as a set of building blocks rather than a singular format.
Almost everything in Vortex is extensible, enabling it to be used for both
general-purpose columnar data processing and niche embedded use-cases where specific
encodings and performance characteristics are required.

```{toctree}
---
maxdepth: 2
caption: Getting Started
---

getting-started/python
getting-started/rust
getting-started/java
getting-started/cli
```

```{toctree}
---
maxdepth: 2
caption: Concepts
---

concepts/overview
concepts/dtypes
concepts/arrays
concepts/encodings
concepts/compute
concepts/layouts
concepts/scan
concepts/extension-types
```

```{toctree}
---
maxdepth: 2
caption: User Guides
---

guides/user/python-integrations
guides/user/datafusion
guides/user/duckdb
guides/user/spark
guides/user/trino
guides/user/ray
```

```{toctree}
---
maxdepth: 2
caption: Extending Vortex
---

guides/extending/writing-an-encoding
guides/extending/writing-a-layout
guides/extending/writing-a-compute-fn
guides/extending/extension-types
```

```{toctree}
---
maxdepth: 2
caption: Embedding Vortex
---

guides/embedding/ffi
guides/embedding/cxx
guides/embedding/scan-api
guides/embedding/gpu
```

```{toctree}
---
maxdepth: 2
caption: Specifications
---

specs/file-format
specs/ipc-format
specs/wire-format
specs/dtype-format
```

```{toctree}
---
maxdepth: 2
caption: API Reference
---

Python API <api/python/index>
Rust API <https://docs.rs/vortex>
C FFI API <api/c/index>
Java API <api/java/index>
```

```{toctree}
---
maxdepth: 2
caption: Internals
---

internals/architecture
internals/session
internals/async-runtime
internals/flatbuffers
internals/benchmarking
internals/lazy-evaluation
internals/datafusion
internals/duckdb
internals/spark
internals/trino
```

```{toctree}
---
maxdepth: 1
caption: Project
---

project/roadmap
project/contributing
project/changelog
references
```

```{toctree}
---
hidden:
caption: Project Links
---

Spiral <https://spiraldb.com>
GitHub <https://github.com/spiraldb/vortex>
PyPI <https://pypi.org/project/vortex-data>
Crates <https://crates.io/crates/vortex>
```
