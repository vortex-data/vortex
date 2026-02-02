---
sd_hide_title: true
---

# Vortex

:::{image} _static/vortex_wordmark.svg
:class: only-light vortex-wordmark
:alt: Vortex
:align: center
:::

:::{image} _static/vortex_wordmark_dark_theme.svg
:class: only-dark vortex-wordmark
:alt: Vortex
:align: center
:::

An extensible ecosystem for compressed columnar data. Spans in-memory arrays,
on-disk file formats, over-the-wire protocols, and integrations with query engines — all built
around the latest research from the database community.

## Where to start

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Read & write Vortex files
:link: getting-started/index
:link-type: doc

Get started with Vortex in **Python**, **Rust**, or **Java**. Convert from Parquet, compress
your data, and query it.
:::

:::{grid-item-card} Use with a query engine
:link: user-guide/index
:link-type: doc

Integrate Vortex with **DataFusion**, **DuckDB**, **Spark**, **Trino**, or **Ray** for
accelerated queries over compressed data.
:::

:::{grid-item-card} Understand the architecture
:link: concepts/index
:link-type: doc

Learn how **DTypes**, **Arrays**, **Encodings**, **Layouts**, and the **Scan API** fit together
as building blocks.
:::

:::{grid-item-card} Extend Vortex
:link: developer-guide/index
:link-type: doc

Write your own **encodings**, **layouts**, **compute functions**, or **extension types** from
Rust or Python.
:::

:::{grid-item-card} Create an engine integration
:link: developer-guide/index
:link-type: doc

Build a **query engine connector** or **data source** using the **Scan API**, **C FFI**, or
**C++ wrapper**.
:::

:::{grid-item-card} Internals
:link: developer-guide/index
:link-type: doc

Explore the **crate architecture**, **async runtime**, **session system**, and integration
internals. Build and benchmark locally.
:::

::::

## Highlights

- **Compressed arrays**: Operate directly on compressed data with encodings like
  [FastLanes](https://github.com/spiraldb/fastlanes),
  [FSST](https://github.com/spiraldb/fsst), and
  [ALP](https://github.com/spiraldb/alp) — no decompression needed for many operations.

- **Extensible file format**: Zero-allocation reads, FlatBuffer metadata for O(1) column access,
  and optional WASM decompression kernels for forward compatibility.

- **Query engine integration**: Filter and projection pushdown through the Scan API, with native
  integrations for DataFusion, DuckDB, Spark, Trino, and Ray.

- **Language bindings**: First-class support for Python (PyO3), Java (JNI + Spark/Trino connectors),
  and C/C++ (FFI).

```{toctree}
---
hidden:
---

getting-started/index
concepts/index
user-guide/index
developer-guide/index
specs/index
api/index
project/index
```
