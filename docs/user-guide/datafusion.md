# DataFusion

The `vortex-datafusion` crate integrates Vortex as a native DataFusion `FileFormat`, supporting
both reads and writes with filter, projection, and limit pushdown.

## Setup

Add the dependency:

```toml
[dependencies]
vortex-datafusion = "<version>"
```

Register the Vortex format with a `SessionContext`:

:::{literalinclude} ../../vortex-datafusion/src/persistent/mod.rs
:language: rust
:dedent:
:start-after: [setup]
:end-before: [setup]
:::

## Reading Vortex Files

### SQL

Create an external table and query it:

:::{literalinclude} ../../vortex-datafusion/src/persistent/mod.rs
:language: rust
:dedent:
:start-after: [create]
:end-before: [create]
:::

### Rust API

You can also register a `ListingTable` directly:

:::{literalinclude} ../../vortex-datafusion/examples/vortex_table.rs
:language: rust
:dedent:
:start-after: [register]
:end-before: [register]
:::

## Writing Vortex Files

Write query results to Vortex using `INSERT INTO`:

:::{literalinclude} ../../vortex-datafusion/src/persistent/mod.rs
:language: rust
:dedent:
:start-after: [write]
:end-before: [write]
:::

Partitioned writes are supported — DataFusion automatically creates subdirectories for each
partition value.

## Querying

Filters and projections are pushed down into the Vortex scan:

:::{literalinclude} ../../vortex-datafusion/src/persistent/mod.rs
:language: rust
:dedent:
:start-after: [query]
:end-before: [query]
:::

### Pushdown Support

The integration pushes the following operations into the Vortex scan:

- **Projections** — only referenced columns are read and decompressed.
- **Filters** — comparison (`=`, `<`, `>`), logical (`AND`, `OR`, `NOT`), `IN`, `LIKE`,
  `IS NULL`, and cast expressions are evaluated during the scan. Unsupported filters fall back
  to DataFusion post-scan evaluation.
- **Limits** — applied at the scan level when no filter is present.
- **File pruning** — files are eliminated without being opened based on partition values and
  file-level column statistics (min/max).
