# DataFusion

> If you're a developer willing to integrate Vortex and Datafusion, see
> [`vortex-datafusion` developer
> guide](docs/developer-guide/integrations/datafusion.md) instead.

Vortex provides a
[DataFusion CLI](https://github.com/vortex-data/vortex-datafusion-cli) which
allows you to query `.vortex` files via DataFusion.

## Reading Vortex Files

### SQL

Create an external table and query it:

:::{literalinclude} ../../vortex-datafusion/src/persistent/tests.rs
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

:::{literalinclude} ../../vortex-datafusion/src/persistent/tests.rs
:language: rust
:dedent:
:start-after: [write]
:end-before: [write]
:::

Partitioned writes are supported — DataFusion automatically creates subdirectories for each
partition value.

## Querying

Filters and projections are pushed down into the Vortex scan:

:::{literalinclude} ../../vortex-datafusion/src/persistent/tests.rs
:language: rust
:dedent:
:start-after: [query]
:end-before: [query]
:::
