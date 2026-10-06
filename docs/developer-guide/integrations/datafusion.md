# DataFusion

Vortex provides a
[`vortex-datafusion`](https://crates.io/crates/vortex-datafusion) crate for
`.vortex` file support in DataFusion.

## Setup

Add the dependency:

```toml
[dependencies]
vortex-datafusion = "<version>"
```

Register the Vortex format with a `SessionContext`:

:::{literalinclude} ../../vortex-datafusion/src/persistent/tests.rs
:language: rust
:dedent:
:start-after: [setup]
:end-before: [setup]
:::

## Registration

Vortex registers itself through DataFusion's `FileFormatFactory` interface. Once registered,
DataFusion can create `ListingTable` providers that automatically discover `.vortex` files in a
directory or object store prefix. The format factory creates a `VortexFormat` instance that
carries the Vortex session and its associated options (segment cache, read sizes, etc.).

Schema inference reads the footer of one of the discovered files to extract the Arrow schema.
DataFusion then uses this schema when planning queries against the table.

## Multiple Files

DataFusion handles multi-file scans through its own file listing and partitioning layer. Each
discovered file becomes a `PartitionedFile` that DataFusion assigns to execution partitions.
Vortex implements the `FileOpener` trait to open individual files on demand as DataFusion's
executor schedules them.

Layout readers are cached across partitions using a shared concurrent map keyed by file path.
This avoids redundant footer parsing when the same file is accessed by multiple partitions or
repeated queries.

## Filter and Projection Pushdown

The integration converts DataFusion physical expressions into Vortex expressions using an
`ExpressionConverter` trait. Supported predicates (comparisons, LIKE, NOT, IS NULL, IN lists, casts)
are pushed into the Vortex scan where they participate in pruning and filter evaluation at the
layout level. Unsupported predicates remain in the DataFusion plan and are evaluated after the
scan.

Filter pushdown operates at two levels. The full predicate is used to prune entire files before
they are opened, using file-level statistics. The subset of predicates that Vortex can evaluate
efficiently is pushed into the per-file scan for row-level filtering.

Fully supported projections are evaluated in the scan. If any expression is unsupported, the
complete projection runs in DataFusion over the required raw columns.

Searched `CASE WHEN` expressions can be pushed down when all their children can be converted,
including branches that may fail. Conditions and values execute only on the rows that reach
them, so guards such as `CASE WHEN divisor <> 0 THEN value / divisor ELSE 0 END` remain safe.
Simple `CASE value WHEN ...` expressions currently remain in DataFusion.

The integration supports pluggable expression conversion via a custom `ExpressionConverter`,
allowing engine-specific rewrites or schema adaptation when file schemas diverge from the table
schema.

## Data Export

Vortex arrays produced by the scan are converted to Arrow `RecordBatch`es for consumption by
DataFusion. Batches are sliced to respect DataFusion's configured batch size preference.
