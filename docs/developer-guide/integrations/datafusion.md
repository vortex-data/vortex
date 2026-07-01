# DataFusion

The `vortex-datafusion` crate integrates Vortex as a native file format in Apache DataFusion.
It registers a `FileFormat` and `FileSource` so that DataFusion's query planner can discover,
partition, and scan Vortex files using the same machinery it uses for Parquet and CSV.

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

Opened file metadata and scan preparation state are shared where possible across partitions keyed
by file path. This avoids redundant footer parsing and repeated layout expansion when the same file
is accessed by multiple partitions or repeated queries.

## Threading Model

DataFusion runs on Tokio, and the Vortex integration operates entirely within that async
context. The Vortex session is configured with `with_tokio()` to capture the current Tokio
runtime handle. All I/O -- file opens, segment reads, object store fetches -- is dispatched as
Tokio tasks and scheduled across Tokio's multi-threaded executor.

DataFusion's physical executor manages parallelism by assigning partitions to its own task pool.
Each partition opens its files and drives a Vortex file scan backed by layout expansion and
`ScanPlan` prepared reads. The scan returns an async stream of record batches. Multiple partitions
execute concurrently, with DataFusion controlling the degree of parallelism.

## Filter and Projection Pushdown

The integration converts DataFusion physical expressions into Vortex expressions using an
`ExpressionConvertor` trait. Supported predicates (comparisons, LIKE, IS NULL, IN lists, casts)
are pushed into the Vortex scan where they participate in layout-level evidence, pruning, and
residual filter evaluation. Unsupported predicates remain in the DataFusion plan and are evaluated
after the scan.

Filter pushdown operates at two levels. The full predicate is used to prune entire files before
they are opened, using file-level statistics. The subset of predicates that Vortex can evaluate
efficiently is pushed into the per-file scan for row-level filtering.

Projection pushdown maps DataFusion's requested column indices to Vortex field names and passes
them as projection expressions to the scan. Struct layouts route those expressions to the requested
field children, so only the requested columns are read from storage.

The integration supports pluggable expression conversion via a custom `ExpressionConvertor`,
allowing engine-specific rewrites or schema adaptation when file schemas diverge from the table
schema.

## Data Export

Vortex arrays produced by the scan are converted to Arrow `RecordBatch`es for consumption by
DataFusion. Batches are sliced to respect DataFusion's configured batch size preference.

## Dynamic Filters

Dynamic expressions support use-cases like top-k queries, where the query engine discovers tighter
bounds during execution. When a dynamic predicate version changes, cheap prepared evidence handles can recheck
in-flight morsels before projection so the scan avoids reading output rows that are no longer
needed.
