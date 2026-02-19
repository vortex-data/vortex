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

Layout readers are cached across partitions using a shared concurrent map keyed by file path.
This avoids redundant footer parsing when the same file is accessed by multiple partitions or
repeated queries.

## Threading Model

DataFusion runs on Tokio, and the Vortex integration operates entirely within that async
context. The Vortex session is configured with `with_tokio()` to capture the current Tokio
runtime handle. All I/O -- file opens, segment reads, object store fetches -- is dispatched as
Tokio tasks and scheduled across Tokio's multi-threaded executor.

DataFusion's physical executor manages parallelism by assigning partitions to its own task pool.
Each partition opens its files and drives a `ScanBuilder` that returns an async stream of
record batches. Multiple partitions execute concurrently, with DataFusion controlling the degree
of parallelism.

## Filter and Projection Pushdown

The integration converts DataFusion physical expressions into Vortex expressions using an
`ExpressionConvertor` trait. Supported predicates (comparisons, LIKE, IS NULL, IN lists, casts)
are pushed into the Vortex scan where they participate in pruning and filter evaluation at the
layout level. Unsupported predicates remain in the DataFusion plan and are evaluated after the
scan.

Filter pushdown operates at two levels. The full predicate is used to prune entire files before
they are opened, using file-level statistics. The subset of predicates that Vortex can evaluate
efficiently is pushed into the per-file scan for row-level filtering.

Projection pushdown maps DataFusion's requested column indices to Vortex field names and passes
them as a projection expression to the scan. Only the requested columns are read from storage.

The integration supports pluggable expression conversion via a custom `ExpressionConvertor`,
allowing engine-specific rewrites or schema adaptation when file schemas diverge from the table
schema.

## Data Export

Vortex arrays produced by the scan are converted to Arrow `RecordBatch`es for consumption by
DataFusion. Batches are sliced to respect DataFusion's configured batch size preference.

## Future Work

The current integration builds directly on the `ScanBuilder` and layout reader APIs. Future work
will migrate it to use the [Scan API](/concepts/scanning) `Source` trait, which will simplify
the integration by providing a standard interface for file discovery, partitioning, and pushdown
that is shared across all engine integrations.

Other planned improvements include projection expression pushdown, which would allow DataFusion
to push complex projection expressions (such as extracting nested struct fields) into the Vortex
scan rather than materializing entire columns and projecting afterwards. Additionally, better
support for dynamic expressions would enable use-cases like top-k queries, where the scan's
filter expression is updated during execution as the query engine discovers tighter bounds.
