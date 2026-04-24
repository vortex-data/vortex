# Spark

The `vortex-spark` connector implements Apache Spark's DataSource V2 API, allowing Spark to read
and write Vortex files as a native data source registered under the format name `vortex`.

## Registration

The connector implements Spark's `TableProvider` and `DataSourceRegister` interfaces. When a
query references the `vortex` format, Spark creates a `VortexTable` that supports both batch
reads (`SupportsRead`) and writes (`SupportsWrite`). Schema inference reads the footer of a
discovered file to extract the Arrow schema and map it to Spark's schema representation.

## Multiple Files

Spark's scan builder enumerates Vortex files by scanning the provided path. If the path is a
directory, native code lists all `.vortex` files within it. Each file becomes an independent
input partition, and Spark's task scheduler distributes partitions across executors in the
cluster.

Each partition creates its own file handle and scan state, so there is no shared mutable state
between partitions. This maps naturally to Spark's execution model where each task runs
independently on a separate JVM thread.

## Threading Model

The Spark integration crosses the JNI boundary between Java and Rust. Each Spark partition
reader opens a Vortex file and creates a native scan via JNI. The native side manages its own
async runtime and drives I/O internally, returning results to Java as Arrow-compatible columnar
batches.

Because each partition reader owns its native resources exclusively, there is no contention
across Spark threads. The JNI boundary is crossed once per batch rather than once per row,
keeping overhead low. A prefetching iterator on the Java side buffers upcoming batches to
overlap I/O with Spark's processing.

## Filter and Projection Pushdown

Projection pushdown is supported through Spark's `SupportsPushDownRequiredColumns` interface.
The scan builder prunes the column list to only those referenced by the query, and the pruned
column set is passed to the native scan via `ScanOptions`.

Filter pushdown infrastructure exists in the `ScanOptions` type but is not yet connected to
Spark's `SupportsPushDownFilters` interface. This is planned future work.

## Data Export

Native Vortex arrays are exported to Arrow via the C Data Interface, then wrapped in Spark's
columnar batch format using custom `ArrowColumnVector` wrappers. This avoids a copy between
the Rust and JVM heaps -- the Arrow buffers remain in native memory and are accessed from Java
through direct byte buffers.

## Future Work

The current integration builds directly on the native file and scan APIs via JNI. Future work
will migrate it to use the [Scan API](/concepts/scanning) `Source` trait, which will provide a
standard interface for file discovery, partitioning, and pushdown. This will also enable
connecting Spark's filter pushdown to Vortex's expression-based filtering.
