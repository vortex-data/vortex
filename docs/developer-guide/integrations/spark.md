# Spark

The `vortex-spark` connector is built on Spark's file-source DataSource V2 framework. Shared
sources compile into Spark 3.5/Scala 2.12, Spark 3.5/Scala 2.13, and Spark 4.0/Scala 2.13
artifacts; the Spark 4 artifact is also tested on Spark 4.1.

## File-source integration

`VortexDataSourceV2` extends `FileDataSourceV2`, while `VortexTable`, `VortexScan`, and
`VortexWrite` use Spark's `FileTable`, `FileScan`, and `FileWrite` abstractions. Spark therefore
owns file listing, partition discovery and pruning, input bin-packing, output commit, and
overwrite behavior. `VortexFileFormat` supplies the functional V1 fallback used by catalog
tables and direct path queries.

Vortex files are not internally split. A Spark file partition may contain several files, and the
reader factory opens each `PartitionedFile` in turn. Hive partition values are appended with
constant column vectors.

## I/O and JNI

Spark lists paths through Hadoop. Content reads go through pooled Hadoop input streams exposed to
native Vortex through the JNI `NativeReadable` interface. Writes expose the committer's Hadoop
task path through `NativeWritable`. Vortex's own object-store clients are not used, so the
connector sees the same schemes and credentials as Spark's file index and commit protocol.

Native arrays cross into Spark through the Arrow C Data Interface. Each partition reader owns its
native scan, Arrow allocator, and exported batches and closes them at task completion.

## Pushdown and statistics

Schema inference merges every footer, so the dataset schema is the union of the top-level fields its
files carry. The partition reader projects only the fields the file it opened actually holds and
fills the rest with constant null vectors, and it converts filters against those same fields, so a
filter on a column the file lacks stays a Spark residual rather than reaching the native scan.

Spark's required schema becomes the Vortex scan projection. Convertible V1 filters become Vortex
expressions; filters the converter rejects remain Spark residuals.

The scan accepts `COUNT(*)` aggregation when there are no data filters and grouping uses only
partition columns, and only from footers that state their row count exactly. Readers return one
footer count per file and Spark performs the final merge; the scan then reports one row per file as
its statistics rather than reading those footers again on the driver. Footer row counts from files
left after partition pruning are reported through Spark scan statistics when no aggregate is
pushed. MIN/MAX and `COUNT(column)` need read-side column statistics in the JNI API and are
not pushed down.

## Source layout

Shared Java and Scala live in `java/vortex-spark/common`. Thin projects in `v3.5` and `v4.0`
select the corresponding Spark and Scala dependencies. The Scala shims isolate the binary
differences in Spark's Scala APIs; nothing in Java depends on them, so javac compiles the Java
sources and ErrorProne and Nopen keep covering them.
