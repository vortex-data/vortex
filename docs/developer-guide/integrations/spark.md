# Spark

The `vortex-spark` connector is built on Spark's file-source DataSource V2 framework. Shared
sources compile into Spark 3.5/Scala 2.12, Spark 3.5/Scala 2.13, Spark 4.0/Scala 2.13, Spark
4.1/Scala 2.13, and Spark 4.2/Scala 2.13 artifacts. Each artifact is also tested on the later Spark
lines it supports.

## File-source integration

`VortexDataSourceV2` extends `FileDataSourceV2`, while `VortexTable`, `VortexScan`, and
`VortexWrite` use Spark's `FileTable`, `FileScan`, and `FileWrite` abstractions. Spark therefore
owns file listing, partition discovery and pruning, input bin-packing, output commit, and
overwrite behavior. `VortexFileFormat` supplies the functional V1 fallback used by catalog
tables and direct path queries.

Spark's file listing keeps every file that is not hidden. The scan, the statistics, and both
reader factories skip files without the `.vortex` extension, so other files in a dataset directory
are not read.

Spark splits large files into byte ranges. The partition reader maps the byte range
`[start, end)` of a file with `size` bytes and `rows` rows to the row range
`[rows * start / size, rows * end / size)` and passes it to the native scan, which reads only the
chunks the range overlaps. Integer division makes the ranges of a file's splits tile its rows. A
Spark file partition may contain several files or splits, and the reader factory opens each
`PartitionedFile` in turn. Hive partition values are appended with constant column vectors.

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

Spark's required schema becomes the Vortex scan projection. On the V2 path, struct columns are
pruned to the nested fields the query uses, at any depth: the projection is a `select` when only
leaf fields are pruned, and a `pack` masked by the source struct's validity when a child struct is
pruned too. Fields inside arrays and maps are read whole. Spark's V1 schema pruning rule applies
only to Parquet and ORC, so the V1 fallback reads whole top-level columns.

The V1 generated column `_tmp_metadata_row_index` becomes the native `row_idx` expression, which
numbers rows in file coordinates, so a split and a filter keep each row's index.

Convertible V1 filters become Vortex expressions; filters the converter rejects remain Spark
residuals. `IN` becomes a native `list_contains` over a literal list with SQL null semantics, which
the scan prepares once as a hash set and falsifies against zone statistics through the list's
bounds. A pushed `LIMIT` is applied per file, and Spark applies it again.

`VortexScan` implements `SupportsRuntimeV2Filtering`. Spark calls `filter` after it built the
reader factory, so runtime filters on data columns travel with the planned `VortexFilePartition`s,
and each task's copy of the factory notes its partition's filters before Spark asks it for file
readers. Runtime filters on partition columns drop files from the planned partitions on the driver.

`SupportsPushDownTopN` pushes the first sort key of an `ORDER BY ... LIMIT n`. Each task keeps a
heap of the best `n` keys its readers returned and publishes the worst of them through a JNI
`DynamicValue`, which a native dynamic comparison reads on every evaluation, zone pruning included.
The comparison is strict for a single sort key and inclusive otherwise, since a tie can still win
on a later key, and nulls that sort first are always kept. The driver also seeds the threshold from the footers: a file with at
least `n` non-null keys bounds the result's last key by its minimum, or maximum when ascending, and
the best such bound reaches every task as a second, inclusive dynamic comparison.

`SupportsPushDownOffset` takes an `OFFSET` that Spark then drops, so the scan skips exactly that
many rows. The driver reads every planned file's exact footer row count, drops the files the offset
covers whole, and hands the file it ends in a first row to start from. The scan refuses an offset
alongside a sample, an aggregate, or a top-N, and stops offering runtime filters, since each changes
the rows the counts describe.

`SupportsPushDownTableSample` turns a sample without replacement into a Roaring row selection per
file, drawn from a hash of the seed, the path, and the row index. A pushed sample rules out
aggregate pushdown, since footer statistics describe every row.

`SupportsReportPartitioning` reports a `KeyGroupedPartitioning` over identity transforms of the read
partition columns when v2 bucketing is enabled. The scan then splits each planned `FilePartition`
by partition value into `VortexKeyedFilePartition`s implementing `HasPartitionKey`, and Spark groups
them by key.

JNI writers record the `IsSorted` aggregate besides the default file statistics when the enabled
editions include `vortex.is_sorted`, which only the `unstable` edition does. JNI sessions enable
the core edition, so for now the statistic is not recorded and the scan reports no order. Its
partial state keeps each chunk's first and last value, so merging chunks checks the order across
their boundaries. `NativeFiles.aggregate` reports a column sorted only when the footer also counts no
nulls or NaN in it.
`SupportsReportOrdering` reads it through `NativeFiles.aggregate` and reports the first read column
every footer states sorted. Each file or split is then its own input partition and the native scan
keeps file order. Spark itself drops the order when it groups several partitions by key.

Data filters without a V1 form reach the reader as Catalyst expressions. `CatalystPredicates`
converts `LIKE` and column-to-column comparisons, and delegates parts with a V1 form to the filter
converter, so mixed conjunctions and disjunctions convert in full.

The scan accepts `COUNT(*)`, `COUNT`, `MIN`, `MAX`, and integer `SUM` when there are no data
filters and grouping uses only partition columns. The JNI entry point `NativeFiles.aggregate`
answers ungrouped aggregates over one file: from the footer's file statistics when they state the
answer exactly, and otherwise, unless the caller forbids it, from a scan of only the columns
involved, through Vortex's aggregate kernels. It returns a one-row Arrow batch and marks the
aggregates it left unanswered. Floating-point extrema count `NaN` as the largest value, and integer
sums wrap to 64 bits, so the per-file partials combine as Spark's own aggregates would. Readers
return one partial row per file and Spark performs the final merge; the scan then reports one row
per file as its statistics rather than reading those footers again on the driver.

`VortexScanBuilder.supportCompletePushDown` reads every footer on the driver, through
`VortexAggregateReaderFactory.footerRows`, which never computes from data. When every footer
answers, `CompleteAggregation` merges the partial rows of each partition group, and `build` returns
a `VortexLocalScan`, which Spark plans as a `LocalTableScanExec`. Spark asks before and after it
pushes the aggregation, so the builder keeps the answer for the last aggregation it was asked
about.

`VortexSparkSessionExtensions` adds `NestedDynamicPruning` to the session's
`experimental.extraOptimizations`, once. Spark runs these rules after `PartitionPruning`, and no
extension injection point does: rules from `injectOptimizerRule` run before it. The extension
therefore installs the rule from the builder of a post-hoc resolution rule, which Spark calls when
it builds the session's analyzer, before the session optimizes a query. `filterAttributes` offers
data columns only when `NestedDynamicPruning.installed` finds the rule in the session. With data
columns, `PartitionPruning` adds a `DynamicPruningSubquery` for each join key, and its build plan
is the other side of the join, with the pruning subqueries already added there. A self-joined
common table expression then nests them exponentially: TPC-DS q64 planned 122,883 subqueries
before the rule existed. The rule drops each pruning filter whose build plan holds another one.

Reader factories carry a `VortexIo` from `VortexIo.broadcast`, which broadcasts the Hadoop
configuration once, so a task does not deserialize the whole configuration from its own bytes. The
V2 scan reports `VortexScanMetrics`: each task counts into one `VortexScanMetrics.Task` that the
reader factory wraps around Spark's `FilePartitionReader`, because that reader does not pass on the
metrics of the readers of its files.

Footer row counts from files left after partition pruning are reported through Spark scan
statistics when no aggregate is pushed. With the cost-based optimizer enabled, the driver calls
`NativeFiles.aggregate` without computing anything from data, and merges each column's exact null
counts and extrema across footers.

## Variants

Spark 4's `VariantType` maps to Arrow's `arrow.parquet.variant` extension, a struct of binary
`metadata` and `value`, through the version shim `SparkVariants`: the Spark 4 builds carry the
mapping and the Spark 3.5 build a class of the same name that reports no variant type. A null
variant keeps empty, non-null parts, as Vortex requires its parts to be valid. Spark reads a
variant's value from child 0 and its metadata from child 1, the reverse of Arrow's order, so the
column vector swaps them. Zone maps and file statistics skip variant columns, including variants
nested in structs and lists.

On Spark 4.1, `VortexScanBuilder41` accepts pushed variant extractions. Spark replaces the variant
column in the read schema with a struct of one field per extraction, named by ordinal. The reader
projects a native `variant_get` and an `IS NOT NULL` of the column for each such field, and
`SparkVariants.extract` casts the extracted variants with Spark's own `variant_get` cast, so cast
errors and `try_variant_get` behave as they do in Spark.

On Spark 4.2, `VortexScanBuilder42` maps `TABLESAMPLE SYSTEM` to a sample of whole splits, which
`NativeFiles.splits` returns from the footer. `VortexScan42` supports iterative runtime filtering:
it reports the predicates it acted on through `pushedPredicates`, and Spark offers it the rest of
the partition-column runtime filters as `PartitionPredicate`s, which it evaluates on each file's
full partition values.

## Source layout

Shared Java and Scala live in `java/vortex-spark/common`. Sources for Spark 4.0 and later live
in `spark4`, and sources for Spark 4.1 and later in `spark41`. Thin projects in `v3.5`, `v4.0`,
`v4.1`, and `v4.2` select the corresponding Spark and Scala dependencies and the shared source
directories, and each holds a `VortexScanBuilders` that creates the scan builder of its line. The Scala shims isolate the binary
differences in Spark's Scala APIs; nothing in Java depends on them, so javac compiles the Java
sources and ErrorProne and Nopen keep covering them.
