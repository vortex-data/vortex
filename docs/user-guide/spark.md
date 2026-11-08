# Spark

Vortex provides a Spark file data source for reading and writing Vortex files. Choose the
artifact matching both the Spark and Scala versions in the application:

| Artifact                           | Spark           | Scala |
|------------------------------------|-----------------|-------|
| `dev.vortex:vortex-spark-3.5_2.12` | 3.5.x           | 2.12  |
| `dev.vortex:vortex-spark-3.5_2.13` | 3.5.x           | 2.13  |
| `dev.vortex:vortex-spark-4.0_2.13` | 4.0.x and later | 2.13  |
| `dev.vortex:vortex-spark-4.1_2.13` | 4.1.x and later | 2.13  |
| `dev.vortex:vortex-spark-4.2_2.13` | 4.2.x           | 2.13  |

An artifact also runs on the later Spark lines listed, but only the artifact of the Spark line in
use pushes down what that line adds: variant extraction on 4.1, and `TABLESAMPLE SYSTEM` and
partition predicates on 4.2.

Use the `all` classifier JAR. It bundles the Vortex JNI bindings, native libraries for Linux
(x86_64 and aarch64), macOS (aarch64), and Windows (x86_64), and relocated Arrow, Guava, and
Jackson dependencies that avoid conflicts with Spark. The unclassified thin JAR is not usable by
itself because it references relocated classes that only ship in the `all` JAR.

## Installation

Pass the classified JAR to `spark-shell`, `spark-submit`, or `pyspark` with `--jars`. For
example, for connector version `VERSION`:

```shell
spark-shell --jars https://repo1.maven.org/maven2/dev/vortex/vortex-spark-4.0_2.13/VERSION/vortex-spark-4.0_2.13-VERSION-all.jar
```

Or configure the JAR on a PySpark session:

```python
spark = (
    SparkSession.builder
    .config("spark.jars", "/path/to/vortex-spark-4.0_2.13-VERSION-all.jar")
    .getOrCreate()
)
```

```{note}
Spark's `--packages` option cannot select the `all` classifier. It resolves the thin JAR,
which fails at runtime because the relocated dependencies are only in the classified JAR.
```

For a JVM build, specify the classifier explicitly.

Gradle (Kotlin):

```kotlin
implementation("dev.vortex:vortex-spark-4.0_2.13:VERSION:all")
```

Maven:

```xml
<dependency>
    <groupId>dev.vortex</groupId>
    <artifactId>vortex-spark-4.0_2.13</artifactId>
    <version>VERSION</version>
    <classifier>all</classifier>
</dependency>
```

The connector registers itself as `vortex`; no catalog configuration is required. To let joins
prune Vortex scans on data columns, also apply the Vortex session extensions (see
[Runtime Filtering](#runtime-filtering)):

```shell
spark-shell --jars /path/to/vortex-spark-4.0_2.13-VERSION-all.jar \
  --conf spark.sql.extensions=dev.vortex.spark.VortexSparkSessionExtensions
```

If `spark.sql.extensions` already names other extensions, add the Vortex class to the
comma-separated list.

## Reading Vortex Files

Read a file, directory, or set of paths with the DataFrame API:

```java
Dataset<Row> df = spark.read()
    .format("vortex")
    .load("/path/to/data");
```

Spark's file-source framework provides recursive file listing, split bin-packing, Hive-style
partition discovery and pruning, and the standard `pathGlobFilter` and
`recursiveFileLookup` options. As with other Spark file formats, hidden files whose names begin
with `_` or `.` are ignored.

Spark splits large files into byte ranges, as it does for Parquet, and each task reads the matching
share of the file's rows: a split of bytes `[start, end)` reads rows
`[rows * start / size, rows * end / size)`, so the splits of a file read each row exactly once.
Only the chunks a range overlaps are read. Set `vortex.splitFiles` to `false` to read each file in
one task.

On the V2 path the scan plans the splits itself. It splits a file only at chunk boundaries, so each
task reads whole chunks and no two tasks decode the same chunk. Each split holds at least
`vortex.minSplitBytes` (16 MiB by default), or Spark's own split size when that is larger. A smaller
`spark.sql.files.maxPartitionBytes` lowers this minimum to it. Fewer, larger splits cost fewer file
opens on a cluster with many cores.

On the V2 path the driver also drops every file whose footer statistics prove that no row satisfies
the pushed filters, before any task starts. The runtime filters on data columns, such as the join
keys of dynamic pruning, drop files the same way. Comparisons, `IN`, null checks, and `AND` and `OR`
of them prune on integer, decimal, date, and timestamp columns. Set `vortex.pruneFiles` to `false`
to turn this off.

Only required columns are read. On the V2 path a struct column is also pruned to the fields the
query reads, one level deep: `SELECT s.a` reads the field `a` of `s` but no other field, and keeps
the rows where `s` itself is null. A `LIMIT` with no filter above the scan stops each file after
that many rows.

`ORDER BY col LIMIT n` with no filter above the scan is pushed too, when `col` is an integer,
decimal, date, timestamp, or string data column. Each task keeps the best `n` values of `col` it
has read and filters the rest of its files on the worst of them. Vortex checks that threshold
against the statistics of each file and chunk as the scan proceeds, so data sorted or clustered by
`col` skips most of what it would otherwise read. Spark still sorts the rows the scan returns.

An `OFFSET` with no filter above the scan, alone or followed by a `LIMIT`, is applied inside the
scan: footer row counts tell which files the offset covers whole, and those are never read. Without
an `ORDER BY` Spark leaves the skipped rows unspecified, and the scan skips them in the order it
plans the files. A pushed offset reads each file in one task and turns off runtime filtering, which
would drop rows the offset has already counted.

`TABLESAMPLE (p PERCENT)` and `Dataset.sample(fraction)` without replacement are applied inside the
scan. Each row is kept when a hash of the seed, the file's path, and the row's index falls in the
sampled range, so a seed selects the same rows on every run however the files are split. The
native scan receives the kept rows as a row selection and never decodes a chunk without one.

On Spark 4.2, `TABLESAMPLE SYSTEM (p PERCENT)` keeps or drops whole chunks of rows, chosen by a
hash of the file's path and the chunk's index, so a reader decodes only the chunks it keeps. Spark
does not accept a seed for `SYSTEM` sampling.

Vortex files written by Spark record whether each column is sorted, when the enabled Vortex editions
include that statistic. The core edition that Spark uses does not include it yet, so for now no
footer states an order. With `vortex.reportOrdering`
set, the scan finds the first read column that every footer states sorted ascending without nulls,
reports that order to Spark, and reads each file or split in its own task, in file order, so a
`sortWithinPartitions` or a sort-based operator over it needs no sort. It is off by default, since
one task per file costs parallelism planning for files of mixed size.

With `spark.sql.sources.v2.bucketing.enabled`, the scan reports its Hive partition columns to
Spark as the keys of its partitions, so joins and aggregations on them, and storage-partitioned
joins between two Vortex datasets partitioned the same way, need no shuffle.

Only files that end with `.vortex` belong to a Vortex dataset. Writes produce that extension.
Reads skip every other file in the dataset's directories. If a path holds files but none of them
is a Vortex file, schema inference fails.

### Schema Inference and Merging

The schema of a dataset is the merge of every file's footer schema, so a column added by a later
write is part of the dataset and the files written before it read as null. A field that only some
files carry is nullable in the merged schema.

Merging reads one footer per file on the driver before the job starts. Set `vortex.mergeSchema` to
`false` to read a single footer and let one file's schema stand for the whole dataset, which is
worth doing for a large dataset of uniform files. Passing an explicit `.schema(...)` skips
inference altogether.

Only top-level columns are merged. A struct column that gained or lost a field cannot be merged,
because the reader projects a struct as the file stores it and cannot widen one file's struct to
match another's; inference fails and names the field. Two files that give the same column
different types fail the same way.

Comparisons (`=`, `<>`, `>`, `>=`, `<`, `<=`), `IS NULL`, `IS NOT NULL`, `IN`, and the string
predicates `STARTS_WITH`, `ENDS_WITH`, and `CONTAINS` can be pushed into the Vortex scan. An `IN`
list becomes one native set lookup, which costs the same for any number of values and keeps SQL's
null semantics under `NOT`. Comparisons on `FLOAT` and `DOUBLE` columns stay in Spark, because
Vortex orders `-0.0` below `0.0` while Spark treats them as equal.

On the V2 path, predicates with no Spark V1 filter form are pushed too: `LIKE` with any pattern,
such as `name LIKE 'a%b_c'`, and comparisons between two columns of the same type, such as
`a < b`, alone or combined with other pushable predicates under `AND`, `OR`, and `NOT`.
Nested filter pushdown through the V1 fallback also requires `vortex` in
`spark.sql.optimizer.nestedPredicatePushdown.supportedFileSources`. A filter reading a column
that some file does not carry is evaluated by Spark above the scan rather than pushed into that
file.

### Aggregate Pushdown

On the V2 path, aggregates without data filters are answered once per file rather than by reading
rows, and Spark combines the partial result of each file:

| Aggregate          | Columns                                                                            |
|--------------------|------------------------------------------------------------------------------------|
| `COUNT(*)`         | —                                                                                  |
| `COUNT(col)`       | Any column.                                                                        |
| `MIN(col)`, `MAX(col)` | Boolean, integer, floating-point, decimal, date, timestamp, string, and binary columns, and partition columns. |
| `SUM(col)`, `AVG(col)` | `TINYINT`, `SMALLINT`, `INT`, and `BIGINT` columns, outside ANSI mode.         |

Each aggregate is answered from the file footer's statistics when they state it exactly, at the
cost of the footer read alone. Otherwise only the columns it reads are scanned, and Vortex computes
it from the compressed data, often without decompressing it. `MIN` and `MAX` order `NaN` above
every number, as Spark does. Grouping by any set of partition columns is pushed too, as is an
aggregate over a partition column. A file written before a column was added counts as all nulls
in it.

`SUM` wraps on overflow like Spark's own `BIGINT` addition. In ANSI mode, which Spark 4 enables by
default, an overflow must fail, and a per-file sum cannot tell, so `SUM` and `AVG` stay in Spark.

When the footer of every file states each aggregate exactly, the driver reads the footers, merges
the results of the files, and gives Spark the final rows. Spark then runs no task for the query.
If a footer does not state an aggregate exactly, for example the `MIN` of a long string, each file
answers in its own task as above. Set `vortex.completeAggregatePushdown` to `false` to always
answer in tasks.

Set `vortex.aggregatePushdown` to `false` to compute every aggregate from the rows.

### Runtime Filtering

On the V2 path, Spark's dynamic pruning filters narrow a scan to the keys of the other side of a
join at run time. A filter on a partition column drops whole files on the driver.

In a session with the Vortex session extensions
(`spark.sql.extensions=dev.vortex.spark.VortexSparkSessionExtensions`), the scan also accepts
these filters on data columns. A filter on a data column reaches every reader as a native set
lookup, so Vortex skips each file and chunk whose statistics rule out all of the keys before
decoding it. Beyond `vortex.runtimeFilter.maxValues` keys, only the range between the smallest and
largest key is pushed, because the keys travel with every task. Data columns of type `FLOAT`,
`DOUBLE`, and nested types are not filtered this way. Other runtime filters with a V1 filter form
over data columns, such as a comparison with the value of a scalar subquery, also reach every
reader.

The data column filters need the extensions because Spark adds one pruning subquery for each join
key that a scan accepts, and the subquery holds the plan of the other side of the join. When that
side holds pruned scans too, such as a common table expression that is joined with itself, the
subqueries copy each other and the plan can become too large for the driver's memory. The
extensions add an optimizer rule that removes each pruning filter whose other side holds pruning
filters of its own. Such a side is not a broadcast dimension, so its pruning does not run, and the
result does not change.

On Spark 4.2, Spark wraps the runtime filters on
partition columns that have no such form, such as `p % 7 = (SELECT ...)`, in partition predicates.
The scan evaluates each on the driver once per file, and drops the files it rules out.

### Scan Metrics

On the V2 path, the Spark UI shows these metrics for each Vortex scan, summed over its tasks: the
files opened, the splits that hold no rows to read, the batches read, the time spent to open files
and plan their scans, and the time spent in the native scan.

### Statistics

Footer row counts feed Spark's scan statistics unless `vortex.stats.rowCount` is disabled. With
the cost-based optimizer enabled (`spark.sql.cbo.enabled`), the footers also supply each column's
null count and, for numeric, date, and timestamp columns, its minimum and maximum. Only what every
footer states exactly is reported, and no data is read.

## Writing Vortex Files

```java
df.write()
    .format("vortex")
    .mode(SaveMode.Overwrite)
    .save("/path/to/output");
```

Spark's commit protocol owns output naming, task retries, append, truncate, and static or dynamic
partition overwrite. Partitioned writes use the normal DataFrame API:

```java
df.write()
    .format("vortex")
    .partitionBy("date")
    .mode(SaveMode.Overwrite)
    .save("/path/to/output");
```

Catalog tables can also be bucketed, with `bucketBy(...).saveAsTable(...)`, `CLUSTERED BY`, or
`PARTITIONED BY (bucket(n, col))`. Spark writes these as its standard bucket files, so joins and
aggregations on the bucket columns can skip the shuffle. Spark's session catalog rejects the
`years`, `months`, `days`, and `hours` transforms for every file format. To partition by time,
derive the column and partition by it, for example `withColumn("year", year(col("ts")))`.

| Option                    | Default | Description |
|---------------------------|---------|-------------|
| `batch.size`              | `2048`  | Rows buffered per write batch; range 1–65536. |
| `vortex.write.batch.size` | —       | Rows buffered per write batch for Vortex alone, overriding `batch.size`. |

## Spark SQL

Vortex works as a native file format on Spark 3.5 and 4.x:

```sql
CREATE TEMPORARY VIEW people
USING vortex
OPTIONS (path '/path/to/data');

CREATE TABLE student (id INT, name STRING, age INT)
USING vortex;

INSERT INTO student VALUES (1, 'Alice', 20), (2, 'Bob', 21);
SELECT * FROM student;
```

With a `LOCATION` clause, the table is external. Without one, Spark stores it below the warehouse
directory and deletes its files on `DROP TABLE`. `CREATE TABLE ... AS SELECT` is also supported.

Existing files can be queried directly without registering a catalog:

```sql
SELECT * FROM vortex.`/path/to/data`;
```

As with Spark's built-in file formats, direct path queries are read-only. To write to a path from
SQL, create a table over it with `LOCATION` and insert into the table.

## Choosing the V2 or V1 Code Path

By default Vortex reads and writes through Spark's DataSource V2 file API. Catalog tables use the V1
file format instead, and so does every path when `vortex` is listed in
`spark.sql.sources.useV1SourceList`:

```python
spark.conf.set("spark.sql.sources.useV1SourceList", "vortex")
```

Spark lists all of its own file formats there by default. The `_metadata` column is only available
on the V1 path, because Spark's V2 file API does not expose it. Besides Spark's file fields, it
holds `_metadata.row_index`, each row's index in its file before any filter, as Parquet provides it.

Filter pushdown, column pruning, partition pruning, file splitting, dynamic partition pruning, and
partitioned writes work on both paths. Aggregate pushdown, `LIMIT`, `OFFSET`, `ORDER BY ... LIMIT`,
and `TABLESAMPLE` pushdown, `LIKE` and column-comparison pushdown, variant extraction, nested column pruning, runtime
filters on data columns, reported partition keys and ordering, and footer-backed scan statistics
are V2 only.

Spark's bloom-filter runtime filters (`spark.sql.optimizer.runtime.bloomFilter.enabled`) are
evaluated by Spark above the scan: Spark adds them after it has pushed filters into the scan, so no
file format receives them.

## I/O and Remote Storage

Spark lists files and discovers partitions through Hadoop, and all file content is read and
written through Hadoop streams, so the connector inherits Spark's filesystem implementations,
credentials, and retry behavior. Configure remote storage through Spark's Hadoop configuration
and use the matching Hadoop scheme, for example `s3a://` for S3 and `abfs://` for Azure. Vortex's
own storage clients are not used by the Spark connector.

| Option                     | Default | Description |
|----------------------------|---------|-------------|
| `vortex.readConcurrency`   | `0`     | Maximum Hadoop read upcalls per file; `0` uses the native default. |
| `vortex.splitConcurrency`  | `0`     | Most row splits of a file decoded at once by one task; `0` uses the native default of four per core. Lowering this reduces memory held by concurrent split decoding and Arrow conversion. A scan with both a filter and a limit may also retain one trimmed array per split while preparing output. This does not cap total native memory or I/O read-ahead. |
| `vortex.workerThreads`     | `4`     | Background threads driving Vortex futures. JVM-wide, and set once per JVM by the first read or write. |
| `vortex.session.provider`  | —       | Fully-qualified name of a `VortexSessionProvider` supplying a custom session on the driver and on every executor. |
| `vortex.aggregatePushdown` | `true`  | Answer aggregates once per file rather than from the rows. |
| `vortex.completeAggregatePushdown` | `true` | Answer aggregates on the driver when every footer states them exactly. |
| `vortex.limitPushdown`     | `true`  | Stop reading each file once it has returned the rows a `LIMIT` needs. |
| `vortex.topNPushdown`      | `true`  | Skip rows that cannot reach the result of an `ORDER BY ... LIMIT`. |
| `vortex.samplePushdown`    | `true`  | Apply `TABLESAMPLE` without replacement inside the scan. |
| `vortex.offsetPushdown`    | `true`  | Skip the rows of an `OFFSET` inside the scan, using footer row counts. |
| `vortex.predicatePushdown` | `true`  | Push `LIKE` patterns and column comparisons that have no V1 filter form. |
| `vortex.reportPartitioning` | `true` | Report Hive partition keys to Spark under v2 bucketing. |
| `vortex.variantPushdown`   | `false` | Extract the paths of `variant_get` inside the scan, on Spark 4.1 and later. |
| `vortex.reportOrdering`    | `false` | Report the order the files store their rows in, so Spark need not sort again. |
| `vortex.splitFiles`        | `true`  | Split large files into row ranges read by separate tasks. |
| `vortex.minSplitBytes`     | `16m`   | Smallest split on the V2 path, at most `spark.sql.files.maxPartitionBytes`. |
| `vortex.pruneFiles`        | `true`  | Drop files on the driver whose footer statistics rule out every row. |
| `vortex.runtimeFilter`     | `true`  | Accept Spark's runtime filters, such as dynamic partition pruning. |
| `vortex.runtimeFilter.maxValues` | `10000` | Most runtime filter keys pushed as a set; a larger set is pushed as its range. |
| `vortex.mergeSchema`       | `true`  | Merge every file's footer schema, rather than reading one footer. |
| `vortex.stats.rowCount`    | `true`  | Read footer row counts for Spark statistics. |
| `vortex.stats.columns`     | `true`  | Read footer column statistics when the cost-based optimizer is enabled. |
| `vortex.footerParallelism` | `8`     | Maximum concurrent footer reads, for schema merging and statistics. |
| `vortex.stats.maxFiles`    | `1000`  | Skip footer row counts above this file count; `0` removes the bound. |

Option names are matched without regard to case, as they are everywhere else in Spark.

Like Spark's own vectorized readers, the connector turns off the bounds and null checks of Arrow
value access, as `arrow.enable_unsafe_memory_access=true` and
`arrow.enable_null_check_for_get=false`. A value set as a JVM system property or as the environment
variables `ARROW_ENABLE_UNSAFE_MEMORY_ACCESS` and `ARROW_ENABLE_NULL_CHECK_FOR_GET` takes precedence.

Scan statistics cost one read per file on the driver, before the job starts. `vortex.stats.maxFiles`
keeps planning bounded on a large dataset by reporting no row count instead. Raise it when exact
row counts matter more than planning time, or set `vortex.stats.rowCount` to `false` to stop
reading footers at all.

A split file's footer is read again on executors that haven't cached it. On the V1 path, a chunk
that spans a cut is decoded by both tasks next to it. A file whose footer states no exact row count
is read whole by its first split.

## Supported Types

| Spark Type         | Vortex Type                           |
|--------------------|---------------------------------------|
| `BooleanType`      | Bool                                  |
| `ByteType`         | Int8 / UInt8                          |
| `ShortType`        | Int16 / UInt16                        |
| `IntegerType`      | Int32 / UInt32                        |
| `LongType`         | Int64 / UInt64                        |
| `FloatType`        | Float32                               |
| `DoubleType`       | Float64                               |
| `StringType`       | Utf8                                  |
| `BinaryType`       | Binary                                |
| `DecimalType`      | Decimal                               |
| `DateType`         | Date (days)                           |
| `TimestampType`    | Timestamp (microseconds, UTC)         |
| `TimestampNTZType` | Timestamp (microseconds, no timezone) |
| `ArrayType`        | List                                  |
| `StructType`       | Struct                                |
| `MapType`          | Map                                   |
| `VariantType`      | Variant (Spark 4 only)                |

`VARIANT` columns, top-level or nested, are stored as Vortex variants through Arrow's
`arrow.parquet.variant` extension. Spark evaluates `variant_get` over them itself by default.

On Spark 4.1 and later, set `vortex.variantPushdown` to `true` to push `variant_get` and
`try_variant_get` over top-level variant columns into the scan. Vortex extracts each path
natively, and the scan returns only the extracted values instead of whole variants. Spark casts
them to the requested type, so the results do not change. It is off by default for two reasons.
Spark builds a scan with pushed extractions without pruning its columns, so the scan reads every
column of the table. Also, for unshredded variants, which Spark writes, native extraction parses
each variant as Spark's own `variant_get` does, and it is not faster.
