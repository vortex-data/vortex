# Spark

Vortex provides a Spark file data source for reading and writing Vortex files. Choose the
artifact matching both the Spark and Scala versions in the application:

| Artifact                           | Spark           | Scala |
|------------------------------------|-----------------|-------|
| `dev.vortex:vortex-spark-3.5_2.12` | 3.5.x           | 2.12  |
| `dev.vortex:vortex-spark-3.5_2.13` | 3.5.x           | 2.13  |
| `dev.vortex:vortex-spark-4.0_2.13` | 4.0.x and 4.1.x | 2.13  |

Use the `all` classifier JAR. It bundles the Vortex JNI bindings, native libraries for Linux
(x86_64 and aarch64) and macOS (aarch64), and relocated Arrow, Guava, and Jackson dependencies that avoid
conflicts with Spark. The unclassified thin JAR is not usable by itself because it references
relocated classes that only ship in the `all` JAR.

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

The connector registers itself as `vortex`; no session extension or catalog configuration is
required.

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
with `_` or `.` are ignored. Vortex files are not split internally, and only required columns
are read.

Every file in a Vortex dataset must end with `.vortex`. Writes produce that extension. A dataset
that holds any other file is rejected: schema inference reports that it found no Vortex file, and
a scan names the offending path. Use `pathGlobFilter` to read Vortex files out of a directory that
holds other things too.

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
predicates `STARTS_WITH`, `ENDS_WITH`, and `CONTAINS` can be pushed into the Vortex scan.
Nested filter pushdown through the V1 fallback also requires `vortex` in
`spark.sql.optimizer.nestedPredicatePushdown.supportedFileSources`. A filter reading a column
that some file does not carry is evaluated by Spark above the scan rather than pushed into that
file.

`COUNT(*)` without data filters is computed from file footers. Spark combines the partial count
from each file, including counts grouped by partition columns. Footer row counts also feed
Spark's scan statistics unless `vortex.stats.rowCount` is disabled.

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

## Choosing the V2 or V1 Code Path

By default Vortex reads and writes through Spark's DataSource V2 file API. Catalog tables use the V1
file format instead, and so does every path when `vortex` is listed in
`spark.sql.sources.useV1SourceList`:

```python
spark.conf.set("spark.sql.sources.useV1SourceList", "vortex")
```

Spark lists all of its own file formats there by default. Two features are only available on the V1
path:

- the `_metadata` column, which Spark's V2 file API does not expose;
- dynamic partition pruning, which Spark applies to V1 relations only.

Filter pushdown, column pruning, partition pruning, and partitioned writes work on both paths.
`COUNT(*)` footer pushdown and footer-backed scan statistics are V2 only.

## I/O and Remote Storage

Spark lists files and discovers partitions through Hadoop, and all file content is read and
written through Hadoop streams, so the connector inherits Spark's filesystem implementations,
credentials, and retry behavior. Configure remote storage through Spark's Hadoop configuration
and use the matching Hadoop scheme, for example `s3a://` for S3 and `abfs://` for Azure. Vortex's
own storage clients are not used by the Spark connector.

| Option                     | Default | Description |
|----------------------------|---------|-------------|
| `vortex.readConcurrency`   | `0`     | Maximum Hadoop read upcalls per file; `0` uses the native default. |
| `vortex.workerThreads`     | `4`     | Background threads driving Vortex futures. JVM-wide, and set once per JVM by the first read or write. |
| `vortex.session.provider`  | —       | Fully-qualified name of a `VortexSessionProvider` supplying a custom session on the driver and on every executor. |
| `vortex.aggregatePushdown` | `true`  | Answer `COUNT(*)` from file footers. |
| `vortex.mergeSchema`       | `true`  | Merge every file's footer schema, rather than reading one footer. |
| `vortex.stats.rowCount`    | `true`  | Read footer row counts for Spark statistics. |
| `vortex.footerParallelism` | `8`     | Maximum concurrent footer reads, for schema merging and statistics. |
| `vortex.stats.maxFiles`    | `1000`  | Skip footer row counts above this file count; `0` removes the bound. |

Option names are matched without regard to case, as they are everywhere else in Spark.

Scan statistics cost one read per file on the driver, before the job starts. `vortex.stats.maxFiles`
keeps planning bounded on a large dataset by reporting no row count instead. Raise it when exact
row counts matter more than planning time, or set `vortex.stats.rowCount` to `false` to stop
reading footers at all.

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

## Migrating from earlier artifacts

- Replace `vortex-spark_2.12` or `vortex-spark_2.13` with the Spark-versioned artifact listed
  above.
- Remove `spark.sql.catalog.spark_catalog=dev.vortex.spark.VortexSessionCatalog` and
  `spark.sql.catalog.vortex=dev.vortex.spark.VortexCatalog`. The file-source integration makes
  both catalogs unnecessary, and their classes were removed.
- Listing now follows Spark semantics. Hidden files are skipped, and `pathGlobFilter` and
  `recursiveFileLookup` are available.
- Remote storage requires a Hadoop filesystem connector. OpenDAL-only schemes such as `cos://`
  and `oss://` are not available through Spark's file index.
