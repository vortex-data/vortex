# Spark

Vortex provides a Spark DataSource V2 connector for reading and writing Vortex files. The
connector is published to Maven Central in two flavors:

- `dev.vortex:vortex-spark_2.13` for Spark 4.x (Scala 2.13)
- `dev.vortex:vortex-spark_2.12` for Spark 3.5.x (Scala 2.12)

Use the `all` classifier JAR (e.g. `vortex-spark_2.13-0.78.0-all.jar`). It is self-contained:
it bundles the Vortex JNI bindings, native libraries for Linux (x86_64 and aarch64) and macOS
(aarch64), and relocates its Arrow, Guava, and Jackson dependencies to avoid classpath
conflicts with Spark. The thin (unclassified) JAR does not work on its own because it
references relocated classes that only ship in the `all` JAR.

## Getting Vortex into Spark

For `spark-shell`, `spark-submit`, or `pyspark`, pass the `all` JAR with `--jars`. Spark
accepts either a local path or a URL, so you can point directly at Maven Central:

```shell
spark-shell --jars https://repo1.maven.org/maven2/dev/vortex/vortex-spark_2.13/0.78.0/vortex-spark_2.13-0.78.0-all.jar
```

Or equivalently when building a session programmatically, e.g. in PySpark:

```python
spark = (
    SparkSession.builder
    .config("spark.jars", "/path/to/vortex-spark_2.13-0.78.0-all.jar")
    .getOrCreate()
)
```

```{note}
`--packages dev.vortex:vortex-spark_2.13:0.78.0` does not work: `--packages` cannot select
the `all` classifier and resolves the thin JAR, which fails at runtime with
`NoClassDefFoundError: dev/vortex/relocated/...`.
```

Once the JAR is on the classpath, the connector registers itself automatically under the
format name `vortex` — no session configuration is required.

## Installation as a Build Dependency

To depend on the connector from a JVM project, add the `all` classifier to the dependency:

Gradle (Kotlin):

```kotlin
implementation("dev.vortex:vortex-spark_2.13:0.78.0:all")
```

Maven:

```xml
<dependency>
    <groupId>dev.vortex</groupId>
    <artifactId>vortex-spark_2.13</artifactId>
    <version>0.78.0</version>
    <classifier>all</classifier>
</dependency>
```

## Reading Vortex Files

Paths may be local filesystem paths (`/path/to/data`) or URLs (`file:///path/to/data`,
`s3://bucket/path/to/data`). Use the `vortex` format to read a single file or a directory of
Vortex files:

```java
Dataset<Row> df = spark.read()
    .format("vortex")
    .option("path", "/path/to/data.vortex")
    .load();
```

When pointed at a directory, the connector discovers all `.vortex` files and creates one read
partition per file.

Column pruning is pushed down — only the columns referenced by the query are read from the file.

## Writing Vortex Files

```java
df.write()
    .format("vortex")
    .option("path", "/path/to/output")
    .mode(SaveMode.Overwrite)
    .save();
```

Each Spark partition produces one output file named `part-{partitionId}-{taskId}.vortex`.

### Write Options

| Option                    | Default | Description                        |
|---------------------------|---------|------------------------------------|
| `vortex.write.batch.size` | 2048    | Number of rows per batch (1–65536) |

### Save Modes

The connector supports all standard Spark save modes: `Overwrite`, `Append`, `Ignore`, and
`ErrorIfExists`.

## Spark SQL

The connector can also be used from pure SQL. To query existing Vortex files, register them
as a temporary view:

```sql
CREATE TEMPORARY VIEW people
USING vortex
OPTIONS (path '/path/to/data');

SELECT name, age FROM people WHERE age > 30;
```

Tables can be created with `USING vortex`, then written to and read back with plain SQL.
With a `LOCATION` clause the table is external, backed by the files at that path; without
one the table is managed, and Spark stores its data under the warehouse directory (and
deletes it on `DROP TABLE`):

```sql
CREATE TABLE student (id INT, name STRING, age INT)
USING vortex;

INSERT INTO student VALUES (1, 'Alice', 20), (2, 'Bob', 21);

SELECT * FROM student;
```

`CREATE TABLE ... AS SELECT` works the same way:

```sql
CREATE TABLE adults
USING vortex
AS SELECT * FROM people WHERE age >= 18;
```

```{note}
On Spark 3.5, `CREATE TABLE ... USING vortex` additionally requires replacing the session
catalog, because Spark 3.5's built-in catalog cannot read tables backed by a DataSource
V2-only connector:

    spark.sql.catalog.spark_catalog=dev.vortex.spark.VortexSessionCatalog

The extension delegates everything to the built-in session catalog (including the Hive
metastore, if configured) and only changes how `vortex` tables are resolved; tables of other
providers are untouched. It is not needed on Spark 4, though setting it is harmless.
```

## Direct File Queries

Spark's built-in ``SELECT * FROM format.`path` `` syntax only works for built-in file
formats, so the connector ships a path-based catalog that provides the equivalent for
Vortex. Register it in the session configuration under the name `vortex`:

```shell
spark-sql --conf spark.sql.catalog.vortex=dev.vortex.spark.VortexCatalog
```

Then query a Vortex file, or a directory of Vortex files, directly by path — no view or
table required:

```sql
SELECT * FROM vortex.`/path/to/data`;

INSERT INTO vortex.`/path/to/data` VALUES (1, 'Alice', 20);
```

## Supported Types

| Spark Type         | Vortex Type                            |
|--------------------|----------------------------------------|
| `BooleanType`      | Bool                                   |
| `ByteType`         | Int8 / UInt8                           |
| `ShortType`        | Int16 / UInt16                         |
| `IntegerType`      | Int32 / UInt32                         |
| `LongType`         | Int64 / UInt64                         |
| `FloatType`        | Float32                                |
| `DoubleType`       | Float64                                |
| `StringType`       | Utf8                                   |
| `BinaryType`       | Binary                                 |
| `DecimalType`      | Decimal                                |
| `DateType`         | Date (days)                            |
| `TimestampType`    | Timestamp (microseconds, UTC)          |
| `TimestampNTZType` | Timestamp (microseconds, no timezone)  |
| `ArrayType`        | List                                   |
| `StructType`       | Struct                                 |

## S3 Support

The connector supports reading and writing to S3 paths:

```java
Dataset<Row> df = spark.read()
    .format("vortex")
    .option("path", "s3://bucket/path/to/data")
    .load();
```
