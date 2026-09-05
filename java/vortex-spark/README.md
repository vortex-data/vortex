# vortex-spark

A Spark file data source for reading and writing [Vortex](https://vortex.dev) files. It
registers itself as `vortex` and supports the DataFrame API and Spark SQL without session
extensions or custom catalogs.

Choose the artifact matching both Spark and Scala:

| Artifact                           | Spark         | Scala |
|------------------------------------|---------------|-------|
| `dev.vortex:vortex-spark-3.5_2.12` | 3.5.x         | 2.12  |
| `dev.vortex:vortex-spark-3.5_2.13` | 3.5.x         | 2.13  |
| `dev.vortex:vortex-spark-4.0_2.13` | 4.0.x and 4.1.x | 2.13 |

Use the `all` classifier JAR. It bundles the Vortex JNI bindings, native libraries, and
relocated Arrow, Guava, and Jackson dependencies. The unclassified thin JAR is not usable by
itself.

For example, with version `VERSION`:

```shell
spark-shell --jars https://repo1.maven.org/maven2/dev/vortex/vortex-spark-4.0_2.13/VERSION/vortex-spark-4.0_2.13-VERSION-all.jar
```

Or add the classified artifact to a JVM project:

```kotlin
implementation("dev.vortex:vortex-spark-4.0_2.13:VERSION:all")
```

Spark's `--packages` flag cannot select the `all` classifier, so pass the classified JAR with
`--jars` instead.

## Usage

```java
df.write()
    .format("vortex")
    .mode(SaveMode.Overwrite)
    .save("/path/to/output");

Dataset<Row> result = spark.read()
    .format("vortex")
    .load("/path/to/output");
```

Spark discovers files, Hive-style partitions, and storage credentials through its Hadoop
configuration. This also enables standard file-source options such as `pathGlobFilter` and
`recursiveFileLookup`. All file content is read and written through Hadoop streams.

Every file in a Vortex dataset must end with `.vortex`. Writes produce that extension, and a
dataset holding any other file is rejected rather than read in part.

```sql
CREATE TABLE student (id INT, name STRING) USING vortex;
INSERT INTO student VALUES (1, 'Alice'), (2, 'Bob');
SELECT * FROM student;

SELECT * FROM vortex.`/path/to/output`;
```

`CREATE TABLE ... USING vortex` and direct path queries work on Spark 3.5 and 4.x without
catalog configuration. Catalog tables read and write through the V1 file format;
`spark.sql.sources.useV1SourceList=vortex` puts every path on it, which is what the `_metadata`
column and dynamic partition pruning need.

## Benchmarks

JMH benchmarks live in `common/src/jmh/java`. They measure the per-file open cost of the
framework's file-splitting model and the footer reads behind statistics and `COUNT(*)` pushdown.
Run them for one variant and pass JMH arguments through `-PjmhArgs`:

```bash
cd java
./gradlew :vortex-spark-4.0_2.13:jmh -PjmhArgs="SparkScanBenchmark -p fileCount=1,128"
./gradlew :vortex-spark-4.0_2.13:jmh -PjmhArgs="FooterReadBenchmark -f 1 -wi 2 -i 5"
```

## Migrating from the old connector

- Replace `vortex-spark_2.12` or `vortex-spark_2.13` with the versioned artifact above.
- Remove `spark.sql.catalog.spark_catalog=dev.vortex.spark.VortexSessionCatalog` and
  `spark.sql.catalog.vortex=dev.vortex.spark.VortexCatalog`; those classes no longer exist.
- Configure remote filesystems through Hadoop and use their Hadoop schemes, such as `s3a://`
  and `abfs://`. Spark's file index now owns listing and skips hidden `_`-prefixed files.
- All reads and writes use Hadoop streams. The `vortex.io` option and Vortex's native storage
  clients (with their `aws_*`/`azure_*` options) are no longer available from Spark.

See the [Spark user guide](https://docs.vortex.dev/user-guide/spark.html) for supported types,
options, partitioning, and remote storage details.
