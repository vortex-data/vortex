# vortex-spark

A Spark DataSource V2 connector for reading and writing [Vortex](https://vortex.dev) files.
It registers itself under the format name `vortex` and supports both the DataFrame API and
Spark SQL.

Two flavors are published to Maven Central:

| Artifact                      | Spark     | Scala |
|-------------------------------|-----------|-------|
| `dev.vortex:vortex-spark_2.13` | Spark 4.x | 2.13  |
| `dev.vortex:vortex-spark_2.12` | Spark 3.5.x | 2.12 |

Use the `all` classifier JAR (e.g. `vortex-spark_2.13-0.85.0-all.jar`). It is self-contained:
it bundles the Vortex JNI bindings, native libraries for Linux (x86_64 and aarch64) and macOS
(aarch64), and relocates its Arrow, Guava, and Jackson dependencies to avoid classpath
conflicts with Spark. The thin (unclassified) JAR does not work on its own because it
references relocated classes that only ship in the `all` JAR.

## Getting Vortex into Spark

Pass the `all` JAR to `spark-shell`, `spark-submit`, or `pyspark` with `--jars`. Spark accepts
either a local path or a URL, so you can point directly at Maven Central:

```shell
spark-shell --jars https://repo1.maven.org/maven2/dev/vortex/vortex-spark_2.13/0.85.0/vortex-spark_2.13-0.85.0-all.jar
```

Or configure it on the session builder, e.g. in PySpark:

```python
spark = (
    SparkSession.builder
    .config("spark.jars", "/path/to/vortex-spark_2.13-0.85.0-all.jar")
    .getOrCreate()
)
```

Note that `--packages dev.vortex:vortex-spark_2.13:0.85.0` does not work: `--packages` cannot
select the `all` classifier and resolves the thin JAR, which fails at runtime with
`NoClassDefFoundError: dev/vortex/relocated/...`.

To depend on the connector from a JVM project instead, add the `all` classifier to the
dependency:

```kotlin
implementation("dev.vortex:vortex-spark_2.13:0.85.0:all")
```

## Usage

Paths may be local filesystem paths (`/path/to/data`) or URLs (`file:///path/to/data`,
`s3://bucket/path/to/data`).

### DataFrame API

```java
// Write
df.write()
    .format("vortex")
    .option("path", "/path/to/output")
    .mode(SaveMode.Overwrite)
    .save();

// Read a single file or a directory of .vortex files
Dataset<Row> df = spark.read()
    .format("vortex")
    .option("path", "/path/to/output")
    .load();
```

### Spark SQL

```sql
-- Query existing Vortex files through a temporary view
CREATE TEMPORARY VIEW people
USING vortex
OPTIONS (path '/path/to/data');

SELECT name, age FROM people WHERE age > 30;

-- Create a table and write to it. With a LOCATION clause the table is external,
-- backed by the files at that path; without one it is managed by Spark.
CREATE TABLE student (id INT, name STRING, age INT)
USING vortex;

INSERT INTO student VALUES (1, 'Alice', 20), (2, 'Bob', 21);

SELECT * FROM student;
```

On Spark 3.5, `CREATE TABLE ... USING vortex` additionally requires replacing the session
catalog with `spark.sql.catalog.spark_catalog=dev.vortex.spark.VortexSessionCatalog`,
because Spark 3.5's built-in catalog cannot read tables backed by a DataSource V2-only
connector. The extension delegates everything else to the built-in session catalog and
leaves tables of other providers untouched; it is not needed on Spark 4.

### Direct file queries

Spark's built-in ``SELECT * FROM format.`path` `` syntax only works for built-in file
formats, so the connector ships a path-based catalog that provides the equivalent for
Vortex. Register it under the name `vortex` with this session config:

```
spark.sql.catalog.vortex=dev.vortex.spark.VortexCatalog
```

Then query (or insert into) Vortex files directly by path:

```sql
SELECT * FROM vortex.`/path/to/data`;
```

See the [Spark user guide](https://docs.vortex.dev/user-guide/spark.html) for the full
documentation, including supported types, write options, and S3 configuration.
