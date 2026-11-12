# Java

Vortex has two Java libraries:

- `dev.vortex:vortex-jni` gives direct access to Vortex files. It writes and scans files with
  [Apache Arrow](https://arrow.apache.org/docs/java/) record batches.
- `dev.vortex:vortex-spark-3.5_2.12`, `dev.vortex:vortex-spark-3.5_2.13`, and
  `dev.vortex:vortex-spark-4.0_2.13` are Spark file data sources. Use them to read and write
  Vortex files from Spark.

## Install

The libraries need Java 17 or later. The JARs contain native libraries for Linux (x86_64 and
aarch64), macOS (aarch64), and Windows (x86_64). The JNI bindings load the native library
automatically.

Gradle (Kotlin):

```kotlin
implementation("dev.vortex:vortex-jni:0.93.0")
```

Maven:

```xml
<dependency>
    <groupId>dev.vortex</groupId>
    <artifactId>vortex-jni</artifactId>
    <version>0.93.0</version>
</dependency>
```

The JNI library uses Arrow Java for its memory management. Arrow Java needs this JVM option on
Java 17 and later:

```shell
--add-opens=java.base/java.nio=ALL-UNNAMED
```

## Write a File

A `Session` is the entry point for all operations. A `VortexWriter` accepts Arrow record batches
and writes them to a file. The destination can be a local path or a URI.

```java
import static java.nio.charset.StandardCharsets.UTF_8;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriteSummary;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import java.util.List;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

BufferAllocator allocator = ArrowAllocation.rootAllocator();
Schema schema = new Schema(List.of(
        Field.notNullable("name", new ArrowType.Utf8()),
        Field.notNullable("age", new ArrowType.Int(32, true)),
        Field.nullable("state", new ArrowType.Utf8())));

Session session = Session.create();

try (VortexWriter writer = VortexWriter.builder(session, "/tmp/people.vortex", schema, allocator).build();
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
    root.allocateNew();

    VarCharVector names = (VarCharVector) root.getVector("name");
    IntVector ages = (IntVector) root.getVector("age");
    VarCharVector states = (VarCharVector) root.getVector("state");

    names.setSafe(0, "Alice".getBytes(UTF_8));
    ages.setSafe(0, 34);
    states.setSafe(0, "CA".getBytes(UTF_8));

    names.setSafe(1, "Bob".getBytes(UTF_8));
    ages.setSafe(1, 27);
    states.setNull(1);

    names.setSafe(2, "Carol".getBytes(UTF_8));
    ages.setSafe(2, 45);
    states.setSafe(2, "NY".getBytes(UTF_8));

    root.setRowCount(3);
    writer.writeBatch(root, allocator);

    VortexWriteSummary summary = writer.finish();
    System.out.printf("Wrote %d rows, %d bytes%n", summary.rowCount(), summary.fileSize());
}
```

Call `writeBatch` one time for each batch. All batches must use the schema given to the
builder. `writeBatch(root, allocator)` does not copy the data. After the call, you can clear
or close the root, but do not write new values into its buffers. To fill the next batch,
allocate new buffers, for example with `root.allocateNew()`.

To write a batch that you exported through the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html), give its
`ArrowArray` and `ArrowSchema` addresses to `writeBatch(long, long)`. Use this method also for
dictionary-encoded fields.

`finish()` writes the file footer and returns the row count, the file size, and per-column
statistics. `close()` also finishes the file. Call `finish()` or `close()` explicitly, so that
IO errors come back to the caller. While the writer is open, `bytesWritten()` and
`bufferedBytes()` give an estimate of the file size.

### Writer Options

`VortexWriter.Builder` has these optional settings:

- `putMetadata(key, bytes)` and `metadata(map)` store your own metadata in the file footer.
- `compact(true)` writes with the compact strategy. The file is smaller, but reads are slower.
  Readers do not need a special setting to read the file.
- `options(map)` and `putOption(key, value)` give object store credentials. See
  [Object Storage](#object-storage).

To write files that older Vortex readers can read, enable an older edition on the session
before you open the writer:

```java
session.enableEdition("core2025.10.0");
```

[Using editions](../specs/versioning/using-editions.md) explains how to select an edition.

## Read a File

`DataSource.open` opens one or more files. It reads only the first file immediately, to get the
schema. `DataSource.scan` returns a `Scan`, which is an iterator of `Partition` objects. Each
partition gives an Arrow `ArrowReader`. Close the data source, the scan, and the reader after
you use them.

```java
import dev.vortex.api.DataSource;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import org.apache.arrow.vector.ipc.ArrowReader;

try (DataSource source = DataSource.open(session, "/tmp/people.vortex")) {
    System.out.println(source.arrowSchema(allocator));
    System.out.println(source.rowCount().asOptional());

    try (Scan scan = source.scan(ScanOptions.of())) {
        while (scan.hasNext()) {
            Partition partition = scan.next();

            try (ArrowReader reader = partition.scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot batch = reader.getVectorSchemaRoot();
                    System.out.print(batch.contentToTSVString());
                }
            }
        }
    }
}
```

A partition is single-use: `scanArrow` consumes it. You can scan a `DataSource` more than one
time. When you close a data source or a scan, the partitions and readers that you already have
stay valid. Partitions do not keep the row order of the file by default. To keep the row order, set
`ordered(true)` on the scan options.

The scan can return string columns as Arrow `Utf8View` vectors (`ViewVarCharVector`), even if
you wrote them as `Utf8`. Use `getObject` or the vector type from the batch schema to read
values.

### Read Many Files

Give a list of paths or a glob to read many files as one data source:

```java
DataSource all = DataSource.open(session, "/data/events/*.vortex");

DataSource some = DataSource.open(
        session, List.of("/data/events/a.vortex", "/data/events/b.vortex"), Map.of());
```

The data source opens the first match immediately and opens the other files when the scan
reaches them.

## Projection and Filters

`ScanOptions.builder()` configures a scan. A projection selects columns, and a filter selects
rows. Vortex evaluates both in the native scan, so it reads only the necessary data.

```java
import dev.vortex.api.Expression;
import dev.vortex.api.Expression.BinaryOp;

ScanOptions options = ScanOptions.builder()
        .projection(Expression.select(new String[] {"name", "age"}, Expression.root()))
        .filter(Expression.and(
                Expression.binary(BinaryOp.GT, Expression.column("age"), Expression.literal(30)),
                Expression.isNotNull(Expression.column("state"))))
        .build();

Scan scan = source.scan(options);
```

`Expression` also has `or`, `not`, `isNull`, `like`, `between`, `in`, `notIn`, `listContains`,
and typed literals for decimals, dates, timestamps, UUIDs, and more. Use `Expression.column(String[])`
to get a nested struct field.

## Row Selection and Limits

The scan options can also select rows by position or limit the row count:

```java
// Only rows 0, 3, and 9. The indices must be sorted and unique.
source.scan(ScanOptions.includeRows(0, 3, 9));

// All rows except rows 0, 3, and 9.
source.scan(ScanOptions.excludeRows(0, 3, 9));

// Rows in the range [100, 200).
source.scan(ScanOptions.builder().rowRangeBegin(100).rowRangeEnd(200).build());

// At most 10 rows.
source.scan(ScanOptions.builder().limit(10).build());
```

For large selections, `includeRows` and `excludeRows` also accept a
[RoaringBitmap](https://github.com/RoaringBitmap/RoaringBitmap) `Roaring64NavigableMap`.

## Memory Use

A scan decodes many splits of each partition at the same time. By default, the number of
splits increases with the number of CPU cores. To use less memory when many scans run at the
same time, set a lower limit with `splitConcurrency`:

```java
source.scan(ScanOptions.builder().splitConcurrency(2).build());
```

This option does not limit all native memory or IO read-ahead.

## Object Storage

Paths can be URIs for object storage, for example `s3://bucket/path/file.vortex` or
`abfss://container@account.dfs.core.windows.net/path/file.vortex`. Give credentials and other
object store options as a map of properties:

```java
Map<String, String> properties = Map.of(
        "aws_access_key_id", "...",
        "aws_secret_access_key", "...",
        "aws_region", "us-east-1");

DataSource source = DataSource.open(session, "s3://bucket/path/*.vortex", properties);

VortexWriter writer = VortexWriter.builder(session, "s3://bucket/path/out.vortex", schema, allocator)
        .options(properties)
        .build();
```

Other S3 keys are `aws_session_token`, `aws_endpoint`, `aws_allow_http`, and
`aws_skip_signature`. For Azure, use `azure_storage_account_key` or `azure_storage_sas_key`.
If you do not give credentials, the object store uses the credentials from the environment.

### Custom IO

To read or write through your own IO layer, implement `NativeReadable` or `NativeWritable` from
`dev.vortex.io` and give it to `DataSource.open` or `VortexWriter.builder`. The native reader
calls `NativeReadable.readFully` from many threads at the same time, so the implementation must
be thread-safe. Native code does not close a readable or a writable. Close them yourself after
the data source and all its scans and readers are done, or after the writer is finished.

`DataSource.open(session, readables, readConcurrency)` sets the maximum number of concurrent
`readFully` calls. The value `0` selects the default.

The native reader merges nearby reads into one request. By default, it merges reads with a gap
of at most 1 MiB, to a maximum span of 16 MiB. To change these limits, override
`coalesceDistance()` and `coalesceMaxSize()` in your `NativeReadable`. A larger gap makes fewer
requests but reads more bytes.

## File Metadata and Statistics

`dev.vortex.jni.NativeFiles` reads the footer of a file without a scan:

```java
import dev.vortex.api.VortexColumnStatistics;
import dev.vortex.jni.NativeFiles;

Map<String, byte[]> metadata = NativeFiles.readMetadata(session, "/tmp/people.vortex", Map.of());

for (VortexColumnStatistics stats : NativeFiles.readStatistics(session, "/tmp/people.vortex", Map.of())) {
    System.out.printf("column %d: %d values, %d bytes, min %s, max %s%n",
            stats.columnIndex(), stats.valueCount(), stats.compressedSize(),
            stats.lowerBound().orElse(null), stats.upperBound().orElse(null));
}
```

`readMetadata` returns the metadata that the writer stored with `putMetadata`. `readStatistics`
returns the statistics of each top-level column in Arrow schema order. Both methods also accept
a `NativeReadable`. `NativeFiles` also has `listFiles` and `delete`.

## Spark

The Spark connector uses the JNI library to read and write Vortex files from Spark. Add the
self-contained `all` JAR to Spark:

```shell
spark-shell --jars https://repo1.maven.org/maven2/dev/vortex/vortex-spark-4.0_2.13/0.93.0/vortex-spark-4.0_2.13-0.93.0-all.jar
```

Use `vortex-spark-3.5_2.12` or `vortex-spark-3.5_2.13` for Spark 3.5, and `vortex-spark-4.0_2.13`
for Spark 4.0 to 4.3. The connector registers the format name `vortex`:

```java
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;

Dataset<Row> people = spark.read()
        .format("vortex")
        .option("path", "/tmp/people.vortex")
        .load();

people.filter("age > 30")
        .select("name", "state")
        .write()
        .format("vortex")
        .option("path", "/tmp/people_over_30")
        .mode(SaveMode.Overwrite)
        .save();
```

Spark lists the files and discovers Hive-style partitions. The connector pushes column
projections and many filters down to the native scan. The connector reads and writes all
files through Hadoop, so configure remote storage in the Spark Hadoop configuration. The object
store properties in the previous section do not apply to Spark. It also works from Spark SQL:

```sql
CREATE TEMPORARY VIEW people
USING vortex
OPTIONS (path '/tmp/people.vortex');

SELECT name, age FROM people WHERE age > 30;
```

The [Spark user guide](../user-guide/spark.md) has the full documentation, including the
build dependency setup, tables, direct file queries, write options such as
`vortex.write.compact`, read tuning, supported types, and remote storage through Hadoop.

## Next Steps

- The [Java API reference](../api/java/index.rst) has the Javadoc for `vortex-jni` and
  `vortex-spark`.
- [Concepts](../concepts/index.md) explains Vortex arrays, layouts, and scans.
