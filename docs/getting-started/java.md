# Java

Vortex has two Java libraries:

- `dev.vortex:vortex-jni` writes and scans Vortex files as
  [Apache Arrow](https://arrow.apache.org/docs/java/) record batches.
- `dev.vortex:vortex-spark-3.5_2.12`, `dev.vortex:vortex-spark-3.5_2.13`, and
  `dev.vortex:vortex-spark-4.0_2.13` read and write Vortex files from Spark.

## Install

The libraries need Java 17 or later. Java 21 is the preferred version. The JARs contain native
libraries for Linux (x86_64 and aarch64), macOS (aarch64), and Windows (x86_64). The JNI
bindings load the native library automatically.

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

The JNI library allocates memory through Arrow Java, which needs this JVM option:

```shell
--add-opens=java.base/java.nio=ALL-UNNAMED
```

## Write a File

All operations start from a `Session`. A `VortexWriter` writes Arrow record batches to a file
at a local path or a URI.

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

Call `writeBatch` once for each batch. Every batch must have the schema that the builder
received. `writeBatch(root, allocator)` does not copy the data. After the call, you can clear
or close the root, but do not overwrite its buffers. Allocate new buffers for the next batch,
for example with `root.allocateNew()`.

`writeBatch(long, long)` accepts the `ArrowArray` and `ArrowSchema` addresses of a batch
exported through the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html).
Dictionary-encoded fields need this method.

`finish()` writes the file footer and returns the row count, the file size, and per-column
statistics. `close()` calls `finish()`. Call one of them explicitly, so that the caller
receives IO errors. While the writer is open, `bytesWritten()` and `bufferedBytes()` let you
estimate the file size.

### Writer Options

`VortexWriter.Builder` has these optional settings:

- `putMetadata(key, bytes)` and `metadata(map)` store your own metadata in the file footer.
- `compact(true)` writes a smaller file that is slower to read. Readers need no special
  setting.
- `options(map)` and `putOption(key, value)` set object store credentials. See
  [Object Storage](#object-storage).

To write files for older Vortex readers, enable an older edition on the session before you open
the writer:

```java
session.enableEdition("core2025.10.0");
```

[Using editions](../specs/versioning/using-editions.md) explains how to select an edition.

## Read a File

`DataSource.open` opens one or more files. It reads only the first file immediately, to get the
schema. `DataSource.scan` returns a `Scan`, which iterates over `Partition` objects. Each
partition returns an Arrow `ArrowReader`. Close the data source, the scan, and each reader
after use.

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

`scanArrow` consumes the partition, so you can read each partition only once. You can scan a
`DataSource` many times. Partitions and readers that you already have stay valid after you
close their data source or scan.

By default, a scan does not keep the row order of the file. Set `ordered(true)` on the scan
options to keep it.

A scan can return string columns as Arrow `Utf8View` vectors (`ViewVarCharVector`), even if
you wrote them as `Utf8`. Read values with `getObject`, or use the vector type from the batch
schema.

### Read Many Files

A data source can cover a glob or a list of paths:

```java
DataSource all = DataSource.open(session, "/data/events/*.vortex");

DataSource some = DataSource.open(
        session, List.of("/data/events/a.vortex", "/data/events/b.vortex"), Map.of());
```

The data source opens the first match immediately and opens the other files when the scan
reaches them.

## Projection and Filters

`ScanOptions.builder()` configures a scan. A projection selects columns, and a filter selects
rows. The native scan evaluates both, so it reads only the data that it needs.

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
and typed literals such as decimals, dates, timestamps, and UUIDs. `Expression.column(String[])`
selects a nested struct field.

## Row Selection and Limits

Scan options can also select rows by position or limit the row count:

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

A scan decodes several splits of each partition concurrently. By default, the number of
splits scales with the number of CPU cores. When many scans run concurrently, set a lower limit
with `splitConcurrency` to use less memory:

```java
source.scan(ScanOptions.builder().splitConcurrency(2).build());
```

This option does not limit all native memory or IO read-ahead.

## Object Storage

A path can be an object storage URI, for example `s3://bucket/path/file.vortex` or
`abfss://container@account.dfs.core.windows.net/path/file.vortex`. Pass credentials and other
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
Without credentials in the map, the object store uses the credentials from the environment.

### Custom IO

To read or write through your own IO layer, implement `NativeReadable` or `NativeWritable` from
`dev.vortex.io` and pass it to `DataSource.open` or `VortexWriter.builder`. The native reader
calls `NativeReadable.readFully` from many threads concurrently, so the implementation must be
thread-safe. Native code never closes a readable or a writable. Close a readable after you
close its data source and all scans and readers of that data source. Close a writable after
the writer finishes.

`DataSource.open(session, readables, readConcurrency)` sets the maximum number of concurrent
`readFully` calls. The value `0` selects the default.

The native reader merges nearby reads into one request. By default, it merges reads that are
at most 1 MiB apart, up to a span of 16 MiB. Override `coalesceDistance()` and
`coalesceMaxSize()` in your `NativeReadable` to change these limits. A larger gap results in
fewer requests but more bytes read.

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
returns the statistics of each top-level column, in Arrow schema order. Both methods also
accept a `NativeReadable`. `NativeFiles` also has `listFiles` and `delete`.

## Spark

The Spark connector reads and writes Vortex files through the JNI library. Add the `all` JAR,
which bundles its dependencies, to Spark:

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
projections and many filters down to the native scan. The connector does all file IO through
Hadoop, so configure remote storage in the Spark Hadoop configuration. The properties in
[Object Storage](#object-storage) do not apply to Spark.

The format is also available in Spark SQL:

```sql
CREATE TEMPORARY VIEW people
USING vortex
OPTIONS (path '/tmp/people.vortex');

SELECT name, age FROM people WHERE age > 30;
```

The [Spark user guide](../user-guide/spark.md) covers the build dependency setup, tables,
direct file queries, read and write options, supported types, and remote storage.

## Next Steps

- The [Java API reference](../api/java/index.rst) contains the Javadoc for `vortex-jni` and
  `vortex-spark`.
- [Concepts](../concepts/index.md) explains Vortex arrays, layouts, and scans.
