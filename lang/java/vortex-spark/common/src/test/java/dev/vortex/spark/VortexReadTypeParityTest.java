// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.c.ArrowArray;
import dev.vortex.relocated.org.apache.arrow.c.ArrowSchema;
import dev.vortex.relocated.org.apache.arrow.c.Data;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Reading a column with a supplied type other than the one the file stores acts like Parquet.
 *
 * <p>Each case writes a column of one type to both formats and reads it back with another. Where Parquet fails, Vortex
 * must fail too, naming the column and the type asked for; where both return rows, the rows must match. Parquet's rules
 * change between Spark versions, so a case may let Vortex refuse a conversion Parquet makes.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexReadTypeParityTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexReadTypeParityTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.session.timeZone", "UTC")
                // Microseconds, as Vortex stores timestamps, rather than the INT96 default.
                .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
                .getOrCreate();
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    /**
     * One write and read. {@code write} selects {@code id} and {@code x}, and {@code readType} is the type {@code x} is
     * read with. A Vortex error must name {@code path}, the field whose type differs, and {@code pathType}, the type it
     * is read with.
     */
    record Case(
            String name,
            String write,
            String readType,
            String path,
            String pathType,
            String query,
            boolean vortexMayRefuse) {
        @Override
        public String toString() {
            return name;
        }
    }

    private static Case leaf(String writeType, String first, String second, String readType) {
        return leaf(writeType, first, second, readType, false);
    }

    private static Case leaf(String writeType, String first, String second, String readType, boolean mayRefuse) {
        return new Case(
                writeType + " as " + readType,
                values(writeType, first, second),
                readType,
                "x",
                readType,
                "SELECT id, x FROM t",
                mayRefuse);
    }

    private static String values(String writeType, String first, String second) {
        return String.format(
                Locale.ROOT,
                "SELECT CAST(id AS BIGINT) AS id, CAST(x AS %s) AS x FROM VALUES (0, %s), (1, %s), (2, NULL) AS v(id, x)",
                writeType,
                first,
                second);
    }

    static Stream<Case> cases() {
        Stream<Case> collated =
                org.apache.spark.package$.MODULE$.SPARK_VERSION().startsWith("3.")
                        ? Stream.empty()
                        : Stream.of(leaf("STRING", "'abc'", "'ABC'", "STRING COLLATE UTF8_LCASE"));
        return Stream.concat(common(), collated);
    }

    private static Stream<Case> common() {
        String longs = values("BIGINT", "1000000", "-5");
        return Stream.of(
                leaf("BIGINT", "1000000", "-5", "TIMESTAMP"),
                leaf(
                        "TIMESTAMP",
                        "TIMESTAMP '2026-10-29 12:34:56.789'",
                        "TIMESTAMP '1969-12-31 23:59:59'",
                        "TIMESTAMP_NTZ"),
                leaf(
                        "TIMESTAMP_NTZ",
                        "TIMESTAMP_NTZ '2026-10-29 12:34:56.789'",
                        "TIMESTAMP_NTZ '1969-12-31 23:59:59'",
                        "TIMESTAMP"),
                leaf("TIMESTAMP", "TIMESTAMP '2026-10-29 12:34:56.789'", "TIMESTAMP '1969-12-31 23:59:59'", "BIGINT"),
                leaf("INT", "20000", "-1", "DATE"),
                leaf("DECIMAL(10,2)", "12345678.91", "-0.01", "DECIMAL(12,2)"),
                leaf("DECIMAL(10,2)", "12345678.91", "-0.01", "DECIMAL(12,1)"),
                new Case(
                        "struct field BIGINT as TIMESTAMP",
                        "SELECT id, named_struct('a', x, 'b', CAST(id AS STRING)) AS x FROM (" + longs + ")",
                        "STRUCT<a: TIMESTAMP, b: STRING>",
                        "x.a",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "ARRAY<BIGINT> as ARRAY<TIMESTAMP>",
                        "SELECT id, IF(x IS NULL, NULL, array(x, NULL)) AS x FROM (" + longs + ")",
                        "ARRAY<TIMESTAMP>",
                        "x.element",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "MAP<STRING, BIGINT> as MAP<STRING, TIMESTAMP>",
                        "SELECT id, IF(x IS NULL, NULL, map('k', x)) AS x FROM (" + longs + ")",
                        "MAP<STRING, TIMESTAMP>",
                        "x.value",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "array struct field BIGINT as TIMESTAMP",
                        "SELECT id, IF(x IS NULL, NULL, array(named_struct('a', x, 'b', CAST(id AS STRING)))) AS x"
                                + " FROM (" + longs + ")",
                        "ARRAY<STRUCT<a: TIMESTAMP, b: STRING>>",
                        "x.element.a",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                // The vectors return structs inside arrays by position, so swapped fields of one type must not be
                // read as each other. Parquet matches them by name, which Vortex may refuse to do.
                new Case(
                        "array struct fields in another order",
                        "SELECT id, IF(x IS NULL, NULL, array(named_struct('a', x, 'b', x + 1))) AS x FROM (" + longs
                                + ")",
                        "ARRAY<STRUCT<b: BIGINT, a: BIGINT>>",
                        "x.element",
                        "STRUCT<b: BIGINT, a: BIGINT>",
                        "SELECT id, x FROM t",
                        true),
                new Case(
                        "filtered BIGINT as TIMESTAMP",
                        longs,
                        "TIMESTAMP",
                        "x",
                        "TIMESTAMP",
                        "SELECT id FROM t WHERE x > TIMESTAMP '1970-01-01 00:00:00.5'",
                        false));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void aColumnReadAsAnotherTypeActsLikeParquet(Case c) {
        String slug = c.name().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-");
        Outcome parquet = run(c, "parquet", slug);
        Outcome vortex = run(c, "vortex", slug);
        if (parquet.failure() != null) {
            assertTrue(vortex.failure() != null, "Parquet fails with " + parquet + " but Vortex returns " + vortex);
        } else if (vortex.failure() == null) {
            assertEquals(parquet.rows(), vortex.rows(), "rows");
        } else if (!c.vortexMayRefuse()) {
            fail("Parquet returns " + parquet + " but Vortex fails with " + vortex);
        }
        if (vortex.failure() != null) {
            String message = vortex.failure();
            String readType = spark.sql("SELECT CAST(NULL AS " + c.pathType() + ")")
                    .schema()
                    .fields()[0]
                    .dataType()
                    .catalogString();
            assertTrue(
                    message.contains("column " + c.path() + " ") && message.contains("read as " + readType),
                    "the error names column " + c.path() + " and type " + readType + ": " + message);
            assertTrue(message.contains(".vortex"), "the error names the file: " + message);
        }
    }

    private record Outcome(List<String> rows, String failure) {
        @Override
        public String toString() {
            return failure != null ? "FAILS " + failure : rows.toString();
        }
    }

    private Outcome run(Case c, String format, String slug) {
        String root = tempDir.resolve(slug + "-" + format).toString();
        spark.sql(c.write()).coalesce(1).write().format(format).save(root);
        try {
            spark.read()
                    .format(format)
                    .schema("id BIGINT, x " + c.readType())
                    .load(root)
                    .createOrReplaceTempView("t");
            return new Outcome(spark.sql(c.query()).orderBy("id").toJSON().collectAsList(), null);
        } catch (Exception e) {
            // Spark rethrows a failed job's SparkException unchecked.
            return new Outcome(null, describe(e));
        }
    }

    /** Every message in the cause chain, root cause last, so a check sees past Spark's wrappers. */
    private static String describe(Throwable failure) {
        StringBuilder messages = new StringBuilder();
        Throwable root = failure;
        for (Throwable t = failure; t != null; t = t.getCause()) {
            messages.append(t.getMessage()).append(" | ");
            root = t;
        }
        String rootMessage =
                String.valueOf(root.getMessage()).lines().findFirst().orElse("");
        return root.getClass().getSimpleName() + ": " + rootMessage + " || " + messages;
    }

    @Test
    void aTimestampFilterReadsFilesOfAnyUnitAndZone() throws IOException {
        // The native scan compares timestamps only in the unit and zone of the filter literal.
        Path dir = tempDir.resolve("timestamps");
        writeTimestamps(dir.resolve("a.vortex"), new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC"), 0, 1_000L);
        writeTimestamps(
                dir.resolve("b.vortex"),
                new ArrowType.Timestamp(TimeUnit.NANOSECOND, "America/New_York"),
                1,
                3_000_000_000L);
        writeTimestamps(
                dir.resolve("c.vortex"), new ArrowType.Timestamp(TimeUnit.MICROSECOND, "+01:00"), 2, 5_000_000L);

        List<Row> rows = spark.read()
                .format("vortex")
                .schema("id BIGINT, ts TIMESTAMP")
                .load(dir.toString())
                .where("ts > TIMESTAMP '1970-01-01 00:00:01.5'")
                .selectExpr("id", "unix_micros(ts)")
                .orderBy("id")
                .collectAsList();
        assertEquals("[[1,3000000], [2,5000000]]", rows.toString());
    }

    @Test
    void aTimestampInAnotherUnitIsNotReadAsItsMicroseconds() throws IOException {
        // Parquet reads a millisecond timestamp as BIGINT milliseconds, and the vector would return microseconds.
        Path dir = tempDir.resolve("millis");
        writeFile(
                dir.resolve("a.vortex"),
                new Schema(List.of(Field.nullable("x", new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC")))),
                1,
                root -> ((TimeStampVector) root.getVector("x")).setSafe(0, 1_000L));
        try {
            List<Row> rows = spark.read()
                    .format("vortex")
                    .schema("x BIGINT")
                    .load(dir.toString())
                    .collectAsList();
            fail("read a millisecond timestamp as BIGINT: " + rows);
        } catch (Exception e) {
            String message = describe(e);
            assertTrue(message.contains("column x as timestamp, which cannot be read as bigint"), message);
        }
    }

    private static void writeTimestamps(Path path, ArrowType.Timestamp type, long id, long value) throws IOException {
        writeFile(
                path,
                new Schema(List.of(Field.nullable("id", new ArrowType.Int(64, true)), Field.nullable("ts", type))),
                1,
                root -> {
                    ((BigIntVector) root.getVector("id")).setSafe(0, id);
                    ((TimeStampVector) root.getVector("ts")).setSafe(0, value);
                });
    }

    private static void writeFile(Path path, Schema schema, int rowCount, Consumer<VectorSchemaRoot> fill)
            throws IOException {
        Files.createDirectories(path.getParent());
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
                VortexWriter writer = VortexWriter.builder(Session.create(), path.toString(), schema, allocator)
                        .build();
                ArrowArray array = ArrowArray.allocateNew(allocator);
                ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
            root.allocateNew();
            fill.accept(root);
            root.setRowCount(rowCount);
            Data.exportVectorSchemaRoot(allocator, root, null, array, arrowSchema);
            writer.writeBatch(array.memoryAddress(), arrowSchema.memoryAddress());
        }
    }
}
