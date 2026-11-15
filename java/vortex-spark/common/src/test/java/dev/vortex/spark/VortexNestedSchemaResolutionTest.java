// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.c.ArrowArray;
import dev.vortex.relocated.org.apache.arrow.c.ArrowSchema;
import dev.vortex.relocated.org.apache.arrow.c.Data;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.UInt4Vector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests that nested fields of a supplied schema resolve against each file by name, the way Parquet resolves them.
 *
 * <p>Every file holds {@code id} and {@code s = {a, b, t: {a, b}}}, with {@code s} null in one row and {@code s.t} in
 * another. Each query reads the same rows from Parquet and Vortex with the same supplied schema, with nested schema
 * pruning on and off. The tests of columns Spark cannot read write their own files through the native writer.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaResolutionTest {
    private static final String PRUNING = "spark.sql.optimizer.nestedSchemaPruning.enabled";
    private static final String CASE_SENSITIVE = "spark.sql.caseSensitive";

    private SparkSession spark;
    private String parquetPath;
    private String vortexPath;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaResolutionTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        Dataset<Row> rows = spark.range(4)
                .selectExpr(
                        "id",
                        "IF(id = 1, NULL, named_struct('a', id + 10, 'b', id + 20, 't', IF(id = 2, NULL,"
                                + " named_struct('a', id + 30, 'b', id + 40)))) AS s");
        parquetPath = tempDir.resolve("parquet").toString();
        vortexPath = tempDir.resolve("vortex").toString();
        rows.write().parquet(parquetPath);
        rows.write().format("vortex").save(vortexPath);
    }

    @AfterEach
    public void resetConf() {
        spark.conf().unset(PRUNING);
        spark.conf().unset(CASE_SENSITIVE);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @ParameterizedTest(name = "caseSensitive={0}")
    @ValueSource(booleans = {false, true})
    @DisplayName("A struct field named in another case resolves under Spark's case sensitivity")
    public void fieldInAnotherCase(boolean caseSensitive) {
        spark.conf().set(CASE_SENSITIVE, caseSensitive);
        String schema = "id BIGINT, s STRUCT<A: BIGINT, B: BIGINT, t: STRUCT<a: BIGINT, b: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.B"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.B", "s.t.a"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
        assertMatchesParquet(schema, data -> data.where("s.B > 20").selectExpr("id", "s.t.b"));
    }

    @ParameterizedTest(name = "caseSensitive={0}")
    @ValueSource(booleans = {false, true})
    @DisplayName("A field of a struct inside a struct named in another case resolves under Spark's case sensitivity")
    public void nestedFieldInAnotherCase(boolean caseSensitive) {
        spark.conf().set(CASE_SENSITIVE, caseSensitive);
        String schema = "id BIGINT, s STRUCT<a: BIGINT, b: BIGINT, t: STRUCT<A: BIGINT, B: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.B"));
        assertMatchesParquet(schema, data -> data.where("s.t.B > 40").selectExpr("id", "s.a"));
        if (caseSensitive) {
            // `t` keeps its nulls with none of its read fields in the file. Parquet does too from Spark 4.1.
            assertRows(
                    schema,
                    data -> data.selectExpr("id", "s.a", "s.t"),
                    "[[0,10,[null,null]], [1,null,null], [2,12,null], [3,13,[null,null]]]");
        } else {
            assertMatchesParquet(schema, data -> data.selectExpr("id", "s.a", "s.t"));
        }
    }

    @Test
    @DisplayName("A supplied struct with fewer fields than the file reads them by name")
    public void narrowerStruct() {
        String schema = "id BIGINT, s STRUCT<b: BIGINT, t: STRUCT<b: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
    }

    @Test
    @DisplayName("A supplied struct with its fields in another order reads them by name")
    public void reorderedStruct() {
        String schema = "id BIGINT, s STRUCT<t: STRUCT<b: BIGINT, a: BIGINT>, b: BIGINT, a: BIGINT>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.b", "s.a"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
    }

    @Test
    @DisplayName("A nullable struct field the file does not carry reads as null")
    public void absentField() {
        String schema = "id BIGINT, s STRUCT<a: BIGINT, b: BIGINT, t: STRUCT<a: BIGINT, b: BIGINT, c: BIGINT>,"
                + " c: BIGINT, u: STRUCT<x: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.c"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.c", "s.t.a"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.u.x", "s.u IS NULL", "s.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
        assertMatchesParquet(
                schema, data -> data.where("s.c IS NULL AND s.t.c IS NULL").selectExpr("id", "s.a"));
    }

    @Test
    @DisplayName("Files whose structs differ in field order and in fields read by name in one query")
    public void filesWithDifferentStructs(@TempDir Path tempDir) {
        String parquet = tempDir.resolve("parquet").toString();
        String vortex = tempDir.resolve("vortex").toString();
        Dataset<Row> first = spark.range(0, 2).selectExpr("id", "named_struct('x', id + 10, 'y', id + 20) AS c");
        Dataset<Row> second = spark.range(2, 4)
                .selectExpr("id", "IF(id = 3, NULL, named_struct('y', id + 20, 'z', id + 30, 'x', id + 10)) AS c");
        for (Dataset<Row> rows : List.of(first, second)) {
            rows.write().mode("append").parquet(parquet);
            rows.write().mode("append").format("vortex").save(vortex);
        }

        String schema = "id BIGINT, c STRUCT<x: BIGINT, y: BIGINT, z: BIGINT>";
        assertRows(
                vortex,
                schema,
                data -> data.selectExpr("id", "c"),
                "[[0,[10,20,null]], [1,[11,21,null]], [2,[12,22,32]], [3,null]]");
        assertMatchesParquet(parquet, vortex, schema, data -> data.selectExpr("id", "c"));
        assertMatchesParquet(parquet, vortex, schema, data -> data.selectExpr("id", "c.y", "c.z"));
        assertMatchesParquet(
                parquet, vortex, schema, data -> data.where("c.x > 10").selectExpr("id", "c.y"));
    }

    @Test
    @DisplayName("A struct read over a field the file stores as another type fails and names the field")
    public void structOverAnotherType(@TempDir Path tempDir) {
        String path = tempDir.resolve("vortex").toString();
        spark.range(2)
                .selectExpr("id", "named_struct('a', id) AS s")
                .write()
                .format("vortex")
                .save(path);

        Exception thrown = assertThrows(Exception.class, () -> spark.read()
                .schema("id BIGINT, s STRUCT<a: STRUCT<x: BIGINT>>")
                .format("vortex")
                .load(path)
                .selectExpr("id", "s.a.x")
                .collectAsList());

        String expected = "stores s.a as Int(64, true), but the query reads it as a struct";
        boolean found = false;
        for (Throwable cause = thrown; cause != null && !found; cause = cause.getCause()) {
            found = cause.getMessage() != null && cause.getMessage().contains(expected);
        }
        assertTrue(found, () -> "expected a cause containing \"" + expected + "\" in " + thrown);
    }

    @Test
    @DisplayName("A struct field Spark cannot read does not fail a query that reads its siblings")
    public void unreadStructFieldSparkCannotRead(@TempDir Path tempDir) throws IOException {
        Schema schema = new Schema(List.of(
                Field.nullable("id", new ArrowType.Int(64, true)),
                new Field(
                        "s",
                        FieldType.nullable(ArrowType.Struct.INSTANCE),
                        List.of(
                                Field.nullable("a", new ArrowType.Int(64, true)),
                                Field.nullable("b", new ArrowType.Int(64, true)),
                                Field.nullable("u", new ArrowType.Int(32, false))))));
        String path = tempDir.resolve("unsigned.vortex").toString();
        writeOneRow(path, schema, root -> {
            ((BigIntVector) root.getVector("id")).setSafe(0, 0);
            StructVector s = (StructVector) root.getVector("s");
            ((BigIntVector) s.getChild("a")).setSafe(0, 10);
            ((BigIntVector) s.getChild("b")).setSafe(0, 20);
            ((UInt4Vector) s.getChild("u")).setSafe(0, 30);
            s.setIndexDefined(0);
        });
        String readSchema = "id BIGINT, s STRUCT<a: BIGINT, b: BIGINT>";
        assertRows(path, readSchema, data -> data.selectExpr("id", "s.b"), "[[0,20]]");
        assertRows(path, readSchema, data -> data.selectExpr("id", "s"), "[[0,[10,20]]]");
    }

    @Test
    @DisplayName("Partition columns come from the path under Spark's case sensitivity, without reading data columns")
    public void partitionColumns(@TempDir Path tempDir) throws IOException {
        Schema schema = new Schema(List.of(Field.nullable("u", new ArrowType.Int(32, false))));
        for (String dir : List.of("data/p=7", "ambiguous/p=7/P=8")) {
            Files.createDirectories(tempDir.resolve(dir));
            writeOneRow(tempDir.resolve(dir + "/f.vortex").toString(), schema, root -> {
                ((UInt4Vector) root.getVector("u")).setSafe(0, 1);
            });
        }
        String path = tempDir.resolve("data").toString();
        assertRows(path, "p INT", data -> data.selectExpr("p AS id"), "[[7]]");
        assertRows(path, "P INT", data -> data.selectExpr("P AS id"), "[[7]]");
        assertEquals(1, spark.read().schema("p INT").format("vortex").load(path).count());
        String ambiguous = tempDir.resolve("ambiguous").toString();
        assertThrows(AnalysisException.class, () -> spark.read()
                .schema("p INT")
                .format("vortex")
                .load(ambiguous)
                .collectAsList());
    }

    private static void writeOneRow(String path, Schema schema, Consumer<VectorSchemaRoot> fill) throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
                VortexWriter writer = VortexWriter.builder(Session.create(), path, schema, allocator)
                        .build();
                ArrowArray array = ArrowArray.allocateNew(allocator);
                ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
            root.allocateNew();
            fill.accept(root);
            root.setRowCount(1);
            Data.exportVectorSchemaRoot(allocator, root, null, array, arrowSchema);
            writer.writeBatch(array.memoryAddress(), arrowSchema.memoryAddress());
        }
    }

    /** Asserts that {@code query} returns {@code expected} from Vortex, with pruning off and on. */
    private void assertRows(String schema, Function<Dataset<Row>, Dataset<Row>> query, String expected) {
        assertRows(vortexPath, schema, query, expected);
    }

    private void assertRows(String path, String schema, Function<Dataset<Row>, Dataset<Row>> query, String expected) {
        for (boolean pruning : new boolean[] {false, true}) {
            spark.conf().set(PRUNING, pruning);
            List<Row> actual = query.apply(
                            spark.read().schema(schema).format("vortex").load(path))
                    .orderBy("id")
                    .collectAsList();
            assertEquals(expected, actual.toString(), schema + " with pruning " + (pruning ? "on" : "off"));
        }
    }

    /** Asserts that {@code query} returns the same rows from Vortex as from Parquet, with pruning off and on. */
    private void assertMatchesParquet(String schema, Function<Dataset<Row>, Dataset<Row>> query) {
        assertMatchesParquet(parquetPath, vortexPath, schema, query);
    }

    private void assertMatchesParquet(
            String parquet, String vortex, String schema, Function<Dataset<Row>, Dataset<Row>> query) {
        for (boolean pruning : new boolean[] {false, true}) {
            spark.conf().set(PRUNING, pruning);
            List<Row> expected = query.apply(spark.read().schema(schema).parquet(parquet))
                    .orderBy("id")
                    .collectAsList();
            List<Row> actual = query.apply(
                            spark.read().schema(schema).format("vortex").load(vortex))
                    .orderBy("id")
                    .collectAsList();
            assertEquals(expected, actual, schema + " with pruning " + (pruning ? "on" : "off"));
        }
    }
}
