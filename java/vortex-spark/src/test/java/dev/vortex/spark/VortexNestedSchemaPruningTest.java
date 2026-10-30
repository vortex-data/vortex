// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests that reading one field of a nested column returns that field.
 *
 * <p>With {@code spark.sql.optimizer.nestedSchemaPruning.enabled}, Spark asks the scan for only the nested fields a
 * query uses. Each row of the table holds {@code a = id + 100}, {@code b = id + 200} and {@code c = id + 300}, so a
 * field read from the wrong position shows up as the wrong hundred.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaPruningTest {

    private static final List<Long> C_VALUES = List.of(300L, 301L, 302L, 303L, 304L);

    private SparkSession spark;
    private Dataset<Row> table;
    private Path tempDir;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        this.tempDir = tempDir;
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaPruningTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.optimizer.nestedSchemaPruning.enabled", "true")
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        Column abc = functions.struct(
                functions.col("id").plus(100).alias("a"),
                functions.col("id").plus(200).alias("b"),
                functions.col("id").plus(300).alias("c"));
        Dataset<Row> df = spark.range(5)
                .select(
                        functions.col("id"),
                        abc.alias("s"),
                        functions.array(abc).alias("arr"),
                        functions.map(functions.lit("k"), abc).alias("m"),
                        functions
                                .struct(functions.lit("x").alias("p"), abc.alias("inner"))
                                .alias("outer"),
                        nullWhen(
                                        functions.col("id").equalTo(1),
                                        functions.struct(
                                                functions.lit("x").alias("p"),
                                                nullWhen(functions.col("id").equalTo(2), abc)
                                                        .alias("inner")))
                                .alias("nullable_outer"),
                        nullWhen(
                                        functions.col("id").mod(2).equalTo(1),
                                        functions.struct(
                                                functions.col("id").plus(100).alias("a"),
                                                nullWhen(
                                                                functions
                                                                        .col("id")
                                                                        .equalTo(4),
                                                                functions
                                                                        .col("id")
                                                                        .plus(200))
                                                        .alias("b")))
                                .alias("n"));

        String path = tempDir.resolve("nested").toUri().toString();
        df.write()
                .format("vortex")
                .option("path", path)
                .mode(SaveMode.Overwrite)
                .save();
        table = spark.read().format("vortex").option("path", path).load();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Selecting a struct field other than the first returns that field")
    public void structField() {
        assertEquals(C_VALUES, valuesOf(table, functions.col("s.c")));
    }

    @Test
    @DisplayName("Selecting a field of the structs in an array returns that field")
    public void arrayElementField() {
        assertEquals(C_VALUES, valuesOf(table, functions.element_at(functions.col("arr.c"), 1)));
    }

    @Test
    @DisplayName("Selecting a field of the structs in a map returns that field")
    public void mapValueField() {
        assertEquals(C_VALUES, valuesOf(table, functions.col("m").getItem("k").getField("c")));
    }

    @Test
    @DisplayName("Selecting a field of a struct inside a struct returns that field")
    public void structInStructField() {
        assertEquals(C_VALUES, valuesOf(table, functions.col("outer.inner.c")));
    }

    @Test
    @DisplayName("Null structs and null fields stay null when the struct is pruned")
    public void nullableStructField() {
        // Selecting `n.b` next to `n IS NULL` makes Spark read the pruned `n`, so the null check sees its nulls.
        Dataset<Row> rows = table.select(
                functions.col("id"), functions.col("n.b"), functions.col("n").isNull());
        assertEquals("struct<id:bigint,n:struct<b:bigint>>", readSchema(rows));
        assertEquals(
                Arrays.asList(
                        Arrays.asList(0L, 200L, false),
                        Arrays.asList(1L, null, true),
                        Arrays.asList(2L, 202L, false),
                        Arrays.asList(3L, null, true),
                        Arrays.asList(4L, null, false)),
                rowsOf(rows));
    }

    @Test
    @DisplayName("Nulls at each level of a struct inside a struct survive pruning both levels")
    public void nullableStructInStructField() {
        Dataset<Row> rows = table.select(
                functions.col("id"),
                functions.col("nullable_outer.inner.c"),
                functions.col("nullable_outer").isNull(),
                functions.col("nullable_outer.inner").isNull());
        assertEquals("struct<id:bigint,nullable_outer:struct<inner:struct<c:bigint>>>", readSchema(rows));
        assertEquals(
                Arrays.asList(
                        Arrays.asList(0L, 300L, false, false),
                        Arrays.asList(1L, null, true, true),
                        Arrays.asList(2L, null, false, true),
                        Arrays.asList(3L, 303L, false, false),
                        Arrays.asList(4L, 304L, false, false)),
                rowsOf(rows));
    }

    @Test
    @DisplayName("A pushed filter on a field the query does not select still applies")
    public void filterOnPrunedField() {
        assertEquals(
                List.of(302L, 303L, 304L),
                valuesOf(table.where(functions.col("s.a").gt(101)), functions.col("s.c")));
    }

    @Test
    @DisplayName("Whole struct columns still read in full alongside pruned ones")
    public void wholeAndPrunedStructs() {
        Row row = table.where("id = 3")
                .select(functions.col("s"), functions.col("outer.inner.a"))
                .first();
        assertEquals(
                List.of(103L, 203L, 303L),
                List.of(
                        row.getStruct(0).getLong(0),
                        row.getStruct(0).getLong(1),
                        row.getStruct(0).getLong(2)));
        assertEquals(103L, row.getLong(1));
    }

    @ParameterizedTest(name = "nullable file inferred: {0}")
    @ValueSource(booleans = {true, false})
    @DisplayName("A pruned nested struct keeps its nulls across files that differ only in nullability")
    public void filesDifferingInNullability(boolean nullableInferred) {
        Column inner = functions.struct(
                functions.col("id").plus(100).alias("a"),
                functions.col("id").plus(200).alias("b"));
        Column o = functions.struct(functions.lit("x").alias("p"), inner.alias("inner"));
        String noNulls =
                write(spark.range(0, 2).select(functions.col("id"), o.alias("o")), "no-nulls-" + nullableInferred);
        String withNull = write(
                spark.range(2, 4)
                        .select(
                                functions.col("id"),
                                nullWhen(functions.col("id").equalTo(3), o).alias("o")),
                "with-null-" + nullableInferred);
        // Schema inference reads the last path.
        String[] paths = nullableInferred ? new String[] {noNulls, withNull} : new String[] {withNull, noNulls};
        Dataset<Row> rows = spark.read()
                .format("vortex")
                .load(paths)
                .select(
                        functions.col("id"),
                        functions.col("o.inner.b"),
                        functions.col("o").isNull());

        assertEquals("struct<id:bigint,o:struct<inner:struct<b:bigint>>>", readSchema(rows));
        assertEquals(
                Arrays.asList(
                        Arrays.asList(0L, 200L, false),
                        Arrays.asList(1L, 201L, false),
                        Arrays.asList(2L, 202L, false),
                        Arrays.asList(3L, null, true)),
                rowsOf(rows));
    }

    private String write(Dataset<Row> rows, String name) {
        String path = tempDir.resolve(name).toUri().toString();
        rows.coalesce(1).write().format("vortex").mode(SaveMode.Overwrite).save(path);
        return path;
    }

    private static Column nullWhen(Column condition, Column value) {
        return functions.when(condition, functions.lit(null)).otherwise(value);
    }

    /** Returns the schema Spark asks the Vortex scan for. */
    private static String readSchema(Dataset<Row> rows) {
        return rows.queryExecution()
                .optimizedPlan()
                .collectLeaves()
                .head()
                .schema()
                .simpleString();
    }

    /** Returns each row of {@code rows}, whose first column is the id, ordered by id. */
    private static List<List<Object>> rowsOf(Dataset<Row> rows) {
        return rows.orderBy("id").collectAsList().stream()
                .map(row -> IntStream.range(0, row.size()).mapToObj(row::get).collect(Collectors.toList()))
                .collect(Collectors.toList());
    }

    /** Returns {@code column} for each row of {@code rows}, ordered by id. */
    @SuppressWarnings("unchecked")
    private static <T> List<T> valuesOf(Dataset<Row> rows, Column column) {
        return rows.orderBy("id").select(column).collectAsList().stream()
                .map(row -> (T) row.get(0))
                .collect(Collectors.toList());
    }
}
