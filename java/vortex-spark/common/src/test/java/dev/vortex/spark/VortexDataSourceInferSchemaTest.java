// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** Tests for the schema {@link VortexDataSourceV2} reads tables with. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDataSourceInferSchemaTest {

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexInferSchemaTest")
                .master("local[1]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Reading a directory without Vortex files reports schema inference failure")
    public void inferSchemaFailureUsesSparkFileSourceError() throws IOException {
        String emptyDir = Files.createDirectories(tempDir.resolve("empty")).toString();

        Throwable thrown = assertThrows(
                Throwable.class, () -> spark.read().format("vortex").load(emptyDir));

        String allMessages = messagesOf(thrown);
        assertTrue(allMessages.contains("Unable to infer schema"), "error should report schema inference failure");
        assertTrue(allMessages.contains("VORTEX"), "error should identify the Vortex file format");
    }

    @Test
    @DisplayName("A null struct reads correctly when schema inference picks a file without nulls")
    public void inferFromFileWithoutNulls() {
        assertFilesDifferingInNullabilityRead(false);
    }

    @Test
    @DisplayName("A null struct reads correctly when schema inference picks the file with nulls")
    public void inferFromFileWithNulls() {
        assertFilesDifferingInNullabilityRead(true);
    }

    @Test
    @DisplayName("A null struct reads correctly under a user schema that declares it non-nullable")
    public void userSchemaDeclaringNonNullable() {
        Column s = functions.struct(
                functions.col("id").plus(100).alias("a"),
                functions.col("id").plus(200).alias("b"));
        StructType nonNullable =
                spark.range(0, 1).select(functions.col("id"), s.alias("s")).schema();
        String withNull = write(
                spark.range(2, 4)
                        .select(
                                functions.col("id"),
                                functions
                                        .when(functions.col("id").equalTo(3), functions.lit(null))
                                        .otherwise(s)
                                        .alias("s")),
                "user-schema");

        Dataset<Row> table = spark.read().schema(nonNullable).format("vortex").load(withNull);

        assertEquals(
                "struct<id:bigint,s:struct<a:bigint,b:bigint>>", table.schema().simpleString());
        assertNoNonNullableFields(table.schema());
        assertEquals(
                Arrays.asList(Arrays.asList(2L, 202L, false), Arrays.asList(3L, null, true)),
                rowsOf(table.select(
                        functions.col("id"),
                        functions.col("s.b"),
                        functions.col("s").isNull())));
    }

    /**
     * Reads two files that differ only in whether struct {@code s} has nulls. {@code inferFromNulls} lists the file
     * with nulls last, so both orders of the footers that schema inference reads are covered.
     */
    private void assertFilesDifferingInNullabilityRead(boolean inferFromNulls) {
        Column s = functions.struct(
                functions.col("id").plus(100).alias("a"),
                functions.col("id").plus(200).alias("b"));
        String noNulls =
                write(spark.range(0, 2).select(functions.col("id"), s.alias("s")), "no-nulls-" + inferFromNulls);
        String withNull = write(
                spark.range(2, 4)
                        .select(
                                functions.col("id"),
                                functions
                                        .when(functions.col("id").equalTo(3), functions.lit(null))
                                        .otherwise(s)
                                        .alias("s")),
                "with-null-" + inferFromNulls);

        Dataset<Row> table = inferFromNulls
                ? spark.read().format("vortex").load(noNulls, withNull)
                : spark.read().format("vortex").load(withNull, noNulls);

        assertEquals(
                "struct<id:bigint,s:struct<a:bigint,b:bigint>>", table.schema().simpleString());
        assertNoNonNullableFields(table.schema());
        assertEquals(
                Arrays.asList(
                        Arrays.asList(0L, 200L, false),
                        Arrays.asList(1L, 201L, false),
                        Arrays.asList(2L, 202L, false),
                        Arrays.asList(3L, null, true)),
                rowsOf(table.select(
                        functions.col("id"),
                        functions.col("s.b"),
                        functions.col("s").isNull())));
    }

    /** Asserts that no field, array element or map value in {@code schema} is non-nullable. */
    static void assertNoNonNullableFields(StructType schema) {
        String json = schema.json();
        for (String flag :
                new String[] {"\"nullable\":false", "\"containsNull\":false", "\"valueContainsNull\":false"}) {
            assertFalse(json.contains(flag), "expected every field to be nullable: " + json);
        }
    }

    private String write(Dataset<Row> rows, String name) {
        String path = tempDir.resolve(name).toUri().toString();
        rows.coalesce(1).write().format("vortex").mode(SaveMode.Overwrite).save(path);
        return path;
    }

    private static List<List<Object>> rowsOf(Dataset<Row> rows) {
        return rows.orderBy("id").collectAsList().stream()
                .map(row -> Arrays.asList(row.get(0), row.get(1), row.get(2)))
                .collect(Collectors.toList());
    }

    /** Concatenates the messages of the whole cause chain, since Spark may wrap data source exceptions. */
    private static String messagesOf(Throwable thrown) {
        StringBuilder sb = new StringBuilder();
        for (Throwable t = thrown; t != null; t = t.getCause()) {
            sb.append(t.getMessage()).append('\n');
        }
        return sb.toString();
    }
}
