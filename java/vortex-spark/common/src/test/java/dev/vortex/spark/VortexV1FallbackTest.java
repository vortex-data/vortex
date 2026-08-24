// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * The V1 {@link VortexFileFormat} path.
 *
 * <p>Catalog tables reach it, and so does {@code spark.sql.sources.useV1SourceList}. Within it, Spark asks for rows
 * instead of batches when whole-stage codegen is off or the schema carries more fields than
 * {@code spark.sql.codegen.maxFields}, so both readers in the format need cover.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexV1FallbackTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexV1FallbackTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.sources.useV1SourceList", "vortex")
                .getOrCreate();
        // Set it again at runtime: `getOrCreate` reuses a session another test class in this JVM may have
        // built, and then it ignores the builder options.
        spark.conf().set("spark.sql.sources.useV1SourceList", "vortex");
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void readsAndWritesThroughTheV1FileFormat() {
        Path output = write("v1_roundtrip", false);
        Dataset<Row> data = spark.read().format("vortex").load(output.toString());

        // `FileScan vortex` is the V1 plan node. The V2 path would read `BatchScan`.
        assertTrue(plan(data).contains("FileScan vortex"), plan(data));
        assertEquals(40, data.count());
        assertEquals(
                List.of(0, 1, 2),
                data.orderBy("id").limit(3).collectAsList().stream()
                        .map(row -> row.getInt(0))
                        .toList());
    }

    @Test
    void prunesColumnsAndPushesFiltersThroughTheV1FileFormat() {
        Path output = write("v1_pushdown", false);

        Dataset<Row> data = spark.read()
                .format("vortex")
                .load(output.toString())
                .select("value")
                .filter("value = 'value_7'");

        assertEquals(1, data.count());
        assertEquals("value_7", data.first().getString(0));
    }

    @Test
    void readsHivePartitionValuesThroughTheV1FileFormat() {
        Path output = write("v1_partitioned", true);

        Dataset<Row> data = spark.read().format("vortex").load(output.toString());

        assertEquals(40, data.count());
        assertEquals(20, data.filter("grp = 0").count());
        assertEquals(20, data.select("grp").filter("grp = 1").count());
    }

    @Test
    void readsRowsWhenSparkDeclinesTheColumnarPath() {
        Path output = write("v1_rows", true);

        // Whole-stage codegen off makes `FileSourceScanExec` ask the format for rows, not batches.
        spark.conf().set("spark.sql.codegen.wholeStage", "false");
        try {
            Dataset<Row> data = spark.read().format("vortex").load(output.toString());

            assertEquals(40, data.count());
            List<Row> rows = data.orderBy("id").limit(2).collectAsList();
            assertEquals(0, intOf(rows.get(0), "id"));
            assertEquals("value_0", rows.get(0).getString(rows.get(0).fieldIndex("value")));
            // Partition values are joined onto the row by Spark, not by the reader.
            assertEquals(0, intOf(rows.get(0), "grp"));
            assertEquals(1, intOf(rows.get(1), "id"));
            assertEquals(20, data.filter("grp = 1").count());
        } finally {
            spark.conf().unset("spark.sql.codegen.wholeStage");
        }
    }

    private static int intOf(Row row, String name) {
        return row.getInt(row.fieldIndex(name));
    }

    private String plan(Dataset<Row> data) {
        return data.queryExecution().executedPlan().toString();
    }

    private Path write(String name, boolean partitioned) {
        Path output = tempDir.resolve(name);
        Dataset<Row> data = spark.range(0, 40)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id as string)) as value",
                        "cast(id % 2 as int) as grp")
                .repartition(2);
        if (partitioned) {
            data.write()
                    .format("vortex")
                    .partitionBy("grp")
                    .mode(SaveMode.Overwrite)
                    .save(output.toString());
        } else {
            data.write().format("vortex").mode(SaveMode.Overwrite).save(output.toString());
        }
        return output;
    }
}
