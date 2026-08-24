// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
 * Reads that need no data column at all.
 *
 * <p>A query over partition columns alone, and a count Spark declines to push down, both leave the read data schema
 * empty. The scan must still push an empty projection, or it pulls every column off storage to answer them.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexProjectionTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexProjectionTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "false")
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
    void selectingOnlyAPartitionColumnReadsNoDataColumn() {
        Dataset<Row> data = writeAndRead("partition_only", true).select("group");

        List<Row> rows = data.collectAsList();

        assertEquals(30, rows.size());
        assertEquals(15, rows.stream().filter(row -> row.getInt(0) == 0).count());
        assertEquals(15, rows.stream().filter(row -> row.getInt(0) == 1).count());
        // No data column is named anywhere in the scan, so nothing but the partition value was read.
        assertFalse(plan(data).contains("value"), plan(data));
    }

    @Test
    void countWithAggregatePushdownOffStillCountsEveryRow() {
        Path output = tempDir.resolve("count_no_pushdown");
        write(output, false);
        Dataset<Row> data = spark.read()
                .format("vortex")
                .option("vortex.aggregatePushdown", "false")
                .load(output.toString());

        Dataset<Row> count = data.selectExpr("count(*) AS count");

        assertEquals(30L, count.first().getLong(0));
        assertTrue(plan(count).contains("PushedAggregation: []"), plan(count));
    }

    @Test
    void constantProjectionStillReportsEveryRow() {
        Dataset<Row> data = writeAndRead("constant", false);

        assertEquals(30, data.selectExpr("1 AS one").collectAsList().size());
    }

    private String plan(Dataset<Row> data) {
        return data.queryExecution().executedPlan().toString();
    }

    private Dataset<Row> writeAndRead(String name, boolean partitioned) {
        Path output = tempDir.resolve(name);
        write(output, partitioned);
        return spark.read().format("vortex").load(output.toString());
    }

    private void write(Path output, boolean partitioned) {
        Dataset<Row> data = spark.range(0, 30)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id as string)) as value",
                        "cast(id % 2 as int) as group")
                .repartition(3);
        if (partitioned) {
            data.write()
                    .format("vortex")
                    .partitionBy("group")
                    .mode(SaveMode.Overwrite)
                    .save(output.toString());
        } else {
            data.write().format("vortex").mode(SaveMode.Overwrite).save(output.toString());
        }
    }
}
