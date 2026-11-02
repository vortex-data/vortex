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
import org.apache.spark.sql.catalyst.plans.logical.Statistics;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** COUNT(*) footer pushdown behavior and rejection cases. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexAggregatePushdownTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexAggregatePushdownTest")
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
    void countStarPushesDownAcrossMultipleFiles() {
        Dataset<Row> data = writeAndRead(321, "count", 5, false);
        Dataset<Row> count = data.selectExpr("count(*) AS count");

        assertEquals(321L, count.first().getLong(0));
        String plan = count.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("PushedAggregation: [COUNT(*)]"), plan);
    }

    @Test
    void filteredCountAndCountColumnAreRejected() {
        Dataset<Row> data = writeAndRead(30, "rejected", 3, false);
        Dataset<Row> filtered = data.filter("id > 10").selectExpr("count(*) AS count");
        Dataset<Row> countColumn = data.selectExpr("count(value) AS count");

        assertEquals(19L, filtered.first().getLong(0));
        assertFalse(filtered.queryExecution().executedPlan().toString().contains("PushedAggregation: [count(*)]"));
        assertEquals(30L, countColumn.first().getLong(0));
        assertFalse(
                countColumn.queryExecution().executedPlan().toString().contains("PushedAggregation: [count(value)]"));
    }

    @Test
    void countStarGroupsByHivePartitionValues() {
        Dataset<Row> data = writeAndRead(12, "grouped", 2, true);

        List<Row> rows = data.groupBy("group").count().orderBy("group").collectAsList();

        assertEquals(2, rows.size());
        assertEquals(0, rows.get(0).getInt(0));
        assertEquals(6L, rows.get(0).getLong(1));
        assertEquals(1, rows.get(1).getInt(0));
        assertEquals(6L, rows.get(1).getLong(1));
    }

    @Test
    void countStarPushdownCanBeDisabled() {
        Path output = tempDir.resolve("pushdown_off");
        spark.range(0, 47)
                .selectExpr("cast(id as int) as id")
                .repartition(3)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());

        Dataset<Row> count = spark.read()
                .format("vortex")
                .option("vortex.aggregatePushdown", "false")
                .load(output.toString())
                .selectExpr("count(*) AS count");

        assertEquals(47L, count.first().getLong(0));
        String plan = count.queryExecution().executedPlan().toString();
        assertFalse(plan.contains("PushedAggregation: [COUNT(*)]"), plan);
    }

    @Test
    void aPushedCountReportsTheRowsItEmitsAndReadsNoFooterTwice() {
        Dataset<Row> data = writeAndRead(400, "pushed_stats", 4, false);
        Dataset<Row> count = data.selectExpr("count(*) AS count");

        Statistics statistics =
                count.queryExecution().optimizedPlan().collectLeaves().head().stats();

        // The reader answers from one footer per file and emits one row for each. Summing those footers here as
        // well would pay for every footer twice and describe rows the scan never emits.
        assertTrue(statistics.rowCount().isDefined());
        assertEquals(4L, statistics.rowCount().get().longValue());
        assertEquals(400L, count.first().getLong(0));
    }

    private Dataset<Row> writeAndRead(int count, String name, int partitions, boolean partitioned) {
        Path output = tempDir.resolve(name);
        Dataset<Row> data = spark.range(0, count)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id as string)) as value",
                        "cast(id % 2 as int) as group");
        if (partitioned) {
            data.repartition(partitions)
                    .write()
                    .format("vortex")
                    .partitionBy("group")
                    .mode(SaveMode.Overwrite)
                    .save(output.toString());
        } else {
            data.repartition(partitions)
                    .write()
                    .format("vortex")
                    .mode(SaveMode.Overwrite)
                    .save(output.toString());
        }
        return spark.read().format("vortex").load(output.toString());
    }
}
