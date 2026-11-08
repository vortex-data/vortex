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
import org.apache.spark.sql.functions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** Aggregate pushdown behavior, answers, and rejection cases. */
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
    void countStarIsAnsweredFromTheFootersOnTheDriver() {
        Dataset<Row> data = writeAndRead(321, "count", 5, false);
        Dataset<Row> count = data.selectExpr("count(*) AS count");

        assertEquals(321L, count.first().getLong(0));
        String plan = count.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("LocalTableScan"), plan);
        assertFalse(plan.contains("BatchScan"), plan);
    }

    @Test
    void countStarPushesDownAcrossMultipleFiles() {
        writeAndRead(321, "partial_count", 5, false);
        Dataset<Row> count = spark.read()
                .format("vortex")
                .option("vortex.completeAggregatePushdown", "false")
                .load(tempDir.resolve("partial_count").toString())
                .selectExpr("count(*) AS count");

        assertEquals(321L, count.first().getLong(0));
        String plan = count.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("PushedAggregation: [COUNT(*)]"), plan);
    }

    @Test
    void filteredCountIsRejected() {
        Dataset<Row> data = writeAndRead(30, "rejected", 3, false);
        Dataset<Row> filtered = data.filter("id > 10").selectExpr("count(*) AS count");

        assertEquals(19L, filtered.first().getLong(0));
        assertFalse(filtered.queryExecution().executedPlan().toString().contains("PushedAggregation: [COUNT(*)]"));
    }

    @Test
    void columnAggregatesPushDownAndMatchAScan() {
        Path output = tempDir.resolve("column_aggregates");
        spark.range(0, 500)
                .selectExpr(
                        "cast(id as int) as id",
                        "case when id % 5 = 0 then null else concat('v', lpad(cast(id as string), 4, '0')) end as name",
                        "case when id % 7 = 0 then cast('NaN' as double) else cast(id as double) - 250 end as score",
                        "cast(id as decimal(10, 2)) / 4 as price",
                        "date_add(date'2024-01-01', cast(id as int)) as day",
                        "cast(id * 1000000000000 as long) as big")
                .repartition(4)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());

        String aggregates = "count(*), count(name), min(id), max(id), min(name), max(name), min(score), max(score), "
                + "min(price), max(price), min(day), max(day), sum(id), sum(big), avg(id)";
        withAnsi(false, () -> {
            Dataset<Row> pushed =
                    spark.read().format("vortex").load(output.toString()).selectExpr(aggregates.split(", "));
            Dataset<Row> scanned = spark.read()
                    .format("vortex")
                    .option("vortex.aggregatePushdown", "false")
                    .load(output.toString())
                    .selectExpr(aggregates.split(", "));

            String plan = pushed.queryExecution().executedPlan().toString();
            assertTrue(plan.contains("PushedAggregation: [COUNT(*), COUNT(name), MIN(id), MAX(id)"), plan);
            assertEquals(scanned.collectAsList(), pushed.collectAsList());
        });
    }

    @Test
    void aggregatesOverPartitionColumnsAndColumnsMissingFromOlderFiles() {
        Path output = tempDir.resolve("evolved");
        spark.range(0, 20)
                .selectExpr("cast(id as int) as id", "cast(id % 2 as int) as part")
                .write()
                .format("vortex")
                .partitionBy("part")
                .mode(SaveMode.Overwrite)
                .save(output.toString());
        spark.range(20, 30)
                .selectExpr("cast(id as int) as id", "cast(id * 10 as long) as added", "cast(id % 2 as int) as part")
                .write()
                .format("vortex")
                .partitionBy("part")
                .mode(SaveMode.Append)
                .save(output.toString());

        String[] aggregates = {"count(added)", "min(added)", "max(added)", "sum(added)", "min(part)", "max(part)"};
        Dataset<Row> pushed = spark.read().format("vortex").load(output.toString());
        Dataset<Row> scanned = spark.read()
                .format("vortex")
                .option("vortex.aggregatePushdown", "false")
                .load(output.toString());

        Dataset<Row> grouped = pushed.groupBy("part").agg(functions.expr("count(added)"), functions.expr("max(id)"));
        // The footers state every partial, so the driver merges the files of each partition into its final row.
        String groupedPlan = grouped.queryExecution().executedPlan().toString();
        assertTrue(groupedPlan.contains("LocalTableScan"), groupedPlan);
        assertEquals(
                scanned.groupBy("part")
                        .agg(functions.expr("count(added)"), functions.expr("max(id)"))
                        .orderBy("part")
                        .collectAsList(),
                grouped.orderBy("part").collectAsList());
        withAnsi(
                false,
                () -> assertEquals(
                        scanned.selectExpr(aggregates).collectAsList(),
                        pushed.selectExpr(aggregates).collectAsList()));
    }

    @Test
    void sumsOfNarrowColumnsPushDownUnderAnsiMode() {
        Path output = writeShiftedSumData("ansi_sums");
        withAnsi(true, () -> {
            Dataset<Row> pushed = spark.read().format("vortex").load(output.toString());
            Dataset<Row> scanned = spark.read()
                    .format("vortex")
                    .option("vortex.aggregatePushdown", "false")
                    .load(output.toString());

            // A sum of SMALLINT or INT values cannot overflow 64 bits in a file, so ANSI rules allow it.
            String[] narrow = {"sum(small)", "sum(id)", "count(small)"};
            assertPushed(pushed.selectExpr(narrow));
            assertEquals(
                    scanned.selectExpr(narrow).collectAsList(),
                    pushed.selectExpr(narrow).collectAsList());

            // A sum of BIGINT values can overflow, and a wrapped partial cannot tell.
            assertNotPushed(pushed.selectExpr("sum(big)"));
        });
    }

    @Test
    void sumsOfAColumnPlusAConstantPushDownUnderAnsiMode() {
        Path output = writeShiftedSumData("shifted_sums");
        withAnsi(true, () -> {
            Dataset<Row> pushed = spark.read().format("vortex").load(output.toString());
            Dataset<Row> scanned = spark.read()
                    .format("vortex")
                    .option("vortex.aggregatePushdown", "false")
                    .load(output.toString());

            // ClickBench q30 sums one SMALLINT column plus 0 to 89. The appended file holds no value of small, so its
            // shifted sums are null and add nothing.
            String[] shifted = {
                "sum(small + 1)", "sum(small + 89)", "sum(-7 + small)", "sum(id + 5000000000)", "sum(small)"
            };
            assertPushed(pushed.selectExpr(shifted));
            assertEquals(
                    scanned.selectExpr(shifted).collectAsList(),
                    pushed.selectExpr(shifted).collectAsList());
        });
    }

    @Test
    void sumsOfAColumnPlusAConstantThatCanOverflowStayInSpark() {
        Path output = writeShiftedSumData("overflowing_sums");
        withAnsi(true, () -> {
            Dataset<Row> data = spark.read().format("vortex").load(output.toString());

            // An INT value plus an INT constant can overflow the addition, which ANSI rules make fail.
            assertNotPushed(data.selectExpr("sum(id + 1)"));
            assertNotPushed(data.selectExpr("sum(big + 1)"));
            assertNotPushed(data.selectExpr("sum(small + 2147483647)"));
        });
    }

    /**
     * Writes rows with SMALLINT, INT, and BIGINT columns, and appends a file whose SMALLINT column holds only nulls.
     */
    private Path writeShiftedSumData(String name) {
        Path output = tempDir.resolve(name);
        spark.range(0, 300)
                .selectExpr(
                        "cast(id as int) as id",
                        "case when id % 4 = 0 then null else cast(id - 150 as smallint) end as small",
                        "id * 3 as big")
                .repartition(3)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());
        spark.range(1000, 1010)
                .selectExpr("cast(id as int) as id", "cast(null as smallint) as small", "id * 3 as big")
                .coalesce(1)
                .write()
                .format("vortex")
                .mode(SaveMode.Append)
                .save(output.toString());
        return output;
    }

    /** The aggregate is answered by the source: by a pushed partial per file or from the footers on the driver. */
    private static void assertPushed(Dataset<Row> aggregate) {
        String plan = aggregate.queryExecution().executedPlan().toString();
        assertTrue(isPushed(plan), plan);
    }

    private static void assertNotPushed(Dataset<Row> aggregate) {
        String plan = aggregate.queryExecution().executedPlan().toString();
        assertFalse(isPushed(plan), plan);
    }

    /** Every scan lists its pushed aggregates, so an empty list means that none was pushed. */
    private static boolean isPushed(String plan) {
        return (plan.contains("PushedAggregation: [") && !plan.contains("PushedAggregation: []"))
                || plan.contains("LocalTableScan");
    }

    /** Runs {@code body} with ANSI mode set, which Spark 4 enables by default and Spark 3.5 does not. */
    private void withAnsi(boolean enabled, Runnable body) {
        String previous = spark.conf().get("spark.sql.ansi.enabled");
        spark.conf().set("spark.sql.ansi.enabled", Boolean.toString(enabled));
        try {
            body.run();
        } finally {
            spark.conf().set("spark.sql.ansi.enabled", previous);
        }
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
        writeAndRead(400, "pushed_stats", 4, false);
        Dataset<Row> count = spark.read()
                .format("vortex")
                .option("vortex.completeAggregatePushdown", "false")
                .load(tempDir.resolve("pushed_stats").toString())
                .selectExpr("count(*) AS count");

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
