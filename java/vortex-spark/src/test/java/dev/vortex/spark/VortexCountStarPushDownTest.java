// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.spark.read.VortexCountStarScan;
import dev.vortex.spark.read.VortexScan;
import dev.vortex.spark.read.VortexScanBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Map;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.aggregate.AggregateFunc;
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation;
import org.apache.spark.sql.connector.expressions.aggregate.Count;
import org.apache.spark.sql.connector.expressions.aggregate.CountStar;
import org.apache.spark.sql.connector.expressions.aggregate.Min;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Characterizes {@code COUNT(*)} aggregate pushdown: a global count is answered from file footer metadata via
 * {@link VortexCountStarScan}, while filtered, grouped, or column-counting aggregations fall back to a regular
 * {@link VortexScan} and stay correct.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexCountStarPushDownTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexCountStarPushDownTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
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

    @AfterEach
    public void cleanupTempFiles() throws IOException {
        if (Files.exists(tempDir)) {
            try (Stream<Path> walk = Files.walk(tempDir)) {
                walk.sorted(Comparator.reverseOrder()).forEach(path -> {
                    if (!path.equals(tempDir)) {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException e) {
                            // best-effort cleanup
                        }
                    }
                });
            }
        }
    }

    @Test
    @DisplayName("Global COUNT(*) over a multi-file dataset is answered by VortexCountStarScan and is exact")
    public void testCountStarIsPushedDownAndExact() throws IOException {
        Path outputPath = writeRows(10_000, "count_multi_file", 4);

        Dataset<Row> df = spark.read().format("vortex").load(outputPath.toUri().toString());
        Dataset<Row> counted = df.groupBy().count();

        String plan = counted.queryExecution().executedPlan().toString();
        assertTrue(
                plan.contains("VortexCountStarScan"),
                "expected the physical plan to use the pushed-down count scan, got:\n" + plan);
        assertEquals(
                10_000L, counted.collectAsList().get(0).getLong(0), "pushed-down count must match the written rows");
        assertEquals(10_000L, df.count(), "Dataset.count() must match the written rows");
    }

    @Test
    @DisplayName("COUNT(*) with a data filter is not pushed down and stays correct")
    public void testFilteredCountFallsBackToRegularScan() throws IOException {
        Path outputPath = writeRows(1_000, "count_filtered", 2);

        Dataset<Row> filtered = spark.read()
                .format("vortex")
                .load(outputPath.toUri().toString())
                .filter("id < 100");
        Dataset<Row> counted = filtered.groupBy().count();

        String plan = counted.queryExecution().executedPlan().toString();
        assertFalse(
                plan.contains("VortexCountStarScan"), "a filtered count must not use footer metadata, got:\n" + plan);
        assertEquals(100L, filtered.count(), "filtered count must reflect the predicate, not footer totals");
    }

    @Test
    @DisplayName("Grouped counts are not pushed down and stay correct")
    public void testGroupedCountFallsBackToRegularScan() throws IOException {
        Path outputPath = writeRows(1_000, "count_grouped", 2);

        Dataset<Row> df = spark.read().format("vortex").load(outputPath.toUri().toString());
        Dataset<Row> grouped =
                df.selectExpr("id % 2 as bucket").groupBy("bucket").count();

        String plan = grouped.queryExecution().executedPlan().toString();
        assertFalse(plan.contains("VortexCountStarScan"), "a grouped count must scan data, got:\n" + plan);
        assertEquals(2, grouped.collectAsList().size(), "grouping must produce one row per bucket");
    }

    @Test
    @DisplayName("Builder accepts a single COUNT(*) aggregation and builds a VortexCountStarScan")
    public void testBuilderAcceptsSingleCountStar() {
        VortexScanBuilder builder = new VortexScanBuilder(Map.of());
        builder.addPath("/tmp/example.vortex");

        assertTrue(
                builder.pushAggregation(aggregation(new CountStar())),
                "a lone COUNT(*) with no grouping must be pushed");
        assertInstanceOf(
                VortexCountStarScan.class, builder.build(), "a pushed COUNT(*) must build the metadata count scan");
    }

    @Test
    @DisplayName("Builder declines aggregations it cannot answer from footer metadata")
    public void testBuilderDeclinesUnsupportedAggregations() {
        assertFalse(
                newBuilder().pushAggregation(aggregation(new Count(Expressions.column("id"), false))),
                "COUNT(col) must not be pushed: footer counts do not reflect column nulls");
        assertFalse(
                newBuilder().pushAggregation(aggregation(new Min(Expressions.column("id")))), "MIN must not be pushed");
        assertFalse(
                newBuilder().pushAggregation(aggregation(new CountStar(), new CountStar())),
                "multiple aggregate expressions must not be pushed");
        assertFalse(
                newBuilder()
                        .pushAggregation(new Aggregation(
                                new AggregateFunc[] {new CountStar()}, new Expression[] {Expressions.column("id")})),
                "grouped aggregations must not be pushed");
    }

    @Test
    @DisplayName("Builder declining an aggregation leaves the regular scan path intact")
    public void testDeclinedAggregationBuildsRegularScan() {
        VortexScanBuilder builder = newBuilder();

        assertFalse(builder.pushAggregation(aggregation(new Min(Expressions.column("id")))));
        assertInstanceOf(VortexScan.class, builder.build(), "a declined pushdown must build the regular scan");
    }

    private static VortexScanBuilder newBuilder() {
        VortexScanBuilder builder = new VortexScanBuilder(Map.of());
        builder.addPath("/tmp/example.vortex");
        return builder;
    }

    private static Aggregation aggregation(AggregateFunc... functions) {
        return new Aggregation(functions, new Expression[0]);
    }

    private Path writeRows(int numRows, String name, int partitions) throws IOException {
        Path outputPath = tempDir.resolve(name);
        Dataset<Row> df = spark.range(0, numRows)
                .selectExpr("cast(id as int) as id", "concat('value_', cast(id as string)) as value");

        df.repartition(partitions)
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();
        return outputPath;
    }
}
