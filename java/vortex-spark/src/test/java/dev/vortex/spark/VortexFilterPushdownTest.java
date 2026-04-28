// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.QueryExecution;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that Spark predicate pushdown into the Vortex datasource produces correct results.
 *
 * <p>The tests write a Vortex dataset and then read it back applying various Spark filters. The
 * {@code VortexScanBuilder.pushFilters} path attempts to translate each filter to a Vortex {@code Expression}; filters
 * it cannot translate (or that reference partition columns) are returned to Spark for post-scan evaluation. Either way
 * the final result must match the same query against the original DataFrame.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexFilterPushdownTest {

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexFilterPushdownTest")
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

    @Test
    @DisplayName("Equality, comparison, IS NULL, IN, AND/OR/NOT all return correct rows after pushdown")
    public void testFilterPushdownCorrectness() throws IOException {
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, "alpha", 10L, true),
                        RowFactory.create(2, "beta", 20L, false),
                        RowFactory.create(3, "gamma", 30L, true),
                        RowFactory.create(4, "delta", null, false),
                        RowFactory.create(5, null, 50L, true)),
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("name", DataTypes.StringType, true),
                        DataTypes.createStructField("amount", DataTypes.LongType, true),
                        DataTypes.createStructField("flag", DataTypes.BooleanType, false))));

        Path outputPath = tempDir.resolve("pushdown_basic");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        assertEquals(
                List.of(2), idsOf(readDf.filter(readDf.col("id").equalTo(2)).orderBy("id")));

        assertEquals(
                List.of(3, 4, 5), idsOf(readDf.filter(readDf.col("id").gt(2)).orderBy("id")));

        assertEquals(List.of(1, 2), idsOf(readDf.filter(readDf.col("id").leq(2)).orderBy("id")));

        assertEquals(
                List.of(1, 3),
                idsOf(readDf.filter(readDf.col("name").isin("alpha", "gamma")).orderBy("id")));

        assertEquals(
                List.of(4), idsOf(readDf.filter(readDf.col("amount").isNull()).orderBy("id")));

        assertEquals(
                List.of(1, 2, 3, 5),
                idsOf(readDf.filter(readDf.col("amount").isNotNull()).orderBy("id")));

        assertEquals(
                List.of(1, 3),
                idsOf(readDf.filter(readDf.col("flag")
                                .equalTo(true)
                                .and(readDf.col("amount").lt(40L)))
                        .orderBy("id")));

        assertEquals(
                List.of(1, 4, 5),
                idsOf(readDf.filter(readDf.col("id")
                                .equalTo(1)
                                .or(readDf.col("amount").isNull())
                                .or(readDf.col("name").isNull()))
                        .orderBy("id")));

        // NOT around an unsupported predicate (string startsWith) should still produce
        // correct results — Spark applies it as a post-scan filter.
        assertEquals(
                List.of(2, 3, 4),
                idsOf(readDf.filter(functions.not(readDf.col("name").startsWith("a")))
                        .orderBy("id")));
    }

    @Test
    @DisplayName("Filters on partition columns yield correct results without pushdown")
    public void testFilterOnPartitionColumn() throws IOException {
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, "alpha", "A"),
                        RowFactory.create(2, "beta", "B"),
                        RowFactory.create(3, "gamma", "A"),
                        RowFactory.create(4, "delta", "B")),
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("name", DataTypes.StringType, true),
                        DataTypes.createStructField("group", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("pushdown_partitioned");
        df.write()
                .format("vortex")
                .partitionBy("group")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        assertEquals(
                List.of(1, 3),
                idsOf(readDf.filter(readDf.col("group").equalTo("A")).orderBy("id")));

        // Predicate spanning partition + data columns must still produce the right answer.
        assertEquals(
                List.of(3),
                idsOf(readDf.filter(readDf.col("group")
                                .equalTo("A")
                                .and(readDf.col("id").gt(1)))
                        .orderBy("id")));
    }

    @Test
    @DisplayName("Pushed filters appear in the executed scan node")
    public void testPushedFiltersInPlan() throws IOException {
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(RowFactory.create(1, "x"), RowFactory.create(2, "y"), RowFactory.create(3, "z")),
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("label", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("pushdown_plan");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        Dataset<Row> filtered = readDf.filter(readDf.col("id").gt(1));
        QueryExecution qe = filtered.queryExecution();
        SparkPlan plan = qe.executedPlan();
        String planString = plan.toString();
        assertTrue(
                planString.contains("id > 1"),
                "Expected pushed predicate for id > 1 in the executed plan: " + planString);
    }

    private static List<Integer> idsOf(Dataset<Row> df) {
        return df.collectAsList().stream().map(row -> row.getInt(0)).collect(Collectors.toList());
    }

    @AfterEach
    public void cleanupTempFiles() throws IOException {
        if (tempDir != null && Files.exists(tempDir)) {
            try (Stream<Path> paths = Files.walk(tempDir)) {
                paths.sorted(Comparator.reverseOrder()).forEach(path -> {
                    try {
                        Files.deleteIfExists(path);
                    } catch (IOException e) {
                        System.err.println("Failed to delete: " + path);
                    }
                });
            }
        }
    }
}
