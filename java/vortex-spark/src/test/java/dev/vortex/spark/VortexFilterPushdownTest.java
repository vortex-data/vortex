// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Date;
import java.sql.Timestamp;
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
import org.apache.spark.sql.types.StructType;
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

        // !=
        assertEquals(
                List.of(1, 3, 4, 5),
                idsOf(readDf.filter(readDf.col("id").notEqual(2)).orderBy("id")));

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

        // NOT around a pushed predicate.
        assertEquals(
                List.of(2, 3, 4),
                idsOf(readDf.filter(functions.not(readDf.col("name").startsWith("a")))
                        .orderBy("id")));
    }

    @Test
    @DisplayName("STARTS_WITH / ENDS_WITH / CONTAINS push down via LIKE with metachar escaping")
    public void testStringPredicatePushdown() throws IOException {
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, "alpha_one"),
                        RowFactory.create(2, "alpha_two"),
                        RowFactory.create(3, "beta_one"),
                        RowFactory.create(4, "ab%cd"), // contains a literal %
                        RowFactory.create(5, null)),
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("label", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("pushdown_strings");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // STARTS_WITH
        assertEquals(
                List.of(1, 2),
                idsOf(readDf.filter(readDf.col("label").startsWith("alpha")).orderBy("id")));

        // ENDS_WITH
        assertEquals(
                List.of(1, 3),
                idsOf(readDf.filter(readDf.col("label").endsWith("_one")).orderBy("id")));

        // CONTAINS (literal underscore must match the underscore character).
        assertEquals(
                List.of(1, 3),
                idsOf(readDf.filter(readDf.col("label").contains("_o")).orderBy("id")));

        // CONTAINS with no special chars — verify standard substring search works.
        assertEquals(
                List.of(1, 2),
                idsOf(readDf.filter(readDf.col("label").contains("alpha")).orderBy("id")));

        // Literal "%" should not act as a LIKE wildcard; only id=4 contains it.
        assertEquals(
                List.of(4),
                idsOf(readDf.filter(readDf.col("label").contains("%")).orderBy("id")));

        // STARTS_WITH on an underscore should match the literal underscore character.
        assertEquals(
                List.of(),
                idsOf(readDf.filter(readDf.col("label").startsWith("_")).orderBy("id")));
    }

    @Test
    @DisplayName("Date, timestamp, and decimal literals push down through equality and range comparisons")
    public void testTemporalAndDecimalPushdown() throws IOException {
        StructType schema = DataTypes.createStructType(Arrays.asList(
                DataTypes.createStructField("id", DataTypes.IntegerType, false),
                DataTypes.createStructField("d", DataTypes.DateType, true),
                DataTypes.createStructField("ts", DataTypes.TimestampType, true),
                DataTypes.createStructField("amt", DataTypes.createDecimalType(10, 2), true)));

        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(
                                1,
                                Date.valueOf("2020-01-01"),
                                Timestamp.valueOf("2020-01-01 00:00:00"),
                                new BigDecimal("1.23")),
                        RowFactory.create(
                                2,
                                Date.valueOf("2021-06-15"),
                                Timestamp.valueOf("2021-06-15 12:30:00"),
                                new BigDecimal("99.99")),
                        RowFactory.create(
                                3,
                                Date.valueOf("2022-12-31"),
                                Timestamp.valueOf("2022-12-31 23:59:59"),
                                new BigDecimal("-5.00"))),
                schema);

        Path outputPath = tempDir.resolve("pushdown_temporal");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // Date equality
        assertEquals(
                List.of(2),
                idsOf(readDf.filter(readDf.col("d").equalTo(Date.valueOf("2021-06-15")))
                        .orderBy("id")));

        // Date range
        assertEquals(
                List.of(2, 3),
                idsOf(readDf.filter(readDf.col("d").gt(Date.valueOf("2020-06-01")))
                        .orderBy("id")));

        // Timestamp range
        assertEquals(
                List.of(1, 2),
                idsOf(readDf.filter(readDf.col("ts").lt(Timestamp.valueOf("2022-01-01 00:00:00")))
                        .orderBy("id")));

        // Decimal equality
        assertEquals(
                List.of(2),
                idsOf(readDf.filter(readDf.col("amt").equalTo(new BigDecimal("99.99")))
                        .orderBy("id")));

        // Decimal range
        assertEquals(
                List.of(3),
                idsOf(readDf.filter(readDf.col("amt").lt(new BigDecimal("0.00")))
                        .orderBy("id")));
    }

    @Test
    @DisplayName("Filters on nested struct fields push down")
    public void testNestedFieldPushdown() throws IOException {
        StructType inner = DataTypes.createStructType(Arrays.asList(
                DataTypes.createStructField("category", DataTypes.StringType, true),
                DataTypes.createStructField("score", DataTypes.IntegerType, true)));
        StructType schema = DataTypes.createStructType(Arrays.asList(
                DataTypes.createStructField("id", DataTypes.IntegerType, false),
                DataTypes.createStructField("info", inner, true)));

        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, RowFactory.create("apple", 10)),
                        RowFactory.create(2, RowFactory.create("banana", 20)),
                        RowFactory.create(3, RowFactory.create("cherry", 30)),
                        RowFactory.create(4, RowFactory.create("apple", 40))),
                schema);

        Path outputPath = tempDir.resolve("pushdown_nested");
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
                List.of(1, 4),
                idsOf(readDf.filter(readDf.col("info.category").equalTo("apple"))
                        .orderBy("id")));

        assertEquals(
                List.of(3, 4),
                idsOf(readDf.filter(readDf.col("info.score").gt(20)).orderBy("id")));
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

    @Test
    @DisplayName("Deep nesting of AND/OR/NOT pushes down correctly")
    public void testDeeplyNestedLogicalPushdown() throws IOException {
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, 10, "alpha"),
                        RowFactory.create(2, 20, "beta"),
                        RowFactory.create(3, 30, "gamma"),
                        RowFactory.create(4, 40, "delta"),
                        RowFactory.create(5, 50, "epsilon")),
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("val", DataTypes.IntegerType, false),
                        DataTypes.createStructField("label", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("pushdown_deep");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // ((id > 1 AND val < 50) OR label = 'epsilon') AND NOT(label = 'beta')
        assertEquals(
                List.of(3, 4, 5),
                idsOf(readDf.filter(readDf.col("id")
                                .gt(1)
                                .and(readDf.col("val").lt(50))
                                .or(readDf.col("label").equalTo("epsilon"))
                                .and(functions.not(readDf.col("label").equalTo("beta"))))
                        .orderBy("id")));
    }

    @Test
    @DisplayName("STARTS_WITH/ENDS_WITH/CONTAINS escape LIKE meta-characters in the literal substring")
    public void testStringPredicateEscapeRegression() throws IOException {
        // Cover every LIKE meta-character (`%`, `_`, `\\`) as well as a "safe" string to ensure
        // ordinary substrings still pass through unchanged. Each fixture row carries the literal
        // we will later search for using STARTS_WITH/ENDS_WITH/CONTAINS, plus a "decoy" row that
        // would only match if the meta-character were interpreted as a wildcard.
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, "%pct%"), // contains literal %
                        RowFactory.create(2, "no-percent"),
                        RowFactory.create(3, "abc%def"),
                        RowFactory.create(4, "abXdef"), // would match `%_%` if `_` were a wildcard
                        RowFactory.create(5, "_under"),
                        RowFactory.create(6, "no-under"),
                        RowFactory.create(7, "a\\b"), // single literal backslash between a and b
                        RowFactory.create(8, "ab"), // would match `a\b` if `\` were stripped
                        RowFactory.create(9, "trail\\"), // ends with literal backslash
                        RowFactory.create(10, "%front")), // starts with literal %
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("label", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("pushdown_string_escapes");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // CONTAINS("%") -- must NOT match rows without a literal `%`.
        assertEquals(
                List.of(1, 3, 10),
                idsOf(readDf.filter(readDf.col("label").contains("%")).orderBy("id")));

        // STARTS_WITH("%") -- only rows that start with a literal `%`.
        assertEquals(
                List.of(1, 10),
                idsOf(readDf.filter(readDf.col("label").startsWith("%")).orderBy("id")));

        // ENDS_WITH("%") -- only rows that end with a literal `%`.
        assertEquals(
                List.of(1),
                idsOf(readDf.filter(readDf.col("label").endsWith("%")).orderBy("id")));

        // CONTAINS("_") -- must NOT match every row; only those with a literal `_`.
        assertEquals(
                List.of(5),
                idsOf(readDf.filter(readDf.col("label").contains("_")).orderBy("id")));

        // STARTS_WITH("_") -- only rows that start with a literal `_`.
        assertEquals(
                List.of(5),
                idsOf(readDf.filter(readDf.col("label").startsWith("_")).orderBy("id")));

        // CONTAINS("\\") -- must match rows with a single literal backslash. The Java string
        // literal "\\" is a 1-char string containing just `\`.
        assertEquals(
                List.of(7, 9),
                idsOf(readDf.filter(readDf.col("label").contains("\\")).orderBy("id")));

        // ENDS_WITH("\\") -- only rows ending with a literal backslash.
        assertEquals(
                List.of(9),
                idsOf(readDf.filter(readDf.col("label").endsWith("\\")).orderBy("id")));

        // CONTAINS("abc%def") -- treat `%` literally; only row 3 has the exact substring.
        assertEquals(
                List.of(3),
                idsOf(readDf.filter(readDf.col("label").contains("abc%def")).orderBy("id")));

        // Sanity check: a non-meta substring still works.
        assertEquals(
                List.of(2, 6),
                idsOf(readDf.filter(readDf.col("label").contains("no-")).orderBy("id")));
    }

    @Test
    @DisplayName("Binary literals push down through equality")
    public void testBinaryLiteralPushdown() throws IOException {
        StructType schema = DataTypes.createStructType(Arrays.asList(
                DataTypes.createStructField("id", DataTypes.IntegerType, false),
                DataTypes.createStructField("payload", DataTypes.BinaryType, true)));

        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, new byte[] {0x01, 0x02, 0x03}),
                        RowFactory.create(2, new byte[] {0x04, 0x05}),
                        RowFactory.create(3, new byte[] {0x01, 0x02, 0x03})),
                schema);

        Path outputPath = tempDir.resolve("pushdown_binary");
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
                List.of(1, 3),
                idsOf(readDf.filter(readDf.col("payload").equalTo(new byte[] {0x01, 0x02, 0x03}))
                        .orderBy("id")));
    }

    @Test
    @DisplayName("Bare boolean column reference (e.g. WHERE bool_col) pushes down")
    public void testBareBooleanColumnPushdown() throws IOException {
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, true),
                        RowFactory.create(2, false),
                        RowFactory.create(3, true),
                        RowFactory.create(4, false)),
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("flag", DataTypes.BooleanType, false))));

        Path outputPath = tempDir.resolve("pushdown_bool");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // WHERE flag
        assertEquals(List.of(1, 3), idsOf(readDf.filter(readDf.col("flag")).orderBy("id")));

        // WHERE NOT flag
        assertEquals(
                List.of(2, 4),
                idsOf(readDf.filter(functions.not(readDf.col("flag"))).orderBy("id")));
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
