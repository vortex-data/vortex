// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration test for Vortex DataSource write and read functionality.
 * <p>
 * This test verifies that:
 * 1. Spark DataFrames can be written as Vortex files
 * 2. Multiple partitions create multiple files
 * 3. Data can be read back correctly
 * 4. Schema is preserved during write/read
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDataSourceWriteTest {

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        // Create a local Spark session for testing
        spark = SparkSession.builder()
                .appName("VortexWriteTest")
                .master("local[2]") // Use 2 threads
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false") // Disable AQE for predictable partitioning
                .config("spark.ui.enabled", "false") // Disable UI for tests
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Write and read Vortex files with multiple partitions")
    public void testWriteAndReadVortexFiles() throws IOException {
        // Given: Create a DataFrame with two columns
        int numRows = 100;
        Dataset<Row> originalDf = createTestDataFrame(numRows);

        // Verify original data
        assertEquals(numRows, originalDf.count(), "Original DataFrame should have " + numRows + " rows");
        assertEquals(3, originalDf.columns().length, "Original DataFrame should have 2 columns");

        // When: Repartition to 2 partitions and write as Vortex
        Path outputPath = tempDir.resolve("vortex_output");
        originalDf
                .repartition(2) // Force 2 partitions
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        // Add a small delay to ensure files are fully written and filesystem is synced
        try {
            Thread.sleep(1000); // 1 second delay
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Then: Verify two Vortex files were created
        if (Files.exists(outputPath)) {
            try (Stream<Path> allFiles = Files.walk(outputPath)) {
                allFiles.forEach(p -> System.err.println("Found: " + p));
            }
        }

        List<Path> vortexFiles = findVortexFiles(outputPath);
        System.err.println("Found " + vortexFiles.size() + " vortex files");
        assertEquals(2, vortexFiles.size(), "Should have created 2 Vortex files (one per partition)");

        // Verify files have expected naming pattern
        for (Path file : vortexFiles) {
            assertTrue(
                    file.getFileName().toString().matches("part-\\d{5}-\\d+\\.vortex"),
                    "File should match pattern part-XXXXX-Y.vortex: " + file.getFileName());
        }

        // When: Read the Vortex files back
        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // Then: Verify schema is preserved
        assertSchemaEquals(originalDf.schema(), readDf.schema());

        // Verify row count
        assertEquals(numRows, readDf.count(), "Read DataFrame should have same number of rows as original");

        // Verify data content
        verifyDataContent(originalDf, readDf);
    }

    @Test
    @DisplayName("Write empty DataFrame as Vortex")
    public void testWriteEmptyDataFrame() throws IOException {
        // Given: Create an empty DataFrame with schema
        Dataset<Row> emptyDf = spark.createDataFrame(
                spark.emptyDataFrame().rdd(),
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("value", DataTypes.StringType, true))));

        // When: Write as Vortex
        Path outputPath = tempDir.resolve("empty_vortex");
        emptyDf.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        // Then: Read back and verify
        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        assertEquals(0, readDf.count(), "Empty DataFrame should remain empty after write/read");
        assertSchemaEquals(emptyDf.schema(), readDf.schema());
    }

    @Test
    @DisplayName("Overwrite existing Vortex files")
    public void testOverwriteMode() throws IOException {
        Path outputPath = tempDir.resolve("overwrite_test");

        // First write
        Dataset<Row> df1 = createTestDataFrame(50);
        df1.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        // Second write with different data
        Dataset<Row> df2 = createTestDataFrame(75);
        df2.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        // Read and verify we get the second dataset
        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        assertEquals(75, readDf.count(), "Should have data from second write after overwrite");
    }

    @Test
    @DisplayName("Write and read partitioned Vortex files")
    public void testPartitionedWrite() throws IOException {
        // Given: a DataFrame with a partition column
        List<Row> rows = Arrays.asList(
                RowFactory.create(1, "alpha", "A"),
                RowFactory.create(2, "beta", "B"),
                RowFactory.create(3, "gamma", "A"),
                RowFactory.create(4, "delta", "B"),
                RowFactory.create(5, "epsilon", "A"));

        Dataset<Row> df = spark.createDataFrame(
                rows,
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("name", DataTypes.StringType, true),
                        DataTypes.createStructField("group", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("partitioned_output");

        // When: write with partitionBy
        df.write()
                .format("vortex")
                .partitionBy("group")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        // Then: verify partition directories exist
        assertTrue(Files.exists(outputPath.resolve("group=A")), "Partition directory group=A should exist");
        assertTrue(Files.exists(outputPath.resolve("group=B")), "Partition directory group=B should exist");

        // Verify vortex files inside partition directories
        List<Path> filesA = findVortexFiles(outputPath.resolve("group=A"));
        List<Path> filesB = findVortexFiles(outputPath.resolve("group=B"));
        assertFalse(filesA.isEmpty(), "Partition A should have vortex files");
        assertFalse(filesB.isEmpty(), "Partition B should have vortex files");

        // When: read back
        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // Then: verify all rows are present
        assertEquals(5, readDf.count(), "Should read all 5 rows back");

        // Verify partition values are correct
        Dataset<Row> groupA = readDf.filter(readDf.col("group").equalTo("A")).orderBy("id");
        assertEquals(3, groupA.count(), "Group A should have 3 rows");
        assertEquals(1, (int) groupA.collectAsList().get(0).getAs("id"));
        assertEquals(3, (int) groupA.collectAsList().get(1).getAs("id"));
        assertEquals(5, (int) groupA.collectAsList().get(2).getAs("id"));
    }

    @Test
    @DisplayName("Write and read with multiple partition columns")
    public void testMultiColumnPartitionedWrite() throws IOException {
        List<Row> rows = Arrays.asList(
                RowFactory.create(1, "X", 10),
                RowFactory.create(2, "Y", 20),
                RowFactory.create(3, "X", 20),
                RowFactory.create(4, "Y", 10));

        Dataset<Row> df = spark.createDataFrame(
                rows,
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("category", DataTypes.StringType, true),
                        DataTypes.createStructField("bucket", DataTypes.IntegerType, false))));

        Path outputPath = tempDir.resolve("multi_partition_output");

        df.write()
                .format("vortex")
                .partitionBy("category", "bucket")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        // Verify nested partition directories
        assertTrue(
                Files.exists(outputPath.resolve("category=X/bucket=10")),
                "Partition directory category=X/bucket=10 should exist");
        assertTrue(
                Files.exists(outputPath.resolve("category=Y/bucket=20")),
                "Partition directory category=Y/bucket=20 should exist");

        // Read back and verify
        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        assertEquals(4, readDf.count(), "Should read all 4 rows back");
    }

    @Test
    @DisplayName("Handle special characters and nulls")
    public void testSpecialCharactersAndNulls() throws IOException {
        // Create DataFrame with nulls and special characters
        List<Row> rows = Arrays.asList(
                RowFactory.create(1, "normal"),
                RowFactory.create(2, null),
                RowFactory.create(3, "special!@#$%^&*()"),
                RowFactory.create(4, "unicode_测试_тест"),
                RowFactory.create(5, ""),
                RowFactory.create(6, "multi\nline\nstring"));

        Dataset<Row> specialDf = spark.createDataFrame(
                rows,
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("text", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("special_chars");

        // Write and read
        specialDf
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        // Verify all special cases are preserved
        assertEquals(6, readDf.count());

        // Check null value
        Dataset<Row> nullRows = readDf.filter(readDf.col("text").isNull());
        assertEquals(1, nullRows.count(), "Should have one null value");
        assertEquals(2, nullRows.first().getInt(0), "Null should be for id=2");

        // Check empty string
        Dataset<Row> emptyRows = readDf.filter(readDf.col("text").equalTo(""));
        assertEquals(1, emptyRows.count(), "Should have one empty string");

        // Check special characters preserved
        Dataset<Row> specialRows = readDf.filter(readDf.col("id").equalTo(3));
        assertEquals("special!@#$%^&*()", specialRows.first().getString(1));
    }

    @Test
    @DisplayName("Write and read date, timestamp, and nested struct columns")
    public void testWriteAndReadTemporalAndStructColumns() throws IOException {
        Dataset<Row> originalDf = spark.range(0, 2)
                .selectExpr(
                        "cast(id as int) as id",
                        "CASE WHEN id = 0 THEN CAST('2024-01-02' AS DATE) ELSE CAST('2024-02-03' AS DATE) END AS event_date",
                        """
                                CASE WHEN id = 0 THEN CAST('2024-01-02 03:04:05.123456' AS TIMESTAMP)
                                ELSE CAST('2024-02-03 04:05:06.654321' AS TIMESTAMP) END AS event_ts""",
                        """
                                named_struct(
                                    'event_date', CASE WHEN id = 0 THEN CAST('2024-01-02' AS DATE) ELSE CAST('2024-02-03' AS DATE) END,
                                    'event_ts', CASE WHEN id = 0 THEN CAST('2024-01-02 03:04:05.123456' AS TIMESTAMP)
                                        ELSE CAST('2024-02-03 04:05:06.654321' AS TIMESTAMP) END,
                                    'label', CASE WHEN id = 0 THEN 'alpha' ELSE 'beta' END
                                ) AS payload""");

        Path outputPath = tempDir.resolve("temporal_struct_output");
        originalDf
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        List<String> expectedRows = List.of(
                "{\"id\":0,\"event_date\":\"2024-01-02\",\"event_ts\":\"2024-01-02 03:04:05.123456\","
                        + "\"payload_event_date\":\"2024-01-02\",\"payload_event_ts\":\"2024-01-02 03:04:05.123456\","
                        + "\"payload_label\":\"alpha\"}",
                "{\"id\":1,\"event_date\":\"2024-02-03\",\"event_ts\":\"2024-02-03 04:05:06.654321\","
                        + "\"payload_event_date\":\"2024-02-03\",\"payload_event_ts\":\"2024-02-03 04:05:06.654321\","
                        + "\"payload_label\":\"beta\"}");

        assertEquals(DataTypes.DateType, readDf.schema().fields()[1].dataType());
        assertEquals(DataTypes.TimestampType, readDf.schema().fields()[2].dataType());
        assertInstanceOf(StructType.class, readDf.schema().fields()[3].dataType());
        assertEquals(expectedRows, projectTemporalAndStructRows(readDf));
    }

    @Test
    @DisplayName("Write TimestampNTZ columns and nested structs")
    public void testWriteTimestampNtzColumns() throws IOException {
        Dataset<Row> timestampNtzDf = spark.range(0, 2).selectExpr("cast(id as int) as id", """
                CASE WHEN id = 0 THEN CAST('2024-01-02 03:04:05.123456' AS TIMESTAMP_NTZ)
                ELSE CAST(NULL AS TIMESTAMP_NTZ) END AS event_ntz""", """
                named_struct(
                    'event_ntz', CASE WHEN id = 0 THEN CAST('2024-01-02 03:04:05.123456' AS TIMESTAMP_NTZ)
                        ELSE CAST('2024-02-03 04:05:06.654321' AS TIMESTAMP_NTZ) END
                ) AS payload""");

        Path outputPath = tempDir.resolve("timestamp_ntz_output");
        assertDoesNotThrow(() -> timestampNtzDf
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save());

        assertFalse(findVortexFiles(outputPath).isEmpty(), "TimestampNTZ write should create Vortex files");
    }

    /**
     * Creates a test DataFrame with monotonically increasing integers
     * and their string representations.
     */
    private Dataset<Row> createTestDataFrame(int numRows) {
        // Create DataFrame with monotonically increasing integers
        return spark.range(0, numRows)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id as string)) as value",
                        "array('Alpha', 'Bravo', 'Charlie') AS elements");
    }

    private List<String> projectTemporalAndStructRows(Dataset<Row> df) {
        return df.orderBy("id").selectExpr("""
                        to_json(named_struct(
                            'id', id,
                            'event_date', cast(event_date as string),
                            'event_ts', date_format(event_ts, 'yyyy-MM-dd HH:mm:ss.SSSSSS'),
                            'payload_event_date', cast(payload.event_date as string),
                            'payload_event_ts', date_format(payload.event_ts, 'yyyy-MM-dd HH:mm:ss.SSSSSS'),
                            'payload_label', payload.label
                        )) as json""").collectAsList().stream()
                .map(row -> row.getString(0))
                .collect(Collectors.toList());
    }

    /**
     * Finds all Vortex files in the given directory.
     */
    private List<Path> findVortexFiles(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return List.of();
        }

        try (Stream<Path> paths = Files.walk(directory)) {
            return paths.filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".vortex"))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    /**
     * Verifies that two schemas are equal.
     */
    private void assertSchemaEquals(StructType expected, StructType actual) {
        assertEquals(expected.fields().length, actual.fields().length, "Schemas should have same number of fields");

        for (int i = 0; i < expected.fields().length; i++) {
            StructField expectedField = expected.fields()[i];
            StructField actualField = actual.fields()[i];

            assertEquals(expectedField.name(), actualField.name(), "Field names should match at position " + i);
            assertEquals(
                    expectedField.dataType(),
                    actualField.dataType(),
                    "Field types should match for field: " + expectedField.name());
            assertEquals(
                    expectedField.nullable(),
                    actualField.nullable(),
                    "Field nullability should match for field: " + expectedField.name());
        }
    }

    /**
     * Verifies that the data content of two DataFrames is identical.
     */
    private void verifyDataContent(Dataset<Row> expected, Dataset<Row> actual) {
        // Sort both DataFrames by id to ensure consistent ordering
        Dataset<Row> expectedSorted = expected.orderBy("id");
        Dataset<Row> actualSorted = actual.orderBy("id");

        // Collect and compare
        List<Row> expectedRows = expectedSorted.collectAsList();
        List<Row> actualRows = actualSorted.collectAsList();

        assertEquals(expectedRows.size(), actualRows.size(), "Should have same number of rows");

        for (int i = 0; i < expectedRows.size(); i++) {
            Row expectedRow = expectedRows.get(i);
            Row actualRow = actualRows.get(i);

            assertEquals(expectedRow.getInt(0), actualRow.getInt(0), "ID should match at row " + i);
            assertEquals(expectedRow.getString(1), actualRow.getString(1), "Value should match at row " + i);
        }
    }

    @AfterEach
    public void cleanupTempFiles() throws IOException {
        // Additional cleanup if needed (tempDir is auto-cleaned by JUnit)
        // This is here as a safety measure and for any additional cleanup logic
        if (tempDir != null && Files.exists(tempDir)) {
            try (Stream<Path> paths = Files.walk(tempDir)) {
                paths.sorted(Comparator.reverseOrder()).forEach(path -> {
                    try {
                        Files.deleteIfExists(path);
                    } catch (IOException e) {
                        // Log but don't fail test on cleanup errors
                        System.err.println("Failed to delete: " + path);
                    }
                });
            }
        }
    }
}
