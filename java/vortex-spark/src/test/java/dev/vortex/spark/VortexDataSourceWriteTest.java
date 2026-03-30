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

    /**
     * Finds all Vortex files in the given directory.
     */
    private List<Path> findVortexFiles(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return Arrays.asList();
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
