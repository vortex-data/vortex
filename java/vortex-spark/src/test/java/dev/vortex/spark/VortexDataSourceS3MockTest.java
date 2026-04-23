// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.adobe.testing.s3mock.testcontainers.S3MockContainer;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.*;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

/**
 * Integration test for Vortex DataSource with mocked S3 using Adobe S3Mock.
 *
 * <p>This test verifies that Vortex can correctly read and write files from S3-compatible
 * storage by using S3Mock running as a Testcontainer.
 */
@Testcontainers
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDataSourceS3MockTest {

    private static final String TEST_BUCKET = "vortex-test-bucket";

    @Container
    private static final S3MockContainer S3_MOCK = new S3MockContainer("4.11.0").withInitialBuckets(TEST_BUCKET);

    private SparkSession spark;

    @BeforeAll
    public void setUp() {
        // Get the S3Mock endpoint
        String s3Endpoint = S3_MOCK.getHttpEndpoint();

        // Create a local Spark session configured to use S3Mock
        spark = SparkSession.builder()
                .appName("VortexS3MockTest")
                .master("local[2]")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                // S3A configuration for S3Mock.
                // This should be propagated into our reader
                .config("spark.hadoop.fs.s3a.endpoint", s3Endpoint)
                .config("spark.hadoop.fs.s3a.access.key", "foo")
                .config("spark.hadoop.fs.s3a.secret.key", "bar")
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                // Disable features that S3Mock may not support
                .config("spark.hadoop.fs.s3a.change.detection.version.required", "false")
                .config("spark.hadoop.fs.s3a.change.detection.mode", "none")
                .config("spark.hadoop.fs.s3a.multiobjectdelete.enable", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Write and read Vortex files from mocked S3")
    public void testWriteAndReadVortexFilesFromS3Mock() {
        // Given: Create a DataFrame with test data
        int numRows = 100;
        Dataset<Row> originalDf = createTestDataFrame(numRows);

        // When: Write to S3Mock
        String s3Path = "s3a://" + TEST_BUCKET + "/vortex-test";
        originalDf
                .repartition(2)
                .write()
                .format("vortex")
                .option("path", s3Path)
                .mode(SaveMode.Overwrite)
                .save();

        // Then: Read back from S3Mock
        Dataset<Row> readDf = spark.read().format("vortex").load(s3Path);

        // Verify schema is preserved
        assertSchemaEquals(originalDf.schema(), readDf.schema());

        // Verify row count
        assertEquals(numRows, readDf.count(), "Read DataFrame should have same number of rows as original");

        // Verify data content
        verifyDataContent(originalDf, readDf);
    }

    @Test
    @DisplayName("Write and read Vortex files with format options from S3Mock")
    public void testWriteAndReadWithFormatOptionsFromS3Mock() {
        // Given: Create a DataFrame with test data
        int numRows = 50;
        Dataset<Row> originalDf = createTestDataFrame(numRows);

        // When: Write to S3Mock with format options
        String s3Path = "s3a://" + TEST_BUCKET + "/vortex-options-test";
        originalDf
                .write()
                .format("vortex")
                .option("path", s3Path)
                .mode(SaveMode.Overwrite)
                .save();

        // Then: Read back from S3Mock with format options
        Dataset<Row> readDf =
                spark.read().format("vortex").option("path", s3Path).load();

        // Verify row count
        assertEquals(numRows, readDf.count(), "Read DataFrame should have same number of rows as original");

        // Verify data content
        verifyDataContent(originalDf, readDf);
    }

    /**
     * Creates a test DataFrame with monotonically increasing integers
     * and their string representations.
     */
    private Dataset<Row> createTestDataFrame(int numRows) {
        return spark.range(0, numRows)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id as string)) as value",
                        "array('Alpha', 'Bravo', 'Charlie') AS elements");
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
}
