// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Statistics;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration tests for {@link VortexScan#estimateStatistics()}.
 *
 * <p>Verifies that the Spark V2 scan surfaces both the row count Vortex stores in each file footer and the sum of the
 * on-storage file sizes reported by the filesystem listing.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDataSourceStatsTest {
    private static final String FILE_COMPRESSION_FACTOR_KEY = "spark.sql.sources.fileCompressionFactor";

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexStatsTest")
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
    @DisplayName("VortexScan reports exact row count for single-file scans")
    public void testEstimateStatisticsReportsRowCount() throws IOException {
        int numRows = 250;
        Path outputPath = writeRows(numRows, "single_file");

        VortexScan scan = buildScan(outputPath);
        Statistics stats = scan.estimateStatistics();

        assertTrue(
                stats.numRows().isPresent(),
                "VortexScan should report a row count for a Vortex dataset with a populated footer");
        assertEquals(numRows, stats.numRows().getAsLong(), "Row count should match the rows we wrote");
    }

    @Test
    @DisplayName("VortexScan reports aggregate row count across multi-file scans")
    public void testEstimateStatisticsAcrossMultipleFiles() throws IOException {
        int numRows = 400;
        Path outputPath = writeRows(numRows, "multi_file", 4);

        VortexScan scan = buildScan(outputPath);
        Statistics stats = scan.estimateStatistics();

        assertTrue(stats.numRows().isPresent(), "Row count should be reported for multi-file Vortex datasets");
        assertEquals(numRows, stats.numRows().getAsLong(), "Row count should sum across all files");
    }

    @Test
    @DisplayName("VortexScan reports sizeInBytes equal to the sum of on-storage file sizes")
    public void testEstimateStatisticsReportsSizeInBytes() throws IOException {
        Path outputPath = writeRows(120, "with_size", 3);

        long fileBytes = totalVortexFileBytes(outputPath);
        assertTrue(fileBytes > 0, "Test setup should produce at least one non-empty .vortex file");

        VortexScan scan = buildScan(outputPath);
        Statistics stats = scan.estimateStatistics();

        assertTrue(
                stats.sizeInBytes().isPresent(),
                "VortexScan should surface a sizeInBytes when the filesystem listing reports file sizes");
        // Mirror the scan's Spark-convention scaling (factor 1.0, unpruned schema), which divides and
        // re-multiplies by the schema default size in double arithmetic before truncating; asserting
        // against the raw byte sum would be sensitive to the floating-point round trip.
        StructType schema = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load()
                .schema();
        long expectedSize = (long) (1.0 * fileBytes / schema.defaultSize() * schema.defaultSize());
        assertEquals(
                expectedSize,
                stats.sizeInBytes().getAsLong(),
                "sizeInBytes should equal the sum of on-storage .vortex file sizes");
    }

    @Test
    @DisplayName("VortexScan scales sizeInBytes by the pushed read schema")
    public void testEstimateStatisticsScalesSizeInBytesForProjection() throws IOException {
        Path outputPath = writeRows(120, "projected_size", 3);
        long fileBytes = totalVortexFileBytes(outputPath);

        StructType fullSchema = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load()
                .schema();
        StructType idOnlySchema = new StructType(new StructField[] {fullSchema.fields()[0]});

        String previousCompressionFactor = spark.conf().get(FILE_COMPRESSION_FACTOR_KEY);
        spark.conf().set(FILE_COMPRESSION_FACTOR_KEY, "0.5");
        try {
            VortexScan scan = buildScan(outputPath, idOnlySchema);
            Statistics stats = scan.estimateStatistics();

            long expectedSize = (long) (0.5 * fileBytes / fullSchema.defaultSize() * idOnlySchema.defaultSize());
            assertTrue(stats.sizeInBytes().isPresent(), "Projected scans should still surface sizeInBytes");
            assertEquals(
                    expectedSize,
                    stats.sizeInBytes().getAsLong(),
                    "sizeInBytes should follow Spark FileScan's compression and schema-width scaling");
            assertTrue(
                    stats.sizeInBytes().getAsLong() < fileBytes,
                    "Projected scan stats should be smaller than full file bytes");
        } finally {
            spark.conf().set(FILE_COMPRESSION_FACTOR_KEY, previousCompressionFactor);
        }
    }

    @Test
    @DisplayName("VortexScan caches statistics across repeated calls")
    public void testEstimateStatisticsIsCached() throws IOException {
        Path outputPath = writeRows(50, "cached", 1);

        VortexScan scan = buildScan(outputPath);
        Statistics first = scan.estimateStatistics();
        Statistics second = scan.estimateStatistics();

        // Same instance returned -- the second call hits the cached value.
        assertEquals(first, second, "estimateStatistics() should return the same Statistics object on repeat calls");
        assertInstanceOf(Statistics.class, first);
    }

    private VortexScan buildScan(Path outputPath) {
        return buildScan(outputPath, null);
    }

    private VortexScan buildScan(Path outputPath, StructType requiredSchema) {
        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();
        StructType readSchema = readDf.schema();

        VortexScanBuilder builder = new VortexScanBuilder(Map.of());
        builder.addPath(outputPath.toUri().toString());
        for (StructField field : readSchema.fields()) {
            builder.addColumn(Column.create(field.name(), field.dataType()));
        }
        if (requiredSchema != null) {
            builder.pruneColumns(requiredSchema);
        }
        return (VortexScan) builder.build();
    }

    private Path writeRows(int numRows, String name) throws IOException {
        return writeRows(numRows, name, 1);
    }

    private long totalVortexFileBytes(Path outputPath) throws IOException {
        try (Stream<Path> paths = Files.walk(outputPath)) {
            return paths.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(".vortex"))
                    .mapToLong(path -> {
                        try {
                            return Files.size(path);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    })
                    .sum();
        }
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
