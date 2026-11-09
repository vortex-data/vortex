// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
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

/** Integration tests for row-count and byte-size statistics exposed to Catalyst. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDataSourceStatsTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexStatsTest")
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
    void reportsExactRowCountAcrossFiles() throws IOException {
        Path output = writeRows(400, "row_count", 4);

        Statistics statistics = statistics(read(output));

        assertTrue(statistics.rowCount().isDefined());
        assertEquals(400L, statistics.rowCount().get().longValue());
        assertTrue(statistics.sizeInBytes().longValue() > 0);
    }

    @Test
    void rowCountStatisticsCanBeDisabled() throws IOException {
        Path output = writeRows(25, "disabled", 2);

        Dataset<Row> data = spark.read()
                .format("vortex")
                .option("vortex.stats.rowCount", "false")
                .load(output.toString());

        assertFalse(statistics(data).rowCount().isDefined());
        assertEquals(25, data.count());
    }

    @Test
    void rowCountIsSkippedWhenTheDatasetHasTooManyFiles() throws IOException {
        Path output = writeRows(40, "too_many_files", 4);

        Dataset<Row> capped = spark.read()
                .format("vortex")
                .option("vortex.stats.maxFiles", "2")
                .load(output.toString());

        // Each footer costs a read on the driver, so a large dataset reports no row count rather than
        // paying for the whole listing before the job starts.
        assertFalse(statistics(capped).rowCount().isDefined());
        assertEquals(40, capped.count());
    }

    @Test
    void projectedScanHasSmallerByteEstimate() throws IOException {
        Path output = writeRows(120, "projected", 3);
        Dataset<Row> full = read(output);
        Dataset<Row> projected = full.select("id");

        assertTrue(statistics(projected).sizeInBytes().longValue()
                < statistics(full).sizeInBytes().longValue());
    }

    private Statistics statistics(Dataset<Row> data) {
        return data.queryExecution().optimizedPlan().stats();
    }

    private Dataset<Row> read(Path output) {
        return spark.read().format("vortex").load(output.toString());
    }

    private Path writeRows(int count, String name, int partitions) throws IOException {
        Path output = tempDir.resolve(name);
        spark.range(0, count)
                .selectExpr("cast(id as int) as id", "concat('value_', cast(id as string)) as value")
                .repartition(partitions)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());
        return output;
    }
}
