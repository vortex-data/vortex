// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.spark.read.VortexScan42;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.expressions.filter.PartitionPredicate;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.datasources.FilePartition;
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import scala.collection.JavaConverters;

/** What the Spark 4.2 scan adds: {@code TABLESAMPLE SYSTEM} and partition predicates. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSpark42Test {
    private static final long ROWS = 200_000;

    private SparkSession spark;
    private Path rows;
    private Path partitioned;

    // Static, so it exists before the per-class setup writes the data.
    @TempDir
    static Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexSpark42Test")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        rows = tempDir.resolve("rows");
        // Spark takes no seed for a SYSTEM sample. With 40 files of contiguous ids, a sample of no rows or of every
        // row is too rare to fail the test.
        spark.range(0, ROWS, 1, 40)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(rows.toString());

        partitioned = tempDir.resolve("partitioned");
        spark.range(0, 10_000)
                .selectExpr("id", "id % 20 AS p", "id % 2 AS q")
                .write()
                .format("vortex")
                .partitionBy("p", "q")
                .mode(SaveMode.Overwrite)
                .save(partitioned.toString());
        spark.range(0, 20).selectExpr("id AS k").createOrReplaceTempView("keys");
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void systemSampleKeepsWholeChunks() {
        spark.read().format("vortex").load(rows.toString()).createOrReplaceTempView("rows");
        String query = "SELECT id FROM rows TABLESAMPLE SYSTEM (30 PERCENT)";
        Dataset<Row> sample = spark.sql(query);
        String plan = sample.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("PushedSample: SAMPLE SYSTEM [0.0, 0.3)"), plan);

        long[] ids = sample.collectAsList().stream()
                .mapToLong(row -> row.getLong(0))
                .sorted()
                .toArray();
        assertTrue(ids.length > 0 && ids.length < ROWS, "sampled " + ids.length + " rows");

        // Whole chunks give long runs of consecutive ids, where a row sample gives runs of about one row.
        long runs = 1;
        for (int i = 1; i < ids.length; i++) {
            if (ids[i] != ids[i - 1] + 1) {
                runs++;
            }
        }
        assertTrue(ids.length / runs >= 100, ids.length + " rows in " + runs + " runs");
    }

    @ParameterizedTest
    @CsvSource({"'id, p, q', true", "'id, p', false"})
    void partitionPredicatesDropFiles(String columns, boolean pushed) {
        spark.read().format("vortex").load(partitioned.toString()).createOrReplaceTempView("facts");
        // `p % 7` has no V1 filter form, so Spark wraps the runtime filter in a partition predicate. Spark does so
        // only when the scan reads every partition column.
        String query = "SELECT " + columns + " FROM facts WHERE p % 7 = (SELECT max(k) % 7 FROM keys WHERE k < 4)";
        Dataset<Row> frame = spark.sql(query);
        // Collect this very plan: its scan is the one Spark narrows at run time.
        List<Row> result = frame.collectAsList();
        assertEquals(1500, result.size());
        assertTrue(result.stream().allMatch(row -> row.getInt(1) % 7 == 3));

        VortexScan42 scan = vortexScan(frame.queryExecution().executedPlan());
        assertEquals(
                pushed,
                Arrays.stream(scan.pushedPredicates()).anyMatch(PartitionPredicate.class::isInstance),
                Arrays.toString(scan.pushedPredicates()) + "\n"
                        + frame.queryExecution().executedPlan());
        long directories = Arrays.stream(scan.toBatch().planInputPartitions())
                .flatMap(partition -> Arrays.stream(((FilePartition) partition).files()))
                .map(file -> file.toPath().getParent().getParent().getName())
                .distinct()
                .count();
        // p is 3, 10 or 17.
        assertEquals(pushed ? 3 : 20, directories);
    }

    private static VortexScan42 vortexScan(SparkPlan plan) {
        for (SparkPlan node : JavaConverters.seqAsJavaList(plan.collectLeaves())) {
            if (node instanceof BatchScanExec batch && batch.scan() instanceof VortexScan42 scan) {
                return scan;
            }
        }
        throw new AssertionError("no Vortex scan in " + plan);
    }
}
