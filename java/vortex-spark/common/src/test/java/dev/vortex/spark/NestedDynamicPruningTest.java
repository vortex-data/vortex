// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.spark.read.NestedDynamicPruning;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * The {@link NestedDynamicPruning} rule, which {@link VortexSparkSessionExtensions} installs.
 *
 * <p>The tables are Parquet files, because the rule changes Spark's own dynamic partition pruning and does not depend
 * on the file format.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class NestedDynamicPruningTest {
    private static final String PRUNING_FILTER = "dynamicpruningexpression";
    private static final int MAX_PRUNING_FILTERS = 40;

    // Each level joins the level below with itself. Without the rule, each level multiplies the pruning filters of
    // the level below by three. On Spark 4.0, the plan of this query holds 106 of them without the rule and 16 with
    // it.
    private static final String QUERY =
            "WITH l1 AS (SELECT f.id, f.key FROM fact f JOIN dim d ON f.key = d.key AND d.name = 'one'), "
                    + "l2 AS (SELECT a.id, a.key FROM l1 a JOIN l1 b ON a.key = b.key AND a.id = b.id), "
                    + "l3 AS (SELECT a.id, a.key FROM l2 a JOIN l2 b ON a.key = b.key AND a.id = b.id), "
                    + "l4 AS (SELECT a.id, a.key FROM l3 a JOIN l3 b ON a.key = b.key AND a.id = b.id), "
                    + "l5 AS (SELECT a.id, a.key FROM l4 a JOIN l4 b ON a.key = b.key AND a.id = b.id) "
                    + "SELECT count(*), sum(id) FROM l5";

    private SparkSession spark;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) throws IOException {
        Path warehouseDir = Files.createDirectory(tempDir.resolve("warehouse"));
        spark = SparkSession.builder()
                .appName("NestedDynamicPruningTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.warehouse.dir", warehouseDir.toUri().toString())
                .config("spark.sql.extensions", VortexSparkSessionExtensions.class.getName())
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        String factLocation = tempDir.resolve("fact").toUri().toString();
        spark.range(0, 1000)
                .selectExpr("id", "id % 10 AS key")
                .write()
                .partitionBy("key")
                .parquet(factLocation);
        spark.read().parquet(factLocation).createOrReplaceTempView("fact");

        // Files keep the optimizer from folding the dimension into a constant.
        String dimLocation = tempDir.resolve("dim").toUri().toString();
        spark.range(1, 3)
                .selectExpr("id AS key", "IF(id = 1, 'one', 'two') AS name")
                .write()
                .parquet(dimLocation);
        spark.read().parquet(dimLocation).createOrReplaceTempView("dim");
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("The session extensions install the rule")
    public void testExtensionsInstallTheRule() {
        spark.sql("SELECT 1").collectAsList();

        assertTrue(NestedDynamicPruning.installed(spark));
    }

    @Test
    @DisplayName("Self-joins do not nest pruning subqueries")
    public void testSelfJoinsDoNotNestPruningSubqueries() {
        int pruningFilters = pruningFilters();

        assertTrue(pruningFilters >= 1, "The joins on the partition column should prune: " + pruningFilters);
        assertTrue(pruningFilters <= MAX_PRUNING_FILTERS, "The pruning filters should not nest: " + pruningFilters);

        String pruning = "spark.sql.optimizer.dynamicPartitionPruning.enabled";
        spark.conf().set(pruning, "false");
        List<Row> expected;
        try {
            expected = spark.sql(QUERY).collectAsList();
        } finally {
            spark.conf().unset(pruning);
        }

        assertEquals(expected, spark.sql(QUERY).collectAsList());
    }

    private int pruningFilters() {
        String plan =
                spark.sql("EXPLAIN EXTENDED " + QUERY).collectAsList().get(0).getString(0);
        int count = 0;
        for (int at = plan.indexOf(PRUNING_FILTER); at >= 0; at = plan.indexOf(PRUNING_FILTER, at + 1)) {
            count++;
        }

        return count;
    }
}
