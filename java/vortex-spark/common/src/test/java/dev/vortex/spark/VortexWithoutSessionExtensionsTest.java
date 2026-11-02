// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** Dynamic pruning in a session that does not apply {@link VortexSparkSessionExtensions}. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexWithoutSessionExtensionsTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexWithoutSessionExtensionsTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void onlyPartitionColumnsArePruned() {
        Path facts = tempDir.resolve("facts");
        spark.range(0, 10_000)
                .selectExpr("id", "id % 1000 AS key", "cast(id % 8 AS int) AS part")
                .write()
                .format("vortex")
                .partitionBy("part")
                .save(facts.toString());

        Path dimensions = tempDir.resolve("dimensions");
        spark.range(0, 1000)
                .selectExpr("id AS key", "cast(id % 8 AS int) AS part", "concat('name_', id % 50) AS name")
                .write()
                .format("vortex")
                .save(dimensions.toString());

        assertTrue(plan(facts, dimensions, "part").contains("dynamicpruningexpression"));
        assertFalse(plan(facts, dimensions, "key").contains("dynamicpruningexpression"));
        assertEquals(200, joined(facts, dimensions, "key").count());
    }

    private String plan(Path facts, Path dimensions, String joinColumn) {
        return joined(facts, dimensions, joinColumn)
                .queryExecution()
                .executedPlan()
                .toString();
    }

    private Dataset<Row> joined(Path facts, Path dimensions, String joinColumn) {
        Dataset<Row> fact = spark.read().format("vortex").load(facts.toString());
        Dataset<Row> dimension =
                spark.read().format("vortex").load(dimensions.toString()).filter("name = 'name_7'");
        return fact.join(dimension, fact.col(joinColumn).equalTo(dimension.col(joinColumn)))
                .select(fact.col("id"));
    }
}
