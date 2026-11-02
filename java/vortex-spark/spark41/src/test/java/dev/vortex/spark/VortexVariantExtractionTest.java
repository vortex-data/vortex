// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** Spark 4.1's variant extraction pushdown, which Vortex answers by extracting each path natively. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexVariantExtractionTest {
    private SparkSession spark;
    private Path output;

    // Static, so it exists before the per-class setup writes the data.
    @TempDir
    static Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexVariantExtractionTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        output = tempDir.resolve("variants");
        spark.range(0, 2000)
                .selectExpr(
                        "id",
                        "case when id % 13 = 0 then null else parse_json(concat("
                                + "'{\"a\": ', id, ', \"b\": \"s', id % 7, '\", \"c\": [', id * 2, ', \"x\"], "
                                + "\"d\": {\"e\": ', case when id % 5 = 0 then 'null' else cast(id as string) end, '}}'"
                                + ")) end AS v")
                .repartition(3)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void pushedExtractionsMatchSpark() {
        for (String query : List.of(
                "SELECT id, variant_get(v, '$.a', 'int'), variant_get(v, '$.b', 'string') FROM v",
                "SELECT id, variant_get(v, '$.c[0]', 'long'), variant_get(v, '$.c[1]', 'string') FROM v",
                "SELECT id, variant_get(v, '$.d.e', 'int'), try_variant_get(v, '$.b', 'int') FROM v",
                "SELECT id, variant_get(v, '$.missing', 'int'), to_json(v) FROM v",
                "SELECT id FROM v WHERE variant_get(v, '$.a', 'int') > 1990",
                "SELECT id FROM v WHERE v IS NULL")) {
            List<Row> pushed = run(query, true);
            assertEquals(run(query, false), pushed, query);
        }
    }

    @Test
    void extractionsArePushedOnlyWhenEnabled() {
        String query = "SELECT variant_get(v, '$.a', 'int') FROM v";
        assertTrue(plan(query, true).contains("PushedVariants: [v: struct<0:int>]"), plan(query, true));
        assertFalse(plan(query, false).contains("PushedVariants"), plan(query, false));
    }

    @Test
    void failingCastsFailAsSparkDoes() {
        // `b` holds strings such as "s3", which do not cast to int.
        String query = "SELECT id, variant_get(v, '$.b', 'int') FROM v";
        assertThrows(Exception.class, () -> run(query, true));
        assertThrows(Exception.class, () -> run(query, false));
    }

    private List<Row> run(String query, boolean pushdown) {
        return frame(query, pushdown).orderBy("id").collectAsList();
    }

    private String plan(String query, boolean pushdown) {
        return frame(query, pushdown).queryExecution().executedPlan().toString();
    }

    private Dataset<Row> frame(String query, boolean pushdown) {
        spark.read()
                .format("vortex")
                .option("vortex.variantPushdown", Boolean.toString(pushdown))
                .load(output.toString())
                .createOrReplaceTempView("v");
        return spark.sql(query);
    }
}
