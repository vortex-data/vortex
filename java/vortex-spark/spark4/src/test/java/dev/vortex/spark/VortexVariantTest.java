// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.VariantType$;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** Spark's VARIANT type, which Vortex stores as Arrow's parquet variant extension. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexVariantTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexVariantTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
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
    void variantsRoundTripTopLevelAndNested() {
        Dataset<Row> data = spark.range(0, 500)
                .selectExpr(
                        "id",
                        "case when id % 11 = 0 then null else parse_json(concat("
                                + "'{\"a\": ', id, ', \"b\": \"s', id % 7, '\", \"c\": [', id, ', true, null]}'"
                                + ")) end AS v",
                        "named_struct('inner', parse_json(concat('[', id, ', \"x\"]'))) AS s");
        Path output = tempDir.resolve("variants");
        data.write().format("vortex").mode(SaveMode.Overwrite).save(output.toString());

        Dataset<Row> read = spark.read().format("vortex").load(output.toString());
        assertEquals(VariantType$.MODULE$, read.schema().apply("v").dataType());
        assertEquals(
                DataTypes.createStructType(new org.apache.spark.sql.types.StructField[] {
                    DataTypes.createStructField("inner", VariantType$.MODULE$, true)
                }),
                read.schema().apply("s").dataType());

        String[] projection = {
            "id", "to_json(v)", "variant_get(v, '$.a', 'int')", "variant_get(v, '$.b', 'string')", "to_json(s.inner)"
        };
        assertEquals(
                data.selectExpr(projection).orderBy("id").collectAsList(),
                read.selectExpr(projection).orderBy("id").collectAsList());
    }
}
