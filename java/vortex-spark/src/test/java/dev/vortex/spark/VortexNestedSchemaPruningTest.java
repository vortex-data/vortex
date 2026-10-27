// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that reading one field of a nested column returns that field.
 *
 * <p>With {@code spark.sql.optimizer.nestedSchemaPruning.enabled}, Spark asks the scan for only the nested fields a
 * query uses. Each row of the table holds {@code a = id + 100}, {@code b = id + 200} and {@code c = id + 300}, so a
 * field read from the wrong position shows up as the wrong hundred.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaPruningTest {

    private static final List<Long> C_VALUES = List.of(300L, 301L, 302L, 303L, 304L);

    private SparkSession spark;
    private Dataset<Row> table;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaPruningTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.optimizer.nestedSchemaPruning.enabled", "true")
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        Column abc = functions.struct(
                functions.col("id").plus(100).alias("a"),
                functions.col("id").plus(200).alias("b"),
                functions.col("id").plus(300).alias("c"));
        Dataset<Row> df = spark.range(5)
                .select(
                        functions.col("id"),
                        abc.alias("s"),
                        functions.array(abc).alias("arr"),
                        functions.map(functions.lit("k"), abc).alias("m"),
                        functions
                                .struct(functions.lit("x").alias("p"), abc.alias("inner"))
                                .alias("outer"));

        String path = tempDir.resolve("nested").toUri().toString();
        df.write()
                .format("vortex")
                .option("path", path)
                .mode(SaveMode.Overwrite)
                .save();
        table = spark.read().format("vortex").option("path", path).load();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Selecting a struct field other than the first returns that field")
    public void structField() {
        assertEquals(C_VALUES, longsOf(functions.col("s.c")));
    }

    @Test
    @DisplayName("Selecting a field of the structs in an array returns that field")
    public void arrayElementField() {
        assertEquals(C_VALUES, longsOf(functions.element_at(functions.col("arr.c"), 1)));
    }

    @Test
    @DisplayName("Selecting a field of the structs in a map returns that field")
    public void mapValueField() {
        assertEquals(C_VALUES, longsOf(functions.col("m").getItem("k").getField("c")));
    }

    @Test
    @DisplayName("Selecting a field of a struct inside a struct returns that field")
    public void structInStructField() {
        assertEquals(C_VALUES, longsOf(functions.col("outer.inner.c")));
    }

    private List<Long> longsOf(Column column) {
        return table.orderBy("id").select(column).collectAsList().stream()
                .map(row -> row.getLong(0))
                .collect(Collectors.toList());
    }
}
