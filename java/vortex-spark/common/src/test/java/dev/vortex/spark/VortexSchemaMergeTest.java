// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Datasets whose files do not all carry the same columns.
 *
 * <p>Schema inference merges every footer, so a column added by a later write belongs to the dataset, and the files
 * written before it read as null.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSchemaMergeTest {
    private static final StructType NARROW =
            new StructType().add("a", DataTypes.IntegerType, false).add("b", DataTypes.StringType, true);

    private static final StructType WIDE = new StructType()
            .add("a", DataTypes.IntegerType, false)
            .add("b", DataTypes.StringType, true)
            .add("c", DataTypes.DoubleType, true);

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexSchemaMergeTest")
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
    void aColumnOnlyOneFileCarriesBelongsToTheDataset() {
        Path dir = tempDir.resolve("widened");
        append(dir, NARROW, RowFactory.create(1, "one"));
        append(dir, WIDE, RowFactory.create(2, "two", 2.5));

        Dataset<Row> data = read(dir).orderBy("a");

        assertEquals("struct<a:int,b:string,c:double>", data.schema().simpleString());
        List<Row> rows = data.collectAsList();
        assertEquals(2, rows.size());
        // The file written before `c` existed has no value for it.
        assertNull(rows.get(0).get(2));
        assertEquals(2.5, rows.get(1).getDouble(2));
    }

    @Test
    void aColumnOnlyOneFileCarriesIsNullableEvenWhenThatFileRequiresIt() {
        Path dir = tempDir.resolve("nullability");
        append(dir, NARROW, RowFactory.create(1, "one"));
        append(
                dir,
                new StructType()
                        .add("a", DataTypes.IntegerType, false)
                        .add("b", DataTypes.StringType, true)
                        .add("d", DataTypes.IntegerType, false),
                RowFactory.create(2, "two", 9));

        assertTrue(read(dir).schema().apply("d").nullable());
    }

    @Test
    void selectingOnlyTheAddedColumnStillReadsEveryFile() {
        Path dir = tempDir.resolve("projected");
        append(dir, NARROW, RowFactory.create(1, "one"));
        append(dir, WIDE, RowFactory.create(2, "two", 2.5));

        List<Row> rows = read(dir).selectExpr("c").collectAsList();

        assertEquals(2, rows.size());
        assertEquals(1, rows.stream().filter(row -> row.isNullAt(0)).count());
    }

    @Test
    void filteringOnTheAddedColumnKeepsOnlyTheFileThatCarriesIt() {
        Path dir = tempDir.resolve("filtered");
        append(dir, NARROW, RowFactory.create(1, "one"));
        append(dir, WIDE, RowFactory.create(2, "two", 2.5));

        // The filter cannot be pushed into the file that lacks `c`, so Spark's own filter above the scan is what
        // drops that file's rows.
        List<Row> rows = read(dir).where("c > 1.0").collectAsList();

        assertEquals(1, rows.size());
        assertEquals(2, rows.get(0).getInt(0));
    }

    @Test
    void mergingCanBeTurnedOffToReadOneFooter() {
        Path dir = tempDir.resolve("unmerged");
        append(dir, NARROW, RowFactory.create(1, "one"));
        append(dir, NARROW, RowFactory.create(2, "two"));

        // An unusable footer parallelism proves which path ran: merging reads footers in a pool, while a single
        // footer read never asks for one.
        Dataset<Row> data = spark.read()
                .format("vortex")
                .option("vortex.mergeSchema", "false")
                .option("vortex.stats.rowCount", "false")
                .option("vortex.footerParallelism", "0")
                .load(dir.toString());

        assertEquals("struct<a:int,b:string>", data.schema().simpleString());
        assertEquals(2, data.count());
    }

    @Test
    void mergingIsFoundUnderAnyOptionCasing() {
        Path dir = tempDir.resolve("option_case");
        append(dir, NARROW, RowFactory.create(1, "one"));

        Dataset<Row> data = spark.read()
                .format("vortex")
                .option("VORTEX.MERGESCHEMA", "false")
                .option("VORTEX.STATS.ROWCOUNT", "false")
                .option("VORTEX.FOOTERPARALLELISM", "0")
                .load(dir.toString());

        assertEquals(1, data.count());
    }

    @Test
    void aFieldWithTwoTypesNamesTheFileThatDisagrees() {
        Path dir = tempDir.resolve("conflict");
        append(dir, NARROW, RowFactory.create(1, "one"));
        append(
                dir,
                new StructType().add("a", DataTypes.StringType, false).add("b", DataTypes.StringType, true),
                RowFactory.create("two", "two"));

        String message = assertThrows(
                        IllegalArgumentException.class, () -> read(dir).schema())
                .getMessage();

        assertTrue(message.contains(".vortex"), message);
        assertTrue(message.contains("field a is"), message);
        assertTrue(message.contains("vortex.mergeSchema"), message);
    }

    @Test
    void aStructThatGainedAFieldCannotBeMerged() {
        Path dir = tempDir.resolve("nested_conflict");
        StructType narrowStruct = new StructType()
                .add("id", DataTypes.IntegerType, false)
                .add("s", new StructType().add("x", DataTypes.IntegerType, true), true);
        StructType widerStruct = new StructType()
                .add("id", DataTypes.IntegerType, false)
                .add(
                        "s",
                        new StructType().add("x", DataTypes.IntegerType, true).add("y", DataTypes.StringType, true),
                        true);
        append(dir, narrowStruct, RowFactory.create(1, RowFactory.create(1)));
        append(dir, widerStruct, RowFactory.create(2, RowFactory.create(2, "two")));

        String message = assertThrows(
                        IllegalArgumentException.class, () -> read(dir).schema())
                .getMessage();

        // A struct column is projected whole, as the file stores it, so a widened struct cannot be read back
        // through one merged schema.
        assertTrue(message.contains("nested field s"), message);
    }

    private Dataset<Row> read(Path dir) {
        return spark.read().format("vortex").load(dir.toString());
    }

    private void append(Path dir, StructType schema, Row... rows) {
        spark.createDataFrame(List.of(rows), schema)
                .coalesce(1)
                .write()
                .format("vortex")
                .mode(SaveMode.Append)
                .save(dir.toString());
    }
}
