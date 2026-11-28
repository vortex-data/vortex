// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** File splitting, row indexes, LIMIT, top-N, sample, and nested column pushdown, and runtime filtering. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexScanFeaturesTest {
    private static final int ROWS = 200_000;

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexScanFeaturesTest")
                .master("local[2]")
                .config("spark.sql.extensions", VortexSparkSessionExtensions.class.getName())
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

    @AfterEach
    public void resetConf() {
        spark.conf().unset("spark.sql.files.maxPartitionBytes");
        spark.conf().unset("spark.sql.files.openCostInBytes");
        spark.conf().unset("spark.sql.sources.useV1SourceList");
        spark.conf().unset("spark.sql.optimizer.dynamicPartitionPruning.enabled");
    }

    @Test
    void splitsOfOneFileReadEachRowOnce() {
        Path output = writeOneLargeFile("split");
        splitSmall();

        for (String v1Sources : List.of("", "vortex")) {
            spark.conf().set("spark.sql.sources.useV1SourceList", v1Sources);
            Dataset<Row> data = spark.read()
                    .format("vortex")
                    .option("vortex.aggregatePushdown", "false")
                    .load(output.toString());

            assertTrue(data.rdd().getNumPartitions() > 1, "the file should be split for " + v1Sources);
            Row totals =
                    data.selectExpr("count(*)", "count(distinct id)", "sum(id)").first();
            assertEquals((long) ROWS, totals.getLong(0));
            assertEquals((long) ROWS, totals.getLong(1));
            assertEquals((long) ROWS * (ROWS - 1) / 2, totals.getLong(2));
        }
    }

    @Test
    void rowIndexMetadataCountsRowsAcrossSplits() {
        Path output = writeOneLargeFile("row_index");
        splitSmall();
        spark.conf().set("spark.sql.sources.useV1SourceList", "vortex");

        Dataset<Row> data = spark.read().format("vortex").load(output.toString());
        Dataset<Row> indexed = data.selectExpr("id", "_metadata.row_index AS row_index");

        // The single file holds the ids in order, so each row's index in the file is its id.
        assertEquals(0L, indexed.filter("row_index != id").count());
        assertEquals((long) ROWS, indexed.count());
        // The row index is assigned before filtering, so a filter keeps each surviving row's index.
        assertEquals(
                List.of(RowFactory.create(12345L, 12345L)),
                indexed.filter("id = 12345").collectAsList());
    }

    @Test
    void limitIsPushedToEachFile() {
        Path output = tempDir.resolve("limit");
        spark.range(0, 1000).repartition(4).write().format("vortex").save(output.toString());

        Dataset<Row> limited =
                spark.read().format("vortex").load(output.toString()).limit(7);

        assertTrue(limited.queryExecution().executedPlan().toString().contains("PushedLimit: LIMIT 7"));
        assertEquals(7, limited.collectAsList().size());
    }

    @Test
    void topNIsPushedAndMatchesASortedScan() {
        Path output = tempDir.resolve("top_n");
        spark.range(0, 100_000)
                .selectExpr(
                        "id",
                        "case when id % 97 = 0 then null else (id * 7919) % 100003 end AS key",
                        "cast(id % 13 AS string) AS tie",
                        "concat('s', (id * 31) % 1009) AS name")
                .repartition(6)
                .write()
                .format("vortex")
                .save(output.toString());

        // Rows that tie on every sort key may come back in any order, and either choice is a correct top 25, so
        // single-key orders compare only the keys, and multi-key orders end with the unique id.
        List<String> orders = List.of(
                "key ASC NULLS FIRST",
                "key ASC NULLS LAST",
                "key DESC NULLS FIRST",
                "key DESC NULLS LAST",
                "name ASC, id DESC",
                "key DESC, tie ASC, id ASC");
        for (String order : orders) {
            String columns = order.contains(",") ? "id, key, name" : "key";
            String query = String.format("SELECT %s FROM v ORDER BY %s LIMIT 25", columns, order);

            spark.read().format("vortex").load(output.toString()).createOrReplaceTempView("v");
            Dataset<Row> pushed = spark.sql(query);
            String plan = pushed.queryExecution().executedPlan().toString();
            assertTrue(plan.contains("PushedTopN: ORDER BY"), order + ":\n" + plan);
            List<Row> pushedRows = pushed.collectAsList();

            spark.read()
                    .format("vortex")
                    .option("vortex.topNPushdown", "false")
                    .load(output.toString())
                    .createOrReplaceTempView("v");
            assertEquals(spark.sql(query).collectAsList(), pushedRows, order);
        }
    }

    @Test
    void topNKeepsRowsThatTieTheFooterBound() {
        Path output = tempDir.resolve("top_n_ties");
        // The first file holds exactly the limit of rows, all equal to its minimum, so the footers prove the result's
        // last key is at least 10, and rows equal to 10 must still be read.
        spark.range(0, 25)
                .selectExpr("id", "10L AS key")
                .write()
                .format("vortex")
                .save(output.toString());
        spark.range(25, 125)
                .selectExpr("id", "id % 11 AS key")
                .write()
                .format("vortex")
                .mode(SaveMode.Append)
                .save(output.toString());

        for (String order : List.of("key DESC", "key ASC")) {
            String query = "SELECT key FROM v ORDER BY " + order + " LIMIT 30";
            spark.read().format("vortex").load(output.toString()).createOrReplaceTempView("v");
            List<Row> pushed = spark.sql(query).collectAsList();
            spark.read()
                    .format("vortex")
                    .option("vortex.topNPushdown", "false")
                    .load(output.toString())
                    .createOrReplaceTempView("v");
            assertEquals(spark.sql(query).collectAsList(), pushed, order);
        }
    }

    @Test
    void tableSampleIsPushedAndDoesNotDependOnSplits() {
        Path output = writeOneLargeFile("sample");

        Dataset<Row> sampled =
                spark.read().format("vortex").load(output.toString()).sample(0.1, 42);
        assertTrue(sampled.queryExecution().executedPlan().toString().contains("PushedSample"));
        // A sample is not every row, so COUNT(*) must not come from the footers.
        long count = sampled.count();
        assertTrue(count > ROWS * 0.09 && count < ROWS * 0.11, "sampled " + count + " of " + ROWS);

        List<Row> whole = sampled.select("id").orderBy("id").collectAsList();
        assertEquals(count, whole.size());
        assertEquals(
                whole,
                spark.read()
                        .format("vortex")
                        .load(output.toString())
                        .sample(0.1, 42)
                        .select("id")
                        .orderBy("id")
                        .collectAsList());

        splitSmall();
        Dataset<Row> splitSample =
                spark.read().format("vortex").load(output.toString()).sample(0.1, 42);
        assertTrue(splitSample.rdd().getNumPartitions() > 1);
        assertEquals(whole, splitSample.select("id").orderBy("id").collectAsList());
    }

    @Test
    void offsetSkipsWholeFilesAndEndsMidFile() {
        Path output = tempDir.resolve("offset");
        spark.range(0, 10_000).repartition(7).write().format("vortex").save(output.toString());
        Dataset<Row> data = spark.read()
                .format("vortex")
                .option("vortex.aggregatePushdown", "false")
                .load(output.toString());

        for (int offset : new int[] {1, 1499, 4321, 9999, 10_000, 12_000}) {
            Dataset<Row> rest = data.offset(offset);
            assertTrue(rest.queryExecution().executedPlan().toString().contains("PushedOffset: OFFSET " + offset));
            List<Row> rows = rest.collectAsList();
            assertEquals(Math.max(0, 10_000 - offset), rows.size(), "offset " + offset);
            assertEquals(rows.size(), rows.stream().distinct().count(), "offset " + offset);
        }

        // Spark pushes LIMIT offset + n per file and the offset, and keeps the LIMIT n above the scan.
        assertEquals(25, data.offset(4321).limit(25).collectAsList().size());
    }

    @Test
    void nestedColumnsArePrunedAndKeepStructNulls() {
        Path output = tempDir.resolve("nested");
        spark.range(0, 100)
                .selectExpr(
                        "id",
                        "case when id % 10 = 0 then null "
                                + "else named_struct('a', id, 'b', concat('b', id), 'c', named_struct('x', id)) end AS s")
                .write()
                .format("vortex")
                .save(output.toString());

        Dataset<Row> pruned =
                spark.read().format("vortex").load(output.toString()).selectExpr("id", "s.a AS a");

        String plan = pruned.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("struct<id:bigint,s:struct<a:bigint>>"), plan);
        List<Row> rows = pruned.orderBy("id").collectAsList();
        assertEquals(100, rows.size());
        for (Row row : rows) {
            long id = row.getLong(0);
            if (id % 10 == 0) {
                assertTrue(row.isNullAt(1), "a null struct keeps a null field " + id);
            } else {
                assertEquals(id, row.getLong(1));
            }
        }
    }

    @Test
    void decimalsReadBackAtEveryPrecision() {
        Path output = tempDir.resolve("decimals");
        // Up to 18 digits a decimal crosses the native boundary as its unscaled long, and above as a 128-bit decimal.
        Dataset<Row> written = spark.range(0, 5_000)
                .selectExpr(
                        "id",
                        "CASE WHEN id % 7 = 0 THEN NULL ELSE cast((id - 2500) / 100.0 AS decimal(7, 2)) END AS small",
                        "cast(id * 1000003 - 999999999 AS decimal(18, 0)) AS widest_long",
                        "CASE WHEN id % 5 = 0 THEN NULL ELSE cast(id / 3.0 AS decimal(15, 4)) END AS medium",
                        "cast(id AS decimal(38, 10)) * 1000000 AS wide");
        written.repartition(3).write().format("vortex").save(output.toString());

        Dataset<Row> read = spark.read().format("vortex").load(output.toString());
        assertEquals(written.schema().simpleString(), read.schema().simpleString());
        assertEquals(written.orderBy("id").collectAsList(), read.orderBy("id").collectAsList());
        assertEquals(
                written.selectExpr("sum(small)", "sum(widest_long)", "sum(medium)")
                        .collectAsList(),
                read.selectExpr("sum(small)", "sum(widest_long)", "sum(medium)").collectAsList());
    }

    private Path writeOneLargeFile(String name) {
        Path output = tempDir.resolve(name);
        spark.range(0, ROWS)
                .selectExpr("id", "concat('payload_', id, '_', sha2(cast(id AS string), 256)) AS payload")
                .coalesce(1)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());
        return output;
    }

    /** Splits files into pieces of at most 256 KiB, so one file spans many tasks. */
    private void splitSmall() {
        spark.conf().set("spark.sql.files.maxPartitionBytes", "262144");
        spark.conf().set("spark.sql.files.openCostInBytes", "0");
    }
}
