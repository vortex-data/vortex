// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.DynamicPruningSubquery;
import org.apache.spark.sql.catalyst.expressions.Expression;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import scala.collection.JavaConverters;

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
    void footerStatisticsDropFilesNoRowCanMatch() {
        Path output = tempDir.resolve("pruned");
        // Each of the four files holds a separate range of 1000 ids.
        spark.range(0, 4000, 1, 4)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());

        // An open cost above the split size gives each file a partition of its own.
        spark.conf().set("spark.sql.files.openCostInBytes", "1g");
        Dataset<Row> data = spark.read().format("vortex").load(output.toString());

        assertEquals(4, data.rdd().getNumPartitions());
        assertEquals(1, data.filter("id = 1500").rdd().getNumPartitions());
        assertEquals(List.of(RowFactory.create(1500L)), data.filter("id = 1500").collectAsList());

        assertEquals(2, data.filter("id IN (5, 3500)").rdd().getNumPartitions());
        assertEquals(2, data.filter("id < 1000 OR id >= 3000").rdd().getNumPartitions());
        assertEquals(0, data.filter("id > 3999").rdd().getNumPartitions());
        assertEquals(0L, data.filter("id > 3999").count());

        Dataset<Row> unpruned = spark.read()
                .format("vortex")
                .option("vortex.pruneFiles", "false")
                .load(output.toString());
        assertEquals(4, unpruned.filter("id = 1500").rdd().getNumPartitions());
    }

    @Test
    void filesSplitNoSmallerThanTheMinimumSplit() {
        Path output = writeOneLargeFile("min_split");
        // Spark alone would split the file in two, one piece for each core of the session.
        spark.conf().set("spark.sql.files.openCostInBytes", "0");
        Dataset<Row> whole = spark.read().format("vortex").load(output.toString());
        assertEquals(1, whole.rdd().getNumPartitions());

        Dataset<Row> split = spark.read()
                .format("vortex")
                .option("vortex.minSplitBytes", "256k")
                .option("vortex.aggregatePushdown", "false")
                .load(output.toString());
        assertTrue(split.rdd().getNumPartitions() > 1);
        assertEquals((long) ROWS, split.selectExpr("count(DISTINCT id)").first().getLong(0));
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
    void partitionKeysLetJoinsSkipTheShuffle() {
        Path left = tempDir.resolve("keyed_left");
        Path right = tempDir.resolve("keyed_right");
        spark.range(0, 20_000)
                .selectExpr("id", "cast(id % 5 AS int) AS p")
                .repartition(7)
                .write()
                .format("vortex")
                .partitionBy("p")
                .save(left.toString());
        spark.range(0, 500)
                .selectExpr("id AS rid", "cast(id % 5 AS int) AS p")
                .write()
                .format("vortex")
                .partitionBy("p")
                .save(right.toString());

        spark.conf().set("spark.sql.autoBroadcastJoinThreshold", "-1");
        try {
            spark.conf().set("spark.sql.sources.v2.bucketing.enabled", "true");
            Dataset<Row> keyed = joinOnPartitions(left, right);
            String plan = keyed.queryExecution().executedPlan().toString();
            assertFalse(plan.contains("Exchange hashpartitioning"), "a keyed join needs no shuffle:\n" + plan);
            Row keyedTotals = keyed.first();

            spark.conf().set("spark.sql.sources.v2.bucketing.enabled", "false");
            Dataset<Row> shuffled = joinOnPartitions(left, right);
            assertTrue(shuffled.queryExecution().executedPlan().toString().contains("Exchange hashpartitioning"));
            assertEquals(shuffled.first(), keyedTotals);
        } finally {
            spark.conf().unset("spark.sql.sources.v2.bucketing.enabled");
            spark.conf().unset("spark.sql.autoBroadcastJoinThreshold");
        }
    }

    private Dataset<Row> joinOnPartitions(Path left, Path right) {
        Dataset<Row> facts = spark.read().format("vortex").load(left.toString());
        Dataset<Row> others = spark.read().format("vortex").load(right.toString());
        return facts.join(others, "p").selectExpr("count(*)", "sum(id)", "sum(rid)");
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
    @Disabled("JNI writers record IsSorted only under an edition that includes vortex.is_sorted, and they enable core")
    void sortedFilesReportTheirOrderSoSparkNeedNotSort() {
        Path output = tempDir.resolve("sorted_files");
        spark.range(0, 50_000)
                .selectExpr("id", "id % 7 AS other")
                .repartition(5)
                .sortWithinPartitions("id")
                .write()
                .format("vortex")
                .save(output.toString());

        Dataset<Row> reported = spark.read()
                .format("vortex")
                .option("vortex.reportOrdering", "true")
                .load(output.toString())
                .sortWithinPartitions("id");
        String plan = reported.queryExecution().executedPlan().toString();
        assertFalse(plan.contains("Sort ["), "the files' order makes the sort redundant:\n" + plan);
        for (List<Row> partition : reported.select("id").javaRDD().glom().collect()) {
            for (int i = 1; i < partition.size(); i++) {
                assertTrue(partition.get(i - 1).getLong(0) <= partition.get(i).getLong(0));
            }
        }
        assertEquals(50_000L, reported.count());

        // Without the report, or for a column the footers do not state sorted, Spark sorts.
        assertTrue(spark.read()
                .format("vortex")
                .load(output.toString())
                .sortWithinPartitions("id")
                .queryExecution()
                .executedPlan()
                .toString()
                .contains("Sort ["));
        assertTrue(spark.read()
                .format("vortex")
                .option("vortex.reportOrdering", "true")
                .load(output.toString())
                .sortWithinPartitions("other")
                .queryExecution()
                .executedPlan()
                .toString()
                .contains("Sort ["));
    }

    @Test
    void sparkEvaluatesBloomFilterRuntimeFiltersAboveTheScan() {
        Path facts = tempDir.resolve("bloom_facts");
        spark.range(0, 50_000)
                .selectExpr("id", "id % 1000 AS key")
                .write()
                .format("vortex")
                .save(facts.toString());
        Path dimensions = tempDir.resolve("bloom_dimensions");
        spark.range(0, 1000)
                .selectExpr("id AS key", "concat('name_', id % 50) AS name")
                .write()
                .format("vortex")
                .save(dimensions.toString());

        // A shuffle join with a selective side, and no dynamic pruning, which Spark prefers when it applies.
        String[][] settings = {
            {"spark.sql.autoBroadcastJoinThreshold", "-1"},
            {"spark.sql.optimizer.dynamicPartitionPruning.enabled", "false"},
            {"spark.sql.optimizer.runtime.bloomFilter.enabled", "true"},
            {"spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold", "1"},
            {"spark.sql.optimizer.runtime.bloomFilter.creationSideThreshold", "100MB"}
        };
        for (String[] setting : settings) {
            spark.conf().set(setting[0], setting[1]);
        }
        try {
            Dataset<Row> fact = spark.read().format("vortex").load(facts.toString());
            Dataset<Row> dimension =
                    spark.read().format("vortex").load(dimensions.toString()).filter("name = 'name_7'");
            Dataset<Row> joined = fact.join(dimension, "key").select("id");
            String plan = joined.queryExecution().optimizedPlan().toString();

            // Spark injects the bloom filter after it pushed filters into the scan. Spark 4.2 also lists it among a
            // scan's runtime filters, but has no V2 form to hand the scan, so on a data column it is never among
            // what the scan pushes. If a Spark release ever translates it, this fails, and the connector should
            // take it.
            assertTrue(plan.contains("might_contain"), "Spark should inject a bloom filter:\n" + plan);
            String scans =
                    joined.queryExecution().executedPlan().collectLeaves().mkString("\n");
            Matcher pushed =
                    Pattern.compile("Pushed(Filters|Predicates): \\[[^\\]]*]").matcher(scans);
            while (pushed.find()) {
                assertFalse(pushed.group().contains("might_contain"), "a scan pushed the bloom filter:\n" + scans);
            }
            assertEquals(1000L, joined.count());
        } finally {
            for (String[] setting : settings) {
                spark.conf().unset(setting[0]);
            }
        }
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
    void runtimeFiltersNarrowDataAndPartitionColumns() {
        Path facts = tempDir.resolve("facts");
        spark.range(0, 50_000)
                .selectExpr("id", "id % 1000 AS key", "cast(id % 8 AS int) AS part")
                .repartition(8)
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

        // Reads through the DataFrame API take the V2 scan, which accepts runtime filters on data columns too.
        for (String joinColumn : List.of("key", "part")) {
            spark.conf().set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true");
            Dataset<Row> pruned = joinedIds(facts, dimensions, joinColumn);
            String plan = pruned.queryExecution().executedPlan().toString();
            assertTrue(plan.contains("dynamicpruningexpression"), "runtime filter on " + joinColumn + ":\n" + plan);
            List<Row> prunedRows = pruned.orderBy("id").collectAsList();

            spark.conf().set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "false");
            assertEquals(joinedIds(facts, dimensions, joinColumn).orderBy("id").collectAsList(), prunedRows);
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

    @Test
    void pruningSubqueriesDoNotNestThroughASelfJoin() {
        Path facts = tempDir.resolve("nested_facts");
        spark.range(0, 20_000)
                .selectExpr("id", "id % 500 AS key", "id % 37 AS other")
                .repartition(4)
                .write()
                .format("vortex")
                .save(facts.toString());
        Path dimensions = tempDir.resolve("nested_dimensions");
        spark.range(0, 500)
                .selectExpr("id AS key", "concat('name_', id % 50) AS name")
                .write()
                .format("vortex")
                .save(dimensions.toString());
        spark.read().format("vortex").load(facts.toString()).createOrReplaceTempView("nested_facts");
        spark.read().format("vortex").load(dimensions.toString()).createOrReplaceTempView("nested_dimensions");

        // The join of the two references prunes one by the other, whose own scan the dimension prunes.
        String query = "WITH x AS (SELECT f.other, sum(f.id) AS total FROM nested_facts f "
                + "JOIN nested_dimensions d ON f.key = d.key WHERE d.name = 'name_7' GROUP BY f.other) "
                + "SELECT a.other, a.total, b.total AS other_total FROM x a JOIN x b ON a.other = b.other "
                + "WHERE b.total > 100";
        spark.conf().set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true");
        Dataset<Row> pruned = spark.sql(query);
        LogicalPlan plan = pruned.queryExecution().optimizedPlan();
        List<DynamicPruningSubquery> subqueries = pruningSubqueries(plan);
        assertFalse(subqueries.isEmpty(), "the dimension still prunes the facts:\n" + plan);
        for (DynamicPruningSubquery subquery : subqueries) {
            assertTrue(
                    pruningSubqueries(subquery.buildQuery()).isEmpty(),
                    "a pruning subquery holds another one:\n" + plan);
        }
        List<Row> prunedRows = pruned.orderBy("other").collectAsList();

        spark.conf().set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "false");
        assertEquals(spark.sql(query).orderBy("other").collectAsList(), prunedRows);
    }

    /** The dynamic pruning subqueries in {@code plan}, without those inside their own filtering plans. */
    private static List<DynamicPruningSubquery> pruningSubqueries(LogicalPlan plan) {
        List<DynamicPruningSubquery> found = new ArrayList<>();
        collectPruningSubqueries(plan, found);
        return found;
    }

    private static void collectPruningSubqueries(LogicalPlan plan, List<DynamicPruningSubquery> found) {
        for (Expression expression : JavaConverters.seqAsJavaList(plan.expressions())) {
            collectPruningSubqueries(expression, found);
        }
        for (LogicalPlan child : JavaConverters.seqAsJavaList(plan.children())) {
            collectPruningSubqueries(child, found);
        }
    }

    private static void collectPruningSubqueries(Expression expression, List<DynamicPruningSubquery> found) {
        if (expression instanceof DynamicPruningSubquery subquery) {
            found.add(subquery);
            return;
        }
        for (Expression child : JavaConverters.seqAsJavaList(expression.children())) {
            collectPruningSubqueries(child, found);
        }
    }

    private Dataset<Row> joinedIds(Path facts, Path dimensions, String joinColumn) {
        Dataset<Row> fact = spark.read().format("vortex").load(facts.toString());
        Dataset<Row> dimension =
                spark.read().format("vortex").load(dimensions.toString()).filter("name = 'name_7'");
        return fact.join(dimension, fact.col(joinColumn).equalTo(dimension.col(joinColumn)))
                .select(fact.col("id"));
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
