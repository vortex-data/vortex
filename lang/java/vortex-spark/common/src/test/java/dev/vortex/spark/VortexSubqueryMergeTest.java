// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.ReusedSubqueryExec;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import scala.collection.JavaConverters;

/** Spark merges scalar subqueries that scan the same Vortex table with the same filters. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSubqueryMergeTest {
    private SparkSession spark;

    // Static, so it exists before the per-class setup writes the data.
    @TempDir
    static Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexSubqueryMergeTest")
                .master("local[2]")
                .config("spark.sql.extensions", VortexSparkSessionExtensions.class.getName())
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        Path table = tempDir.resolve("facts");
        spark.range(0, 10_000, 1, 4)
                .selectExpr("id % 100 AS q", "id AS a", "id * 2 AS b", "id % 7 AS c")
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(table.toString());
        spark.read().format("vortex").load(table.toString()).createOrReplaceTempView("facts");
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void subqueriesThatReadDifferentColumnsShareOneScan() {
        // The shape of TPC-DS q9: each subquery reads its own columns with the same filter.
        String query = "SELECT "
                + "(SELECT count(*) FROM facts WHERE q BETWEEN 1 AND 20) AS n, "
                + "(SELECT sum(a) FROM facts WHERE q BETWEEN 1 AND 20) AS sa, "
                + "(SELECT sum(b) FROM facts WHERE q BETWEEN 1 AND 20) AS sb, "
                + "(SELECT sum(b) FROM facts WHERE q BETWEEN 21 AND 40) AS other";
        SparkPlan plan = spark.sql(query).queryExecution().executedPlan();
        List<Row> rows = spark.sql(query).collectAsList();

        Row expected = spark.sql("SELECT count(*), sum(a), sum(b) FROM facts WHERE q BETWEEN 1 AND 20")
                .first();
        Row other =
                spark.sql("SELECT sum(b) FROM facts WHERE q BETWEEN 21 AND 40").first();
        assertEquals(
                List.of(expected.getLong(0), expected.getLong(1), expected.getLong(2), other.getLong(0)),
                List.of(
                        rows.get(0).getLong(0),
                        rows.get(0).getLong(1),
                        rows.get(0).getLong(2),
                        rows.get(0).getLong(3)));

        // One scan for the three subqueries with the same filter, and one for the subquery with another filter.
        List<BatchScanExec> scans = scans(plan);
        assertEquals(2, scans.size(), plan.toString());
    }

    @Test
    void scansThatSparkDoesNotMergeReadOnlyTheirOwnColumns() {
        // Spark cannot merge an aggregate over a grouped aggregate with a plain aggregate.
        String query = "SELECT "
                + "(SELECT sum(a) FROM facts WHERE q < 10) AS sa, "
                + "(SELECT max(s) FROM (SELECT sum(b) AS s FROM facts WHERE q < 10 GROUP BY c)) AS mb";
        SparkPlan plan = spark.sql(query).queryExecution().executedPlan();
        Row row = spark.sql(query).first();

        Row expected = spark.sql("SELECT sum(a) FROM facts WHERE q < 10").first();
        Row grouped = spark.sql("SELECT max(s) FROM (SELECT sum(b) AS s FROM facts WHERE q < 10 GROUP BY c)")
                .first();
        assertEquals(expected.getLong(0), row.getLong(0));
        assertEquals(grouped.getLong(0), row.getLong(1));

        List<BatchScanExec> scans = scans(plan);
        assertEquals(2, scans.size(), plan.toString());

        Set<List<String>> columns = Set.of(List.of("q", "a"), List.of("q", "b", "c"));
        for (BatchScanExec scan : scans) {
            List<String> names = List.of(scan.scan().readSchema().fieldNames());
            assertTrue(columns.contains(names), names + " in " + plan);
        }
    }

    /** The scans of `plan` and its subqueries, each once. */
    private static List<BatchScanExec> scans(SparkPlan plan) {
        List<BatchScanExec> found = new ArrayList<>();
        collect(plan, found, Collections.newSetFromMap(new IdentityHashMap<>()));
        return found;
    }

    private static void collect(SparkPlan plan, List<BatchScanExec> found, Set<SparkPlan> seen) {
        // A reused subquery runs the plan of another subquery.
        if (!seen.add(plan) || plan instanceof ReusedSubqueryExec) {
            return;
        }

        if (plan instanceof BatchScanExec scan) {
            found.add(scan);
        }

        for (SparkPlan child : JavaConverters.seqAsJavaList(plan.children())) {
            collect(child, found, seen);
        }
        for (SparkPlan subquery : JavaConverters.seqAsJavaList(plan.subqueries())) {
            collect(subquery, found, seen);
        }
    }
}
