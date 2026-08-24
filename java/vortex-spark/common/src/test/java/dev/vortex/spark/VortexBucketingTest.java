// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * Bucketed Vortex tables.
 *
 * <p>Spark's session catalog turns {@code bucket(n, col)} and {@code CLUSTERED BY} into a bucket spec, and its V1 file
 * writer gives each bucket its own files. These tests hold Vortex to that layout, and check that Spark reads it back as
 * buckets: a join of two tables bucketed on the join key needs no shuffle.
 *
 * <p>Spark rejects the {@code years}, {@code months}, {@code days}, and {@code hours} transforms in the session catalog
 * for every file format, before a provider sees them.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexBucketingTest {
    private static final Pattern BUCKET_FILE = Pattern.compile("part-.*_(\\d{5})\\.c\\d{3}\\.vortex");

    private SparkSession spark;
    private Path warehouseDir;

    @BeforeAll
    public void setUp() throws IOException {
        warehouseDir = Files.createTempDirectory("vortex-bucketing-warehouse");
        spark = SparkSession.builder()
                .appName("VortexBucketingTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.warehouse.dir", warehouseDir.toUri().toString())
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.autoBroadcastJoinThreshold", "-1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterEach
    public void dropTables() {
        spark.sql("DROP TABLE IF EXISTS bucketed_left");
        spark.sql("DROP TABLE IF EXISTS bucketed_right");
        spark.sql("DROP TABLE IF EXISTS bucketed_sql");
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("bucketBy writes one set of Vortex files per bucket, and the join of two such tables needs no shuffle")
    void bucketByWritesSparkBuckets() throws IOException {
        writeBucketed("bucketed_left", 100);
        writeBucketed("bucketed_right", 50);

        Set<Integer> buckets = bucketIds("bucketed_left");
        assertFalse(buckets.isEmpty(), "bucketed table should hold bucket files");
        assertTrue(buckets.stream().allMatch(id -> id >= 0 && id < 4), "bucket ids: " + buckets);

        Dataset<Row> joined = spark.sql(
                "SELECT l.id FROM bucketed_left l JOIN bucketed_right r ON l.id = r.id");
        String plan = joined.queryExecution().executedPlan().toString();
        assertFalse(plan.contains("Exchange"), "a bucketed join should not shuffle:\n" + plan);
        assertEquals(50, joined.count());
    }

    @Test
    @DisplayName("PARTITIONED BY (bucket(n, col)) creates a bucketed Vortex table")
    void bucketTransformCreatesBucketedTable() throws IOException {
        spark.sql("CREATE TABLE bucketed_sql (id INT, name STRING) USING vortex PARTITIONED BY (bucket(4, id))");
        spark.sql("INSERT INTO bucketed_sql SELECT CAST(id AS INT), CAST(id AS STRING) FROM range(40)");

        List<Row> rows = spark.sql("SELECT count(*), count(DISTINCT id) FROM bucketed_sql").collectAsList();
        assertEquals(40L, rows.get(0).getLong(0));
        assertEquals(40L, rows.get(0).getLong(1));

        Set<Integer> buckets = bucketIds("bucketed_sql");
        assertFalse(buckets.isEmpty(), "bucketed table should hold bucket files");
        assertTrue(buckets.stream().allMatch(id -> id >= 0 && id < 4), "bucket ids: " + buckets);
    }

    @Test
    @DisplayName("Spark rejects time transforms in the session catalog before Vortex sees them")
    void timeTransformsAreRejectedBySpark() {
        assertThrows(
                Exception.class,
                () -> spark.sql("CREATE TABLE bucketed_sql (ts TIMESTAMP) USING vortex PARTITIONED BY (years(ts))"));
    }

    private void writeBucketed(String table, int rows) {
        spark.range(0, rows)
                .selectExpr("cast(id as int) as id", "concat('value_', cast(id as string)) as value")
                .write()
                .format("vortex")
                .bucketBy(4, "id")
                .sortBy("id")
                .saveAsTable(table);
    }

    private Set<Integer> bucketIds(String table) throws IOException {
        try (Stream<Path> files = Files.walk(warehouseDir.resolve(table))) {
            List<String> names = files.filter(Files::isRegularFile)
                    .map(file -> file.getFileName().toString())
                    .filter(name -> !name.startsWith(".") && !name.startsWith("_"))
                    .toList();
            for (String name : names) {
                assertTrue(BUCKET_FILE.matcher(name).matches(), "not a bucket file: " + name);
            }
            return names.stream()
                    .map(BUCKET_FILE::matcher)
                    .filter(Matcher::matches)
                    .map(matcher -> Integer.parseInt(matcher.group(1)))
                    .collect(Collectors.toSet());
        }
    }
}
