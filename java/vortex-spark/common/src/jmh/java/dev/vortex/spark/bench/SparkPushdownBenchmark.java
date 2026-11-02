// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Each pushdown the connector offers, measured with the pushdown on and off over the same data.
 *
 * <p>{@code pushdown=false} turns the feature off through its option, or, for {@code IN}, hides the column behind
 * {@code abs}, which Spark cannot push, so both runs answer the same query over non-negative ids. Spark folds {@code id
 * + 0} back to {@code id}, so that would push in both runs.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 3)
@Fork(1)
public class SparkPushdownBenchmark {
    private static final long ROWS = 8_000_000L;

    @Param({"true", "false"})
    private boolean pushdown;

    private SparkSession spark;
    private Path dataDir;
    private String inList;
    private String clusteredInList;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        spark = BenchmarkSparkSession.create("SparkPushdownBenchmark");
        // Spark 4 enables ANSI mode by default, which keeps SUM in Spark.
        spark.conf().set("spark.sql.ansi.enabled", "false");
        spark.conf().set("spark.sql.autoBroadcastJoinThreshold", "10485760");
        dataDir = Files.createTempDirectory("vortex-pushdown-bench");

        BenchmarkSparkSession.writeDataset(spark, dataDir.resolve("files"), ROWS, 16);
        BenchmarkSparkSession.writeDataset(spark, dataDir.resolve("one_file"), ROWS, 1);
        // Range-partitioned and sorted by id, as tables ordered by time usually are, so statistics can tell files
        // and chunks apart.
        spark.range(0, ROWS)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id % 1000 as string)) as value",
                        "cast(id as double) / 7.0 as measure")
                .repartitionByRange(16, functions.col("id"))
                .sortWithinPartitions("id")
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(dataDir.resolve("sorted").toString());
        spark.range(0, ROWS)
                .selectExpr("cast(id as int) as id", "cast(id % 64 as int) as part")
                .repartition(16)
                .write()
                .format("vortex")
                .partitionBy("part")
                .mode(SaveMode.Overwrite)
                .save(dataDir.resolve("partitioned").toString());
        spark.range(0, 64_000)
                .selectExpr("cast(id as int) as other", "cast(id % 64 as int) as part")
                .repartition(4)
                .write()
                .format("vortex")
                .partitionBy("part")
                .mode(SaveMode.Overwrite)
                .save(dataDir.resolve("partitioned_other").toString());
        spark.range(0, 10_000)
                .selectExpr(
                        "cast(id * 800 as int) as id",
                        "concat('k', id % 100) as tag",
                        "cast(id / 1000 as int) as region")
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(dataDir.resolve("dimension").toString());

        // Spread over the whole id range, so per-chunk statistics cannot rule out most chunks by range alone.
        inList = LongStream.range(0, 500)
                .map(i -> i * (ROWS / 500) + 3)
                .mapToObj(Long::toString)
                .collect(Collectors.joining(", "));
        clusteredInList = LongStream.range(0, 500)
                .map(i -> ROWS / 2 + i * 7)
                .mapToObj(Long::toString)
                .collect(Collectors.joining(", "));
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        spark.stop();
        SparkScanBenchmark.deleteRecursively(dataDir);
    }

    /** MIN, MAX, COUNT, and SUM over data columns, answered per file from footers or native kernels. */
    @Benchmark
    public Row columnAggregates() {
        return read("files", "vortex.aggregatePushdown")
                .selectExpr("min(id)", "max(id)", "count(value)", "min(value)", "max(measure)", "sum(id)")
                .first();
    }

    /** A 500-value IN list, evaluated natively as one set lookup per row. */
    @Benchmark
    public long inListFilter() {
        String column = pushdown ? "id" : "abs(id)";
        return read("files", null)
                .filter(String.format("%s IN (%s)", column, inList))
                .count();
    }

    /** A 500-value IN list over a narrow id range of the sorted data, which statistics prune to a few chunks. */
    @Benchmark
    public long clusteredInListFilter() {
        String column = pushdown ? "id" : "abs(id)";
        return read("sorted", null)
                .filter(String.format("%s IN (%s)", column, clusteredInList))
                .count();
    }

    /** ORDER BY ... LIMIT over the sorted data, where a tightening threshold skips the chunks behind it. */
    @Benchmark
    public List<Row> topN() {
        return read("sorted", "vortex.topNPushdown")
                .orderBy(functions.col("id").desc())
                .limit(10)
                .collectAsList();
    }

    /** ORDER BY ... LIMIT over the shuffled data, where only the rows behind the threshold are dropped. */
    @Benchmark
    public List<Row> topNUnsorted() {
        return read("files", "vortex.topNPushdown")
                .orderBy(functions.col("measure"))
                .limit(10)
                .collectAsList();
    }

    /** A 1% TABLESAMPLE, which hands each scan a row selection so most chunks are never decoded. */
    @Benchmark
    public long sample() {
        return read("files", "vortex.samplePushdown").sample(0.01, 7).count();
    }

    /** A LIKE pattern with no V1 filter form, evaluated natively over the compressed strings. */
    @Benchmark
    public long likePattern() {
        return read("files", "vortex.predicatePushdown")
                .filter("value LIKE 'value_1%3'")
                .count();
    }

    /** An aggregation of a join on Hive partition columns, which reported keys let Spark run without a shuffle. */
    @Benchmark
    public Row partitionKeyedJoin() {
        spark.conf().set("spark.sql.sources.v2.bucketing.enabled", "true");
        spark.conf().set("spark.sql.autoBroadcastJoinThreshold", "-1");
        try {
            Dataset<Row> facts = read("partitioned", "vortex.reportPartitioning");
            Dataset<Row> others = read("partitioned_other", "vortex.reportPartitioning");
            return facts.join(others, "part").selectExpr("count(*)", "sum(id)").first();
        } finally {
            spark.conf().set("spark.sql.autoBroadcastJoinThreshold", "10485760");
            spark.conf().unset("spark.sql.sources.v2.bucketing.enabled");
        }
    }

    /** An OFFSET past three quarters of the rows, which footer row counts let the scan skip unread. */
    @Benchmark
    public long offset() {
        return read("files", "vortex.offsetPushdown")
                .offset((int) (ROWS * 3 / 4))
                .count();
    }

    /** A per-partition sort of data the footers prove sorted, which a reported order makes unnecessary. */
    @Benchmark
    public long sortWithinPartitions() {
        // Counting the physical plan's rows keeps the sort, which an aggregate on top would let Spark drop.
        return read("sorted", "vortex.reportOrdering")
                .select("id", "measure")
                .sortWithinPartitions("id")
                .queryExecution()
                .toRdd()
                .count();
    }

    /** A LIMIT that each file can stop reading at. */
    @Benchmark
    public List<Row> limit() {
        return read("files", "vortex.limitPushdown").limit(100).collectAsList();
    }

    /** A full scan of one large file, which splitting spreads over every core. */
    @Benchmark
    public Row scanOneLargeFile() {
        return read("one_file", "vortex.splitFiles")
                .selectExpr("sum(id)", "count(value)", "sum(measure)")
                .first();
    }

    /** A join with a selective dimension, whose keys a runtime filter pushes into the fact scan. */
    @Benchmark
    public long runtimeJoinFilter() {
        Dataset<Row> facts = read("files", "vortex.runtimeFilter");
        Dataset<Row> dimension = spark.read()
                .format("vortex")
                .load(dataDir.resolve("dimension").toString())
                .filter("tag = 'k7'");
        return facts.join(dimension, "id").count();
    }

    /**
     * A join whose dimension filter, on a column other than the key, selects one contiguous key range of the sorted
     * facts, so the runtime filter prunes whole files. A filter on the key itself would reach the facts statically.
     */
    @Benchmark
    public long clusteredRuntimeJoinFilter() {
        Dataset<Row> facts = read("sorted", "vortex.runtimeFilter");
        Dataset<Row> dimension = spark.read()
                .format("vortex")
                .load(dataDir.resolve("dimension").toString())
                .filter("region = 5");
        return facts.join(dimension, "id").count();
    }

    private Dataset<Row> read(String dataset, String option) {
        var reader = spark.read().format("vortex");
        if (option != null) {
            reader = reader.option(option, Boolean.toString(pushdown));
        }
        if (!"vortex.aggregatePushdown".equals(option)) {
            // Keep COUNT(*) and friends reading rows, so only the feature under test differs.
            reader = reader.option("vortex.aggregatePushdown", "false");
        }
        return reader.load(dataDir.resolve(dataset).toString());
    }
}
