// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
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
 * End-to-end Spark scans over a fixed number of rows split into a varying number of files.
 *
 * <p>The connector opens one Vortex reader per {@code PartitionedFile}, so the {@code fileCount} axis measures the
 * per-file open overhead that bin-packing adds on top of the pure decode cost.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(1)
public class SparkScanBenchmark {
    private static final long ROWS = 2_000_000L;

    @Param({"1", "16", "128"})
    private int fileCount;

    private SparkSession spark;
    private Path dataDir;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        spark = BenchmarkSparkSession.create("SparkScanBenchmark");
        dataDir = Files.createTempDirectory("vortex-scan-bench");
        BenchmarkSparkSession.writeDataset(spark, dataDir.resolve("data"), ROWS, fileCount);
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        spark.stop();
        deleteRecursively(dataDir);
    }

    /** COUNT(*) is pushed down to footer row counts, so this measures footer reads plus planning. */
    @Benchmark
    public long countStar() {
        return read().count();
    }

    /** Reads every column of every file and aggregates, measuring the full decode path. */
    @Benchmark
    public Row fullScanAggregate() {
        return read().selectExpr("sum(id)", "count(value)", "sum(measure)").first();
    }

    /** Pushes a predicate into each per-file reader and counts the survivors. */
    @Benchmark
    public long filteredScan() {
        return read().filter("id % 97 = 0").count();
    }

    /** Projects a single column to isolate per-file open cost from decode volume. */
    @Benchmark
    public Row singleColumnScan() {
        return read().selectExpr("sum(id)").first();
    }

    private Dataset<Row> read() {
        return spark.read().format("vortex").load(dataDir.resolve("data").toString());
    }

    static void deleteRecursively(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.delete(path);
            }
        }
    }
}
