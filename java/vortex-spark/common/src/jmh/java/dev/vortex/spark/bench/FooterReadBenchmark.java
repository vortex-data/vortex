// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.bench;

import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import dev.vortex.spark.read.VortexFooterReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.apache.hadoop.conf.Configuration;
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
 * Cost of opening a Vortex file just to read its footer.
 *
 * <p>Footer reads drive schema inference, row-count statistics, and COUNT(*) pushdown, so each of those pays this price
 * once per file. The {@code ioMode} axis compares Hadoop stream upcalls with native file reads.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(1)
public class FooterReadBenchmark {
    private static final long ROWS = 500_000L;
    private static final int FILE_COUNT = 32;

    @Param({"1", "8"})
    private int statsParallelism;

    private SparkSession spark;
    private Path dataDir;
    private VortexIo io;
    private VortexOptions options;
    private List<VortexFile> files;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        spark = BenchmarkSparkSession.create("FooterReadBenchmark");
        dataDir = Files.createTempDirectory("vortex-footer-bench");
        Path data = dataDir.resolve("data");
        BenchmarkSparkSession.writeDataset(spark, data, ROWS, FILE_COUNT);

        options = VortexOptions.of(
                Map.of(VortexFooterReader.FOOTER_PARALLELISM_OPTION, Integer.toString(statsParallelism)));
        io = VortexIo.create(options, new Configuration());

        files = new ArrayList<>();
        try (Stream<Path> paths = Files.list(data)) {
            for (Path path : paths.toList()) {
                if (path.getFileName().toString().endsWith(".vortex")) {
                    files.add(new VortexFile(path.toUri().toString(), Files.size(path)));
                }
            }
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        spark.stop();
        SparkScanBenchmark.deleteRecursively(dataDir);
    }

    /** One footer open and row-count read, the unit of work behind every statistic. */
    @Benchmark
    public OptionalLong singleFooterRowCount() {
        return VortexFooterReader.estimatedRowCount(files.get(0), io, options);
    }

    /** Footer row counts summed over all files through the bounded pool the scan uses for statistics. */
    @Benchmark
    public OptionalLong sumRowCountsAcrossFiles() {
        return VortexFooterReader.sumRowCounts(files, io, options);
    }
}
