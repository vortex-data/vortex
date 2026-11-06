// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Spark's concurrent partition writer closes its open writers one after another when a task aborts and stops at the
 * first close that throws, so a writer whose write already failed must not throw again from close.
 */
final class VortexOutputWriterAbortTest {
    private static final List<VortexOutputWriter> WRITERS = new ArrayList<>();
    private static final List<BufferAllocator> ALLOCATORS = new ArrayList<>();
    private static final int PARTITIONS = 8;

    @TempDir
    Path tempDir;

    // The oversized row fails in write as it is appended, whatever the batch size.
    @ParameterizedTest
    @CsvSource({"1", "2048"})
    void aFailedWriteLetsAbortCloseEveryPartitionWriter(String batchSize) throws IOException {
        SparkSession spark = SparkSession.builder()
                .master("local[1,1]")
                .appName("VortexOutputWriterAbortTest")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        spark.conf().set("spark.sql.adaptive.enabled", "false");
        spark.conf().set("spark.sql.maxConcurrentOutputFileWriters", "16");
        StructType schema = new StructType().add("s", DataTypes.StringType).add("p", DataTypes.IntegerType);
        List<String> stranded = new ArrayList<>();
        try {
            // The order Spark closes its writers in decides whether the failed one comes first, so fail each in turn.
            for (int failed = 0; failed < PARTITIONS; failed++) {
                WRITERS.clear();
                ALLOCATORS.clear();
                List<Row> rows = new ArrayList<>();
                for (int p = 0; p < PARTITIONS; p++) {
                    rows.add(RowFactory.create("small", p));
                }
                // Larger than the allocator limit set below, so appending it fails.
                rows.add(RowFactory.create("x".repeat(2 << 20), failed));
                Path output = tempDir.resolve(batchSize + "-failed=" + failed);

                Exception thrown = assertThrows(Exception.class, () -> spark.createDataFrame(rows, schema)
                        .coalesce(1)
                        .write()
                        .format("dev.vortex.spark.write.LimitedVortexFileFormat")
                        .option("vortex.write.batch.size", batchSize)
                        .partitionBy("p")
                        .save(output.toString()));

                assertEquals(PARTITIONS, WRITERS.size(), () -> stackTrace(thrown));
                for (int i = 0; i < WRITERS.size(); i++) {
                    VortexOutputWriter writer = WRITERS.get(i);
                    long held = ALLOCATORS.get(i).getAllocatedMemory();
                    if (writer.allocator() != null || held != 0) {
                        String partition = Path.of(URI.create(writer.path()))
                                .getParent()
                                .getFileName()
                                .toString();
                        stranded.add("failed=" + failed + " " + partition + ": " + held + " bytes");
                        try {
                            writer.close();
                        } catch (RuntimeException ignored) {
                            // Only the release matters here; the assertion below reports the stranding.
                        }
                    }
                }
                if (Files.exists(output)) {
                    try (Stream<Path> files = Files.walk(output)) {
                        assertEquals(
                                List.of(),
                                files.filter(f -> f.toString().endsWith(".vortex") || f.endsWith("_SUCCESS"))
                                        .toList());
                    }
                }
            }
        } finally {
            spark.stop();
            WRITERS.clear();
            ALLOCATORS.clear();
        }
        assertEquals(List.of(), stranded, "writers left open after the task aborted");
    }

    private static String stackTrace(Throwable t) {
        StringWriter out = new StringWriter();
        t.printStackTrace(new PrintWriter(out));
        return out.toString();
    }

    /** Called for every writer LimitedVortexFileFormat opens, which is Scala and compiled after this class. */
    static VortexOutputWriter opened(VortexOutputWriter writer) {
        writer.allocator().setLimit(1 << 20);
        WRITERS.add(writer);
        ALLOCATORS.add(writer.allocator());
        return writer;
    }
}
