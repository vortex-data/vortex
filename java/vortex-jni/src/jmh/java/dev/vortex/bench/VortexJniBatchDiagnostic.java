// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.bench;

import dev.vortex.api.DataSource;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.jni.NativeLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * Standalone diagnostic (not a JMH benchmark): writes the shared {@link BenchData} table at several writer chunk sizes
 * and prints the resulting read-batch row-count distribution, showing that Vortex coalesces to a stable read-batch
 * granularity (~64K rows) independent of how the file was written.
 *
 * <p>Run it with {@code ./gradlew :vortex-jni:batchDiagnostic}.
 */
public final class VortexJniBatchDiagnostic {

    private VortexJniBatchDiagnostic() {}

    public static void main(String[] args) throws Exception {
        NativeLoader.loadJni();
        for (int chunk : new int[] {8192, 65536, 131072}) {
            BufferAllocator alloc = new RootAllocator(Long.MAX_VALUE);
            Session sess = Session.create();
            Path f = Files.createTempFile("vortex-jni-diag-" + chunk + "-", ".vortex");
            Files.deleteIfExists(f);
            String uri = f.toAbsolutePath().toUri().toString();
            BenchData.writeTable(sess, alloc, uri, chunk);
            DataSource ds = DataSource.open(sess, uri);
            long batches = 0;
            long rowsSeen = 0;
            long minRows = Long.MAX_VALUE;
            long maxRows = 0;
            Scan scan = ds.scan(ScanOptions.of());
            while (scan.hasNext()) {
                Partition partition = scan.next();
                try (ArrowReader reader = partition.scanArrow(alloc)) {
                    while (reader.loadNextBatch()) {
                        int rows = reader.getVectorSchemaRoot().getRowCount();
                        batches++;
                        rowsSeen += rows;
                        minRows = Math.min(minRows, rows);
                        maxRows = Math.max(maxRows, rows);
                    }
                }
            }
            System.out.printf(
                    "writeChunkRows=%d -> %d read batches over %d rows (min=%d, max=%d, avg=%d)%n",
                    chunk, batches, rowsSeen, minRows, maxRows, batches == 0 ? 0 : rowsSeen / batches);
            Files.deleteIfExists(f);
        }
    }
}
