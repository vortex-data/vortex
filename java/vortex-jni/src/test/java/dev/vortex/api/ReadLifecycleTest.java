// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.jni.NativeRuntime;
import java.io.IOException;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.NoSuchElementException;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.OutOfMemoryException;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

final class ReadLifecycleTest {
    private static int previousWorkers;
    private final BufferAllocator allocator = ArrowAllocation.rootAllocator();
    private final Session session = Session.create();

    @TempDir
    Path directory;

    private String uri;

    @BeforeAll
    static void driveCleanupTasks() {
        // Canceled I/O drivers must be polled to drop their futures, as Spark's worker pool does.
        previousWorkers = NativeRuntime.workerCount();
        NativeRuntime.setWorkerThreads(1);
    }

    @AfterAll
    static void restoreWorkers() {
        NativeRuntime.setWorkerThreads(previousWorkers);
    }

    @BeforeEach
    void writeFile() throws IOException {
        uri = directory.resolve("values.vortex").toString();
        Schema schema = new Schema(List.of(Field.notNullable("value", new ArrowType.Int(32, true))));
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
                VortexWriter writer =
                        VortexWriter.builder(session, uri, schema, allocator).build()) {
            IntVector values = (IntVector) root.getVector(0);
            values.allocateNew(3);
            for (int row = 0; row < 3; row++) {
                values.set(row, row + 10);
            }
            root.setRowCount(3);
            writer.writeBatch(root, allocator);
        }
    }

    @Test
    void sourceCloseIsIdempotentAndRejectsFurtherUse() {
        DataSource source = DataSource.open(session, uri);
        source.close();
        source.close();
        assertThrows(IllegalStateException.class, source::rowCount);
        assertThrows(IllegalStateException.class, source::byteSize);
        assertThrows(IllegalStateException.class, () -> source.arrowSchema(allocator));
        assertThrows(IllegalStateException.class, () -> source.scan(ScanOptions.of()));
    }

    @Test
    void sourceCloseReleasesItsNativeReadableWhileTheWrapperRemainsReachable() throws Exception {
        NativeInput input = openNativeInput();
        input.source().close();
        assertReadableReleased(input);
    }

    @Test
    void primedScanCloseReleasesThePartitionItHasNotReturned() throws Exception {
        NativeInput input = openNativeInput();
        Scan scan = input.source().scan(ScanOptions.of());
        assertTrue(scan.hasNext());
        input.source().close();
        assertFalse(input.readable().refersTo(null));
        scan.close();
        try {
            assertReadableReleased(input);
        } finally {
            Reference.reachabilityFence(scan);
        }
    }

    @ParameterizedTest
    @ValueSource(longs = {0, 64})
    void failedStreamAllocationOrImportReleasesTheConsumedPartition(long limit) throws Exception {
        NativeInput input = openNativeInput();
        Scan scan = input.source().scan(ScanOptions.of());
        Partition partition = scan.next();
        scan.close();
        input.source().close();
        assertFalse(input.readable().refersTo(null));
        // Arrow 19 allocates two 40-byte stream structs, rounded to 64 bytes each.
        try (BufferAllocator limited = new RootAllocator(limit)) {
            assertThrows(OutOfMemoryException.class, () -> partition.scanArrow(limited));
            assertEquals(0, limited.getAllocatedMemory());
            assertThrows(IllegalStateException.class, () -> partition.scanArrow(allocator));
            assertReadableReleased(input);
        } finally {
            Reference.reachabilityFence(partition);
            Reference.reachabilityFence(scan);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void closingUnusedOrPrimedScanMakesItTerminal(boolean prime) {
        try (DataSource source = DataSource.open(session, uri)) {
            Scan scan = source.scan(ScanOptions.of());
            if (prime) {
                assertTrue(scan.hasNext());
            }
            scan.close();
            scan.close();
            assertFalse(scan.hasNext());
            assertThrows(NoSuchElementException.class, scan::next);
            assertThrows(IllegalStateException.class, () -> scan.arrowSchema(allocator));
            try (Scan another = source.scan(ScanOptions.of())) {
                assertTrue(another.hasNext());
            }
        }
    }

    @Test
    void scanSurvivesItsSourceAndReturnedPartitionSurvivesItsScan() throws IOException {
        DataSource source = DataSource.open(session, uri);
        Scan scan = source.scan(ScanOptions.of());
        source.close();
        Partition partition = scan.next();
        scan.close();
        assertEquals(3, partition.rowCount().orElseThrow());
        try (ArrowReader reader = partition.scanArrow(allocator)) {
            assertValues(reader);
        }
    }

    @Test
    void closedArrowReaderLeavesNothingAllocated() throws IOException {
        try (DataSource source = DataSource.open(session, uri);
                Scan scan = source.scan(ScanOptions.of());
                BufferAllocator tracked = new RootAllocator()) {
            try (ArrowReader reader = scan.next().scanArrow(tracked)) {
                assertValues(reader);
            }
            assertEquals(0, tracked.getAllocatedMemory());
        }
    }

    @Test
    void loadedBatchSurvivesClosingItsParentHandles() throws IOException {
        DataSource source = DataSource.open(session, uri);
        Scan scan = source.scan(ScanOptions.of());
        try (ArrowReader reader = scan.next().scanArrow(allocator)) {
            assertTrue(reader.loadNextBatch());
            scan.close();
            source.close();
            IntVector values = (IntVector) reader.getVectorSchemaRoot().getVector(0);
            assertEquals(3, values.getValueCount());
            assertEquals(10, values.get(0));
            assertEquals(12, values.get(2));
            assertFalse(reader.loadNextBatch());
        }
    }

    private static void assertValues(ArrowReader reader) throws IOException {
        int row = 0;
        while (reader.loadNextBatch()) {
            IntVector values = (IntVector) reader.getVectorSchemaRoot().getVector(0);
            for (int i = 0; i < values.getValueCount(); i++) {
                assertEquals(row + 10, values.get(i));
                row++;
            }
        }
        assertEquals(3, row);
    }

    private NativeInput openNativeInput() throws IOException {
        NativeReadable readable = new ByteArrayReadable(uri, Files.readAllBytes(Path.of(uri)));
        return new NativeInput(DataSource.open(session, readable), new WeakReference<>(readable));
    }

    private static void assertReadableReleased(NativeInput input) throws InterruptedException {
        try {
            for (int attempt = 0; attempt < 100 && !input.readable().refersTo(null); attempt++) {
                System.gc();
                Thread.sleep(10);
            }
            assertTrue(input.readable().refersTo(null), "native handles still own the readable");
        } finally {
            // GC observes release of native global references, not Cleaner release of these wrappers.
            Reference.reachabilityFence(input);
        }
    }

    private record NativeInput(DataSource source, WeakReference<NativeReadable> readable) {}

    private static final class ByteArrayReadable implements NativeReadable {
        private final String name;
        private final byte[] bytes;

        private ByteArrayReadable(String name, byte[] bytes) {
            this.name = name;
            this.bytes = bytes;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public long length() {
            return bytes.length;
        }

        @Override
        public void readFully(long position, ByteBuffer buffer) {
            buffer.put(bytes, Math.toIntExact(position), buffer.remaining());
        }

        @Override
        public void close() {}
    }
}
