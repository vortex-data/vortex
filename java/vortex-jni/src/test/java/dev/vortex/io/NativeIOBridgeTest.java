// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DataSource;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.api.VortexWriteSummary;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.jni.NativeLoader;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ViewVarCharVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Round-trip tests for the caller-provided I/O bridge ({@link NativeReadable} / {@link NativeWritable}). */
public final class NativeIOBridgeTest {
    @TempDir
    Path tempDir;

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    private static Schema personSchema() {
        return new Schema(List.of(
                Field.notNullable("name", new ArrowType.Utf8()),
                Field.notNullable("age", new ArrowType.Int(32, true))));
    }

    /** A {@link NativeReadable} over a local file, safe for concurrent positional reads. */
    private static final class FileChannelReadable implements NativeReadable {
        private final String name;
        private final FileChannel channel;
        private final long length;

        FileChannelReadable(Path path) throws IOException {
            this(path.toString(), path);
        }

        FileChannelReadable(String name, Path path) throws IOException {
            this.name = name;
            this.channel = FileChannel.open(path, StandardOpenOption.READ);
            this.length = channel.size();
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public long length() {
            return length;
        }

        @Override
        public void readFully(long position, ByteBuffer buffer) throws IOException {
            int len = buffer.remaining();
            long pos = position;
            while (buffer.hasRemaining()) {
                int read = channel.read(buffer, pos);
                if (read < 0) {
                    throw new EOFException("EOF reading " + len + " bytes at position " + position);
                }
                pos += read;
            }
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    /** A {@link NativeWritable} over a local file. */
    private static final class StreamWritable implements NativeWritable {
        private final OutputStream out;

        StreamWritable(Path path) throws IOException {
            this.out = Files.newOutputStream(
                    path, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        }

        @Override
        public void write(byte[] buffer, int offset, int length) throws IOException {
            out.write(buffer, offset, length);
        }

        @Override
        public void flush() throws IOException {
            out.flush();
        }

        @Override
        public void close() throws IOException {
            out.close();
        }
    }

    private void writePeopleFile(Session session, BufferAllocator allocator, Path path) throws IOException {
        Schema schema = personSchema();
        VortexWriteSummary summary;
        try (StreamWritable writable = new StreamWritable(path)) {
            try (VortexWriter writer = VortexWriter.create(session, writable, schema, allocator);
                    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
                VarCharVector nameVec = (VarCharVector) root.getVector("name");
                IntVector ageVec = (IntVector) root.getVector("age");
                nameVec.allocateNew(3);
                ageVec.allocateNew(3);
                nameVec.setSafe(0, "Alice".getBytes(UTF_8));
                nameVec.setSafe(1, "Bob".getBytes(UTF_8));
                nameVec.setSafe(2, "Carol".getBytes(UTF_8));
                ageVec.setSafe(0, 30);
                ageVec.setSafe(1, 25);
                ageVec.setSafe(2, 40);
                root.setRowCount(3);

                try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                        ArrowSchema arrowSchemaFfi = ArrowSchema.allocateNew(allocator)) {
                    Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchemaFfi);
                    writer.writeBatch(arrowArray.memoryAddress(), arrowSchemaFfi.memoryAddress());
                }

                // The byte counter must be live for stream writers too: bounded while
                // the write task is still flushing, exact once finished.
                long bytesWhileOpen = writer.bytesWritten();
                summary = writer.finish();
                assertTrue(bytesWhileOpen >= 0 && bytesWhileOpen <= summary.fileSize());
                assertEquals(summary.fileSize(), writer.bytesWritten());
            }
        }
        // Everything the writer counted must have reached the caller-provided sink.
        assertEquals(Files.size(path), summary.fileSize());
    }

    @Test
    public void testWritableThenReadableRoundTrip() throws IOException {
        Path path = tempDir.resolve("bridge_roundtrip.vortex");
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();

        writePeopleFile(session, allocator, path);
        assertTrue(Files.size(path) > 0, "stream writer should have produced bytes");

        try (FileChannelReadable readable = new FileChannelReadable(path)) {
            DataSource ds = DataSource.open(session, readable);
            assertEquals(new DataSource.RowCount.Exact(3L), ds.rowCount());

            Scan scan = ds.scan(ScanOptions.of());
            int rows = 0;
            while (scan.hasNext()) {
                Partition p = scan.next();
                try (ArrowReader reader = p.scanArrow(allocator)) {
                    while (reader.loadNextBatch()) {
                        VectorSchemaRoot root = reader.getVectorSchemaRoot();
                        ViewVarCharVector nameOut = (ViewVarCharVector) root.getVector("name");
                        IntVector ageOut = (IntVector) root.getVector("age");
                        if (rows == 0) {
                            assertEquals("Alice", nameOut.getObject(0).toString());
                            assertEquals(30, ageOut.get(0));
                        }
                        rows += root.getRowCount();
                    }
                }
            }
            assertEquals(3, rows);
        }
    }

    @Test
    public void testMultipleReadables() throws IOException {
        Path first = tempDir.resolve("bridge_a.vortex");
        Path second = tempDir.resolve("bridge_b.vortex");
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();

        writePeopleFile(session, allocator, first);
        writePeopleFile(session, allocator, second);

        try (FileChannelReadable firstReadable = new FileChannelReadable(first);
                FileChannelReadable secondReadable = new FileChannelReadable(second)) {
            DataSource ds = DataSource.open(session, List.of(firstReadable, secondReadable));
            // Only the first file is opened eagerly, so the count is an estimate until scanned.
            assertEquals(6L, ds.rowCount().asOptional().orElseThrow());

            Scan scan = ds.scan(ScanOptions.of());
            long rows = 0;
            while (scan.hasNext()) {
                Partition p = scan.next();
                try (ArrowReader reader = p.scanArrow(allocator)) {
                    while (reader.loadNextBatch()) {
                        rows += reader.getVectorSchemaRoot().getRowCount();
                    }
                }
            }
            assertEquals(6, rows);
        }
    }

    @Test
    public void testDuplicateReadableNamesAreRejected() throws IOException {
        Path path = tempDir.resolve("bridge_duplicate.vortex");
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();

        writePeopleFile(session, allocator, path);

        try (FileChannelReadable first = new FileChannelReadable("/bridge_duplicate.vortex", path);
                FileChannelReadable second = new FileChannelReadable("bridge_duplicate.vortex", path)) {
            RuntimeException thrown =
                    assertThrows(RuntimeException.class, () -> DataSource.open(session, List.of(first, second)));
            assertTrue(
                    thrown.getMessage().contains("multiple Java readables normalize to path"),
                    "error should identify the normalized path collision, got: " + thrown.getMessage());
        }
    }

    @Test
    public void testLargeRoundTrip() throws IOException {
        Path path = tempDir.resolve("bridge_large.vortex");
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        Schema schema = personSchema();

        int batches = 20;
        int rowsPerBatch = 5_000;
        try (StreamWritable writable = new StreamWritable(path)) {
            try (VortexWriter writer = VortexWriter.create(session, writable, schema, allocator);
                    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
                VarCharVector nameVec = (VarCharVector) root.getVector("name");
                IntVector ageVec = (IntVector) root.getVector("age");
                for (int batch = 0; batch < batches; batch++) {
                    nameVec.allocateNew(rowsPerBatch);
                    ageVec.allocateNew(rowsPerBatch);
                    for (int row = 0; row < rowsPerBatch; row++) {
                        nameVec.setSafe(row, ("name-" + batch + "-" + row).getBytes(UTF_8));
                        ageVec.setSafe(row, batch * rowsPerBatch + row);
                    }
                    root.setRowCount(rowsPerBatch);
                    try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                            ArrowSchema arrowSchemaFfi = ArrowSchema.allocateNew(allocator)) {
                        Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchemaFfi);
                        writer.writeBatch(arrowArray.memoryAddress(), arrowSchemaFfi.memoryAddress());
                    }
                }
            }
        }

        try (FileChannelReadable readable = new FileChannelReadable(path)) {
            DataSource ds = DataSource.open(session, List.of(readable), 4);
            Scan scan = ds.scan(ScanOptions.of());
            long rows = 0;
            long ageSum = 0;
            while (scan.hasNext()) {
                Partition p = scan.next();
                try (ArrowReader reader = p.scanArrow(allocator)) {
                    while (reader.loadNextBatch()) {
                        VectorSchemaRoot root = reader.getVectorSchemaRoot();
                        IntVector ageOut = (IntVector) root.getVector("age");
                        for (int i = 0; i < root.getRowCount(); i++) {
                            ageSum += ageOut.get(i);
                        }
                        rows += root.getRowCount();
                    }
                }
            }
            long total = (long) batches * rowsPerBatch;
            assertEquals(total, rows);
            assertEquals(total * (total - 1) / 2, ageSum, "ages should be 0..N-1 exactly once");
        }
    }

    @Test
    public void testReadableExceptionPropagates() {
        Session session = Session.create();
        NativeReadable failing = new NativeReadable() {
            @Override
            public String name() {
                return "memory/failing.vortex";
            }

            @Override
            public long length() {
                return 1024;
            }

            @Override
            public void readFully(long position, ByteBuffer buffer) throws IOException {
                throw new IOException("boom: injected read failure");
            }

            @Override
            public void close() {}
        };

        RuntimeException thrown = assertThrows(RuntimeException.class, () -> DataSource.open(session, failing));
        assertTrue(
                thrown.getMessage().contains("boom: injected read failure"),
                "error should carry the Java exception message, got: " + thrown.getMessage());
    }
}
