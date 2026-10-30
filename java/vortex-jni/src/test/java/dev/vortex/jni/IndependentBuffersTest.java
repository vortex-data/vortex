// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DataSource;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.io.NativeWritable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ViewVarCharVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

public final class IndependentBuffersTest {
    @TempDir
    Path directory;

    private static Schema schema() {
        return new Schema(List.of(new Field(
                "nested",
                FieldType.nullable(ArrowType.Struct.INSTANCE),
                List.of(Field.nullable("value", new ArrowType.Utf8())))));
    }

    private static void fill(VectorSchemaRoot root, int rows) {
        StructVector nested = (StructVector) root.getVector(0);
        VarCharVector values = nested.getChild("value", VarCharVector.class);
        nested.setInitialCapacity(rows);
        nested.allocateNew();
        byte[] value = new byte[512];
        Arrays.fill(value, (byte) 97);
        for (int row = 0; row < rows; row++) {
            value[0] = (byte) (row % 26 + 97);
            values.setSafe(row, value);
            if (row % 2 == 0) {
                nested.setIndexDefined(row);
            } else {
                nested.setNull(row);
            }
        }
        root.setRowCount(rows);
    }

    @Test
    public void releasesLargeColumnsWhileSmallBitmapsRemain() throws Exception {
        Schema schema = schema();
        int batches = 16;
        int rows = 8192;
        try (BufferAllocator allocator = new RootAllocator();
                BufferAllocator batchesAllocator = allocator.newChildAllocator("batches", 0, Long.MAX_VALUE);
                VortexWriter writer = VortexWriter.builder(
                                Session.create(),
                                directory.resolve("memory.vortex").toString(),
                                schema,
                                allocator)
                        .build()) {
            for (int batch = 0; batch < batches; batch++) {
                try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, batchesAllocator)) {
                    fill(root, rows);
                    writer.writeBatch(root, allocator);
                }
            }
            long allValueBytes = (long) batches * rows * 512;
            // Arrow export can transfer accounting to its allocator; measure the common parent.
            long held = allocator.getAllocatedMemory();
            assertTrue(held < allValueBytes * 3 / 4, "unrelated allocations must be released; held " + held);
            assertEquals((long) batches * rows, writer.finish().rowCount());
            assertEquals(0, batchesAllocator.getAllocatedMemory());
            assertEquals(0, allocator.getAllocatedMemory());
        }
    }

    @Test
    public void failedImportReleasesPartialRetains() throws Exception {
        Schema schema = schema();
        try (BufferAllocator allocator = new RootAllocator();
                BufferAllocator batches = allocator.newChildAllocator("batch", 0, Long.MAX_VALUE);
                VectorSchemaRoot root = VectorSchemaRoot.create(schema, batches);
                ArrowSchema ffiSchema = ArrowSchema.allocateNew(allocator);
                ArrowArray array = ArrowArray.allocateNew(allocator)) {
            fill(root, 32);
            Data.exportSchema(allocator, schema, null, ffiSchema);
            long writer = NativeWriter.create(
                    Session.create().nativePointer(),
                    directory.resolve("failure.vortex").toString(),
                    ffiSchema.memoryAddress(),
                    Map.of(),
                    Map.of());
            try {
                Data.exportVectorSchemaRoot(allocator, root, null, array);
                NativeBuffer[] owners = NativeBuffer.collect(root);
                assertThrows(
                        RuntimeException.class,
                        () -> NativeWriter.writeBatchBuffers(
                                writer, array.memoryAddress(), new NativeBuffer[] {owners[0]}));
                root.clear();
                assertEquals(0, batches.getAllocatedMemory());
            } finally {
                NativeWriter.close(writer);
                array.release();
                ffiSchema.release();
            }
        }
    }

    @Test
    public void sinkFailureReleasesBuffers() throws Exception {
        try (BufferAllocator allocator = new RootAllocator()) {
            NativeWritable sink = new NativeWritable() {
                @Override
                public void write(byte[] buffer, int offset, int length) throws IOException {
                    throw new IOException("injected write failure");
                }

                @Override
                public void flush() {}

                @Override
                public void close() {}
            };
            VortexWriter writer = VortexWriter.builder(Session.create(), sink, schema(), allocator)
                    .build();
            try (VectorSchemaRoot root = VectorSchemaRoot.create(schema(), allocator)) {
                fill(root, 32);
                writer.writeBatch(root, allocator);
            }
            IOException error = assertThrows(IOException.class, writer::finish);
            assertTrue(error.getCause().getMessage().contains("injected write failure"), error.toString());
            assertEquals(0, allocator.getAllocatedMemory());
        }
    }

    @Test
    public void emptyAndViewBatchesRoundTrip() throws Exception {
        Schema schema = new Schema(List.of(Field.nullable("text", new ArrowType.Utf8View())));
        Session session = Session.create();
        String path = directory.resolve("views.vortex").toString();
        List<String> expected = Arrays.asList("short", null, "a long out-of-line string value", "");
        try (BufferAllocator allocator = new RootAllocator()) {
            try (VortexWriter writer = VortexWriter.builder(session, path, schema, allocator)
                            .build();
                    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
                writer.writeBatch(root, allocator);
                ViewVarCharVector values = (ViewVarCharVector) root.getVector(0);
                values.allocateNew();
                for (int row = 0; row < expected.size(); row++) {
                    if (expected.get(row) == null) {
                        // Arrow 19 leaves null view slots uninitialized. The existing Vortex
                        // view concatenation path reads them, so initialize the physical slot.
                        values.setSafe(row, new byte[0]);
                        values.setNull(row);
                    } else {
                        values.setSafe(row, expected.get(row).getBytes(StandardCharsets.UTF_8));
                    }
                }
                root.setRowCount(expected.size());
                writer.writeBatch(root, allocator);
                root.clear();
            }
            DataSource source = DataSource.open(session, path);
            Scan scan = source.scan(ScanOptions.of());
            List<String> actual = new ArrayList<>();
            while (scan.hasNext()) {
                try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                    while (reader.loadNextBatch()) {
                        VectorSchemaRoot root = reader.getVectorSchemaRoot();
                        for (int row = 0; row < root.getRowCount(); row++) {
                            Object value = root.getVector(0).getObject(row);
                            actual.add(value == null ? null : value.toString());
                        }
                    }
                }
            }
            assertEquals(expected, actual);
        }
    }

    @Test
    public void unalignedInputKeepsArrowAlignmentCopies() throws Exception {
        Schema schema = new Schema(List.of(Field.nullable("number", new ArrowType.Int(32, true))));
        Session session = Session.create();
        String path = directory.resolve("unaligned.vortex").toString();
        try (BufferAllocator allocator = new RootAllocator()) {
            try (VortexWriter writer = VortexWriter.builder(session, path, schema, allocator)
                            .build();
                    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
                    ArrowBuf validity = allocator.buffer(1);
                    ArrowBuf data = allocator.buffer(13)) {
                validity.setByte(0, 7);
                data.setInt(1, 11);
                data.setInt(5, 22);
                data.setInt(9, 33);
                IntVector vector = (IntVector) root.getVector(0);
                vector.loadFieldBuffers(new ArrowFieldNode(3, 0), List.of(validity, data.slice(1, 12)));
                root.setRowCount(3);
                assertEquals(1, vector.getDataBuffer().memoryAddress() % Integer.BYTES);
                writer.writeBatch(root, allocator);
                root.clear();
            }
            Scan scan = DataSource.open(session, path).scan(ScanOptions.of());
            List<Integer> actual = new ArrayList<>();
            while (scan.hasNext()) {
                try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                    while (reader.loadNextBatch()) {
                        IntVector vector =
                                (IntVector) reader.getVectorSchemaRoot().getVector(0);
                        for (int row = 0; row < vector.getValueCount(); row++) {
                            actual.add(vector.get(row));
                        }
                    }
                }
            }
            assertEquals(List.of(11, 22, 33), actual);
        }
    }

    @Test
    public void mismatchedSchemaIsRejectedBeforeImport() throws Exception {
        Schema different = new Schema(List.of(Field.nullable("number", new ArrowType.Int(32, true))));
        try (BufferAllocator allocator = new RootAllocator();
                VortexWriter writer = VortexWriter.builder(
                                Session.create(),
                                directory.resolve("schema.vortex").toString(),
                                schema(),
                                allocator)
                        .build();
                VectorSchemaRoot root = VectorSchemaRoot.create(different, allocator)) {
            IOException error = assertThrows(IOException.class, () -> writer.writeBatch(root, allocator));
            assertTrue(error.getMessage().contains("schema"));
            try (VectorSchemaRoot valid = VectorSchemaRoot.create(schema(), allocator)) {
                fill(valid, 32);
                writer.writeBatch(valid, allocator);
            }
            assertEquals(32, writer.finish().rowCount());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    public void writeAndFinishSerializeNativeLifetime(boolean independent) throws Exception {
        var executor = Executors.newFixedThreadPool(2);
        try (BufferAllocator allocator = new RootAllocator()) {
            for (int iteration = 0; iteration < 16; iteration++) {
                try (VortexWriter writer = VortexWriter.builder(
                                        Session.create(),
                                        directory
                                                .resolve("race-" + iteration + ".vortex")
                                                .toString(),
                                        schema(),
                                        allocator)
                                .build();
                        VectorSchemaRoot root = VectorSchemaRoot.create(schema(), allocator);
                        ArrowArray array = ArrowArray.allocateNew(allocator);
                        ArrowSchema ffi = ArrowSchema.allocateNew(allocator)) {
                    fill(root, 32);
                    Data.exportVectorSchemaRoot(allocator, root, null, array, ffi);
                    CyclicBarrier start = new CyclicBarrier(2);
                    try {
                        var write = executor.submit(() -> {
                            start.await();
                            try {
                                if (independent) {
                                    writer.writeBatch(root, allocator);
                                } else {
                                    writer.writeBatch(array.memoryAddress(), ffi.memoryAddress());
                                }
                                return true;
                            } catch (IllegalStateException closed) {
                                assertTrue(closed.getMessage().contains("closed"));
                                return false;
                            }
                        });
                        var finish = executor.submit(() -> {
                            start.await();
                            return writer.finish();
                        });
                        boolean accepted = write.get(10, TimeUnit.SECONDS);
                        assertEquals(
                                accepted ? 32 : 0,
                                finish.get(10, TimeUnit.SECONDS).rowCount());
                    } finally {
                        array.release();
                        ffi.release();
                    }
                }
            }
        } finally {
            executor.shutdownNow();
        }
    }
}
