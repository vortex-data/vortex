// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Contract tests for {@link DataSource#open} and its precision-aware counts.
 *
 * <p>The argument checks in {@code open} are the guard on the JNI boundary: every one of them runs before the native
 * call, so dropping one does not surface as an {@link IllegalArgumentException} but as a null or a negative length
 * crossing into native code. They are asserted here by message so that a refactor which keeps the check but
 * reclassifies the failure is still caught.
 *
 * <p>{@link DataSource.RowCount} and {@link DataSource.ByteSize} are two separate ladders over the same three
 * precisions. Only {@code Exact} was covered before; {@code Unknown} losing its empty {@link OptionalLong} would let a
 * caller read a non-empty file as zero rows.
 */
public final class DataSourceContractTest {
    @TempDir
    static Path tempDir;

    private static String vortexFile;

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @BeforeAll
    static void writeFixture() throws IOException {
        vortexFile = tempDir.resolve("counts.vortex").toAbsolutePath().toUri().toString();
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = new Schema(List.of(Field.notNullable("id", new ArrowType.Int(32, true))));
        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.builder(session, vortexFile, schema, allocator)
                        .build();
                VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            ids.allocateNew(3);
            for (int i = 0; i < 3; i++) {
                ids.setSafe(i, i);
            }
            root.setRowCount(3);
            try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                    ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema);
                writer.writeBatch(arrowArray.memoryAddress(), arrowSchema.memoryAddress());
            }
        }
    }

    @Test
    public void openRejectsAnEmptyUriList() {
        Session session = Session.create();
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class, () -> DataSource.open(session, List.of(), Collections.emptyMap()));
        assertTrue(e.getMessage().contains("at least one uri is required"), e::getMessage);
    }

    @Test
    public void openRejectsANullUriInTheList() {
        Session session = Session.create();
        List<String> uris = Arrays.asList(vortexFile, null);
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class, () -> DataSource.open(session, uris, Collections.emptyMap()));
        assertTrue(e.getMessage().contains("uris must not contain null values"), e::getMessage);
    }

    @Test
    public void openRejectsANullSessionAndANullUriList() {
        Session session = Session.create();
        assertThrows(
                NullPointerException.class, () -> DataSource.open(null, List.of(vortexFile), Collections.emptyMap()));
        assertThrows(NullPointerException.class, () -> DataSource.open(session, (List<String>) null, Map.of()));
    }

    @Test
    public void openRejectsAnEmptyReadableList() {
        Session session = Session.create();
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> DataSource.open(session, List.of(), 0));
        assertTrue(e.getMessage().contains("at least one readable is required"), e::getMessage);
    }

    @Test
    public void openRejectsNegativeReadConcurrency() {
        Session session = Session.create();
        List<NativeReadable> readables = List.of(new StubReadable("a.vortex", 1));
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> DataSource.open(session, readables, -1));
        assertTrue(e.getMessage().contains("readConcurrency must not be negative"), e::getMessage);
    }

    @Test
    public void openRejectsANullReadableInTheList() {
        Session session = Session.create();
        List<NativeReadable> readables = Arrays.asList(new StubReadable("a.vortex", 1), null);
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> DataSource.open(session, readables, 0));
        assertTrue(e.getMessage().contains("readables must not contain null values"), e::getMessage);
    }

    @Test
    public void openRejectsAReadableWithoutAName() {
        Session session = Session.create();
        List<NativeReadable> readables = List.of(new StubReadable(null, 1));
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> DataSource.open(session, readables, 0));
        assertTrue(e.getMessage().contains("readable at index 0 returned a null name"), e::getMessage);
    }

    @Test
    public void openRejectsAReadableReportingNegativeLength() {
        Session session = Session.create();
        List<NativeReadable> readables = List.of(new StubReadable("short.vortex", -1));
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> DataSource.open(session, readables, 0));
        assertTrue(e.getMessage().contains("readable for short.vortex reported negative length"), e::getMessage);
    }

    @Test
    public void openRejectsANullReadableList() {
        Session session = Session.create();
        assertThrows(NullPointerException.class, () -> DataSource.open(session, (List<NativeReadable>) null, 0));
    }

    @Test
    public void openAcceptsAValidReadableAndReportsExactCounts() throws IOException {
        Session session = Session.create();
        DataSource source = DataSource.open(session, vortexFile);
        assertEquals(new DataSource.RowCount.Exact(3L), source.rowCount());
        assertEquals(OptionalLong.of(3L), source.rowCount().asOptional());
        assertTrue(source.byteSize() instanceof DataSource.ByteSize.Exact, () -> String.valueOf(source.byteSize()));
        assertTrue(source.byteSize().asOptional().orElseThrow() > 0);
    }

    @Test
    public void unknownCountsCarryNoValue() {
        assertEquals(OptionalLong.empty(), DataSource.RowCount.Unknown.INSTANCE.asOptional());
        assertEquals(OptionalLong.empty(), DataSource.ByteSize.Unknown.INSTANCE.asOptional());
        assertFalse(DataSource.RowCount.Unknown.INSTANCE.asOptional().isPresent());
        assertFalse(DataSource.ByteSize.Unknown.INSTANCE.asOptional().isPresent());
    }

    @Test
    public void estimateAndExactCarryTheirValue() {
        assertEquals(OptionalLong.of(7L), new DataSource.RowCount.Estimate(7L).asOptional());
        assertEquals(OptionalLong.of(7L), new DataSource.RowCount.Exact(7L).asOptional());
        assertEquals(OptionalLong.of(9L), new DataSource.ByteSize.Estimate(9L).asOptional());
        assertEquals(OptionalLong.of(9L), new DataSource.ByteSize.Exact(9L).asOptional());
        // An estimate is not interchangeable with an exact count of the same magnitude.
        assertNotEquals(new DataSource.RowCount.Exact(7L), new DataSource.RowCount.Estimate(7L));
        assertNotEquals(new DataSource.ByteSize.Exact(9L), new DataSource.ByteSize.Estimate(9L));
    }

    private static final class StubReadable implements NativeReadable {
        private final String name;
        private final long length;

        StubReadable(String name, long length) {
            this.name = name;
            this.length = length;
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
            throw new IOException("StubReadable is only used to reach the argument checks");
        }

        @Override
        public void close() {}
    }
}
