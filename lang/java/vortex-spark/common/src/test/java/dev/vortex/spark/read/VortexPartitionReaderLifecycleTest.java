// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DataSource;
import dev.vortex.api.Scan;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.relocated.org.apache.arrow.vector.IntVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.paths.SparkPath;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

final class VortexPartitionReaderLifecycleTest {
    private static final VortexOptions OPTIONS = VortexOptions.of(Map.of("vortex.splitConcurrency", "1"));
    private static final StructType SCHEMA = new StructType().add("value", DataTypes.IntegerType, false);

    @TempDir
    Path directory;

    private Path file;

    @BeforeEach
    void writeFile() throws IOException {
        file = directory.resolve("values.vortex");
        var allocator = ArrowAllocation.rootAllocator();
        Schema arrowSchema = new Schema(List.of(Field.notNullable("value", new ArrowType.Int(32, true))));
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator);
                VortexWriter writer = VortexWriter.builder(
                                VortexSparkSession.get(OPTIONS), file.toString(), arrowSchema, allocator)
                        .build()) {
            IntVector values = (IntVector) root.getVector(0);
            values.allocateNew(3);
            for (int row = 0; row < 3; row++) {
                values.set(row, row + 10);
            }
            root.setRowCount(3);
            writer.writeBatch(root, allocator);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"unstarted", "early", "exhausted"})
    void closesNativeHandlesWithoutWaitingForGc(String state) throws Exception {
        VortexPartitionReader reader = open();
        DataSource source = (DataSource) field(reader, "dataSource");
        Scan scan = (Scan) field(reader, "scan");
        NativeReadable readable = (NativeReadable) field(reader, "readable");
        if (!state.equals("unstarted")) {
            assertTrue(reader.next());
            assertEquals(10, reader.get().column(0).getInt(0));
            if (state.equals("exhausted")) {
                assertFalse(reader.next());
            }
        }
        reader.close();
        reader.close();
        assertClosed(source, scan, readable);
        assertFalse(reader.next());
        assertNull(field(reader, "dataSource"));
        assertNull(field(reader, "scan"));
        assertNull(field(reader, "currentReader"));
        assertNull(field(reader, "currentBatch"));
    }

    @Test
    void exhaustionLeavesNativeHandlesForTheOwnerToClose() throws Exception {
        try (VortexPartitionReader reader = open()) {
            DataSource source = (DataSource) field(reader, "dataSource");
            Scan scan = (Scan) field(reader, "scan");
            NativeReadable readable = (NativeReadable) field(reader, "readable");
            assertTrue(reader.next());
            assertEquals(10, reader.get().column(0).getInt(0));
            assertFalse(reader.next());
            assertFalse(reader.next());
            assertEquals(3L, source.rowCount().asOptional().orElseThrow());
            assertFalse(scan.hasNext());
            readable.readFully(0, ByteBuffer.allocateDirect(1));
            reader.close();
            assertClosed(source, scan, readable);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failingBatchReadStillClosesEveryResource(boolean failClose) throws Exception {
        VortexPartitionReader reader = open();
        DataSource source = (DataSource) field(reader, "dataSource");
        Scan scan = (Scan) field(reader, "scan");
        NativeReadable readable = (NativeReadable) field(reader, "readable");
        IOException readFailure = new IOException("batch read failed");
        IOException closeFailure = new IOException("batch close failed");
        AtomicBoolean attemptedClose = new AtomicBoolean();
        ArrowReader failing = new ArrowReader(ArrowAllocation.rootAllocator()) {
            @Override
            public boolean loadNextBatch() throws IOException {
                throw readFailure;
            }

            @Override
            public long bytesRead() {
                return 0;
            }

            @Override
            protected void closeReadSource() throws IOException {
                attemptedClose.set(true);
                if (failClose) {
                    throw closeFailure;
                }
            }

            @Override
            protected Schema readSchema() {
                return new Schema(List.of());
            }
        };
        setField(reader, "currentReader", failing);
        UncheckedIOException failure = assertThrows(UncheckedIOException.class, reader::next);
        assertSame(readFailure, failure.getCause());
        assertTrue(attemptedClose.get());
        assertEquals(failClose ? 1 : 0, failure.getSuppressed().length);
        if (failClose) {
            assertSame(closeFailure, failure.getSuppressed()[0].getCause());
        }
        assertClosed(source, scan, readable);
        reader.close();
    }

    @Test
    void closeFailureStillReleasesSourceAndReadable() throws Exception {
        VortexPartitionReader reader = open();
        DataSource source = (DataSource) field(reader, "dataSource");
        Scan scan = (Scan) field(reader, "scan");
        NativeReadable readable = (NativeReadable) field(reader, "readable");
        RuntimeException failure = new IllegalStateException("batch close failed");
        setField(reader, "currentBatch", new ColumnarBatch(new ColumnVector[0], 0) {
            @Override
            public void close() {
                throw failure;
            }
        });
        assertSame(failure, assertThrows(IllegalStateException.class, reader::close));
        assertClosed(source, scan, readable);
        reader.close();
    }

    private VortexPartitionReader open() throws IOException {
        long length = Files.size(file);
        PartitionedFile split = new PartitionedFile(
                InternalRow.empty(),
                SparkPath.fromPathString(file.toString()),
                0,
                length,
                new String[0],
                0,
                length,
                scala.collection.immutable.Map$.MODULE$.empty());
        return new VortexPartitionReader(
                split,
                SCHEMA,
                SCHEMA,
                new StructType(),
                VortexIo.create(OPTIONS, new Configuration()),
                OPTIONS,
                new Filter[0],
                true);
    }

    private static void assertClosed(DataSource source, Scan scan, NativeReadable readable) {
        assertThrows(IllegalStateException.class, source::rowCount);
        assertThrows(IllegalStateException.class, scan::hasNext);
        assertThrows(IllegalStateException.class, () -> readable.readFully(0, ByteBuffer.allocateDirect(1)));
    }

    private static Object field(VortexPartitionReader reader, String name) throws ReflectiveOperationException {
        var field = VortexPartitionReader.class.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(reader);
    }

    private static void setField(VortexPartitionReader reader, String name, Object value)
            throws ReflectiveOperationException {
        var field = VortexPartitionReader.class.getDeclaredField(name);
        field.setAccessible(true);
        field.set(reader, value);
    }
}
