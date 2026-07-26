// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ViewVarBinaryVector;
import org.apache.arrow.vector.ViewVarCharVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public final class JNIWriterTest {
    private static final String ARROW_EXTENSION_NAME = "ARROW:extension:name";
    private static final String PARQUET_VARIANT_EXTENSION_NAME = "arrow.parquet.variant";
    private static final byte[] VARIANT_METADATA = new byte[] {0x01, 0x00};
    private static final byte[] VARIANT_INT8_42 = new byte[] {0x0c, 0x2a};
    private static final byte[] VARIANT_TRUE = new byte[] {0x04};

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

    private static Schema parquetVariantSchema() {
        Field variant = new Field(
                "variant",
                new FieldType(
                        true,
                        ArrowType.Struct.INSTANCE,
                        null,
                        Map.of(ARROW_EXTENSION_NAME, PARQUET_VARIANT_EXTENSION_NAME)),
                List.of(
                        Field.notNullable("metadata", new ArrowType.Binary()),
                        Field.nullable("value", new ArrowType.Binary())));
        return new Schema(List.of(variant));
    }

    private static void populateParquetVariantRoot(VectorSchemaRoot root) {
        StructVector variant = (StructVector) root.getVector("variant");
        VarBinaryVector metadata = variant.getChild("metadata", VarBinaryVector.class);
        VarBinaryVector value = variant.getChild("value", VarBinaryVector.class);

        variant.allocateNew();
        metadata.allocateNew(3);
        value.allocateNew(3);

        metadata.setSafe(0, VARIANT_METADATA);
        metadata.setSafe(1, VARIANT_METADATA);
        metadata.setSafe(2, VARIANT_METADATA);
        value.setSafe(0, VARIANT_INT8_42);
        value.setSafe(1, VARIANT_TRUE);
        value.setNull(2);
        variant.setIndexDefined(0);
        variant.setIndexDefined(1);
        variant.setNull(2);

        metadata.setValueCount(3);
        value.setValueCount(3);
        variant.setValueCount(3);
        root.setRowCount(3);
    }

    @Test
    public void testCreateWriter() throws IOException {
        Path outputPath = tempDir.resolve("test_create.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Map<String, String> options = new HashMap<>();

        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.builder(session, writePath, personSchema(), allocator)
                .options(options)
                .build()) {
            assertNotNull(writer);
        }

        assertTrue(Files.exists(outputPath), "output file should exist");
    }

    @Test
    public void testCreateWriterPlainLocalPath() throws IOException {
        Path outputPath = tempDir.resolve("test_create_plain_path.vortex");
        String writePath = outputPath.toAbsolutePath().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Map<String, String> options = new HashMap<>();

        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.builder(session, writePath, personSchema(), allocator)
                .options(options)
                .build()) {
            assertNotNull(writer);
        }

        assertTrue(Files.exists(outputPath), "output file should exist");
    }

    @Test
    public void testCreateWriterCreatesParentDirectories() throws IOException {
        Path outputPath = tempDir.resolve("nested/sub/dir/test_create_nested.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Map<String, String> options = new HashMap<>();

        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.builder(session, writePath, personSchema(), allocator)
                .options(options)
                .build()) {
            assertNotNull(writer);
        }

        assertTrue(Files.exists(outputPath), "output file should exist");
    }

    @Test
    public void testWriteBatch() throws IOException {
        Path outputPath = tempDir.resolve("test_ffi.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = personSchema();

        Session session = Session.create();
        VortexWriteSummary summary;
        long bytesWhileOpen;
        try (VortexWriter writer = VortexWriter.builder(session, writePath, schema, allocator)
                        .build();
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
            bytesWhileOpen = writer.bytesWritten();
            summary = writer.finish();
            assertEquals(summary.fileSize(), writer.bytesWritten());
        }

        assertTrue(Files.exists(outputPath), "output file should exist");
        assertTrue(bytesWhileOpen >= 0);
        assertTrue(bytesWhileOpen <= summary.fileSize());
        assertEquals(Files.size(outputPath), summary.fileSize());
        assertEquals(3L, summary.rowCount());
        assertEquals(2, summary.columnStatistics().size());
        assertEquals(0, summary.columnStatistics().get(0).columnIndex());
        assertTrue(summary.columnStatistics().get(0).compressedSize() > 0);
        assertEquals(3L, summary.columnStatistics().get(0).valueCount());
        assertEquals(0L, summary.columnStatistics().get(0).nullValueCount().orElseThrow());
        assertEquals("Alice", summary.columnStatistics().get(0).lowerBound().orElseThrow());
        assertEquals("Carol", summary.columnStatistics().get(0).upperBound().orElseThrow());
        assertEquals(25, summary.columnStatistics().get(1).lowerBound().orElseThrow());
        assertEquals(40, summary.columnStatistics().get(1).upperBound().orElseThrow());

        DataSource ds = DataSource.open(session, writePath);
        assertEquals(new DataSource.RowCount.Exact(3L), ds.rowCount());

        Scan scan = ds.scan(ScanOptions.of());
        while (scan.hasNext()) {
            Partition p = scan.next();
            try (ArrowReader reader = p.scanArrow(allocator)) {
                reader.loadNextBatch();
                VectorSchemaRoot resultRoot = reader.getVectorSchemaRoot();
                ViewVarCharVector nameOut = (ViewVarCharVector) resultRoot.getVector("name");
                IntVector ageOut = (IntVector) resultRoot.getVector("age");
                assertEquals("Alice", nameOut.getObject(0).toString());
                assertEquals("Bob", nameOut.getObject(1).toString());
                assertEquals("Carol", nameOut.getObject(2).toString());
                assertEquals(30, ageOut.get(0));
                assertEquals(25, ageOut.get(1));
                assertEquals(40, ageOut.get(2));
            }
        }
    }

    /** Write a single three-row batch of {@link #personSchema()} data. */
    private static void writePeopleBatch(VortexWriter writer, BufferAllocator allocator) throws IOException {
        try (VectorSchemaRoot root = VectorSchemaRoot.create(personSchema(), allocator)) {
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
        }
    }

    @Test
    public void testFileMetadataRoundTrip() throws IOException {
        Path outputPath = tempDir.resolve("test_metadata.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        byte[] schemaJson = "{\"type\":\"struct\",\"fields\":[]}".getBytes(UTF_8);
        byte[] deleteType = "position".getBytes(UTF_8);
        Map<String, byte[]> metadata = Map.of("iceberg.schema", schemaJson, "delete-type", deleteType);

        try (VortexWriter writer = VortexWriter.builder(session, writePath, personSchema(), allocator)
                .metadata(metadata)
                .build()) {
            writePeopleBatch(writer, allocator);
        }

        Map<String, byte[]> read = NativeFiles.readMetadata(session, writePath, new HashMap<>());
        assertEquals(metadata.keySet(), read.keySet());
        assertArrayEquals(schemaJson, read.get("iceberg.schema"));
        assertArrayEquals(deleteType, read.get("delete-type"));

        // Metadata is stored out of band: the file still scans as usual.
        DataSource ds = DataSource.open(session, writePath);
        assertEquals(new DataSource.RowCount.Exact(3L), ds.rowCount());
    }

    @Test
    public void testFileWithoutMetadataReadsEmpty() throws IOException {
        Path outputPath = tempDir.resolve("test_no_metadata.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.builder(session, writePath, personSchema(), allocator)
                .build()) {
            writePeopleBatch(writer, allocator);
        }

        assertTrue(NativeFiles.readMetadata(session, writePath, new HashMap<>()).isEmpty());
    }

    @Test
    public void testInvalidMetadataKeyRejectedAtCreate() {
        Path outputPath = tempDir.resolve("test_bad_metadata.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        // Keys are capped well below this; the writer must reject the set before any bytes
        // are produced, rather than failing the first writeBatch with a send error.
        Map<String, byte[]> metadata = Map.of("k".repeat(1024), new byte[] {1});

        IOException thrown = assertThrows(
                IOException.class, () -> VortexWriter.builder(session, writePath, personSchema(), allocator)
                        .metadata(metadata)
                        .build());
        Throwable cause = thrown.getCause();
        assertNotNull(cause, "native failure should be retained as the cause");
        assertTrue(
                cause.getMessage().contains("metadata key"),
                "error should identify the offending key, got: " + cause.getMessage());
        // Rejected before the sink is touched, so no partial file is left behind.
        assertFalse(Files.exists(outputPath));
    }

    @Test
    public void testParquetVariantRoundTrip() throws IOException {
        Path outputPath = tempDir.resolve("test_parquet_variant.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = parquetVariantSchema();

        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.builder(session, writePath, schema, allocator)
                        .build();
                VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            populateParquetVariantRoot(root);

            try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                    ArrowSchema arrowSchemaFfi = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchemaFfi);
                writer.writeBatch(arrowArray.memoryAddress(), arrowSchemaFfi.memoryAddress());
            }
        }

        assertTrue(Files.exists(outputPath), "output file should exist");

        DataSource ds = DataSource.open(session, writePath);
        Field dataSourceField = ds.arrowSchema(allocator).findField("variant");
        assertEquals(
                PARQUET_VARIANT_EXTENSION_NAME, dataSourceField.getMetadata().get(ARROW_EXTENSION_NAME));

        Scan scan = ds.scan(ScanOptions.of());
        Field scanField = scan.arrowSchema(allocator).findField("variant");
        assertEquals(PARQUET_VARIANT_EXTENSION_NAME, scanField.getMetadata().get(ARROW_EXTENSION_NAME));

        while (scan.hasNext()) {
            Partition p = scan.next();
            try (ArrowReader reader = p.scanArrow(allocator)) {
                assertTrue(reader.loadNextBatch());
                VectorSchemaRoot resultRoot = reader.getVectorSchemaRoot();
                StructVector variant = (StructVector) resultRoot.getVector("variant");
                // Binary columns cross the boundary as their native view types.
                ViewVarBinaryVector metadata = variant.getChild("metadata", ViewVarBinaryVector.class);
                ViewVarBinaryVector value = variant.getChild("value", ViewVarBinaryVector.class);

                assertArrayEquals(VARIANT_METADATA, metadata.get(0));
                assertArrayEquals(VARIANT_INT8_42, value.get(0));
                assertArrayEquals(VARIANT_METADATA, metadata.get(1));
                assertArrayEquals(VARIANT_TRUE, value.get(1));
                assertTrue(variant.isNull(2));
            }
        }
    }
}
