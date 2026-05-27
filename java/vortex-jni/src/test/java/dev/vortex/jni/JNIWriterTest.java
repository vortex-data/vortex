// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DataSource;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
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
        try (VortexWriter writer = VortexWriter.create(session, writePath, personSchema(), options, allocator)) {
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
        try (VortexWriter writer = VortexWriter.create(session, writePath, personSchema(), options, allocator)) {
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
        try (VortexWriter writer = VortexWriter.create(session, writePath, personSchema(), options, allocator)) {
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
        try (VortexWriter writer = VortexWriter.create(session, writePath, schema, new HashMap<>(), allocator);
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
        }

        assertTrue(Files.exists(outputPath), "output file should exist");

        DataSource ds = DataSource.open(session, writePath);
        assertEquals(new DataSource.RowCount.Exact(3L), ds.rowCount());

        Scan scan = ds.scan(ScanOptions.of());
        while (scan.hasNext()) {
            Partition p = scan.next();
            try (ArrowReader reader = p.scanArrow(allocator)) {
                reader.loadNextBatch();
                VectorSchemaRoot resultRoot = reader.getVectorSchemaRoot();
                VarCharVector nameOut = (VarCharVector) resultRoot.getVector("name");
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

    @Test
    public void testParquetVariantRoundTrip() throws IOException {
        Path outputPath = tempDir.resolve("test_parquet_variant.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = parquetVariantSchema();

        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.create(session, writePath, schema, new HashMap<>(), allocator);
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
                VarBinaryVector metadata = variant.getChild("metadata", VarBinaryVector.class);
                VarBinaryVector value = variant.getChild("value", VarBinaryVector.class);

                assertArrayEquals(VARIANT_METADATA, metadata.get(0));
                assertArrayEquals(VARIANT_INT8_42, value.get(0));
                assertArrayEquals(VARIANT_METADATA, metadata.get(1));
                assertArrayEquals(VARIANT_TRUE, value.get(1));
                assertTrue(variant.isNull(2));
            }
        }
    }
}
