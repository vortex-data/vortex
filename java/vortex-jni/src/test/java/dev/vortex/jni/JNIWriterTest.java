// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DType;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Direct test of JNI writer to isolate pointer alignment issue.
 */
public final class JNIWriterTest {

    @TempDir
    Path tempDir;

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @Test
    public void testCreateWriter() throws IOException {
        // Test just creating and closing a writer
        Path outputPath = tempDir.resolve("test_create.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        // Make a new file writer with a very simple schema.
        var writeSchema = DType.newStruct(
                new String[] {
                    "name", "age",
                },
                new DType[] {DType.newUtf8(false), DType.newInt(false)},
                false);

        // Minimal Arrow schema
        Map<String, String> options = new HashMap<>();

        System.err.println("Creating writer for path: " + writePath);

        try (VortexWriter writer = VortexWriter.create(writePath, writeSchema, options)) {
            assertNotNull(writer);
            System.err.println("Writer created successfully");
        }

        // Verify file was created
        assertTrue(Files.exists(outputPath), "Output file should exist");
        System.err.println("File created at: " + outputPath);
    }

    @Test
    public void testWriteBatchFfi() throws IOException {
        Path outputPath = tempDir.resolve("test_ffi.vortex");
        String writePath = outputPath.toAbsolutePath().toUri().toString();

        var writeSchema = DType.newStruct(
                new String[] {"name", "age"}, new DType[] {DType.newUtf8(false), DType.newInt(false)}, false);

        BufferAllocator allocator = ArrowAllocation.rootAllocator();

        Schema arrowSchema = new Schema(java.util.List.of(
                new Field("name", FieldType.notNullable(new ArrowType.Utf8()), null),
                new Field("age", FieldType.notNullable(new ArrowType.Int(32, true)), null)));

        try (VortexWriter writer = VortexWriter.create(writePath, writeSchema, new HashMap<>())) {
            // Build a batch with Arrow Java
            try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
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

                // Export to C Data Interface
                try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                        ArrowSchema arrowSchemaFfi = ArrowSchema.allocateNew(allocator)) {
                    Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchemaFfi);

                    writer.writeBatchFfi(arrowArray.memoryAddress(), arrowSchemaFfi.memoryAddress());
                }
            }
        }

        assertTrue(Files.exists(outputPath), "Output file should exist");

        // Read back and verify
        try (var file = dev.vortex.api.Files.open(outputPath.toAbsolutePath().toString());
                var scan = file.newScan(ScanOptions.of())) {
            assertEquals(3, file.rowCount());

            var batch = scan.next();
            var nameField = batch.getField(0);
            var ageField = batch.getField(1);

            assertEquals("Alice", nameField.getUTF8(0));
            assertEquals("Bob", nameField.getUTF8(1));
            assertEquals("Carol", nameField.getUTF8(2));
            assertEquals(30, ageField.getInt(0));
            assertEquals(25, ageField.getInt(1));
            assertEquals(40, ageField.getInt(2));
        }
    }
}
