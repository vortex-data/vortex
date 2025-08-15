// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DType;
import dev.vortex.api.VortexWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
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
                new DType[] {
                    DType.newUtf8(false), DType.newInt(false),
                },
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
    public void testWriteEmptyBatch() throws IOException {
        // Test writing an empty Arrow IPC batch
        Path outputPath = tempDir.resolve("test_empty.vortex");
        String filePath = outputPath.toAbsolutePath().toString();

        String schemaJson =
                "{\"fields\":[{\"name\":\"id\",\"type\":{\"name\":\"int\",\"bitWidth\":32,\"isSigned\":true}}]}";
        Map<String, String> options = new HashMap<>();

        // Create minimal valid Arrow IPC data (empty record batch)
        // This is a minimal Arrow IPC stream with schema and zero records
        byte[] emptyArrowIPC = createMinimalArrowIPC();

        System.err.println("Writing empty batch to: " + filePath);
        System.err.println("Arrow IPC size: " + emptyArrowIPC.length);

        try (VortexWriter writer = VortexWriter.create(filePath, schemaJson, options)) {
            writer.writeBatch(emptyArrowIPC);
            System.err.println("Empty batch written successfully");
        }

        assertTrue(Files.exists(outputPath), "Output file should exist");
        assertTrue(Files.size(outputPath) > 0, "Output file should not be empty");
    }

    private byte[] createMinimalArrowIPC() {
        return new byte[] {
            // Arrow IPC magic number and empty stream markers
            (byte) 0xFF,
            (byte) 0xFF,
            (byte) 0xFF,
            (byte) 0xFF, // continuation marker
            0,
            0,
            0,
            0, // metadata size (0 for empty)
        };
    }
}
