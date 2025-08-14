// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import static org.junit.jupiter.api.Assertions.*;

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
        String filePath = outputPath.toAbsolutePath().toString();

        // Minimal Arrow schema
        String schemaJson =
                "{\"fields\":[{\"name\":\"id\",\"type\":{\"name\":\"int\",\"bitWidth\":32,\"isSigned\":true}}]}";
        Map<String, String> options = new HashMap<>();

        System.err.println("Creating writer for path: " + filePath);
        System.err.println("Schema JSON: " + schemaJson);

        try (VortexWriter writer = VortexWriter.create(filePath, schemaJson, options)) {
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
        // Create a minimal valid Arrow IPC stream
        // This is the binary format for an empty Arrow stream with a simple schema
        // Format: [continuation][metadata_size][metadata_flatbuffer][padding][body]

        // For now, return a simple placeholder that should at least not crash
        // We'll need to construct proper Arrow IPC format
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
