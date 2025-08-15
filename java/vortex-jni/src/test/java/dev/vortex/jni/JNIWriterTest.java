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
}
