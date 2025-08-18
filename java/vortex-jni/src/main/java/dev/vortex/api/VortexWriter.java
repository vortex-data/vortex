// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.jni.JNIDType;
import dev.vortex.jni.JNIWriter;
import dev.vortex.jni.NativeWriterMethods;
import java.io.IOException;
import java.util.Map;

/**
 * Writer for creating Vortex files from Arrow data.
 * <p>
 * This class provides methods to write Arrow VectorSchemaRoot batches
 * to Vortex format files.
 */
public interface VortexWriter extends AutoCloseable {

    /**
     * Creates a new VortexWriter for writing to the specified file path.
     *
     * @param uri     The URI for where the file is opened
     * @param dtype   The Vortex DType for data that gets written
     * @param options additional writer options
     * @return a new VortexWriter instance
     * @throws IOException if the writer cannot be created
     */
    static VortexWriter create(String uri, DType dtype, Map<String, String> options) throws IOException {
        long ptr = NativeWriterMethods.create(uri, ((JNIDType) dtype).getPointer(), options);
        if (ptr <= 0) {
            throw new IOException("Failed to create Vortex writer for: " + uri + " (got ptr=" + ptr + ")");
        }
        return new JNIWriter(ptr);
    }

    /**
     * Writes a batch of Arrow data to the Vortex file.
     *
     * @param arrowData the Arrow data in IPC format as byte array
     * @throws IOException if writing fails
     */
    void writeBatch(byte[] arrowData) throws IOException;

    /**
     * Closes the writer and finalizes the Vortex file.
     * <p>
     * This method must be called to ensure the file is properly written
     * with all necessary metadata and footers.
     *
     * @throws IOException if closing fails
     */
    @Override
    void close() throws IOException;
}
