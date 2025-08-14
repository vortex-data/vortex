// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.jni.JNIWriter;
import dev.vortex.jni.NativeWriterMethods;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.io.IOException;
import java.util.Map;

/**
 * Writer for creating Vortex files from Arrow data.
 * 
 * This class provides methods to write Arrow VectorSchemaRoot batches
 * to Vortex format files.
 */
public interface VortexWriter extends AutoCloseable {
    
    /**
     * Creates a new VortexWriter for writing to the specified file path.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schema the Arrow schema for the data to be written
     * @param options additional writer options
     * @return a new VortexWriter instance
     * @throws IOException if the writer cannot be created
     */
    static VortexWriter create(String filePath, Schema schema, Map<String, String> options) 
            throws IOException {
        long ptr = NativeWriterMethods.create(filePath, schema.toJson(), options);
        if (ptr <= 0) {
            throw new IOException("Failed to create Vortex writer for: " + filePath);
        }
        return new JNIWriter(ptr);
    }
    
    /**
     * Writes a batch of Arrow data to the Vortex file.
     *
     * @param batch the Arrow VectorSchemaRoot containing the data batch
     * @throws IOException if writing fails
     */
    void writeBatch(VectorSchemaRoot batch) throws IOException;
    
    /**
     * Closes the writer and finalizes the Vortex file.
     * 
     * This method must be called to ensure the file is properly written
     * with all necessary metadata and footers.
     *
     * @throws IOException if closing fails
     */
    @Override
    void close() throws IOException;
}