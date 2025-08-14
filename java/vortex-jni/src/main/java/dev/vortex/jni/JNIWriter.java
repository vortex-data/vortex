// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.memory.ArrowBuf;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowStreamWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;

/**
 * JNI implementation of VortexWriter.
 */
public final class JNIWriter implements VortexWriter {
    
    private long ptr;
    private boolean closed = false;
    
    /**
     * Creates a new JNIWriter with the given native pointer.
     *
     * @param ptr the native writer pointer
     */
    public JNIWriter(long ptr) {
        this.ptr = ptr;
    }
    
    /**
     * Writes a batch of Arrow data to the Vortex file.
     *
     * @param batch the Arrow VectorSchemaRoot containing the data batch
     * @throws IOException if writing fails
     */
    @Override
    public void writeBatch(VectorSchemaRoot batch) throws IOException {
        if (closed) {
            throw new IOException("Writer is already closed");
        }
        
        // Serialize the VectorSchemaRoot to Arrow IPC format
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer = new ArrowStreamWriter(
                batch, null, Channels.newChannel(baos))) {
            writer.start();
            writer.writeBatch();
            writer.end();
        }
        
        byte[] arrowData = baos.toByteArray();
        
        // Write the Arrow data to Vortex through JNI
        boolean success = NativeWriterMethods.writeBatch(ptr, arrowData);
        if (!success) {
            throw new IOException("Failed to write batch to Vortex file");
        }
    }
    
    /**
     * Closes the writer and finalizes the Vortex file.
     *
     * @throws IOException if closing fails
     */
    @Override
    public void close() throws IOException {
        if (!closed) {
            boolean success = NativeWriterMethods.close(ptr);
            if (!success) {
                throw new IOException("Failed to close Vortex writer");
            }
            ptr = 0;
            closed = true;
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        if (ptr != 0) {
            // Attempt to close if not already closed
            NativeWriterMethods.close(ptr);
        }
        super.finalize();
    }
}