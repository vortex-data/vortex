// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import dev.vortex.api.VortexWriter;
import java.io.IOException;

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
     * @param arrowData the Arrow data in IPC format as byte array
     * @throws IOException if writing fails
     */
    @Override
    public void writeBatch(byte[] arrowData) throws IOException {
        if (closed) {
            throw new IOException("Writer is already closed");
        }
        
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