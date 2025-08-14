// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import dev.vortex.api.VortexWriter;
import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JNI implementation of VortexWriter.
 * 
 * This class implements AutoCloseable to ensure proper resource cleanup
 * when used with try-with-resources.
 */
public final class JNIWriter implements VortexWriter, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(JNIWriter.class);

    private long ptr;
    private boolean closed = false;

    /**
     * Creates a new JNIWriter with the given native pointer.
     *
     * @param ptr the native writer pointer
     */
    public JNIWriter(long ptr) {
        this.ptr = ptr;
        logger.debug("Created JNIWriter with ptr={}", ptr);
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

        logger.trace("Writing batch with {} bytes", arrowData.length);
        
        // Write the Arrow data to Vortex through JNI
        boolean success = NativeWriterMethods.writeBatch(ptr, arrowData);
        if (!success) {
            logger.error("Failed to write batch to Vortex file");
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
        if (!closed && ptr > 0) {
            logger.debug("Closing JNIWriter with ptr={}", ptr);
            boolean success = NativeWriterMethods.close(ptr);
            if (!success) {
                logger.error("Failed to close Vortex writer");
                throw new IOException("Failed to close Vortex writer");
            }
            ptr = 0;
            closed = true;
        } else if (closed) {
            logger.trace("JNIWriter already closed");
        }
    }

    // Removed deprecated finalize() method - proper cleanup should be done via close()
}
