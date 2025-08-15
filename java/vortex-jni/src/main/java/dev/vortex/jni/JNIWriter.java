// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import dev.vortex.api.VortexWriter;
import java.io.IOException;
import java.util.OptionalLong;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * JNI implementation of VortexWriter.
 * <p>
 * This class implements AutoCloseable to ensure proper resource cleanup
 * when used with try-with-resources.
 */
public final class JNIWriter implements VortexWriter, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(JNIWriter.class);

    private OptionalLong ptr;

    /**
     * Creates a new JNIWriter with the given native pointer.
     *
     * @param ptr the native writer pointer
     */
    public JNIWriter(long ptr) {
        this.ptr = OptionalLong.of(ptr);
        logger.debug("Created JNIWriter with ptr={}", ptr);
    }

    /**
     * Writes a batch of Arrow data to the Vortex file.
     *
     * @param arrowData the Arrow data in IPC format as byte array
     * @throws NullPointerException if this is called after the writer has been closed.
     */
    @Override
    public void writeBatch(byte[] arrowData) throws IOException {
        logger.trace("Writing batch with {} bytes", arrowData.length);

        // Write the Arrow data to Vortex through JNI
        boolean success = NativeWriterMethods.writeBatch(ptr.getAsLong(), arrowData);
        if (!success) {
            logger.error("Failed to write batch to Vortex file");
            throw new IOException("Failed to write batch to Vortex file");
        }
    }

    /**
     * Closes the writer and finalizes the Vortex file.
     *
     * @throws RuntimeException if closing fails
     */
    @Override
    public void close() {
        if (this.ptr.isEmpty()) {
            logger.debug("Attempted to close already closed JNIWriter, skipping");
            return;
        }

        long ptr = this.ptr.getAsLong();

        logger.debug("Closing JNIWriter with ptr={}", ptr);
        NativeWriterMethods.close(ptr);
        this.ptr = OptionalLong.empty();
    }
}
