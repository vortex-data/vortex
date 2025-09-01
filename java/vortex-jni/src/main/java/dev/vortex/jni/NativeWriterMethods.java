// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.Map;

/**
 * Native JNI methods for writing Vortex files.
 */
public final class NativeWriterMethods {

    static {
        NativeLoader.loadJni();
    }

    private NativeWriterMethods() {}

    /**
     * Creates a new native Vortex writer.
     *
     * @param uri     the URI to the file,  e.g. "file://path/to/file".
     * @param dtype   native pointer to a writer schema (Vortex DType)
     * @param options additional writer options. For cloud storage this includes things like credentials.
     * @return a native pointer to the writer, or 0 on failure
     */
    public static native long create(String uri, long dtype, Map<String, String> options);

    /**
     * Writes a batch of Arrow data to the Vortex file.
     *
     * @param writerPtr the native writer pointer
     * @param arrowData the Arrow IPC format data
     * @return true if successful, false otherwise
     */
    public static native boolean writeBatch(long writerPtr, byte[] arrowData);

    /**
     * Close and flush the writer, finalizing it to the storage system.
     *
     * @param writerPtr the native writer pointer
     * @throws RuntimeException if the writer fails to close
     */
    public static native void close(long writerPtr);
}
