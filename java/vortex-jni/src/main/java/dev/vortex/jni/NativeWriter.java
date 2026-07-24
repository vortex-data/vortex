// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import dev.vortex.api.VortexWriteSummary;
import java.util.Map;

/** JNI boundary for {@link dev.vortex.api.VortexWriter}. */
public final class NativeWriter {
    static {
        NativeLoader.loadJni();
    }

    private NativeWriter() {}

    /** Open a writer at {@code uri} that accepts batches matching the Arrow schema at {@code arrowSchemaAddress}. */
    public static native long create(
            long sessionPointer, String uri, long arrowSchemaAddress, Map<String, String> options);

    /**
     * Open a writer that streams the file into a caller-provided {@link dev.vortex.io.NativeWritable}. The native side
     * writes and flushes but never closes the writable; the caller must close it after {@link #close}.
     */
    public static native long createStream(long sessionPointer, Object writable, long arrowSchemaAddress);

    /**
     * Write a batch directly from Arrow C Data Interface addresses.
     *
     * @param writerPointer pointer from {@link #create}
     * @param arrowArrayAddress address of an {@code ArrowArray} struct
     * @param arrowSchemaAddress address of an {@code ArrowSchema} struct
     * @return {@code true} on success
     */
    public static native boolean writeBatch(long writerPointer, long arrowArrayAddress, long arrowSchemaAddress);

    /** Number of bytes successfully written to the underlying sink so far. */
    public static native long bytesWritten(long writerPointer);

    /** Number of uncompressed bytes buffered by the native writer that have not yet reached the sink. */
    public static native long bufferedBytes(long writerPointer);

    /** Flush and close the writer. Must be called exactly once. */
    public static native void close(long writerPointer);

    /** Flush and close the writer, returning its completed-file summary. Must be called exactly once. */
    public static native VortexWriteSummary finish(long writerPointer);
}
