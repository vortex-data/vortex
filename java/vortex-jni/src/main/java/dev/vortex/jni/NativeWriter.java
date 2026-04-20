// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.Map;

/** JNI boundary for {@link dev.vortex.api.VortexWriter}. */
public final class NativeWriter {
    static {
        NativeLoader.loadJni();
    }

    private NativeWriter() {}

    /**
     * Open a writer at {@code uri} that accepts batches matching the Arrow schema at
     * {@code arrowSchemaAddress}.
     */
    public static native long create(
            long sessionPointer, String uri, long arrowSchemaAddress, Map<String, String> options);

    /**
     * Write a batch directly from Arrow C Data Interface addresses.
     *
     * @param writerPointer pointer from {@link #create}
     * @param arrowArrayAddress address of an {@code ArrowArray} struct
     * @param arrowSchemaAddress address of an {@code ArrowSchema} struct
     * @return {@code true} on success
     */
    public static native boolean writeBatch(long writerPointer, long arrowArrayAddress, long arrowSchemaAddress);

    /** Flush and close the writer. Must be called exactly once. */
    public static native void close(long writerPointer);
}
