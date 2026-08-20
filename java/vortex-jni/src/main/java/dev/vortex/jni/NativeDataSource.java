// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.Map;

/** JNI boundary for {@link dev.vortex.api.DataSource}. */
public final class NativeDataSource {
    static {
        NativeLoader.loadJni();
    }

    private NativeDataSource() {}

    /**
     * Open a data source from one or more URIs or globs.
     *
     * @param sessionPointer pointer from {@link NativeSession#newSession()}
     * @param uris paths or globs (for example {@code ["file:///a.vortex", "file:///b.vortex"]})
     * @param options object-store properties (may be null)
     */
    public static native long open(long sessionPointer, String[] uris, Map<String, String> options);

    /**
     * Open a data source over caller-provided {@link dev.vortex.io.NativeReadable} objects. All reads become upcalls
     * into the supplied readables; no native storage client is created.
     *
     * @param sessionPointer pointer from {@link NativeSession#newSession()}
     * @param readables one {@link dev.vortex.io.NativeReadable} per file
     * @param names unique name per file (from {@link dev.vortex.io.NativeReadable#name()}), parallel to
     *     {@code readables}
     * @param lengths file sizes in bytes, parallel to {@code readables}
     * @param readConcurrency maximum in-flight {@code readFully} upcalls across all files of this data source;
     *     {@code <= 0} selects the default
     */
    public static native long openFiles(
            long sessionPointer, Object[] readables, String[] names, long[] lengths, int readConcurrency);

    /** Free a data source pointer. */
    public static native void free(long pointer);

    /**
     * Export the data source's schema into the Arrow C Data Interface struct at {@code schemaAddress}. Extension dtypes
     * are dispatched through the session's registered Arrow export plugins.
     *
     * @param sessionPointer pointer from {@link NativeSession#newSession()}
     */
    public static native void arrowSchema(long sessionPointer, long pointer, long schemaAddress);

    /**
     * Populate {@code out} with {@code [rows, cardinality]}. Cardinality is one of {@code 0=unknown},
     * {@code 1=estimate}, {@code 2=exact}.
     */
    public static native void rowCount(long pointer, long[] out);

    /**
     * Populate {@code out} with {@code [bytes, precision]}, the sum of on-storage file sizes for the data source.
     * Precision is one of {@code 0=unknown}, {@code 1=estimate}, {@code 2=exact}.
     */
    public static native void byteSize(long pointer, long[] out);
}
