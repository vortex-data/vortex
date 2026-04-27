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

    /** Free a data source pointer. */
    public static native void free(long pointer);

    /** Export the data source's schema into the Arrow C Data Interface struct at {@code schemaAddress}. */
    public static native void arrowSchema(long pointer, long schemaAddress);

    /**
     * Populate {@code out} with {@code [rows, cardinality]}. Cardinality is one of
     * {@code 0=unknown}, {@code 1=estimate}, {@code 2=exact}.
     */
    public static native void rowCount(long pointer, long[] out);
}
