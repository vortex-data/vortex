// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.List;
import java.util.Map;

public final class NativeFileMethods {
    static {
        NativeLoader.loadJni();
    }

    private NativeFileMethods() {}

    /**
     * Open a file using the native library with the provided URI and options.
     * @param uri     The URI of the file to open. e.g. "file://path/to/file".
     * @param options A map of options to provide for opening the file.
     * @return A native pointer to the opened file. This will be 0 if the open call failed.
     */
    public static native long open(String uri, Map<String, String> options);

    /**
     * Get the total row count contained in the file associated with the given pointer.
     * @param pointer The native pointer to a file. Must be a value returned by {@link #open(String, Map)}.
     * @return The number of rows of data encoded in the file. This includes null values.
     */
    public static native long rowCount(long pointer);

    /**
     * Get the data type of the file associated with the given pointer.
     * @param pointer The native pointer to a file. Must be a value returned by {@link #open(String, Map)}.
     * @return Native pointer to the DType of the file. This pointer is owned by the file and should not be freed.
     */
    public static native long dtype(long pointer);

    /**
     * Close the file associated with the given pointer.
     * @param pointer The native pointer to a file. Must be a value returned by {@link #open(String, Map)}.
     */
    public static native void close(long pointer);

    /**
     * Build a new native scan operator that will materialize Arrays from the file, pushing down the optional
     * predicate, row range or row indices to perform data skipping.
     */
    public static native long scan(
            long pointer, List<String> columns, byte[] predicateProto, long[] rowRange, long[] rowIndices);
}
