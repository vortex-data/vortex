// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/** JNI boundary for {@link dev.vortex.api.Scan}. */
public final class NativeScan {
    static {
        NativeLoader.loadJni();
    }

    private NativeScan() {}

    /**
     * Create a new scan from a data source. The scan is lazy: no I/O happens until
     * {@link #nextPartition(long)} is called.
     *
     * @param dataSourcePointer pointer from {@link NativeDataSource#open}
     * @param projectionPointer native expression pointer, or 0 for "all columns"
     * @param filterPointer native filter expression pointer, or 0 for "no filter"
     * @param rowRangeBegin inclusive start of the row range, 0 for "unbounded"
     * @param rowRangeEnd exclusive end of the row range, 0 for "unbounded"
     * @param selectionIndices sorted row indices; may be null
     * @param selectionInclude {@code 0} (all), {@code 1} (include {@code selectionIndices}),
     *     {@code 2} (exclude {@code selectionIndices})
     * @param limit max rows to return, or {@code 0} for "no limit"
     * @param ordered true to preserve row order across partitions
     */
    public static native long create(
            long dataSourcePointer,
            long projectionPointer,
            long filterPointer,
            long rowRangeBegin,
            long rowRangeEnd,
            long[] selectionIndices,
            byte selectionInclude,
            long limit,
            boolean ordered);

    /** Free a scan pointer. */
    public static native void free(long pointer);

    /** Export the scan's schema into the Arrow C Data Interface struct at {@code schemaAddress}. */
    public static native void arrowSchema(long pointer, long schemaAddress);

    /** Fill {@code out} with {@code [count, cardinality]}. */
    public static native void partitionCount(long pointer, long[] out);

    /** Advance the scan and return the next partition pointer. Returns {@code 0} when exhausted. */
    public static native long nextPartition(long pointer);
}
