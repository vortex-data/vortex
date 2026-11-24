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
     * Create a new scan from a data source. The scan is lazy: no I/O happens until {@link #nextPartition(long)} is
     * called.
     *
     * @param dataSourcePointer pointer from {@link NativeDataSource#open}
     * @param projectionPointer native expression pointer, or 0 for "all columns"
     * @param filterPointer native filter expression pointer, or 0 for "no filter"
     * @param hasRowRangeBegin whether {@code rowRangeBegin} is set
     * @param rowRangeBegin inclusive start of the row range; ignored when {@code hasRowRangeBegin} is {@code false}
     * @param hasRowRangeEnd whether {@code rowRangeEnd} is set
     * @param rowRangeEnd exclusive end of the row range; ignored when {@code hasRowRangeEnd} is {@code false}
     * @param selectionIndices sorted row indices; may be null
     * @param selectionRoaringBitmap portable serialized Roaring64 bitmap; may be null
     * @param selectionInclude {@code 0} (all), {@code 1} (include {@code selectionIndices}), {@code 2} (exclude
     *     {@code selectionIndices}), {@code 3} (include {@code selectionRoaringBitmap}), {@code 4} (exclude
     *     {@code selectionRoaringBitmap})
     * @param hasLimit whether {@code limit} is set
     * @param limit max rows to return; ignored when {@code hasLimit} is {@code false}
     * @param ordered true to preserve row order across partitions
     * @param splitConcurrency the most splits of each partition read at once, or {@code 0} for the native default
     */
    public static native long create(
            long dataSourcePointer,
            long projectionPointer,
            long filterPointer,
            boolean hasRowRangeBegin,
            long rowRangeBegin,
            boolean hasRowRangeEnd,
            long rowRangeEnd,
            long[] selectionIndices,
            byte[] selectionRoaringBitmap,
            byte selectionInclude,
            boolean hasLimit,
            long limit,
            boolean ordered,
            int splitConcurrency);

    /** Free a scan pointer. */
    public static native void free(long pointer);

    /**
     * Export the scan's schema into the Arrow C Data Interface struct at {@code schemaAddress}. Extension dtypes are
     * dispatched through the session's registered Arrow export plugins.
     *
     * @param sessionPointer pointer from {@link NativeSession#newSession()}
     */
    public static native void arrowSchema(long sessionPointer, long pointer, long schemaAddress);

    /** Fill {@code out} with {@code [count, cardinality]}. */
    public static native void partitionCount(long pointer, long[] out);

    /** Advance the scan and return the next partition pointer. Returns {@code 0} when exhausted. */
    public static native long nextPartition(long pointer);
}
