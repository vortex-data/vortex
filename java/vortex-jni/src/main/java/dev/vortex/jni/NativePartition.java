// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/** JNI boundary for {@link dev.vortex.api.Partition}. */
public final class NativePartition {
    static {
        NativeLoader.loadJni();
    }

    private NativePartition() {}

    /** Free a partition pointer that was not consumed by {@link #scanArrow}. */
    public static native void free(long pointer);

    /** Fill {@code out} with {@code [rows, cardinality]}. */
    public static native void rowCount(long pointer, long[] out);

    /**
     * Consume the partition into the {@code FFI_ArrowArrayStream} at {@code streamAddress}. The partition pointer is
     * invalidated by this call.
     *
     * @param sessionPointer native session pointer used for execution context
     * @param partitionPointer partition pointer to consume
     * @param streamAddress address of an allocated {@code FFI_ArrowArrayStream} struct
     */
    public static native void scanArrow(long sessionPointer, long partitionPointer, long streamAddress);
}
