// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.math.BigDecimal;

public final class NativeArrayMethods {
    static {
        NativeLoader.loadJni();
    }

    private NativeArrayMethods() {}

    public static native long nbytes(long pointer);

    public static native void exportToArrow(long pointer, long[] schemaPointer, long[] arrayPointer);

    public static native void dropArrowSchema(long arrowSchemaPtr);

    public static native void dropArrowArray(long arrowArrayPtr);

    public static native void free(long pointer);

    public static native long getLen(long pointer);

    public static native long getDataType(long pointer);

    public static native long getField(long pointer, int index);

    public static native long slice(long pointer, int start, int stop);

    public static native boolean getNull(long pointer, int index);

    public static native int getNullCount(long pointer);

    public static native byte getByte(long pointer, int index);

    public static native short getShort(long pointer, int index);

    public static native int getInt(long pointer, int index);

    public static native long getLong(long pointer, int index);

    public static native boolean getBool(long pointer, int index);

    public static native float getFloat(long pointer, int index);

    public static native double getDouble(long pointer, int index);

    public static native BigDecimal getBigDecimal(long pointer, int index);

    public static native String getUTF8(long pointer, int index);

    /**
     * Raw-pointer variant of {@link #getUTF8(long, int)} that accepts an array to hold
     * a pointer and an output length.
     * <p>
     * For Java query engines that use Unsafe to manipulate native memory, this allows working with the string
     * inside of the JVM without copying it into Java heap memory.
     */
    public static native void getUTF8_ptr_len(long pointer, int index, long[] outPtr, int[] outLen);

    public static native byte[] getBinary(long pointer, int index);
}
