// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.List;

public final class NativeDTypeMethods {
    static {
        NativeLoader.loadJni();
    }

    private NativeDTypeMethods() {}

    public static native void free(long pointer);

    public static native byte getVariant(long pointer);

    public static native boolean isNullable(long pointer);

    public static native List<String> getFieldNames(long pointer);

    // Returns a list of DType pointers.
    public static native List<Long> getFieldTypes(long pointer);

    public static native long getElementType(long pointer);

    public static native boolean isDate(long pointer);

    public static native boolean isTime(long pointer);

    public static native boolean isTimestamp(long pointer);

    public static native byte getTimeUnit(long pointer);

    public static native String getTimeZone(long pointer);

    public static native boolean isDecimal(long pointer);

    public static native int getDecimalPrecision(long pointer);

    public static native byte getDecimalScale(long pointer);
}
