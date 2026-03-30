// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.List;

public final class NativeDTypeMethods {
    static {
        NativeLoader.loadJni();
    }

    private NativeDTypeMethods() {}

    /**
     * Create a new native DType for a PType::I8. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newByte(boolean isNullable);

    /**
     * Create a new native DType for a PType::I16. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newShort(boolean isNullable);

    /**
     * Create a new native DType for a PType::I32. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newInt(boolean isNullable);

    /**
     * Create a new native DType for a PType::I64. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newLong(boolean isNullable);

    /**
     * Create a new native DType for a PType::F32. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newFloat(boolean isNullable);

    /**
     * Create a new native DType for a PType::F64. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newDouble(boolean isNullable);

    /**
     * Create a new native DType for a decimal type. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @param precision  decimal precision
     * @param scale      decimal scale
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newDecimal(int precision, int scale, boolean isNullable);

    /**
     * Create a new native DType for a UTF-8 string type. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newUtf8(boolean isNullable);

    /**
     * Create a new native DType for a Binary type. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newBinary(boolean isNullable);

    /**
     * Create a new native DType for a boolean type. The created object lives in native memory.
     *
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newBool(boolean isNullable);

    /**
     * Create a new native DType for a List type. The created object lives in native memory.
     *
     * @param elementTypePtr A native pointer to a DType containing the type of the elements
     * @param isNullable     true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newList(long elementTypePtr, boolean isNullable);

    /**
     * Create a new native DType for a FixedSizeList type. The created object lives in native memory.
     *
     * @param elementTypePtr A native pointer to a DType containing the type of the elements
     * @param size           The fixed size of each list
     * @param isNullable     true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newFixedSizeList(long elementTypePtr, int size, boolean isNullable);

    /**
     * Create a new native DType for a Struct type. The created object lives in native memory.
     *
     * @param fieldNames An array of field names
     * @param fieldTypes An array of native pointers to the DType for each field
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newStruct(String[] fieldNames, long[] fieldTypes, boolean isNullable);

    /**
     * Create a new native DType for a List type. The created object lives in native memory.
     *
     * @param timeUnit   A byte that represents a {@link dev.vortex.api.DType.TimeUnit}
     * @param zone       The time zone or offset string
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newTimestamp(byte timeUnit, String zone, boolean isNullable);

    /**
     * Create a new native DType for a List type. The created object lives in native memory.
     *
     * @param timeUnit   A byte that represents a {@link dev.vortex.api.DType.TimeUnit}
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newDate(byte timeUnit, boolean isNullable);

    /**
     * Create a new native DType for a List type. The created object lives in native memory.
     *
     * @param timeUnit   A byte that represents a {@link dev.vortex.api.DType.TimeUnit}
     * @param isNullable true if the values can be null
     * @return Pointer to a new heap-allocated {@code DType}.
     */
    public static native long newTime(byte timeUnit, boolean isNullable);

    public static native void free(long pointer);

    public static native byte getVariant(long pointer);

    public static native boolean isNullable(long pointer);

    public static native List<String> getFieldNames(long pointer);

    // Returns a list of DType pointers.
    public static native List<Long> getFieldTypes(long pointer);

    public static native long getElementType(long pointer);

    public static native int getFixedSizeListSize(long pointer);

    public static native boolean isDate(long pointer);

    public static native boolean isTime(long pointer);

    public static native boolean isTimestamp(long pointer);

    public static native byte getTimeUnit(long pointer);

    public static native String getTimeZone(long pointer);

    public static native boolean isDecimal(long pointer);

    public static native int getDecimalPrecision(long pointer);

    public static native byte getDecimalScale(long pointer);
}
