// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/** JNI boundary for a typed {@link dev.vortex.api.BoundExpression}. */
public final class NativeBoundExpression {
    static {
        NativeLoader.loadJni();
    }

    private NativeBoundExpression() {}

    /** Type check an authored expression against the data source schema. */
    public static native long bind(long expressionPointer, long dataSourcePointer);

    /** Construct a typed scope root from the data source schema. */
    public static native long root(long dataSourcePointer);

    /** Construct the typed row index used in Vortex scans. */
    public static native long rowIdx();

    /** Read a struct field from a typed child. */
    public static native long getItem(String name, long childPointer);

    /** Select fields from a typed struct child. */
    public static native long select(String[] names, long childPointer);

    /** Pack typed children into a struct. */
    public static native long pack(String[] fieldNames, long[] expressionPointers, boolean nullable);

    /** Merge typed structs using the given duplicate field policy. */
    public static native long merge(long[] expressionPointers, byte duplicateHandling);

    public static native long literalBool(boolean value, boolean isNull);

    public static native long literalI8(byte value, boolean isNull);

    public static native long literalI16(short value, boolean isNull);

    public static native long literalI32(int value, boolean isNull);

    public static native long literalI64(long value, boolean isNull);

    public static native long literalU64(long bits);

    public static native long literalF32(float value, boolean isNull);

    public static native long literalF64(double value, boolean isNull);

    public static native long literalString(String value);

    public static native long literalBinary(byte[] value);

    public static native long literalNull(byte dtypeTag);

    public static native long literalDate(long value, byte timeUnit, boolean isNull);

    public static native long literalTimestamp(long value, byte timeUnit, String timezone, boolean isNull);

    public static native long literalDecimal(byte[] unscaledBigEndian, int precision, int scale, boolean isNull);

    public static native long literalUuid(byte[] bigEndianBytes, boolean isNull);

    /** Construct a typed binary call. */
    public static native long binary(byte operator, long lhsPointer, long rhsPointer);

    /** Explicitly cast a typed value to a signed 64-bit integer. */
    public static native long castToI64(long childPointer);

    /** Construct an exact-typed BETWEEN call. */
    public static native long between(
            long valuePointer, long lowerPointer, long upperPointer, boolean lowerStrict, boolean upperStrict);

    /** Construct typed boolean negation. */
    public static native long not(long childPointer);

    /** Construct a typed null test. */
    public static native long isNull(long childPointer);

    /** Construct a typed non-null test. */
    public static native long isNotNull(long childPointer);

    /** Construct a typed LIKE call. */
    public static native long like(long valuePointer, long patternPointer, boolean negated, boolean caseInsensitive);

    /** Release an owned bound expression pointer. */
    public static native void free(long pointer);
}
