// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/** JNI boundary for {@link dev.vortex.api.Expression}. */
public final class NativeExpression {
    static {
        NativeLoader.loadJni();
    }

    private NativeExpression() {}

    public static native long root();

    public static native long getItem(String fieldName, long childPointer);

    public static native long select(String[] fieldNames, long childPointer);

    public static native long and(long[] operandPointers);

    public static native long or(long[] operandPointers);

    public static native long binary(byte operator, long lhs, long rhs);

    public static native long not(long childPointer);

    public static native long isNull(long childPointer);

    public static native long isNotNull(long childPointer);

    public static native long like(long childPointer, long patternPointer, boolean negated, boolean caseInsensitive);

    public static native long between(
            long valuePointer, long lowerPointer, long upperPointer, boolean lowerStrict, boolean upperStrict);

    public static native long literalBool(boolean value, boolean isNull);

    public static native long literalI8(byte value, boolean isNull);

    public static native long literalI16(short value, boolean isNull);

    public static native long literalI32(int value, boolean isNull);

    public static native long literalI64(long value, boolean isNull);

    public static native long literalF32(float value, boolean isNull);

    public static native long literalF64(double value, boolean isNull);

    public static native long literalString(String value);

    public static native long literalBinary(byte[] value);

    public static native long literalDecimal(byte[] unscaledBigEndian, int precision, int scale, boolean isNull);

    public static native long literalDate(long value, byte timeUnitTag, boolean isNull);

    public static native long literalTimestamp(long value, byte timeUnitTag, String timezone, boolean isNull);

    public static native long literalNull(byte dtypeTag);

    public static native void free(long pointer);
}
