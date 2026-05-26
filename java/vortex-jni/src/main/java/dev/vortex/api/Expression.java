// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeExpression;
import java.math.BigInteger;
import java.util.Arrays;

/**
 * A Vortex expression node backed by a native pointer.
 *
 * <p>Expressions are composed via the static factories ({@link #root()}, {@link #getItem(String, Expression)}, etc.).
 * Each returned {@code Expression} owns its native pointer; the pointer is released automatically when the
 * {@code Expression} is no longer reachable. Passing an expression as an input to a builder does <em>not</em> transfer
 * ownership — the resulting expression is an independent copy on the native side.
 */
public final class Expression {
    private final long pointer;

    private Expression(long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid expression pointer");
        this.pointer = pointer;
        VortexCleaner.register(this, () -> NativeExpression.free(pointer));
    }

    long nativePointer() {
        return pointer;
    }

    /** The root expression: applying it to an array yields the array itself. */
    public static Expression root() {
        return new Expression(NativeExpression.root());
    }

    /** Access a named field from a struct expression. */
    public static Expression getItem(String fieldName, Expression child) {
        return new Expression(NativeExpression.getItem(fieldName, child.nativePointer()));
    }

    /** Shortcut for {@code getItem(fieldName, root())}. */
    public static Expression column(String fieldName) {
        return getItem(fieldName, root());
    }

    /**
     * Access a nested field by walking {@code fieldNames} starting from the root of the array. With a single name this
     * is equivalent to {@link #column(String)}.
     */
    public static Expression column(String[] fieldNames) {
        Preconditions.checkArgument(fieldNames.length > 0, "column requires at least one field name");
        Expression expr = root();
        for (String name : fieldNames) {
            expr = getItem(name, expr);
        }
        return expr;
    }

    /** Project a subset of fields out of a struct expression. */
    public static Expression select(String[] fieldNames, Expression child) {
        return new Expression(NativeExpression.select(fieldNames, child.nativePointer()));
    }

    /** Logical AND. Requires at least one operand. */
    public static Expression and(Expression... operands) {
        Preconditions.checkArgument(operands.length > 0, "and requires at least one operand");
        return new Expression(NativeExpression.and(nativePointers(operands)));
    }

    /** Logical OR. Requires at least one operand. */
    public static Expression or(Expression... operands) {
        Preconditions.checkArgument(operands.length > 0, "or requires at least one operand");
        return new Expression(NativeExpression.or(nativePointers(operands)));
    }

    public static Expression binary(BinaryOp op, Expression lhs, Expression rhs) {
        return new Expression(NativeExpression.binary(op.code(), lhs.nativePointer(), rhs.nativePointer()));
    }

    public static Expression not(Expression child) {
        return new Expression(NativeExpression.not(child.nativePointer()));
    }

    public static Expression isNull(Expression child) {
        return new Expression(NativeExpression.isNull(child.nativePointer()));
    }

    public static Expression isNotNull(Expression child) {
        return new Expression(NativeExpression.isNotNull(child.nativePointer()));
    }

    /**
     * SQL {@code LIKE} pattern match.
     *
     * @param negated whether to invert the result (i.e. {@code NOT LIKE})
     * @param caseInsensitive whether to perform case-insensitive matching ({@code ILIKE})
     */
    public static Expression like(Expression child, Expression pattern, boolean negated, boolean caseInsensitive) {
        return new Expression(
                NativeExpression.like(child.nativePointer(), pattern.nativePointer(), negated, caseInsensitive));
    }

    /**
     * {@code value BETWEEN lower AND upper}.
     *
     * @param lowerStrict {@code true} for {@code lower < value}; {@code false} for {@code lower <= value}.
     * @param upperStrict {@code true} for {@code value < upper}; {@code false} for {@code value <= upper}.
     */
    public static Expression between(
            Expression value, Expression lower, Expression upper, boolean lowerStrict, boolean upperStrict) {
        return new Expression(NativeExpression.between(
                value.nativePointer(), lower.nativePointer(), upper.nativePointer(), lowerStrict, upperStrict));
    }

    public static Expression literal(boolean value) {
        return new Expression(NativeExpression.literalBool(value, false));
    }

    public static Expression nullLiteralBool() {
        return new Expression(NativeExpression.literalBool(false, true));
    }

    public static Expression literal(byte value) {
        return new Expression(NativeExpression.literalI8(value, false));
    }

    public static Expression literal(short value) {
        return new Expression(NativeExpression.literalI16(value, false));
    }

    public static Expression literal(int value) {
        return new Expression(NativeExpression.literalI32(value, false));
    }

    public static Expression literal(long value) {
        return new Expression(NativeExpression.literalI64(value, false));
    }

    public static Expression literal(float value) {
        return new Expression(NativeExpression.literalF32(value, false));
    }

    public static Expression literal(double value) {
        return new Expression(NativeExpression.literalF64(value, false));
    }

    public static Expression literal(String value) {
        return new Expression(NativeExpression.literalString(value));
    }

    public static Expression literal(byte[] value) {
        Preconditions.checkArgument(value != null, "use nullLiteral(DType.BINARY) for a null binary literal");
        return new Expression(NativeExpression.literalBinary(value));
    }

    /**
     * Create a decimal literal from its unscaled two's-complement big-endian byte representation (i.e. the value
     * returned by {@link BigInteger#toByteArray()}).
     */
    public static Expression literalDecimal(BigInteger unscaledValue, int precision, int scale) {
        Preconditions.checkArgument(unscaledValue != null, "unscaledValue must not be null");
        return new Expression(NativeExpression.literalDecimal(unscaledValue.toByteArray(), precision, scale, false));
    }

    /** Create a null decimal literal with the specified precision and scale. */
    public static Expression nullLiteralDecimal(int precision, int scale) {
        return new Expression(NativeExpression.literalDecimal(new byte[] {0}, precision, scale, true));
    }

    /**
     * Create a Date literal. The {@code value} is the number of {@code unit} units since the Unix epoch.
     *
     * @param unit only {@link TimeUnit#DAYS} and {@link TimeUnit#MILLISECONDS} are valid for Date.
     */
    public static Expression literalDate(long value, TimeUnit unit) {
        return new Expression(NativeExpression.literalDate(value, unit.tag(), false));
    }

    /** Null Date literal. See {@link #literalDate(long, TimeUnit)} for the {@code unit} constraints. */
    public static Expression nullLiteralDate(TimeUnit unit) {
        return new Expression(NativeExpression.literalDate(0L, unit.tag(), true));
    }

    /**
     * Create a Timestamp literal. The {@code value} is the number of {@code unit} units since the Unix epoch.
     *
     * @param timezone optional IANA timezone identifier (e.g. {@code "UTC"}, {@code "America/Los_Angeles"}). Pass
     *     {@code null} for a local (zone-naive) timestamp.
     */
    public static Expression literalTimestamp(long value, TimeUnit unit, String timezone) {
        return new Expression(NativeExpression.literalTimestamp(value, unit.tag(), timezone, false));
    }

    /** Null Timestamp literal. See {@link #literalTimestamp(long, TimeUnit, String)} for parameter semantics. */
    public static Expression nullLiteralTimestamp(TimeUnit unit, String timezone) {
        return new Expression(NativeExpression.literalTimestamp(0L, unit.tag(), timezone, true));
    }

    /** Create a typed null literal of the given primitive {@link DType}. */
    public static Expression nullLiteral(DType dtype) {
        return new Expression(NativeExpression.literalNull(dtype.tag()));
    }

    private static long[] nativePointers(Expression[] exprs) {
        return Arrays.stream(exprs).mapToLong(Expression::nativePointer).toArray();
    }

    /** Binary operator codes; must match the Rust {@code parse_op} table. */
    public enum BinaryOp {
        EQ((byte) 0),
        NOT_EQ((byte) 1),
        GT((byte) 2),
        GTE((byte) 3),
        LT((byte) 4),
        LTE((byte) 5),
        AND((byte) 6),
        OR((byte) 7),
        ADD((byte) 8),
        SUB((byte) 9),
        MUL((byte) 10),
        DIV((byte) 11);

        private final byte code;

        BinaryOp(byte code) {
            this.code = code;
        }

        public byte code() {
            return code;
        }
    }

    /** Time units for Date/Timestamp literals. Tag values must match the Rust {@code parse_time_unit} table. */
    public enum TimeUnit {
        NANOSECONDS((byte) 0),
        MICROSECONDS((byte) 1),
        MILLISECONDS((byte) 2),
        SECONDS((byte) 3),
        DAYS((byte) 4);

        private final byte tag;

        TimeUnit(byte tag) {
            this.tag = tag;
        }

        public byte tag() {
            return tag;
        }
    }

    /** Primitive {@link DType}s that can be used to construct typed null literals via {@link #nullLiteral(DType)}. */
    public enum DType {
        BOOL((byte) 0),
        I8((byte) 1),
        I16((byte) 2),
        I32((byte) 3),
        I64((byte) 4),
        F32((byte) 5),
        F64((byte) 6),
        UTF8((byte) 7),
        BINARY((byte) 8);

        private final byte tag;

        DType(byte tag) {
            this.tag = tag;
        }

        public byte tag() {
            return tag;
        }
    }
}
