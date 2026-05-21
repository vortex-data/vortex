// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeExpression;
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
}
