// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeBoundExpression;
import java.lang.ref.Reference;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Objects;
import java.util.UUID;

/** A Vortex expression whose fields and function argument types were checked against a data source. */
public final class BoundExpression {
    private final long pointer;

    BoundExpression(long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid bound expression pointer");
        this.pointer = pointer;
        VortexCleaner.register(this, () -> NativeBoundExpression.free(pointer));
    }

    long nativePointer() {
        return pointer;
    }

    /** The root of the given data source's typed input scope. */
    public static BoundExpression root(DataSource source) {
        try {
            return new BoundExpression(NativeBoundExpression.root(source.nativePointer()));
        } finally {
            Reference.reachabilityFence(source);
        }
    }

    /** Read the original file row index within a scan. */
    public static BoundExpression rowIdx() {
        return new BoundExpression(NativeBoundExpression.rowIdx());
    }

    /** Read a named field from a typed struct child. */
    public static BoundExpression getItem(String name, BoundExpression child) {
        Objects.requireNonNull(name, "name");
        try {
            return new BoundExpression(NativeBoundExpression.getItem(name, child.pointer));
        } finally {
            Reference.reachabilityFence(child);
        }
    }

    /** Read a named field from the data source root. */
    public static BoundExpression column(DataSource source, String... names) {
        Preconditions.checkArgument(names.length > 0, "column requires at least one field name");
        BoundExpression result = root(source);
        for (String name : names) {
            result = getItem(name, result);
        }
        return result;
    }

    /** Select fields from a typed struct child. */
    public static BoundExpression select(String[] names, BoundExpression child) {
        try {
            return new BoundExpression(NativeBoundExpression.select(names, child.pointer));
        } finally {
            Reference.reachabilityFence(child);
        }
    }

    /** Pack typed values into a struct with named fields. */
    public static BoundExpression pack(String[] fieldNames, BoundExpression[] expressions, boolean nullable) {
        Objects.requireNonNull(fieldNames, "fieldNames");
        Objects.requireNonNull(expressions, "expressions");
        Preconditions.checkArgument(fieldNames.length == expressions.length, "pack requires one name per expression");
        try {
            return new BoundExpression(NativeBoundExpression.pack(fieldNames, nativePointers(expressions), nullable));
        } finally {
            Reference.reachabilityFence(expressions);
        }
    }

    /** Merge typed structs, applying the requested duplicate field policy. */
    public static BoundExpression merge(
            Expression.DuplicateHandling duplicateHandling, BoundExpression... expressions) {
        Objects.requireNonNull(duplicateHandling, "duplicateHandling");
        try {
            return new BoundExpression(NativeBoundExpression.merge(nativePointers(expressions), duplicateHandling.tag()));
        } finally {
            Reference.reachabilityFence(expressions);
        }
    }

    /** Merge typed structs, failing if any field name is duplicated. */
    public static BoundExpression merge(BoundExpression... expressions) {
        return merge(Expression.DuplicateHandling.ERROR, expressions);
    }

    public static BoundExpression literal(boolean value) {
        return new BoundExpression(NativeBoundExpression.literalBool(value, false));
    }

    public static BoundExpression literal(byte value) {
        return new BoundExpression(NativeBoundExpression.literalI8(value, false));
    }

    public static BoundExpression literal(short value) {
        return new BoundExpression(NativeBoundExpression.literalI16(value, false));
    }

    public static BoundExpression literal(int value) {
        return new BoundExpression(NativeBoundExpression.literalI32(value, false));
    }

    public static BoundExpression literal(long value) {
        return new BoundExpression(NativeBoundExpression.literalI64(value, false));
    }

    /** Create an unsigned 64 bit literal from the raw bits of a Java {@code long}. */
    public static BoundExpression literalUnsigned64(long bits) {
        return new BoundExpression(NativeBoundExpression.literalU64(bits));
    }

    public static BoundExpression literal(float value) {
        return new BoundExpression(NativeBoundExpression.literalF32(value, false));
    }

    public static BoundExpression literal(double value) {
        return new BoundExpression(NativeBoundExpression.literalF64(value, false));
    }

    public static BoundExpression literal(String value) {
        return new BoundExpression(NativeBoundExpression.literalString(value));
    }

    public static BoundExpression literal(byte[] value) {
        return new BoundExpression(NativeBoundExpression.literalBinary(value));
    }

    public static BoundExpression nullLiteral(Expression.DType dtype) {
        return new BoundExpression(NativeBoundExpression.literalNull(dtype.tag()));
    }

    public static BoundExpression nullLiteralBool() {
        return new BoundExpression(NativeBoundExpression.literalBool(false, true));
    }

    public static BoundExpression literalDate(long value, Expression.TimeUnit unit) {
        return new BoundExpression(NativeBoundExpression.literalDate(value, unit.tag(), false));
    }

    public static BoundExpression nullLiteralDate(Expression.TimeUnit unit) {
        return new BoundExpression(NativeBoundExpression.literalDate(0L, unit.tag(), true));
    }

    public static BoundExpression literalTimestamp(long value, Expression.TimeUnit unit, String timezone) {
        return new BoundExpression(NativeBoundExpression.literalTimestamp(value, unit.tag(), timezone, false));
    }

    public static BoundExpression nullLiteralTimestamp(Expression.TimeUnit unit, String timezone) {
        return new BoundExpression(NativeBoundExpression.literalTimestamp(0L, unit.tag(), timezone, true));
    }

    public static BoundExpression literalDecimal(BigInteger unscaled, int precision, int scale) {
        return new BoundExpression(
                NativeBoundExpression.literalDecimal(unscaled.toByteArray(), precision, scale, false));
    }

    public static BoundExpression nullLiteralDecimal(int precision, int scale) {
        return new BoundExpression(NativeBoundExpression.literalDecimal(new byte[] {0}, precision, scale, true));
    }

    /** Create a UUID literal from a Java UUID. */
    public static BoundExpression literal(UUID value) {
        Objects.requireNonNull(value, "value");
        return literalUuid(Expression.uuidToBigEndianBytes(value));
    }

    /** Create a UUID literal from its 16-byte big-endian representation. */
    public static BoundExpression literalUuid(byte[] bigEndianBytes) {
        Objects.requireNonNull(bigEndianBytes, "bigEndianBytes");
        Preconditions.checkArgument(
                bigEndianBytes.length == Expression.UUID_BYTE_LEN,
                "UUID literal must be exactly %s bytes, got %s",
                Expression.UUID_BYTE_LEN,
                bigEndianBytes.length);
        return new BoundExpression(NativeBoundExpression.literalUuid(bigEndianBytes, false));
    }

    /** Create a null UUID literal. */
    public static BoundExpression nullLiteralUuid() {
        return new BoundExpression(NativeBoundExpression.literalUuid(new byte[Expression.UUID_BYTE_LEN], true));
    }

    /** Construct an exact-typed binary operation without Vortex coercion. */
    public static BoundExpression binary(Expression.BinaryOp op, BoundExpression lhs, BoundExpression rhs) {
        try {
            return new BoundExpression(NativeBoundExpression.binary(op.code(), lhs.pointer, rhs.pointer));
        } finally {
            Reference.reachabilityFence(lhs);
            Reference.reachabilityFence(rhs);
        }
    }

    /** Explicitly cast a typed value to a signed 64-bit integer. */
    public static BoundExpression castToI64(BoundExpression child) {
        Objects.requireNonNull(child, "child");
        try {
            return new BoundExpression(NativeBoundExpression.castToI64(child.pointer));
        } finally {
            Reference.reachabilityFence(child);
        }
    }

    /** Construct a typed conjunction. */
    public static BoundExpression and(BoundExpression lhs, BoundExpression rhs) {
        return binary(Expression.BinaryOp.AND, lhs, rhs);
    }

    /** Construct a typed disjunction. */
    public static BoundExpression or(BoundExpression lhs, BoundExpression rhs) {
        return binary(Expression.BinaryOp.OR, lhs, rhs);
    }

    /** Test whether a typed value lies between two bounds, with optional strict comparisons. */
    public static BoundExpression between(
            BoundExpression value, BoundExpression lower, BoundExpression upper, boolean lowerStrict, boolean upperStrict) {
        try {
            return new BoundExpression(NativeBoundExpression.between(
                    value.pointer, lower.pointer, upper.pointer, lowerStrict, upperStrict));
        } finally {
            Reference.reachabilityFence(value);
            Reference.reachabilityFence(lower);
            Reference.reachabilityFence(upper);
        }
    }

    /** Negate a typed boolean expression. */
    public static BoundExpression not(BoundExpression child) {
        try {
            return new BoundExpression(NativeBoundExpression.not(child.pointer));
        } finally {
            Reference.reachabilityFence(child);
        }
    }

    /** Test whether a typed expression is null. */
    public static BoundExpression isNull(BoundExpression child) {
        try {
            return new BoundExpression(NativeBoundExpression.isNull(child.pointer));
        } finally {
            Reference.reachabilityFence(child);
        }
    }

    /** Test whether a typed expression is non-null. */
    public static BoundExpression isNotNull(BoundExpression child) {
        try {
            return new BoundExpression(NativeBoundExpression.isNotNull(child.pointer));
        } finally {
            Reference.reachabilityFence(child);
        }
    }

    /** Construct a typed SQL LIKE expression. */
    public static BoundExpression like(
            BoundExpression value, BoundExpression pattern, boolean negated, boolean caseInsensitive) {
        try {
            return new BoundExpression(
                    NativeBoundExpression.like(value.pointer, pattern.pointer, negated, caseInsensitive));
        } finally {
            Reference.reachabilityFence(value);
            Reference.reachabilityFence(pattern);
        }
    }

    private static long[] nativePointers(BoundExpression[] expressions) {
        return Arrays.stream(expressions)
                .map(expression -> Objects.requireNonNull(expression, "expression"))
                .mapToLong(BoundExpression::nativePointer)
                .toArray();
    }
}
