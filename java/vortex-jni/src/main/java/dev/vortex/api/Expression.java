// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeExpression;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.UUID;

/**
 * A Vortex expression node backed by a native pointer.
 *
 * <p>Expressions are composed via the static factories ({@link #root()}, {@link #getItem(String, Expression)}, etc.).
 * Each returned {@code Expression} owns its native pointer; the pointer is released automatically when the
 * {@code Expression} is no longer reachable. Passing an expression as an input to a builder does <em>not</em> transfer
 * ownership — the resulting expression is an independent copy on the native side.
 */
public final class Expression {
    /** Number of bytes in a UUID's big-endian representation. */
    private static final int UUID_BYTE_LEN = 16;

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

    /**
     * The row-index expression. When evaluated as part of a Vortex scan it yields, as a non-nullable {@code u64}, each
     * row's index in the file <em>before</em> filtering: the index is assigned to the unfiltered rows, so filtered-out
     * rows leave gaps and the surviving rows keep their original positions rather than being renumbered. It cannot be
     * evaluated outside of a scan.
     */
    public static Expression rowIdx() {
        return new Expression(NativeExpression.rowIdx());
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

    /** Creates an expression that packs values into a struct with named fields. */
    public static Expression pack(String[] fieldNames, Expression[] expressions, boolean nullable) {
        return new Expression(NativeExpression.pack(fieldNames, nativePointers(expressions), nullable));
    }

    /**
     * Merge struct expressions into a single struct, combining their fields in order.
     *
     * <p>Every input must evaluate to a non-nullable struct. When the same field name appears in more than one input,
     * {@code duplicateHandling} decides the result. Fields are <em>not</em> merged recursively — a later field replaces
     * an earlier one with the same name. Merging zero expressions yields an empty struct.
     */
    public static Expression merge(DuplicateHandling duplicateHandling, Expression... expressions) {
        return new Expression(NativeExpression.merge(nativePointers(expressions), duplicateHandling.tag()));
    }

    /**
     * Merge struct expressions, failing if any field name is duplicated. Equivalent to {@link #merge(DuplicateHandling,
     * Expression...)} with {@link DuplicateHandling#ERROR}.
     */
    public static Expression merge(Expression... expressions) {
        return merge(DuplicateHandling.ERROR, expressions);
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

    /**
     * Apply a spatial function to native geometry operands, each a column or a geometry literal (see
     * {@link #literalGeometry(byte[])}). The number of operands must match the function's arity.
     */
    public static Expression spatial(SpatialFunction function, Expression... operands) {
        Preconditions.checkArgument(
                operands.length == function.arity(),
                "%s takes %s operands, got %s",
                function,
                function.arity(),
                operands.length);
        return new Expression(NativeExpression.spatial(function.code(), nativePointers(operands)));
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

    /** Create an unsigned 8-bit integer literal. {@code value} must be in {@code [0, 255]}. */
    public static Expression literalU8(int value) {
        Preconditions.checkArgument(value >= 0 && value <= 0xFF, "u8 literal out of range: %s", value);
        return new Expression(NativeExpression.literalU8((byte) value, false));
    }

    /** Create an unsigned 16-bit integer literal. {@code value} must be in {@code [0, 65535]}. */
    public static Expression literalU16(int value) {
        Preconditions.checkArgument(value >= 0 && value <= 0xFFFF, "u16 literal out of range: %s", value);
        return new Expression(NativeExpression.literalU16((short) value, false));
    }

    /** Create an unsigned 32-bit integer literal. {@code value} must be in {@code [0, 2^32 - 1]}. */
    public static Expression literalU32(long value) {
        Preconditions.checkArgument(value >= 0 && value <= 0xFFFF_FFFFL, "u32 literal out of range: %s", value);
        return new Expression(NativeExpression.literalU32((int) value, false));
    }

    /**
     * Create an unsigned 64-bit integer literal from its bit pattern, so values above {@link Long#MAX_VALUE} are passed
     * as negative longs (for example via {@link Long#parseUnsignedLong(String)}).
     */
    public static Expression literalU64(long bits) {
        return new Expression(NativeExpression.literalU64(bits, false));
    }

    /** Create an unsigned 64-bit integer literal. {@code value} must be in {@code [0, 2^64 - 1]}. */
    public static Expression literalU64(BigInteger value) {
        Preconditions.checkArgument(value != null, "use nullLiteral(DType.U64) for a null u64 literal");
        Preconditions.checkArgument(
                value.signum() >= 0 && value.bitLength() <= Long.SIZE, "u64 literal out of range: %s", value);
        return literalU64(value.longValue());
    }

    /** Create a half-precision float literal, rounding {@code value} to the nearest representable half. */
    public static Expression literalF16(float value) {
        return new Expression(NativeExpression.literalF16(value, false));
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

    /**
     * Create a Time (time-of-day) literal. The {@code value} is the number of {@code unit} units since midnight.
     *
     * @param unit any unit except {@link TimeUnit#DAYS}. {@link TimeUnit#SECONDS} and {@link TimeUnit#MILLISECONDS}
     *     values must fit in an {@code int}.
     */
    public static Expression literalTime(long value, TimeUnit unit) {
        return new Expression(NativeExpression.literalTime(value, unit.tag(), false));
    }

    /** Null Time literal. See {@link #literalTime(long, TimeUnit)} for the {@code unit} constraints. */
    public static Expression nullLiteralTime(TimeUnit unit) {
        return new Expression(NativeExpression.literalTime(0L, unit.tag(), true));
    }

    /**
     * Create a geometry literal from its OGC Well-Known Binary (WKB) encoding, for use with Vortex's spatial functions
     * and predicate pushdown over geometry columns.
     *
     * <p>The value is decoded into the native Vortex geometry type matching its kind: Point, LineString, Polygon,
     * MultiPoint, MultiLineString or MultiPolygon, in XY with no coordinate reference system. Geometry collections and
     * malformed WKB are rejected.
     */
    public static Expression literalGeometry(byte[] wkb) {
        Preconditions.checkArgument(wkb != null, "geometry literal WKB must not be null");
        return new Expression(NativeExpression.literalGeometry(wkb));
    }

    /**
     * Create a UUID literal, enabling predicate pushdown over UUID columns. The value is stored as its 16-byte
     * big-endian (network order) representation, matching Vortex's UUID extension type and Arrow's canonical UUID type.
     */
    public static Expression literal(UUID value) {
        Preconditions.checkArgument(value != null, "use nullLiteralUuid() for a null UUID literal");
        return literalUuid(uuidToBigEndianBytes(value));
    }

    /**
     * Create a UUID literal from its 16-byte big-endian (network order) representation, for example the bytes of
     * Arrow's canonical UUID type or a {@link UUID} serialized most-significant-bits first.
     *
     * @param bigEndianBytes exactly 16 bytes; use {@link #nullLiteralUuid()} for a null literal
     */
    public static Expression literalUuid(byte[] bigEndianBytes) {
        Preconditions.checkArgument(bigEndianBytes != null, "use nullLiteralUuid() for a null UUID literal");
        Preconditions.checkArgument(
                bigEndianBytes.length == UUID_BYTE_LEN,
                "UUID literal must be exactly %s bytes, got %s",
                UUID_BYTE_LEN,
                bigEndianBytes.length);
        return new Expression(NativeExpression.literalUuid(bigEndianBytes, false));
    }

    /** Create a null UUID literal. */
    public static Expression nullLiteralUuid() {
        return new Expression(NativeExpression.literalUuid(new byte[UUID_BYTE_LEN], true));
    }

    private static byte[] uuidToBigEndianBytes(UUID value) {
        return ByteBuffer.allocate(UUID_BYTE_LEN)
                .order(ByteOrder.BIG_ENDIAN)
                .putLong(value.getMostSignificantBits())
                .putLong(value.getLeastSignificantBits())
                .array();
    }

    /** Create a typed null literal of the given primitive {@link DType}. */
    public static Expression nullLiteral(DType dtype) {
        return new Expression(NativeExpression.literalNull(dtype.tag()));
    }

    /**
     * Create a list literal.
     *
     * <p>Composite literals are assembled from other literal expressions. The element dtype is the dtype of the
     * {@code elementType} literal (typically a typed null such as {@code nullLiteral(DType.I32)}) with nullability
     * {@code elementsNullable}, and every element, which must itself be a literal, is cast to it.
     */
    public static Expression literalList(Expression elementType, boolean elementsNullable, Expression... elements) {
        return new Expression(NativeExpression.literalList(
                nativePointers(elements), elementType.nativePointer(), elementsNullable, false));
    }

    /** Null list literal. See {@link #literalList(Expression, boolean, Expression...)} for the element dtype. */
    public static Expression nullLiteralList(Expression elementType, boolean elementsNullable) {
        return new Expression(
                NativeExpression.literalList(new long[0], elementType.nativePointer(), elementsNullable, true));
    }

    /**
     * Create a fixed-size list literal whose size is the number of {@code elements}. The element dtype follows
     * {@link #literalList(Expression, boolean, Expression...)}.
     */
    public static Expression literalFixedSizeList(
            Expression elementType, boolean elementsNullable, Expression... elements) {
        return new Expression(NativeExpression.literalFixedSizeList(
                nativePointers(elements), elementType.nativePointer(), elementsNullable, elements.length, false));
    }

    /** Null fixed-size list literal of {@code size} elements. */
    public static Expression nullLiteralFixedSizeList(Expression elementType, boolean elementsNullable, int size) {
        Preconditions.checkArgument(size >= 0, "fixed-size list size must not be negative: %s", size);
        return new Expression(NativeExpression.literalFixedSizeList(
                new long[0], elementType.nativePointer(), elementsNullable, size, true));
    }

    /** Create a struct literal. Each field must be a literal expression; its dtype becomes the field's dtype. */
    public static Expression literalStruct(String[] fieldNames, Expression[] fields) {
        Preconditions.checkArgument(
                fieldNames.length == fields.length,
                "struct literal has %s field names but %s fields",
                fieldNames.length,
                fields.length);
        return new Expression(NativeExpression.literalStruct(fieldNames, nativePointers(fields), false));
    }

    /**
     * Create a null struct literal. {@code fieldTypes} are literals (typically typed nulls) whose dtypes become the
     * struct's field dtypes.
     */
    public static Expression nullLiteralStruct(String[] fieldNames, Expression[] fieldTypes) {
        Preconditions.checkArgument(
                fieldNames.length == fieldTypes.length,
                "struct literal has %s field names but %s field types",
                fieldNames.length,
                fieldTypes.length);
        return new Expression(NativeExpression.literalStruct(fieldNames, nativePointers(fieldTypes), true));
    }

    /**
     * Create a map literal from parallel arrays of literal keys and values.
     *
     * <p>Keys are cast to the non-nullable dtype of the {@code keyType} literal, and values to the dtype of the
     * {@code valueType} literal with nullability {@code valuesNullable}. Keys are not asserted to be sorted.
     */
    public static Expression literalMap(
            Expression keyType, Expression valueType, boolean valuesNullable, Expression[] keys, Expression[] values) {
        Preconditions.checkArgument(
                keys.length == values.length, "map literal has %s keys but %s values", keys.length, values.length);
        return new Expression(NativeExpression.literalMap(
                nativePointers(keys),
                nativePointers(values),
                keyType.nativePointer(),
                valueType.nativePointer(),
                valuesNullable,
                false));
    }

    /** Null map literal. See {@link #literalMap} for how the key and value dtypes are derived. */
    public static Expression nullLiteralMap(Expression keyType, Expression valueType, boolean valuesNullable) {
        return new Expression(NativeExpression.literalMap(
                new long[0], new long[0], keyType.nativePointer(), valueType.nativePointer(), valuesNullable, true));
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

    /** Spatial functions over native geometries; codes must match the Rust {@code spatial} table. */
    public enum SpatialFunction {
        /** Planar area of each geometry; zero for points and line strings. */
        AREA((byte) 0, 1),
        /** Collect each list of homogeneous geometries into the matching multi-geometry. */
        COLLECT((byte) 1, 1),
        /** Whether the second geometry lies completely inside the first. */
        CONTAINS((byte) 2, 2),
        /** Convex hull of each multipoint, as a polygon. */
        CONVEX_HULL((byte) 3, 1),
        /** Planar (Euclidean) distance between two geometries. */
        DISTANCE((byte) 4, 2),
        /** Axis-aligned bounding box of each geometry. */
        ENVELOPE((byte) 5, 1),
        /** Whether two geometries intersect; boundary contact counts. */
        INTERSECTS((byte) 6, 2),
        /** Planar length of each lineal geometry. */
        LENGTH((byte) 7, 1),
        /** Line string from two points. */
        MAKE_LINE((byte) 8, 2);

        private final byte code;
        private final int arity;

        SpatialFunction(byte code, int arity) {
            this.code = code;
            this.arity = arity;
        }

        public byte code() {
            return code;
        }

        public int arity() {
            return arity;
        }
    }

    /**
     * Strategy for resolving duplicate field names in {@link #merge(DuplicateHandling, Expression...)}. Tag values must
     * match the Rust {@code parse_duplicate_handling} table.
     */
    public enum DuplicateHandling {
        /** When two structs share a field name, keep the value from the right-most (later) struct. */
        RIGHT_MOST((byte) 0),
        /** When two structs share a field name, fail with an error. */
        ERROR((byte) 1);

        private final byte tag;

        DuplicateHandling(byte tag) {
            this.tag = tag;
        }

        public byte tag() {
            return tag;
        }
    }

    /** Time units for Date/Time/Timestamp literals. Tag values must match the Rust {@code parse_time_unit} table. */
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
        BINARY((byte) 8),
        U8((byte) 9),
        U16((byte) 10),
        U32((byte) 11),
        U64((byte) 12),
        F16((byte) 13);

        private final byte tag;

        DType(byte tag) {
            this.tag = tag;
        }

        public byte tag() {
            return tag;
        }
    }
}
