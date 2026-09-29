// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.jni.NativeLoader;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public final class ExpressionTest {
    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @Test
    public void rowIdxBuildsAndComposes() {
        assertNotNull(Expression.rowIdx());
        // Mirrors `gt(row_idx(), lit(...))` on the Rust side: the row-index expression
        // composes like any other.
        assertNotNull(Expression.binary(Expression.BinaryOp.LT, Expression.rowIdx(), Expression.literal(5L)));
    }

    @Test
    public void literalDecimalRejectsValuesLargerThan32Bytes() {
        BigInteger tooLarge = BigInteger.ONE.shiftLeft(256);
        assertEquals(33, tooLarge.toByteArray().length);

        RuntimeException exception =
                assertThrows(RuntimeException.class, () -> Expression.literalDecimal(tooLarge, 76, 0));
        assertTrue(exception.getMessage().contains("Decimal value must fit with 32 bytes"));
    }

    @Test
    public void packComposes() {
        assertNotNull(Expression.pack(
                new String[] {"x", "y", "z"},
                new Expression[] {Expression.column("a"), Expression.literal(5L), Expression.rowIdx()},
                true));
    }

    @Test
    public void mergeComposes() {
        // Default duplicate handling (ERROR).
        assertNotNull(Expression.merge(Expression.column("a"), Expression.column("b")));
        // Explicit duplicate handling.
        assertNotNull(Expression.merge(
                Expression.DuplicateHandling.RIGHT_MOST, Expression.column("a"), Expression.column("b")));
        // Merging zero expressions is valid and yields an empty struct.
        assertNotNull(Expression.merge());
    }

    @Test
    public void literalTimeAcceptsEveryUnitExceptDays() {
        for (Expression.TimeUnit unit : new Expression.TimeUnit[] {
            Expression.TimeUnit.NANOSECONDS,
            Expression.TimeUnit.MICROSECONDS,
            Expression.TimeUnit.MILLISECONDS,
            Expression.TimeUnit.SECONDS
        }) {
            assertNotNull(Expression.literalTime(3_600L, unit), unit::name);
            assertNotNull(Expression.nullLiteralTime(unit), unit::name);
        }
        RuntimeException exception =
                assertThrows(RuntimeException.class, () -> Expression.literalTime(0L, Expression.TimeUnit.DAYS));
        assertTrue(
                exception.getMessage().contains("Time type does not support time unit"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void literalTimeRejectsSecondsOutsideI32() {
        RuntimeException exception = assertThrows(
                RuntimeException.class,
                () -> Expression.literalTime((long) Integer.MAX_VALUE + 1, Expression.TimeUnit.SECONDS));
        assertTrue(
                exception.getMessage().contains("does not fit in i32"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void literalGeometryBuildsFromWkbPoint() {
        Expression point = Expression.literalGeometry(wkbPoint(1.0, 2.0));
        assertNotNull(Expression.binary(Expression.BinaryOp.EQ, Expression.column("geom"), point));
    }

    @Test
    public void spatialFunctionsComposeWithGeometryLiterals() {
        Expression point = Expression.literalGeometry(wkbPoint(1.0, 2.0));
        Expression intersects =
                Expression.spatial(Expression.SpatialFunction.INTERSECTS, Expression.column("geom"), point);
        Expression distance = Expression.spatial(Expression.SpatialFunction.DISTANCE, Expression.column("geom"), point);
        assertNotNull(Expression.and(
                intersects, Expression.binary(Expression.BinaryOp.LT, distance, Expression.literal(5.0))));
    }

    @Test
    public void spatialRejectsTheWrongArity() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Expression.spatial(
                        Expression.SpatialFunction.AREA, Expression.column("a"), Expression.column("b")));
    }

    @Test
    public void literalGeometryRejectsMalformedWkb() {
        assertThrows(RuntimeException.class, () -> Expression.literalGeometry(new byte[] {1, 2, 3}));
    }

    @Test
    public void unsignedLiteralsCoverTheirFullRange() {
        assertNotNull(Expression.literalU8(0));
        assertNotNull(Expression.literalU8(255));
        assertNotNull(Expression.literalU16(65_535));
        assertNotNull(Expression.literalU32(0xFFFF_FFFFL));
        assertNotNull(Expression.literalU64(Long.parseUnsignedLong("18446744073709551615")));
        assertNotNull(Expression.literalU64(BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE)));
    }

    @Test
    public void unsignedLiteralsRejectOutOfRangeValues() {
        assertThrows(IllegalArgumentException.class, () -> Expression.literalU8(256));
        assertThrows(IllegalArgumentException.class, () -> Expression.literalU8(-1));
        assertThrows(IllegalArgumentException.class, () -> Expression.literalU16(65_536));
        assertThrows(IllegalArgumentException.class, () -> Expression.literalU32(1L << 32));
        assertThrows(IllegalArgumentException.class, () -> Expression.literalU64(BigInteger.ONE.shiftLeft(64)));
        assertThrows(IllegalArgumentException.class, () -> Expression.literalU64(BigInteger.valueOf(-1)));
    }

    @Test
    public void literalF16Builds() {
        assertNotNull(Expression.binary(Expression.BinaryOp.LT, Expression.column("h"), Expression.literalF16(1.5f)));
    }

    @Test
    public void listLiteralsCastElementsToTheElementType() {
        Expression i32 = Expression.nullLiteral(Expression.DType.I32);
        // Long elements are cast down to the i32 element type.
        assertNotNull(Expression.literalList(i32, false, Expression.literal(1L), Expression.literal(2L)));
        assertNotNull(
                Expression.literalList(i32, true, Expression.literal(1), Expression.nullLiteral(Expression.DType.I32)));
        assertNotNull(Expression.literalList(i32, false));
        assertNotNull(Expression.nullLiteralList(i32, false));
    }

    @Test
    public void listLiteralsRejectNonLiteralElements() {
        Expression i32 = Expression.nullLiteral(Expression.DType.I32);
        RuntimeException exception =
                assertThrows(RuntimeException.class, () -> Expression.literalList(i32, false, Expression.column("a")));
        assertTrue(
                exception.getMessage().contains("must be a literal expression"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void fixedSizeListLiteralsBuild() {
        Expression u8 = Expression.nullLiteral(Expression.DType.U8);
        assertNotNull(Expression.literalFixedSizeList(u8, false, Expression.literalU8(1), Expression.literalU8(2)));
        assertNotNull(Expression.nullLiteralFixedSizeList(u8, false, 2));
    }

    @Test
    public void structLiteralsBuild() {
        String[] names = {"a", "b"};
        assertNotNull(
                Expression.literalStruct(names, new Expression[] {Expression.literal(1), Expression.literal("x")}));
        assertNotNull(Expression.nullLiteralStruct(names, new Expression[] {
            Expression.nullLiteral(Expression.DType.I32), Expression.nullLiteral(Expression.DType.UTF8)
        }));
    }

    @Test
    public void mapLiteralsBuild() {
        Expression keyType = Expression.nullLiteral(Expression.DType.UTF8);
        Expression valueType = Expression.nullLiteral(Expression.DType.I64);
        assertNotNull(Expression.literalMap(
                keyType,
                valueType,
                true,
                new Expression[] {Expression.literal("a"), Expression.literal("b")},
                new Expression[] {Expression.literal(1L), Expression.nullLiteral(Expression.DType.I64)}));
        assertNotNull(Expression.nullLiteralMap(keyType, valueType, false));
    }

    /** Little-endian WKB for {@code POINT(x y)}. */
    private static byte[] wkbPoint(double x, double y) {
        return ByteBuffer.allocate(21)
                .order(ByteOrder.LITTLE_ENDIAN)
                .put((byte) 1)
                .putInt(1)
                .putDouble(x)
                .putDouble(y)
                .array();
    }
}
