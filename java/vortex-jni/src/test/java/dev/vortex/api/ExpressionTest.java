// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.jni.NativeLoader;
import java.math.BigInteger;
import java.util.UUID;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public final class ExpressionTest {
    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @Test
    public void rowIdxBuildsAndComposes() {
        assertNotNull(BoundExpression.binary(
                Expression.BinaryOp.LT, BoundExpression.rowIdx(), BoundExpression.literalUnsigned64(5L)));
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
                new Expression[] {Expression.column("a"), Expression.literal(5L), Expression.literal(6L)},
                true));
    }

    @Test
    public void boundPackComposesWithRowIdx() {
        assertNotNull(BoundExpression.pack(
                new String[] {"row_idx", "value"},
                new BoundExpression[] {BoundExpression.rowIdx(), BoundExpression.literal(5L)},
                false));
        assertThrows(
                IllegalArgumentException.class,
                () -> BoundExpression.pack(new String[] {"row_idx"}, new BoundExpression[] {}, false));
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
    public void boundMergeComposes() {
        BoundExpression first = BoundExpression.pack(
                new String[] {"x"}, new BoundExpression[] {BoundExpression.literal(1L)}, false);
        BoundExpression second = BoundExpression.pack(
                new String[] {"x"}, new BoundExpression[] {BoundExpression.literal(2L)}, false);

        assertNotNull(BoundExpression.merge());
        assertThrows(RuntimeException.class, () -> BoundExpression.merge(first, second));
        assertNotNull(BoundExpression.merge(Expression.DuplicateHandling.RIGHT_MOST, first, second));
    }

    @Test
    public void boundBetweenAndUuidLiteralsCompose() {
        assertNotNull(BoundExpression.between(
                BoundExpression.rowIdx(),
                BoundExpression.literalUnsigned64(1L),
                BoundExpression.literalUnsigned64(5L),
                false,
                true));
        assertThrows(
                RuntimeException.class,
                () -> BoundExpression.between(
                        BoundExpression.rowIdx(),
                        BoundExpression.literal(1L),
                        BoundExpression.literal(5L),
                        false,
                        false));

        assertNotNull(BoundExpression.literal(UUID.fromString("123e4567-e89b-12d3-a456-426614174000")));
        assertNotNull(BoundExpression.nullLiteralUuid());
        assertThrows(IllegalArgumentException.class, () -> BoundExpression.literalUuid(new byte[15]));
    }

    @Test
    public void boundIntegerCastComposes() {
        assertNotNull(BoundExpression.castToI64(BoundExpression.literal(7)));
    }
}
