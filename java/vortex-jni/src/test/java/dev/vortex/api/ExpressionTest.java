// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.jni.NativeLoader;
import java.math.BigInteger;
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
    public void literalListComposesWithListContains() {
        Expression set = Expression.literalList(Expression.literal(1L), Expression.literal(2L));
        assertNotNull(Expression.listContains(set, Expression.column("id")));
        assertNotNull(Expression.in(Expression.column("id"), set));
        assertNotNull(Expression.notIn(Expression.column("id"), set));
    }

    @Test
    public void literalListUnifiesElementNullability() {
        // A null element makes the element type nullable rather than rejecting the set; the non-null elements are
        // cast up to it.
        assertNotNull(Expression.literalList(
                Expression.literal(1L), Expression.nullLiteral(Expression.DType.I64), Expression.literal(3L)));
    }

    @Test
    public void literalListRequiresAtLeastOneElement() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, Expression::literalList);
        assertTrue(
                exception.getMessage().contains("literalEmptyList"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void literalListRejectsMixedElementTypes() {
        RuntimeException exception = assertThrows(
                RuntimeException.class,
                () -> Expression.literalList(Expression.literal(1L), Expression.literal("two")));
        assertTrue(
                exception.getMessage().contains("must share a dtype"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void literalListRejectsNonLiteralElements() {
        RuntimeException exception = assertThrows(
                RuntimeException.class, () -> Expression.literalList(Expression.literal(1L), Expression.column("id")));
        assertTrue(
                exception.getMessage().contains("must themselves be literals"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void emptyAndNullListLiteralsAcceptEveryNullLiteralDType() {
        for (Expression.DType dtype : Expression.DType.values()) {
            assertNotNull(Expression.literalEmptyList(dtype), () -> "native side rejected empty list of " + dtype);
            assertNotNull(Expression.nullLiteralList(dtype), () -> "native side rejected null list of " + dtype);
        }
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
}
