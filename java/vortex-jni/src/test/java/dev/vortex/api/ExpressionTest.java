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
}
