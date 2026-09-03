// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Expression.BinaryOp;
import dev.vortex.api.Expression.DType;
import dev.vortex.api.Expression.DuplicateHandling;
import dev.vortex.api.Expression.TimeUnit;
import dev.vortex.jni.NativeExpression;
import dev.vortex.jni.NativeLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Parity tests for the byte tags {@link Expression} hands to the native side.
 *
 * <p>Four enums carry a tag that the Rust side switches on: {@link BinaryOp} against {@code parse_op},
 * {@link DuplicateHandling} against {@code parse_duplicate_handling}, {@link TimeUnit} against
 * {@code TimeUnit::try_from}, and {@link DType} against the table in {@code literalNull}. Three of the four say in
 * their javadoc that the values must match the Rust table, but nothing checked it. Drift compiles on both sides, and
 * the two failure modes are not equally loud: a tag past the end of a table reaches the {@code other =>} arm and
 * throws, while a tag that collides with a sibling decodes to the wrong operator or the wrong time unit and returns an
 * expression that reads valid.
 *
 * <p>So each table is pinned twice: the constants against the bytes Rust matches, and every constant against the native
 * call that consumes it. Both temporal types are exercised because between them they reject enough units to tell the
 * five {@code TimeUnit} tags apart — Timestamp takes everything but {@code DAYS}, Date only {@code DAYS} and
 * {@code MILLISECONDS} — and Date names the unit it rejected, so a swapped tag reports the wrong name here instead of
 * reading a value at the wrong scale.
 */
public final class ExpressionTagParityTest {
    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @Test
    public void binaryOpCodesMatchTheParseOpTable() {
        assertEquals(0, BinaryOp.EQ.code());
        assertEquals(1, BinaryOp.NOT_EQ.code());
        assertEquals(2, BinaryOp.GT.code());
        assertEquals(3, BinaryOp.GTE.code());
        assertEquals(4, BinaryOp.LT.code());
        assertEquals(5, BinaryOp.LTE.code());
        assertEquals(6, BinaryOp.AND.code());
        assertEquals(7, BinaryOp.OR.code());
        assertEquals(8, BinaryOp.ADD.code());
        assertEquals(9, BinaryOp.SUB.code());
        assertEquals(10, BinaryOp.MUL.code());
        assertEquals(11, BinaryOp.DIV.code());
        assertEquals(12, BinaryOp.values().length);
    }

    @Test
    public void everyBinaryOpIsAcceptedByTheNativeSide() {
        // parse_op bails on a code it does not know, so an operator Java has and Rust lacks fails here.
        for (BinaryOp op : BinaryOp.values()) {
            assertNotNull(
                    Expression.binary(op, Expression.column("a"), Expression.literal(1L)),
                    () -> "native side rejected " + op);
        }
    }

    @Test
    public void aBinaryOpCodePastTheTableIsRejectedByName() {
        Expression lhs = Expression.column("a");
        Expression rhs = Expression.literal(1L);
        RuntimeException exception = assertThrows(
                RuntimeException.class,
                () -> NativeExpression.binary(
                        (byte) BinaryOp.values().length, lhs.nativePointer(), rhs.nativePointer()));
        assertTrue(
                exception.getMessage().contains("unknown binary operator code: 12"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void duplicateHandlingTagsMatchTheRustTable() {
        assertEquals(0, DuplicateHandling.RIGHT_MOST.tag());
        assertEquals(1, DuplicateHandling.ERROR.tag());
        assertEquals(2, DuplicateHandling.values().length);
    }

    @Test
    public void everyDuplicateHandlingIsAcceptedByTheNativeSide() {
        for (DuplicateHandling handling : DuplicateHandling.values()) {
            assertNotNull(
                    Expression.merge(handling, Expression.column("a"), Expression.column("b")),
                    () -> "native side rejected " + handling);
        }
    }

    @Test
    public void aDuplicateHandlingTagPastTheTableIsRejectedByName() {
        RuntimeException exception = assertThrows(
                RuntimeException.class,
                () -> NativeExpression.merge(new long[0], (byte) DuplicateHandling.values().length));
        assertTrue(
                exception.getMessage().contains("unknown duplicate handling code: 2"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void timeUnitTagsMatchTheRustTable() {
        assertEquals(0, TimeUnit.NANOSECONDS.tag());
        assertEquals(1, TimeUnit.MICROSECONDS.tag());
        assertEquals(2, TimeUnit.MILLISECONDS.tag());
        assertEquals(3, TimeUnit.SECONDS.tag());
        assertEquals(4, TimeUnit.DAYS.tag());
        assertEquals(5, TimeUnit.values().length);
    }

    @Test
    public void timestampAcceptsEveryUnitExceptDays() {
        for (TimeUnit unit :
                new TimeUnit[] {TimeUnit.NANOSECONDS, TimeUnit.MICROSECONDS, TimeUnit.MILLISECONDS, TimeUnit.SECONDS}) {
            assertNotNull(Expression.literalTimestamp(0L, unit, "UTC"), unit::name);
            assertNotNull(Expression.nullLiteralTimestamp(unit, null), unit::name);
        }
        RuntimeException exception =
                assertThrows(RuntimeException.class, () -> Expression.literalTimestamp(0L, TimeUnit.DAYS, "UTC"));
        assertTrue(
                exception.getMessage().contains("Timestamp does not support Days time unit"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void dateAcceptsOnlyDaysAndMilliseconds() {
        assertNotNull(Expression.literalDate(0L, TimeUnit.DAYS));
        assertNotNull(Expression.literalDate(0L, TimeUnit.MILLISECONDS));
        assertNotNull(Expression.nullLiteralDate(TimeUnit.DAYS));
        assertDateRejects(TimeUnit.NANOSECONDS);
        assertDateRejects(TimeUnit.MICROSECONDS);
        assertDateRejects(TimeUnit.SECONDS);
    }

    @Test
    public void aTimeUnitTagPastTheTableIsRejected() {
        assertThrows(
                RuntimeException.class,
                () -> NativeExpression.literalTimestamp(0L, (byte) TimeUnit.values().length, "UTC", false));
    }

    @Test
    public void nullLiteralDTypeTagsMatchTheRustTable() {
        assertEquals(0, DType.BOOL.tag());
        assertEquals(1, DType.I8.tag());
        assertEquals(2, DType.I16.tag());
        assertEquals(3, DType.I32.tag());
        assertEquals(4, DType.I64.tag());
        assertEquals(5, DType.F32.tag());
        assertEquals(6, DType.F64.tag());
        assertEquals(7, DType.UTF8.tag());
        assertEquals(8, DType.BINARY.tag());
        assertEquals(9, DType.values().length);
    }

    @Test
    public void everyNullLiteralDTypeIsAcceptedByTheNativeSide() {
        for (DType dtype : DType.values()) {
            assertNotNull(Expression.nullLiteral(dtype), () -> "native side rejected " + dtype);
        }
    }

    @Test
    public void aDTypeTagPastTheTableIsRejectedByName() {
        RuntimeException exception =
                assertThrows(RuntimeException.class, () -> NativeExpression.literalNull((byte) DType.values().length));
        assertTrue(
                exception.getMessage().contains("unknown null dtype tag: 9"),
                () -> "unexpected message: " + exception.getMessage());
    }

    @Test
    public void theTagTablesAreReadIndependently() {
        // literalNull's javadoc says its tags intentionally do not overlap with parse_time_unit. They do
        // overlap numerically — the guarantee is that each tag is only ever read by its own parser, so the
        // byte 4 means I64 to literalNull and DAYS to literalDate.
        assertEquals(TimeUnit.DAYS.tag(), DType.I64.tag());
        assertNotNull(Expression.nullLiteral(DType.I64));
        assertNotNull(Expression.literalDate(0L, TimeUnit.DAYS));
    }

    private static void assertDateRejects(TimeUnit unit) {
        RuntimeException exception = assertThrows(RuntimeException.class, () -> Expression.literalDate(0L, unit));
        assertTrue(
                exception.getMessage().contains("Date type does not support time unit"),
                () -> "unexpected message for " + unit + ": " + exception.getMessage());
    }
}
