// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the value semantics of {@link VortexWriteSummary} and {@link VortexColumnStatistics}.
 *
 * <p>Both constructors are public only so {@code writer.rs} can call them through JNI, which means the boundary is
 * untyped: counts arrive as bare {@code jlong}s and bounds as {@code Object}. {@code exact_count_jlong} sends
 * {@code -1} when Vortex has no exact statistic, and the accessors turn that into an empty {@link OptionalLong} — a
 * caller reading the field directly would see a negative count instead. These tests pin that translation and the
 * defensive copy that keeps the statistics list immutable.
 */
public final class VortexWriteSummaryTest {

    @Test
    public void summaryReportsTheSizeAndRowCountItWasGiven() {
        VortexWriteSummary summary = new VortexWriteSummary(4096L, 10L, new VortexColumnStatistics[0]);

        assertEquals(4096L, summary.fileSize());
        assertEquals(10L, summary.rowCount());
        assertTrue(summary.columnStatistics().isEmpty());
    }

    @Test
    public void summaryCopiesTheStatisticsArray() {
        VortexColumnStatistics first = statistics(0, 1L, 0L);
        VortexColumnStatistics[] columns = {first};
        VortexWriteSummary summary = new VortexWriteSummary(1L, 1L, columns);
        columns[0] = statistics(1, 2L, 0L);

        assertEquals(1, summary.columnStatistics().size());
        assertSame(first, summary.columnStatistics().get(0));
    }

    @Test
    public void statisticsListRejectsModification() {
        VortexWriteSummary summary =
                new VortexWriteSummary(1L, 1L, new VortexColumnStatistics[] {statistics(0, 1L, 0L)});
        List<VortexColumnStatistics> columns = summary.columnStatistics();

        assertThrows(UnsupportedOperationException.class, columns::clear);
        assertThrows(UnsupportedOperationException.class, () -> columns.add(statistics(1, 1L, 0L)));
    }

    @Test
    public void statisticsListIsStableAcrossCalls() {
        VortexWriteSummary summary = new VortexWriteSummary(1L, 1L, new VortexColumnStatistics[0]);

        assertSame(summary.columnStatistics(), summary.columnStatistics());
    }

    @Test
    public void aNegativeCountMeansVortexComputedNone() {
        // `exact_count_jlong` returns -1 when the statistic is absent or inexact.
        VortexColumnStatistics absent = statistics(0, 5L, 5L, -1L, -1L, null, null);

        assertEquals(OptionalLong.empty(), absent.nullValueCount());
        assertEquals(OptionalLong.empty(), absent.nanValueCount());
    }

    @Test
    public void zeroIsAnExactCountAndNotASentinel() {
        VortexColumnStatistics none = statistics(0, 5L, 5L, 0L, 0L, null, null);

        assertEquals(OptionalLong.of(0L), none.nullValueCount());
        assertEquals(OptionalLong.of(0L), none.nanValueCount());
    }

    @Test
    public void exactCountsAreReportedAsGiven() {
        VortexColumnStatistics counted = statistics(2, 128L, 10L, 3L, 1L, null, null);

        assertEquals(2, counted.columnIndex());
        assertEquals(128L, counted.compressedSize());
        assertEquals(10L, counted.valueCount());
        assertEquals(OptionalLong.of(3L), counted.nullValueCount());
        assertEquals(OptionalLong.of(1L), counted.nanValueCount());
    }

    @Test
    public void oneCountCanBeExactWhileTheOtherIsAbsent() {
        // A non-floating-point column has a null count but no NaN count.
        VortexColumnStatistics integral = statistics(0, 64L, 10L, 2L, -1L, null, null);

        assertEquals(OptionalLong.of(2L), integral.nullValueCount());
        assertEquals(OptionalLong.empty(), integral.nanValueCount());
    }

    @Test
    public void absentBoundsAreEmpty() {
        VortexColumnStatistics unbounded = statistics(0, 1L, 1L, 0L, -1L, null, null);

        assertTrue(unbounded.lowerBound().isEmpty());
        assertTrue(unbounded.upperBound().isEmpty());
    }

    @Test
    public void boundsAreReturnedWithoutConversion() {
        // `scalar_to_java` picks the Java type per Arrow scalar; the accessor must not narrow or box further.
        for (Object[] pair : new Object[][] {
            {Integer.valueOf(1), Integer.valueOf(9)},
            {Long.valueOf(1L), Long.valueOf(9L)},
            {BigInteger.ONE, BigInteger.TEN},
            {Float.valueOf(1.5f), Float.valueOf(9.5f)},
            {Double.valueOf(1.5d), Double.valueOf(9.5d)},
            {new BigDecimal("1.50"), new BigDecimal("9.50")},
            {"a", "z"}
        }) {
            VortexColumnStatistics bounded = statistics(0, 1L, 1L, 0L, -1L, pair[0], pair[1]);

            assertSame(pair[0], bounded.lowerBound().orElseThrow());
            assertSame(pair[1], bounded.upperBound().orElseThrow());
        }
    }

    @Test
    public void binaryBoundsAreSharedNotCopied() {
        // Documented as `byte[]` and handed straight back, so a reader that mutates the array changes what
        // every later reader sees. Callers must copy before writing.
        VortexColumnStatistics bounded = statistics(0, 1L, 1L, 0L, -1L, new byte[] {1, 2}, new byte[] {3, 4});

        byte[] lower = (byte[]) bounded.lowerBound().orElseThrow();
        assertArrayEquals(new byte[] {1, 2}, lower);
        lower[0] = 9;

        assertSame(lower, bounded.lowerBound().orElseThrow());
        assertArrayEquals(new byte[] {9, 2}, (byte[]) bounded.lowerBound().orElseThrow());
    }

    private static VortexColumnStatistics statistics(int columnIndex, long compressedSize, long valueCount) {
        return statistics(columnIndex, compressedSize, valueCount, 0L, -1L, null, null);
    }

    private static VortexColumnStatistics statistics(
            int columnIndex,
            long compressedSize,
            long valueCount,
            long nullValueCount,
            long nanValueCount,
            Object lowerBound,
            Object upperBound) {
        return new VortexColumnStatistics(
                columnIndex, compressedSize, valueCount, nullValueCount, nanValueCount, lowerBound, upperBound);
    }
}
