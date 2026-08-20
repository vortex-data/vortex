// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.util.Optional;
import java.util.OptionalLong;

/** Statistics and physical size reported for one top-level column after a Vortex write. */
public final class VortexColumnStatistics {
    private final int columnIndex;
    private final long compressedSize;
    private final long valueCount;
    private final long nullValueCount;
    private final long nanValueCount;
    private final Object lowerBound;
    private final Object upperBound;

    /**
     * Construct native write statistics.
     *
     * <p>This constructor is public so the JNI implementation can instantiate the value without reflective access.
     * Applications should obtain instances from {@link VortexWriteSummary#columnStatistics()}.
     */
    public VortexColumnStatistics(
            int columnIndex,
            long compressedSize,
            long valueCount,
            long nullValueCount,
            long nanValueCount,
            Object lowerBound,
            Object upperBound) {
        this.columnIndex = columnIndex;
        this.compressedSize = compressedSize;
        this.valueCount = valueCount;
        this.nullValueCount = nullValueCount;
        this.nanValueCount = nanValueCount;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    /** Zero-based position of this column in the writer's Arrow schema. */
    public int columnIndex() {
        return columnIndex;
    }

    /** Compressed bytes referenced by this column's physical layout. */
    public long compressedSize() {
        return compressedSize;
    }

    /** Number of values in this top-level column. */
    public long valueCount() {
        return valueCount;
    }

    /** Exact null count, or empty when Vortex did not compute one for this data type. */
    public OptionalLong nullValueCount() {
        return nullValueCount >= 0 ? OptionalLong.of(nullValueCount) : OptionalLong.empty();
    }

    /** Exact NaN count, or empty for non-floating-point columns. */
    public OptionalLong nanValueCount() {
        return nanValueCount >= 0 ? OptionalLong.of(nanValueCount) : OptionalLong.empty();
    }

    /**
     * Lower bound represented using the corresponding Arrow scalar's Java type.
     *
     * <p>Integers use {@link Integer}, {@link Long}, or {@link java.math.BigInteger}; floating-point values use
     * {@link Float} or {@link Double}; decimal values use {@link java.math.BigDecimal}; strings use {@link String}; and
     * binary values use {@code byte[]}.
     */
    public Optional<Object> lowerBound() {
        return Optional.ofNullable(lowerBound);
    }

    /** Upper bound represented using the corresponding Arrow scalar's Java type. */
    public Optional<Object> upperBound() {
        return Optional.ofNullable(upperBound);
    }
}
