// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.util.Optional;
import java.util.OptionalLong;
import org.immutables.value.Value;

/**
 * Scan configuration passed to {@link DataSource#scan(ScanOptions)}.
 *
 * <p>All fields are optional. A call to {@link #of()} returns a default that reads every
 * row and column.
 */
@Value.Immutable
public interface ScanOptions {

    /** Projection expression. If empty, all columns are returned. */
    Optional<Expression> projection();

    /** Filter expression applied before returning rows. */
    Optional<Expression> filter();

    /** Inclusive start of the row range to read. */
    OptionalLong rowRangeBegin();

    /** Exclusive end of the row range to read. */
    OptionalLong rowRangeEnd();

    /**
     * Sorted ascending row indices that should be included in (or excluded from) the scan,
     * depending on {@link #selectionMode()}.
     */
    Optional<long[]> selectionIndices();

    /** Meaning of {@link #selectionIndices()}. */
    @Value.Default
    default SelectionMode selectionMode() {
        return SelectionMode.INCLUDE_ALL;
    }

    /** Maximum row count to return. Absent means "no limit". */
    OptionalLong limit();

    /** If {@code true}, the scan preserves the original row order across partitions. */
    @Value.Default
    default boolean ordered() {
        return false;
    }

    static ScanOptions of() {
        return ImmutableScanOptions.builder().build();
    }

    static ImmutableScanOptions.Builder builder() {
        return ImmutableScanOptions.builder();
    }

    /** How to interpret {@link #selectionIndices()}. */
    enum SelectionMode {
        /** Ignore {@link #selectionIndices()}. */
        INCLUDE_ALL((byte) 0),
        /** Return only rows at the indices. */
        INCLUDE((byte) 1),
        /** Return rows except those at the indices. */
        EXCLUDE((byte) 2);

        private final byte code;

        SelectionMode(byte code) {
            this.code = code;
        }

        public byte code() {
            return code;
        }
    }
}
