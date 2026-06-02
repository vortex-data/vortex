// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import org.immutables.value.Value;
import org.roaringbitmap.longlong.Roaring64NavigableMap;

/**
 * Scan configuration passed to {@link DataSource#scan(ScanOptions)}.
 *
 * <p>All fields are optional. A call to {@link #of()} returns a default that reads every row and column.
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
     * Sorted ascending, unique row indices that should be included in (or excluded from) the scan, depending on
     * {@link #selectionMode()}.
     */
    Optional<long[]> selectionIndices();

    /** Portable serialized {@link Roaring64NavigableMap} row selection. */
    Optional<byte[]> selectionRoaringBitmap();

    /** Meaning of the row selection payload. */
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

    @Value.Check
    default void validateSelectionPayload() {
        boolean hasIndices = selectionIndices().isPresent();
        boolean hasRoaringBitmap = selectionRoaringBitmap().isPresent();
        if (hasIndices && hasRoaringBitmap) {
            throw new IllegalArgumentException("row selection must use either indices or roaring bitmap, not both");
        }
        if (hasIndices) {
            validateSelectionIndices(selectionIndices().orElseThrow());
        }

        switch (selectionMode()) {
            case INCLUDE_ALL -> {
                if (hasIndices || hasRoaringBitmap) {
                    throw new IllegalArgumentException("row selection payload requires a selection mode");
                }
            }
            case INCLUDE, EXCLUDE -> {
                if (!hasIndices) {
                    throw new IllegalArgumentException("selection indices are required for index selection modes");
                }
            }
            case INCLUDE_ROARING, EXCLUDE_ROARING -> {
                if (!hasRoaringBitmap) {
                    throw new IllegalArgumentException(
                            "selection roaring bitmap is required for roaring selection modes");
                }
                if (selectionRoaringBitmap().orElseThrow().length == 0) {
                    throw new IllegalArgumentException("selection roaring bitmap must not be empty");
                }
            }
        }
    }

    private static void validateSelectionIndices(long[] selectionIndices) {
        long previous = -1L;
        for (int i = 0; i < selectionIndices.length; i++) {
            long index = selectionIndices[i];
            if (index < 0) {
                throw new IllegalArgumentException("selection indices must be non-negative");
            }
            if (i > 0 && index <= previous) {
                throw new IllegalArgumentException("selection indices must be sorted ascending and unique");
            }
            previous = index;
        }
    }

    static ScanOptions of() {
        return ImmutableScanOptions.builder().build();
    }

    static ImmutableScanOptions.Builder builder() {
        return ImmutableScanOptions.builder();
    }

    /** Scan only the rows at the given sorted ascending, unique row indices. */
    static ScanOptions includeRows(long... rowIndices) {
        return builder()
                .selectionIndices(rowIndices.clone())
                .selectionMode(SelectionMode.INCLUDE)
                .build();
    }

    /** Scan all rows except the given sorted ascending, unique row indices. */
    static ScanOptions excludeRows(long... rowIndices) {
        return builder()
                .selectionIndices(rowIndices.clone())
                .selectionMode(SelectionMode.EXCLUDE)
                .build();
    }

    /** Scan only the rows in the given Roaring bitmap. */
    static ScanOptions includeRows(Roaring64NavigableMap rowSelection) {
        return builder()
                .selectionRoaringBitmap(serializeRoaringBitmap(rowSelection))
                .selectionMode(SelectionMode.INCLUDE_ROARING)
                .build();
    }

    /** Scan all rows except the rows in the given Roaring bitmap. */
    static ScanOptions excludeRows(Roaring64NavigableMap rowSelection) {
        return builder()
                .selectionRoaringBitmap(serializeRoaringBitmap(rowSelection))
                .selectionMode(SelectionMode.EXCLUDE_ROARING)
                .build();
    }

    private static byte[] serializeRoaringBitmap(Roaring64NavigableMap rowSelection) {
        Objects.requireNonNull(rowSelection, "rowSelection");
        try (ByteArrayOutputStream output = new ByteArrayOutputStream();
                DataOutputStream dataOutput = new DataOutputStream(output)) {
            rowSelection.serializePortable(dataOutput);
            dataOutput.flush();
            return output.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** How to interpret the row selection payload. */
    enum SelectionMode {
        /** Ignore row selection payloads. */
        INCLUDE_ALL((byte) 0),
        /** Return only rows at the indices. */
        INCLUDE((byte) 1),
        /** Return rows except those at the indices. */
        EXCLUDE((byte) 2),
        /** Return only rows in the Roaring bitmap. */
        INCLUDE_ROARING((byte) 3),
        /** Return rows except those in the Roaring bitmap. */
        EXCLUDE_ROARING((byte) 4);

        private final byte code;

        SelectionMode(byte code) {
            this.code = code;
        }

        public byte code() {
            return code;
        }
    }
}
