// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.ScanOptions.SelectionMode;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;
import org.roaringbitmap.longlong.Roaring64NavigableMap;

/**
 * Unit tests for {@link ScanOptions#validateSelectionPayload()} and the factories that build a row selection.
 *
 * <p>Validation runs in an Immutables {@code @Value.Check}, so it fires at {@code build()} time on the driver rather
 * than inside the native scan. That is the only place a mismatched mode and payload is reported with a message a caller
 * can act on: {@code scan.rs} sees the mode as a bare byte alongside two possibly-empty arrays, so an absent payload
 * reaches it as an empty selection instead of an error.
 */
public final class ScanOptionsTest {

    @Test
    public void defaultsReadEveryRowAndColumn() {
        ScanOptions options = ScanOptions.of();

        assertTrue(options.projection().isEmpty());
        assertTrue(options.filter().isEmpty());
        assertEquals(OptionalLong.empty(), options.rowRangeBegin());
        assertEquals(OptionalLong.empty(), options.rowRangeEnd());
        assertEquals(OptionalLong.empty(), options.limit());
        assertTrue(options.selectionIndices().isEmpty());
        assertTrue(options.selectionRoaringBitmap().isEmpty());
        assertEquals(SelectionMode.INCLUDE_ALL, options.selectionMode());
        assertFalse(options.ordered());
    }

    @Test
    public void indexSelectionModesRequireIndices() {
        for (SelectionMode mode : new SelectionMode[] {SelectionMode.INCLUDE, SelectionMode.EXCLUDE}) {
            IllegalArgumentException exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> ScanOptions.builder().selectionMode(mode).build(),
                    () -> "no exception for " + mode);
            assertEquals("selection indices are required for index selection modes", exception.getMessage());
        }
    }

    @Test
    public void indexSelectionModesRejectARoaringPayload() {
        // The payload is present but of the wrong kind, so the indices check is what rejects it.
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> ScanOptions.builder()
                .selectionMode(SelectionMode.INCLUDE)
                .selectionRoaringBitmap(new byte[] {1})
                .build());
        assertEquals("selection indices are required for index selection modes", exception.getMessage());
    }

    @Test
    public void roaringSelectionModesRequireABitmap() {
        for (SelectionMode mode : new SelectionMode[] {SelectionMode.INCLUDE_ROARING, SelectionMode.EXCLUDE_ROARING}) {
            IllegalArgumentException exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> ScanOptions.builder().selectionMode(mode).build(),
                    () -> "no exception for " + mode);
            assertEquals("selection roaring bitmap is required for roaring selection modes", exception.getMessage());
        }
    }

    @Test
    public void roaringSelectionModesRejectAnIndexPayload() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> ScanOptions.builder()
                .selectionMode(SelectionMode.INCLUDE_ROARING)
                .selectionIndices(new long[] {0})
                .build());
        assertEquals("selection roaring bitmap is required for roaring selection modes", exception.getMessage());
    }

    @Test
    public void roaringSelectionModesRejectAnEmptyBitmap() {
        // `deserialize_roaring_selection` in scan.rs also rejects an empty buffer; catching it here keeps the
        // failure on the driver, where the caller still has a stack that names the option.
        for (SelectionMode mode : new SelectionMode[] {SelectionMode.INCLUDE_ROARING, SelectionMode.EXCLUDE_ROARING}) {
            IllegalArgumentException exception = assertThrows(
                    IllegalArgumentException.class,
                    () -> ScanOptions.builder()
                            .selectionMode(mode)
                            .selectionRoaringBitmap(new byte[0])
                            .build(),
                    () -> "no exception for " + mode);
            assertEquals("selection roaring bitmap must not be empty", exception.getMessage());
        }
    }

    @Test
    public void anEmptyIndexArrayIsAcceptedForIndexModes() {
        // Only the roaring payload has a non-empty requirement; an empty index array selects no rows.
        ScanOptions options = ScanOptions.includeRows();

        assertEquals(SelectionMode.INCLUDE, options.selectionMode());
        assertArrayEquals(new long[0], options.selectionIndices().orElseThrow());
    }

    @Test
    public void includeAllRejectsEitherPayload() {
        IllegalArgumentException fromIndices = assertThrows(
                IllegalArgumentException.class,
                () -> ScanOptions.builder().selectionIndices(new long[] {0}).build());
        assertEquals("row selection payload requires a selection mode", fromIndices.getMessage());

        IllegalArgumentException fromBitmap = assertThrows(IllegalArgumentException.class, () -> ScanOptions.builder()
                .selectionRoaringBitmap(new byte[] {1})
                .build());
        assertEquals("row selection payload requires a selection mode", fromBitmap.getMessage());
    }

    @Test
    public void bothPayloadsTogetherAreRejectedBeforeTheModeIsConsidered() {
        // Checked ahead of the mode switch, so it reports the same message for a roaring mode.
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> ScanOptions.builder()
                .selectionMode(SelectionMode.INCLUDE_ROARING)
                .selectionIndices(new long[] {0})
                .selectionRoaringBitmap(new byte[] {1})
                .build());
        assertEquals("row selection must use either indices or roaring bitmap, not both", exception.getMessage());
    }

    @Test
    public void indexFactoriesSetTheMatchingMode() {
        assertEquals(SelectionMode.INCLUDE, ScanOptions.includeRows(0, 3, 9).selectionMode());
        assertEquals(SelectionMode.EXCLUDE, ScanOptions.excludeRows(0, 9).selectionMode());
    }

    @Test
    public void indexFactoriesCopyTheirArgument() {
        long[] rowIndices = {0, 3, 9};
        ScanOptions options = ScanOptions.includeRows(rowIndices);
        rowIndices[0] = 7;

        assertArrayEquals(new long[] {0, 3, 9}, options.selectionIndices().orElseThrow());
    }

    @Test
    public void roaringFactoriesSetTheMatchingModeAndSerializeThePayload() throws Exception {
        Roaring64NavigableMap rows = new Roaring64NavigableMap();
        rows.addLong(0);
        rows.addLong(3);
        rows.addLong(9);

        for (SelectionMode mode : new SelectionMode[] {SelectionMode.INCLUDE_ROARING, SelectionMode.EXCLUDE_ROARING}) {
            ScanOptions options = mode == SelectionMode.INCLUDE_ROARING
                    ? ScanOptions.includeRows(rows)
                    : ScanOptions.excludeRows(rows);

            assertEquals(mode, options.selectionMode());
            // Written with `serializePortable`, so `deserializePortable` on the same bytes must round trip.
            Roaring64NavigableMap parsed = new Roaring64NavigableMap();
            parsed.deserializePortable(new DataInputStream(
                    new ByteArrayInputStream(options.selectionRoaringBitmap().orElseThrow())));
            assertEquals(3, parsed.getLongCardinality());
            assertTrue(parsed.contains(9L));
            assertFalse(parsed.contains(1L));
        }
    }

    @Test
    public void anEmptyRoaringBitmapStillSerializesToAHeader() {
        // `serializePortable` writes a container count even for an empty map, so the non-empty check above
        // guards a malformed buffer rather than an empty selection.
        ScanOptions options = ScanOptions.includeRows(new Roaring64NavigableMap());

        assertTrue(options.selectionRoaringBitmap().orElseThrow().length > 0);
    }

    @Test
    public void selectionModeCodesMatchTheNativeSwitch() {
        // scan.rs matches these bytes directly and bails on anything else.
        assertEquals(0, SelectionMode.INCLUDE_ALL.code());
        assertEquals(1, SelectionMode.INCLUDE.code());
        assertEquals(2, SelectionMode.EXCLUDE.code());
        assertEquals(3, SelectionMode.INCLUDE_ROARING.code());
        assertEquals(4, SelectionMode.EXCLUDE_ROARING.code());
        assertEquals(5, SelectionMode.values().length);
    }

    @Test
    public void rowRangeAndLimitAreCarriedThrough() {
        ScanOptions options = ScanOptions.builder()
                .rowRangeBegin(10)
                .rowRangeEnd(20)
                .limit(5)
                .ordered(true)
                .build();

        assertEquals(OptionalLong.of(10), options.rowRangeBegin());
        assertEquals(OptionalLong.of(20), options.rowRangeEnd());
        assertEquals(OptionalLong.of(5), options.limit());
        assertTrue(options.ordered());
    }
}
