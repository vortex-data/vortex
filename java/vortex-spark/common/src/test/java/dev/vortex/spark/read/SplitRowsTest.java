// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

final class SplitRowsTest {

    @Test
    @DisplayName("Spark's byte splits of a file map to row ranges that tile its rows once, on row-block boundaries")
    void byteSplitsTileTheRows() {
        long[] rowCounts = {0, 1, 2, 8191, 8192, 8193, 50_000, 131_072, 1_000_003};
        long[] fileSizes = {1, 7, 600, 4096, 1_048_576, 3_600_017, 104_857_600};
        long[] splitSizes = {1, 3, 1000, 65_537, 1_048_576, 4_194_304, 134_217_728};
        for (long rows : rowCounts) {
            for (long fileSize : fileSizes) {
                for (long splitSize : splitSizes) {
                    if (fileSize / splitSize > 100_000) {
                        continue;
                    }
                    assertTiles(rows, fileSize, splitSize);
                }
            }
        }
    }

    @Test
    @DisplayName("Offsets map at the same fraction of the file, rounded down to a row block")
    void offsetsMapAtTheSameFraction() {
        // A third of 100k rows is 33333, rounded down to four 8192-row blocks.
        assertEquals(new SplitRows(0, 32_768), SplitRows.of(0, 1_000_000, 3_000_000, 100_000));
        assertEquals(new SplitRows(32_768, 65_536), SplitRows.of(1_000_000, 1_000_000, 3_000_000, 100_000));
        assertEquals(new SplitRows(65_536, 100_000), SplitRows.of(2_000_000, 1_000_000, 3_000_000, 100_000));
        // A split smaller than a row block's worth of bytes may read nothing; a neighbour reads its rows.
        assertEquals(new SplitRows(0, 0), SplitRows.of(0, 1000, 3_000_000, 100_000));
    }

    @Test
    @DisplayName("Offsets times rows beyond a long still map exactly")
    void largeFilesDoNotOverflow() {
        long fileSize = 1L << 42;
        long rows = 1L << 40;
        SplitRows half = SplitRows.of(fileSize / 2, fileSize / 2, fileSize, rows);
        assertEquals(new SplitRows(rows / 2, rows), half);
    }

    private static void assertTiles(long rows, long fileSize, long splitSize) {
        List<SplitRows> ranges = new ArrayList<>();
        // Spark's PartitionedFileUtil.splitFiles: contiguous ranges of splitSize, the last one shorter.
        for (long start = 0; start < fileSize; start += splitSize) {
            ranges.add(SplitRows.of(start, Math.min(splitSize, fileSize - start), fileSize, rows));
        }
        String context = String.format("rows=%d fileSize=%d splitSize=%d", rows, fileSize, splitSize);
        long next = 0;
        for (SplitRows range : ranges) {
            assertEquals(next, range.begin(), context);
            assertTrue(range.end() >= range.begin(), context);
            assertTrue(range.end() == rows || range.end() % SplitRows.ROW_BLOCK == 0, context);
            next = range.end();
        }
        assertEquals(rows, next, context);
    }
}
