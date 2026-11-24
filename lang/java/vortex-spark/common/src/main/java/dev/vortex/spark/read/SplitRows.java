// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.math.BigInteger;

/**
 * The rows that one byte split of a Vortex file reads. Each offset maps to the row at the same fraction of the file, so
 * neighbouring splits share a row boundary and every row is read once whatever offsets Spark picks.
 */
record SplitRows(long begin, long end) {
    // The default writer builds chunks from whole blocks of this many rows, so cuts land on block boundaries.
    static final long ROW_BLOCK = 8192;

    /** The rows that bytes {@code [start, start + length)} of a {@code fileSize}-byte file read. */
    static SplitRows of(long start, long length, long fileSize, long rows) {
        return new SplitRows(rowAt(start, fileSize, rows), rowAt(start + length, fileSize, rows));
    }

    long count() {
        return end - begin;
    }

    private static long rowAt(long offset, long fileSize, long rows) {
        if (offset >= fileSize) {
            return rows;
        }
        if (offset <= 0) {
            return 0;
        }
        long row = BigInteger.valueOf(offset)
                .multiply(BigInteger.valueOf(rows))
                .divide(BigInteger.valueOf(fileSize))
                .longValueExact();
        return row / ROW_BLOCK * ROW_BLOCK;
    }
}
