// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

final class VortexPartitionReaderSplitTest {
    @ParameterizedTest
    @CsvSource({"1000, 1000000", "1000000, 1000000", "999983, 1048576", "3, 7"})
    void splitOffsetStartsTheSplitAtItsRow(long rowCount, long size) {
        for (long row = 0; row < rowCount; row += Math.max(1, rowCount / 97)) {
            long offset = VortexPartitionReader.splitOffset(rowCount, row, size);

            assertEquals(row, VortexPartitionReader.splitRow(rowCount, offset, size));
            assertTrue(offset == 0 || VortexPartitionReader.splitRow(rowCount, offset - 1, size) < row);
        }
    }

    @ParameterizedTest
    @CsvSource({"1000000, 1000", "7, 3"})
    void withMoreRowsThanBytesTheSplitStartsAtTheNextRowItCan(long rowCount, long size) {
        for (long row = 0; row < rowCount; row += Math.max(1, rowCount / 97)) {
            long offset = VortexPartitionReader.splitOffset(rowCount, row, size);

            assertTrue(VortexPartitionReader.splitRow(rowCount, offset, size) >= row);
            assertTrue(offset == 0 || VortexPartitionReader.splitRow(rowCount, offset - 1, size) < row);
        }
    }

    @ParameterizedTest
    @CsvSource({"1000, 1000000", "0, 10"})
    void anOffsetPastTheLastRowIsTheFileSize(long rowCount, long size) {
        assertEquals(size, VortexPartitionReader.splitOffset(rowCount, rowCount, size));
    }
}
