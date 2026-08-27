// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Factory for {@link VortexCountStarPartitionReader}s.
 *
 * <p>Columnar-only, mirroring {@link VortexPartitionReaderFactory}: the count is emitted as a single one-row
 * {@link ColumnarBatch}.
 */
public final class VortexCountStarPartitionReaderFactory implements PartitionReaderFactory {
    private final Map<String, String> formatOptions;

    /**
     * Creates a new factory.
     *
     * @param formatOptions the format options for opening the files
     */
    VortexCountStarPartitionReaderFactory(Map<String, String> formatOptions) {
        this.formatOptions = formatOptions;
    }

    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
        throw new UnsupportedOperationException("row-based reads are not supported");
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        checkArgument(
                partition instanceof VortexCountStarInputPartition,
                "expected VortexCountStarInputPartition, got %s",
                partition.getClass().getName());
        return new VortexCountStarPartitionReader(((VortexCountStarInputPartition) partition).path(), formatOptions);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }
}
