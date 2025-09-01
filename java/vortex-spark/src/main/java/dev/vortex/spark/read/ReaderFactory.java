// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.spark.VortexFilePartition;
import java.io.Serializable;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A {@link PartitionReaderFactory} for Vortex file partitions.
 * <p>
 * This factory creates partition readers for reading Vortex files. It implements the singleton
 * pattern using an enum and only supports columnar reads for optimal performance.
 * Row-based reads are not supported as Vortex is designed for columnar data processing.
 */
enum ReaderFactory implements PartitionReaderFactory, Serializable {
    INSTANCE;

    private static final boolean SUPPORTS_COLUMNAR_READS = true;

    /**
     * Creates a row-based partition reader.
     * <p>
     * This method is not supported as Vortex only supports columnar reads for performance reasons.
     *
     * @param partition the input partition to read from
     * @return never returns, always throws an exception
     * @throws UnsupportedOperationException always, as row-based reading is not supported
     */
    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
        throw new UnsupportedOperationException("ReaderFactory only supports columnar reads");
    }

    /**
     * Creates a columnar partition reader for the given partition.
     * <p>
     * This method creates a VortexPartitionReader that can efficiently read columnar data
     * from a Vortex file partition.
     *
     * @param partition the input partition to read from, must be a VortexFilePartition
     * @return a partition reader that produces ColumnarBatch objects
     * @throws ClassCastException if the partition is not a VortexFilePartition
     */
    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        return new VortexPartitionReader((VortexFilePartition) partition);
    }

    /**
     * Indicates whether this factory supports columnar reads for the given partition.
     * <p>
     * Vortex always supports and prefers columnar reads for optimal performance.
     *
     * @param partition the input partition to check (parameter is ignored)
     * @return always true, indicating columnar reads are supported
     */
    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return SUPPORTS_COLUMNAR_READS;
    }
}
