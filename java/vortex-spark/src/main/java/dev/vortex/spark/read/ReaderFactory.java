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
 */
enum ReaderFactory implements PartitionReaderFactory, Serializable {
    INSTANCE;

    private static final boolean SUPPORTS_COLUMNAR_READS = true;

    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
        throw new UnsupportedOperationException("ReaderFactory only supports columnar reads");
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        return new VortexPartitionReader((VortexFilePartition) partition);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return SUPPORTS_COLUMNAR_READS;
    }
}
