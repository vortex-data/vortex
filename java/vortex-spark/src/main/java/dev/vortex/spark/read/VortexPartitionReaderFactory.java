// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import dev.vortex.spark.VortexFilePartition;
import java.io.Serializable;
import java.util.List;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Factory that produces columnar readers for Vortex files.
 *
 * <p>The set of paths belongs to each {@link VortexFilePartition} — the factory itself is stateless across partitions.
 * For every input partition, {@link VortexPartitionReader} opens a single {@code Session}, {@code DataSource} and
 * {@code Scan} spanning that partition's paths and consumes every Vortex partition produced by that scan before
 * returning.
 */
public final class VortexPartitionReaderFactory implements PartitionReaderFactory, Serializable {
    private static final long serialVersionUID = 1L;

    private final ImmutableList<String> dataColumnNames;
    private final ImmutableMap<String, String> formatOptions;

    public VortexPartitionReaderFactory(List<String> dataColumnNames, java.util.Map<String, String> formatOptions) {
        this.dataColumnNames = ImmutableList.copyOf(dataColumnNames);
        this.formatOptions = ImmutableMap.copyOf(formatOptions);
    }

    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
        throw new UnsupportedOperationException("row-based reads are not supported");
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        VortexFilePartition spark = (VortexFilePartition) partition;
        return new VortexPartitionReader(spark, dataColumnNames, formatOptions);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }
}
