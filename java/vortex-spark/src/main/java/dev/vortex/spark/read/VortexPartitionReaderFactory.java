// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableList;
import dev.vortex.jni.NativeRuntime;
import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexOptions;
import java.io.Serializable;
import java.util.List;
import java.util.Objects;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
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
    private final VortexOptions formatOptions;
    private final Predicate[] pushedPredicates;

    public VortexPartitionReaderFactory(
            List<String> dataColumnNames, VortexOptions formatOptions, Predicate[] pushedPredicates) {
        this.dataColumnNames = ImmutableList.copyOf(dataColumnNames);
        this.formatOptions = Objects.requireNonNull(formatOptions, "formatOptions");
        this.pushedPredicates = pushedPredicates == null ? new Predicate[0] : pushedPredicates.clone();
    }

    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
        throw new UnsupportedOperationException("row-based reads are not supported");
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        NativeRuntime.setWorkerThreads(formatOptions.workerThreads());
        VortexFilePartition spark = (VortexFilePartition) partition;
        return new VortexPartitionReader(spark, dataColumnNames, formatOptions, pushedPredicates);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }
}
