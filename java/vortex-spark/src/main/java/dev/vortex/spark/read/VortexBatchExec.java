// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableList;
import dev.vortex.spark.VortexFilePartition;
import java.util.stream.IntStream;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;

/**
 * Execution source for batch scans of Vortex file tables.
 */
public final class VortexBatchExec implements Batch {
    private final ImmutableList<String> paths;
    private final ImmutableList<Column> columns;

    public VortexBatchExec(ImmutableList<String> paths, ImmutableList<Column> columns) {
        this.paths = paths;
        this.columns = columns;
    }

    @Override
    public InputPartition[] planInputPartitions() {
        return IntStream.range(0, paths.size())
                .mapToObj(partitionId -> new VortexFilePartition(paths.get(partitionId), columns))
                .toArray(InputPartition[]::new);
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        return ReaderFactory.INSTANCE;
    }
}
