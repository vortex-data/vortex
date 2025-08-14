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

    /**
     * Creates a new VortexBatchExec for scanning the specified Vortex files.
     *
     * @param paths the list of file paths to scan
     * @param columns the list of columns to read from the files
     */
    public VortexBatchExec(ImmutableList<String> paths, ImmutableList<Column> columns) {
        this.paths = paths;
        this.columns = columns;
    }

    /**
     * Plans the input partitions for this batch scan.
     * <p>
     * Creates one partition per file path, where each partition is responsible
     * for reading data from a single Vortex file.
     *
     * @return an array of InputPartition objects, one per file path
     */
    @Override
    public InputPartition[] planInputPartitions() {
        return IntStream.range(0, paths.size())
                .mapToObj(partitionId -> new VortexFilePartition(paths.get(partitionId), columns))
                .toArray(InputPartition[]::new);
    }

    /**
     * Creates a factory for creating partition readers.
     * <p>
     * Returns a singleton ReaderFactory instance that can create readers
     * capable of reading Vortex file partitions.
     *
     * @return the PartitionReaderFactory for Vortex files
     */
    @Override
    public PartitionReaderFactory createReaderFactory() {
        return ReaderFactory.INSTANCE;
    }
}
