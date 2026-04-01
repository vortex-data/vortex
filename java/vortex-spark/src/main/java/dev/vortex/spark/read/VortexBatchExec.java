// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import dev.vortex.jni.NativeFileMethods;
import dev.vortex.spark.VortexFilePartition;
import java.util.stream.Stream;
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
    private final ImmutableMap<String, String> formatOptions;

    /**
     * Creates a new VortexBatchExec for scanning the specified Vortex files.
     *
     * @param paths   the list of file paths to scan
     * @param columns the list of columns to read from the files
     */
    public VortexBatchExec(
            ImmutableList<String> paths, ImmutableList<Column> columns, ImmutableMap<String, String> formatOptions) {
        this.paths = paths;
        this.columns = columns;
        this.formatOptions = formatOptions;
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
        // Scan all paths and assign each file its own partition
        return paths.stream()
                .flatMap(path -> {
                    if (path.endsWith(".vortex")) {
                        return Stream.of(path);
                    } else {
                        // Scan and return the paths
                        return NativeFileMethods.listVortexFiles(path, formatOptions).stream();
                    }
                })
                .map(path -> new VortexFilePartition(path, columns, formatOptions))
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
