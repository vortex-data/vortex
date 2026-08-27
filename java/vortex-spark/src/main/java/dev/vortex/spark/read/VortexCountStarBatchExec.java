// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.spark.VortexSparkSession;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;

/**
 * Spark V2 {@link Batch} for a pushed-down {@code COUNT(*)} over Vortex files.
 *
 * <p>Plans one {@link VortexCountStarInputPartition} per resolved file so the per-file footer reads run in parallel on
 * executors, mirroring the per-file partitioning of {@link VortexBatchExec}. Paths are resolved with the same directory
 * expansion as regular scans so a count and a scan over the same relation always see the same file set.
 */
public final class VortexCountStarBatchExec implements Batch {
    private final List<String> paths;
    private final Map<String, String> formatOptions;

    /**
     * Creates a new VortexCountStarBatchExec for the specified file paths. The caller is responsible for passing
     * immutable collections; the constructor does not copy.
     *
     * @param paths the list of Vortex file paths to count over
     * @param formatOptions the format options for opening the files
     */
    VortexCountStarBatchExec(List<String> paths, Map<String, String> formatOptions) {
        this.paths = paths;
        this.formatOptions = formatOptions;
    }

    @Override
    public InputPartition[] planInputPartitions() {
        List<String> resolvedPaths =
                VortexBatchExec.resolveVortexPaths(VortexSparkSession.get(formatOptions), paths, formatOptions);
        return resolvedPaths.stream().map(VortexCountStarInputPartition::new).toArray(InputPartition[]::new);
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        return new VortexCountStarPartitionReaderFactory(formatOptions);
    }
}
