// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkNotNull;

import dev.vortex.api.File;
import dev.vortex.api.Files;
import dev.vortex.api.ScanOptions;
import dev.vortex.spark.VortexFilePartition;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A {@link PartitionReader} that reads columnar batches out of a Vortex file into
 * Vortex memory format.
 */
final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final VortexFilePartition partition;

    private File file;
    private VortexColumnarBatchIterator batches;

    VortexPartitionReader(VortexFilePartition partition) {
        this.partition = partition;
        initNativeResources();
    }

    @Override
    public boolean next() {
        checkNotNull(batches, "batches");

        return batches.hasNext();
    }

    @Override
    public ColumnarBatch get() {
        checkNotNull(batches, "closed ArrayStream");
        return batches.next();
    }

    /**
     * Initialize the Vortex File and ArrayStream resources.
     */
    void initNativeResources() {
        file = Files.open(partition.getPath(), partition.getFormatOptions());
        List<String> pushdownColumns =
                partition.getColumns().stream().map(Column::name).collect(Collectors.toList());
        batches = new VortexColumnarBatchIterator(
                file.newScan(ScanOptions.builder().columns(pushdownColumns).build()));
    }

    @Override
    public void close() {
        checkNotNull(file, "File was closed");
        checkNotNull(batches, "ArrayStream was closed");

        batches.close();
        batches = null;

        file.close();
        file = null;
    }
}
