// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkNotNull;

import dev.vortex.api.File;
import dev.vortex.api.Files;
import dev.vortex.api.ScanOptions;
import dev.vortex.spark.VortexFilePartition;
import java.util.*;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A {@link PartitionReader} that reads columnar batches out of a Vortex file into
 * Vortex memory format.
 * <p>
 * When reading from partitioned directories, partition column values are extracted from the
 * Hive-style file path and materialized as Spark
 * {@link org.apache.spark.sql.execution.vectorized.ConstantColumnVector} instances that are
 * spliced into each output batch.
 */
final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final VortexFilePartition partition;

    private File file;
    private VortexColumnarBatchIterator batches;

    /** Names of columns whose values come from the partition path rather than the data file. */
    private Set<String> partitionColumnNames;

    /** Tracks the last data batch so its native memory can be freed properly. */
    private ColumnarBatch lastDataBatch;

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

        // Free previous data batch native memory
        if (lastDataBatch != null) {
            lastDataBatch.close();
            lastDataBatch = null;
        }

        ColumnarBatch dataBatch = batches.next();

        if (partitionColumnNames.isEmpty()) {
            return dataBatch;
        }

        // Track the data batch for lifecycle management
        lastDataBatch = dataBatch;
        return buildCombinedBatch(dataBatch);
    }

    /**
     * Builds a combined batch with data columns from the file and constant partition columns
     * in the order expected by the full table schema.
     */
    private ColumnarBatch buildCombinedBatch(ColumnarBatch dataBatch) {
        int rowCount = dataBatch.numRows();
        Map<String, String> partVals = partition.getPartitionValues();
        List<Column> allColumns = partition.getColumns();
        ColumnVector[] combined = new ColumnVector[allColumns.size()];

        int dataIdx = 0;
        for (int i = 0; i < allColumns.size(); i++) {
            Column col = allColumns.get(i);
            String partValue = partVals.get(col.name());
            if (partValue != null) {
                combined[i] = PartitionPathUtils.createConstantVector(rowCount, col.dataType(), partValue);
            } else {
                combined[i] = dataBatch.column(dataIdx++);
            }
        }

        return new CombinedColumnarBatch(combined, rowCount);
    }

    /**
     * Initialize the Vortex File and ArrayStream resources.
     * <p>
     * Partition columns are identified by matching requested column names against the
     * partition values from the file path. Only non-partition columns are pushed down
     * to the Vortex scan.
     */
    void initNativeResources() {
        Map<String, String> partVals = partition.getPartitionValues();
        this.partitionColumnNames = new HashSet<>();

        List<String> dataColumnNames = new ArrayList<>();
        for (Column col : partition.getColumns()) {
            if (partVals.containsKey(col.name())) {
                partitionColumnNames.add(col.name());
            } else {
                dataColumnNames.add(col.name());
            }
        }

        file = Files.open(partition.getPath(), partition.getFormatOptions());
        batches = new VortexColumnarBatchIterator(
                file.newScan(ScanOptions.builder().columns(dataColumnNames).build()));
    }

    @Override
    public void close() {
        if (lastDataBatch != null) {
            lastDataBatch.close();
            lastDataBatch = null;
        }

        checkNotNull(file, "File was closed");
        checkNotNull(batches, "ArrayStream was closed");

        batches.close();
        batches = null;

        file.close();
        file = null;
    }

    /**
     * A ColumnarBatch that does not close its column vectors on {@link #close()}.
     * <p>
     * The data column vectors are owned by the underlying {@link VortexColumnarBatch}
     * (tracked via {@link #lastDataBatch}), and the constant partition vectors have trivial
     * lifecycle. Neither should be closed by this wrapper.
     */
    private static final class CombinedColumnarBatch extends ColumnarBatch {
        CombinedColumnarBatch(ColumnVector[] columns, int numRows) {
            super(columns, numRows);
        }

        @Override
        public void close() {
            // Intentionally empty: lifecycle is managed by VortexPartitionReader
        }

        @Override
        public void closeIfFreeable() {
            // Intentionally empty
        }
    }
}
