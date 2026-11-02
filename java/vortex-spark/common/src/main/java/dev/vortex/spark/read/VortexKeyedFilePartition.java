// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.HasPartitionKey;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.sources.Filter;

/**
 * A file partition whose files all share one value of each partition column the scan reads, reported to Spark as the
 * partition's key.
 *
 * <p>With {@code spark.sql.sources.v2.bucketing.enabled}, Spark groups the partitions of a scan by key, so a join or
 * aggregation on the partition columns needs no shuffle.
 */
public final class VortexKeyedFilePartition extends VortexFilePartition implements HasPartitionKey {
    private static final long serialVersionUID = 1L;

    private final InternalRow partitionKey;

    public VortexKeyedFilePartition(
            int index, PartitionedFile[] files, Filter[] runtimeFilters, InternalRow partitionKey) {
        super(index, files, runtimeFilters);
        this.partitionKey = partitionKey;
    }

    @Override
    public InternalRow partitionKey() {
        return partitionKey;
    }
}
