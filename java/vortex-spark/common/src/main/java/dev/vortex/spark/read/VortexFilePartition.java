// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.jakewharton.nopen.annotation.Open;
import org.apache.spark.sql.execution.datasources.FilePartition;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.sources.Filter;

/**
 * A file partition that carries the runtime filters its readers evaluate.
 *
 * <p>Spark hands runtime filters to a scan after it has built the reader factory, so they reach the executors with the
 * partitions planned after them.
 */
@Open
public class VortexFilePartition extends FilePartition {
    private static final long serialVersionUID = 1L;

    private final Filter[] runtimeFilters;

    public VortexFilePartition(int index, PartitionedFile[] files, Filter[] runtimeFilters) {
        super(index, files);
        this.runtimeFilters = runtimeFilters.clone();
    }

    /** The runtime filters every reader of this partition evaluates. */
    public Filter[] runtimeFilters() {
        return runtimeFilters.clone();
    }
}
