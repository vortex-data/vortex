// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.Serializable;
import java.util.Map;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.InputPartition;

/**
 * An {@link InputPartition} for reading a whole Vortex file.
 * <p>
 * This class represents a partition that corresponds to a single Vortex file.
 * It contains the file path and the columns to be read from that file.
 * Each partition can be processed independently by Spark executors.
 */
public final class VortexFilePartition implements InputPartition, Serializable {
    private final String path;
    private final ImmutableList<Column> columns;
    private final ImmutableMap<String, String> formatOptions;

    /**
     * Creates a new Vortex file partition.
     *
     * @param path the file system path to the Vortex file
     * @param columns the list of columns to read from the file
     */
    public VortexFilePartition(String path, ImmutableList<Column> columns, ImmutableMap<String, String> formatOptions) {
        this.path = path;
        this.columns = columns;
        this.formatOptions = formatOptions;
    }

    /**
     * Returns the file system path to the Vortex file for this partition.
     *
     * @return the file path
     */
    public String getPath() {
        return path;
    }

    /**
     * Returns the list of columns to be read from this partition.
     *
     * @return the immutable list of columns
     */
    public ImmutableList<Column> getColumns() {
        return columns;
    }

    public Map<String, String> getFormatOptions() {
        return formatOptions;
    }
}
