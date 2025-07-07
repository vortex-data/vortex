// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.google.common.collect.ImmutableList;
import java.io.Serializable;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.InputPartition;

/**
 * An {@link InputPartition} for reading a whole Vortex file.
 */
public final class VortexFilePartition implements InputPartition, Serializable {
    private final String path;
    private final ImmutableList<Column> columns;

    public VortexFilePartition(String path, ImmutableList<Column> columns) {
        this.path = path;
        this.columns = columns;
    }

    public String getPath() {
        return path;
    }

    public ImmutableList<Column> getColumns() {
        return columns;
    }
}
