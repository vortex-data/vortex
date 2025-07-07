// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableList;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.StructType;

/**
 * Spark V2 {@link Scan} over a table of Vortex files.
 */
public final class VortexScan implements Scan {

    private final ImmutableList<String> paths;
    private final ImmutableList<Column> readColumns;

    public VortexScan(ImmutableList<String> paths, ImmutableList<Column> readColumns) {
        this.paths = paths;
        this.readColumns = readColumns;
    }

    @Override
    public StructType readSchema() {
        return CatalogV2Util.v2ColumnsToStructType(readColumns.toArray(new Column[0]));
    }

    /**
     * Logging-friendly readable description of the scan source.
     */
    @Override
    public String description() {
        return String.format("VortexScan{paths=%s, columns=%s}", paths, readColumns);
    }

    @Override
    public Batch toBatch() {
        return new VortexBatchExec(paths, readColumns);
    }

    // We always provide columnar scans.
    @Override
    public ColumnarSupportMode columnarSupportMode() {
        return ColumnarSupportMode.SUPPORTED;
    }
}
