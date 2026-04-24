// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.StructType;

/**
 * Spark V2 {@link Scan} over a table of Vortex files.
 */
public final class VortexScan implements Scan {

    private final List<String> paths;
    private final List<Column> readColumns;
    private final Map<String, String> formatOptions;

    /**
     * Creates a new VortexScan for the specified file paths and columns. The caller is
     * responsible for passing immutable collections; the constructor does not copy.
     *
     * @param paths       the list of Vortex file paths to scan
     * @param readColumns the list of columns to read from the files
     */
    public VortexScan(List<String> paths, List<Column> readColumns, Map<String, String> formatOptions) {
        this.paths = paths;
        this.readColumns = readColumns;
        this.formatOptions = formatOptions;
    }

    /**
     * Returns the schema for the data that will be read by this scan.
     * <p>
     * The schema is constructed from the read columns that were specified
     * when this scan was created.
     *
     * @return the StructType representing the schema of the read data
     */
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

    /**
     * Converts this scan to a Batch for execution.
     * <p>
     * Creates a VortexBatchExec that will handle the actual reading
     * of the specified files and columns.
     *
     * @return a Batch implementation for executing this scan
     */
    @Override
    public Batch toBatch() {
        return new VortexBatchExec(paths, readColumns, formatOptions);
    }

    /**
     * Returns the columnar support mode for this scan.
     * <p>
     * Vortex always provides columnar data access, so this method
     * always returns SUPPORTED.
     *
     * @return ColumnarSupportMode.SUPPORTED
     */
    @Override
    public ColumnarSupportMode columnarSupportMode() {
        return ColumnarSupportMode.SUPPORTED;
    }
}
