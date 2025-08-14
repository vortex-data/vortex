// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.*;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Spark V2 {@link Table} of Vortex files.
 */
public final class VortexTable implements Table, SupportsRead {
    private static final String SHORT_NAME = "vortex";

    private final ImmutableList<String> paths;
    private final ImmutableList<Column> readColumns;

    /**
     * Creates a new VortexTable for the specified file paths and columns.
     *
     * @param paths the list of Vortex file paths that make up this table
     * @param readColumns the list of columns available for reading from this table
     */
    public VortexTable(ImmutableList<String> paths, ImmutableList<Column> readColumns) {
        this.paths = paths;
        this.readColumns = readColumns;
    }

    /**
     * Creates a new ScanBuilder for this table.
     * <p>
     * The scan builder is pre-configured with all the file paths and columns
     * from this table. The options parameter is currently unused but reserved
     * for future use (e.g., S3 credentials).
     *
     * @param _options scan options (currently unused)
     * @return a new VortexScanBuilder configured for this table
     */
    @Override
    public ScanBuilder newScanBuilder(CaseInsensitiveStringMap _options) {
        // TODO(aduffy): pass any S3 creds from options down into the scan builder.
        //  Or can those be pulled out in the task-side instead?
        return new VortexScanBuilder().addAllPaths(paths).addAllColumns(readColumns);
    }

    /**
     * Returns the name of this table.
     * <p>
     * The name includes the "vortex" prefix and a comma-separated list
     * of all file paths that comprise this table.
     *
     * @return the table name in the format: vortex."path1,path2,..."
     */
    @Override
    public String name() {
        return String.format("%s.\"%s\"", SHORT_NAME, String.join(",", paths));
    }

    /**
     * Returns the schema of this table.
     * <p>
     * The schema is derived from the columns available for reading.
     *
     * @return the StructType representing the table schema
     */
    @Override
    public StructType schema() {
        return CatalogV2Util$.MODULE$.v2ColumnsToStructType(columns());
    }

    /**
     * Returns the columns available for reading from this table.
     *
     * @return an array of Column objects representing the available columns
     */
    @Override
    public Column[] columns() {
        return readColumns.toArray(new Column[0]);
    }

    /**
     * Returns the capabilities supported by this table.
     * <p>
     * Currently, Vortex tables only support batch reading.
     *
     * @return a set containing TableCapability.BATCH_READ
     */
    @Override
    public Set<TableCapability> capabilities() {
        return ImmutableSet.of(TableCapability.BATCH_READ);
    }
}
