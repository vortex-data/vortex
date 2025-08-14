// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import dev.vortex.spark.write.VortexWriteBuilder;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.*;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Spark V2 {@link Table} of Vortex files that supports both reading and writing.
 */
public final class VortexTable implements Table, SupportsRead, SupportsWrite {
    private static final String SHORT_NAME = "vortex";

    private final ImmutableList<String> paths;
    private final ImmutableList<Column> readColumns;
    private final String outputPath;
    private final CaseInsensitiveStringMap writeOptions;

    /**
     * Creates a new VortexTable for the specified file paths and columns.
     *
     * @param paths the list of Vortex file paths that make up this table
     * @param readColumns the list of columns available for reading from this table
     */
    public VortexTable(ImmutableList<String> paths, ImmutableList<Column> readColumns) {
        this(paths, readColumns, null, new CaseInsensitiveStringMap(java.util.Collections.emptyMap()));
    }
    
    /**
     * Creates a new VortexTable with write support.
     *
     * @param paths the list of Vortex file paths that make up this table
     * @param readColumns the list of columns available for reading from this table
     * @param outputPath the path where new Vortex files will be written (optional)
     * @param writeOptions additional options for writing (optional)
     */
    public VortexTable(ImmutableList<String> paths, ImmutableList<Column> readColumns,
                       String outputPath, CaseInsensitiveStringMap writeOptions) {
        this.paths = paths;
        this.readColumns = readColumns;
        this.outputPath = outputPath;
        this.writeOptions = writeOptions;
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
     * Creates a new WriteBuilder for writing data to this table.
     * 
     * The WriteBuilder is responsible for configuring and executing write operations
     * to create new Vortex files.
     *
     * @param info logical information about the write operation
     * @return a new VortexWriteBuilder configured for this table
     */
    @Override
    public WriteBuilder newWriteBuilder(LogicalWriteInfo info) {
        String path = outputPath != null ? outputPath : 
            (paths.isEmpty() ? "." : paths.get(0).substring(0, paths.get(0).lastIndexOf('/')));
        return new VortexWriteBuilder(path, info, writeOptions);
    }
    
    /**
     * Returns the capabilities supported by this table.
     * <p>
     * Vortex tables support both batch reading and batch writing.
     * Note: Write capability temporarily disabled to force V1 write path
     * which handles schema inference better for non-existent files.
     *
     * @return a set containing TableCapability.BATCH_READ
     */
    @Override
    public Set<TableCapability> capabilities() {
        // TODO: Re-enable BATCH_WRITE once schema inference is fixed for write operations
        return ImmutableSet.of(TableCapability.BATCH_READ);
        // return ImmutableSet.of(TableCapability.BATCH_READ, TableCapability.BATCH_WRITE);
    }
}
