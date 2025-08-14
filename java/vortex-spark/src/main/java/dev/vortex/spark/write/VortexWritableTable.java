// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.*;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import dev.vortex.spark.read.VortexTable;

/**
 * Spark V2 {@link Table} of Vortex files that supports both reading and writing.
 * 
 * This table implementation extends the read-only VortexTable to add write capabilities,
 * allowing Spark to write DataFrames as Vortex files.
 */
public final class VortexWritableTable extends VortexTable implements SupportsWrite {
    
    private final String outputPath;
    private final CaseInsensitiveStringMap writeOptions;
    
    /**
     * Creates a new VortexWritableTable for the specified file paths and columns.
     *
     * @param paths the list of Vortex file paths that make up this table (for reading)
     * @param readColumns the list of columns available for reading from this table
     * @param outputPath the path where new Vortex files will be written
     * @param writeOptions additional options for writing
     */
    public VortexWritableTable(
            ImmutableList<String> paths,
            ImmutableList<Column> readColumns,
            String outputPath,
            CaseInsensitiveStringMap writeOptions) {
        super(paths, readColumns);
        this.outputPath = outputPath;
        this.writeOptions = writeOptions;
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
        return new VortexWriteBuilder(outputPath, info, writeOptions);
    }
    
    /**
     * Returns the capabilities supported by this table.
     * 
     * This table supports both batch reading and batch writing.
     *
     * @return a set containing TableCapability.BATCH_READ and TableCapability.BATCH_WRITE
     */
    @Override
    public Set<TableCapability> capabilities() {
        return ImmutableSet.of(
            TableCapability.BATCH_READ,
            TableCapability.BATCH_WRITE
        );
    }
}