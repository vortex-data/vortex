// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import org.apache.spark.sql.connector.write.*;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Builder for configuring Vortex write operations.
 * 
 * This class is responsible for creating BatchWrite instances that execute
 * the actual write operations to create Vortex files from Spark DataFrames.
 */
public final class VortexWriteBuilder implements WriteBuilder, SupportsTruncate, SupportsOverwrite {
    
    private final String outputPath;
    private final LogicalWriteInfo writeInfo;
    private final CaseInsensitiveStringMap options;
    private boolean truncate = false;
    private boolean overwrite = false;
    
    /**
     * Creates a new VortexWriteBuilder.
     *
     * @param outputPath the base path where Vortex files will be written
     * @param writeInfo logical information about the write operation
     * @param options additional write options
     */
    public VortexWriteBuilder(
            String outputPath,
            LogicalWriteInfo writeInfo,
            CaseInsensitiveStringMap options) {
        this.outputPath = outputPath;
        this.writeInfo = writeInfo;
        this.options = options;
    }
    
    /**
     * Builds a BatchWrite for executing the write operation.
     *
     * @return a new VortexBatchWrite configured with this builder's settings
     */
    @Override
    public BatchWrite build() {
        return new VortexBatchWrite(
            outputPath,
            writeInfo.schema(),
            options,
            truncate || overwrite
        );
    }
    
    /**
     * Configures the write operation to truncate existing data.
     * 
     * When truncate is enabled, existing Vortex files at the output path
     * will be removed before writing new data.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder truncate() {
        this.truncate = true;
        return this;
    }
    
    /**
     * Configures the write operation to overwrite existing data.
     * 
     * Similar to truncate, this will remove existing files before writing.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder overwrite(SupportsTruncate.WriteContext context) {
        this.overwrite = true;
        return this;
    }
}