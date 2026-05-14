// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import java.util.Map;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.SupportsTruncate;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriteBuilder;

/**
 * Builder for configuring Vortex write operations.
 *
 * <p>This class is responsible for creating BatchWrite instances that execute the actual write operations to create
 * Vortex files from Spark DataFrames.
 */
public final class VortexWriteBuilder implements WriteBuilder, SupportsTruncate {

    private final String paths;
    private final LogicalWriteInfo writeInfo;
    private final Map<String, String> options;
    private final Transform[] partitionTransforms;
    private boolean truncate = false;

    /**
     * Creates a new VortexWriteBuilder.
     *
     * @param paths root path for write
     * @param writeInfo logical information about the write operation
     * @param options additional write options
     * @param partitionTransforms partition transforms (may be empty)
     */
    public VortexWriteBuilder(
            String paths, LogicalWriteInfo writeInfo, Map<String, String> options, Transform[] partitionTransforms) {
        this.paths = paths;
        this.writeInfo = writeInfo;
        this.options = options;
        this.partitionTransforms = partitionTransforms;
    }

    /**
     * Builds a Write for executing the write operation.
     *
     * @return a new VortexBatchWrite configured with this builder's settings
     */
    @Override
    public Write build() {
        return new VortexBatchWrite(paths, writeInfo.schema(), options, truncate, partitionTransforms);
    }

    /**
     * Configures the write operation to truncate existing data.
     *
     * <p>When truncate is enabled, existing Vortex files at the output path will be removed before writing new data.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder truncate() {
        this.truncate = true;
        return this;
    }
}
