// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Factory for creating VortexDataWriter instances on Spark executors.
 * 
 * This factory is serialized and sent to executors where it creates
 * data writers for each task.
 */
public final class VortexDataWriterFactory implements DataWriterFactory {
    
    private final String outputPath;
    private final StructType schema;
    private final CaseInsensitiveStringMap options;
    
    /**
     * Creates a new VortexDataWriterFactory.
     *
     * @param outputPath the base path where Vortex files will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     */
    public VortexDataWriterFactory(
            String outputPath,
            StructType schema,
            CaseInsensitiveStringMap options) {
        this.outputPath = outputPath;
        this.schema = schema;
        this.options = options;
    }
    
    /**
     * Creates a new data writer for a specific partition and task.
     * 
     * Each task writes its data to a separate Vortex file to avoid conflicts.
     *
     * @param partitionId the partition ID
     * @param taskId the task ID  
     * @param epochId the epoch ID (for streaming, always 0 for batch)
     * @return a new VortexDataWriter instance
     */
    @Override
    public DataWriter<InternalRow> createWriter(int partitionId, long taskId) {
        // Create a unique file name for this task
        String fileName = String.format("part-%05d-%d.vortex", partitionId, taskId);
        String filePath = outputPath + "/" + fileName;
        
        return new VortexDataWriter(filePath, schema, options);
    }