// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import java.io.Serializable;
import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating VortexDataWriter instances on Spark executors.
 * <p>
 * This factory is serialized and sent to executors where it creates
 * data writers for each task.
 */
public final class VortexDataWriterFactory implements DataWriterFactory, Serializable {

    private static final Logger log = LoggerFactory.getLogger(VortexDataWriterFactory.class);

    private final String outputUri;
    private final StructType schema;
    // Store options as a serializable Map instead of CaseInsensitiveStringMap
    private final Map<String, String> options;

    /**
     * Creates a new VortexDataWriterFactory.
     *
     * @param outputUri the base path where Vortex files will be written
     * @param schema    the schema of the data to write
     * @param options   additional write options
     */
    public VortexDataWriterFactory(String outputUri, StructType schema, Map<String, String> options) {
        this.outputUri = outputUri;
        this.schema = schema;
        this.options = options;
    }

    /**
     * Creates a new data writer for a specific partition and task.
     * <p>
     * Each task writes its data to a separate Vortex file to avoid conflicts.
     *
     * @param partitionId the partition ID
     * @param taskId      the task ID
     * @return a new VortexDataWriter instance
     */
    @Override
    public DataWriter<InternalRow> createWriter(int partitionId, long taskId) {
        // Create a unique file name for this task
        String fileName = String.format("part-%05d-%d.vortex", partitionId, taskId);
        String fileUri;
        if (outputUri.endsWith("/")) {
            fileUri = outputUri + fileName;
        } else {
            fileUri = outputUri + "/" + fileName;
        }

        log.debug("Creating writer for partition={} task={}", partitionId, taskId);
        log.debug("Output path: {}", outputUri);
        log.debug("File name: {}", fileName);
        log.debug("Full file path: {}", fileUri);

        // Create a new CaseInsensitiveStringMap from our serializable Map
        CaseInsensitiveStringMap optionsMap = new CaseInsensitiveStringMap(options);
        return new VortexDataWriter(fileUri, schema, optionsMap);
    }
}
