// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.jni.NativeFileMethods;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import org.apache.spark.sql.connector.write.*;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Manages the batch write operation for creating Vortex files.
 * <p>
 * This class coordinates the distributed write operation across Spark executors,
 * handling the creation of data writers and managing commits/aborts.
 */
public final class VortexBatchWrite implements Write, BatchWrite, Serializable {

    private static final Logger log = LoggerFactory.getLogger(VortexBatchWrite.class);
    private final String outputPath;
    private final StructType schema;
    private final Map<String, String> options;
    private final boolean overwrite;

    /**
     * Creates a new VortexBatchWrite.
     *
     * @param outputPath the base path where Vortex files will be written
     * @param schema     the schema of the data to write
     * @param options    additional write options
     * @param overwrite  whether to overwrite existing files
     */
    public VortexBatchWrite(String outputPath, StructType schema, Map<String, String> options, boolean overwrite) {
        this.outputPath = outputPath;
        this.schema = schema;
        this.options = options;
        this.overwrite = overwrite;
    }

    /**
     * Returns this object as a BatchWrite.
     * <p>
     * This method is required by the Write interface to support batch writes.
     *
     * @return this object
     */
    @Override
    public BatchWrite toBatch() {
        return this;
    }

    /**
     * Creates a DataWriterFactory for producing data writers on executors.
     * <p>
     * This method is called once at the start of the write operation,
     * making it the right place to handle overwrite cleanup.
     *
     * @return a new VortexDataWriterFactory
     */
    @Override
    public DataWriterFactory createBatchWriterFactory(PhysicalWriteInfo info) {
        // Handle overwrite cleanup BEFORE writing starts
        if (overwrite) {
            var uris = NativeFileMethods.listVortexFiles(outputPath, options);
            log.info("truncating table with {} files", uris.size());
            NativeFileMethods.delete(uris.toArray(new String[0]), options);
            log.warn("overwrite currently does not do anything for vortex format");
        }

        return new VortexDataWriterFactory(outputPath, schema, options);
    }

    /**
     * Called when a single data writer task completes successfully.
     * <p>
     * This is called for each successful task but individual file commits
     * are handled in the data writer itself.
     *
     * @param message commit message from a successful data writer task
     */
    @Override
    public void onDataWriterCommit(WriterCommitMessage message) {
        // Individual file commits are handled in the data writer
        // This is called for each successful task
        log.debug("Committing DataWriter");
    }

    /**
     * Commits the entire write job after all tasks complete successfully.
     * <p>
     * This finalizes the write operation and ensures all Vortex files
     * are properly written.
     *
     * @param messages commit messages from all successful write tasks
     */
    @Override
    public void commit(WriterCommitMessage[] messages) {
        // Overwrite cleanup should happen BEFORE writing, not after
        // The commit method is called AFTER files are written, so we don't delete them here

        // Extract file paths from commit messages for logging
        String[] writtenFiles = Arrays.stream(messages)
                .filter(msg -> msg instanceof VortexWriterCommitMessage)
                .map(msg -> ((VortexWriterCommitMessage) msg).getFilePath())
                .toArray(String[]::new);

        if (writtenFiles.length > 0) {
            log.info("Successfully wrote {} Vortex files to {}", writtenFiles.length, outputPath);
        }
    }

    /**
     * Aborts the write job due to failures.
     * <p>
     * This method cleans up any partially written files.
     *
     * @param messages commit messages from write tasks (may include failures)
     */
    @Override
    public void abort(WriterCommitMessage[] messages) {
        // Clean up any partially written files
        Arrays.stream(messages)
                .filter(msg -> msg instanceof VortexWriterCommitMessage)
                .map(msg -> ((VortexWriterCommitMessage) msg).getFilePath())
                .forEach(filePath -> {
                    try {
                        Path path = Paths.get(filePath);
                        if (Files.exists(path)) {
                            Files.delete(path);
                        }
                    } catch (IOException e) {
                        // Log but don't throw - we're already in an error state
                        log.error("Failed to clean up file: {}", filePath, e);
                    }
                });
    }
}
