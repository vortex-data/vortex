// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Stream;
import org.apache.spark.sql.connector.write.*;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Manages the batch write operation for creating Vortex files.
 *
 * This class coordinates the distributed write operation across Spark executors,
 * handling the creation of data writers and managing commits/aborts.
 */
public final class VortexBatchWrite implements Write, BatchWrite, Serializable {

    private final String outputPath;
    private final StructType schema;
    private final CaseInsensitiveStringMap options;
    private final boolean overwrite;

    /**
     * Creates a new VortexBatchWrite.
     *
     * @param outputPath the base path where Vortex files will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     * @param overwrite whether to overwrite existing files
     */
    public VortexBatchWrite(String outputPath, StructType schema, CaseInsensitiveStringMap options, boolean overwrite) {
        this.outputPath = outputPath;
        this.schema = schema;
        this.options = options;
        this.overwrite = overwrite;
    }

    /**
     * Returns this object as a BatchWrite.
     *
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
     *
     * This method is called once at the start of the write operation,
     * making it the right place to handle overwrite cleanup.
     *
     * @return a new VortexDataWriterFactory
     */
    @Override
    public DataWriterFactory createBatchWriterFactory(PhysicalWriteInfo info) {
        // Handle overwrite cleanup BEFORE writing starts
        if (overwrite) {
            cleanupExistingFiles();
        }

        return new VortexDataWriterFactory(outputPath, schema, options);
    }

    /**
     * Cleans up existing Vortex files when overwrite mode is enabled.
     */
    private void cleanupExistingFiles() {
        try {
            Path path = Paths.get(outputPath);
            if (Files.exists(path)) {
                try (Stream<Path> walk = Files.walk(path)) {
                    walk.filter(Files::isRegularFile)
                            .filter(p -> p.toString().endsWith(".vortex"))
                            .forEach(p -> {
                                try {
                                    Files.delete(p);
                                } catch (IOException e) {
                                    throw new RuntimeException("Failed to delete file: " + p, e);
                                }
                            });
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to clean up output directory: " + outputPath, e);
        }
    }

    /**
     * Called when the write job is being prepared.
     *
     * If overwrite is enabled, this method will clean up existing files
     * at the output path.
     *
     * @param messages commit messages from the write preparation phase
     */
    @Override
    public void onDataWriterCommit(WriterCommitMessage message) {
        // Individual file commits are handled in the data writer
        // This is called for each successful task
    }

    /**
     * Commits the entire write job after all tasks complete successfully.
     *
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
            System.out.println("Successfully wrote " + writtenFiles.length + " Vortex files to " + outputPath);
        }
    }

    /**
     * Aborts the write job due to failures.
     *
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
                        System.err.println("Failed to clean up file: " + filePath);
                    }
                });
    }
}
