// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.jni.NativeFiles;
import dev.vortex.spark.VortexSparkSession;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.write.BatchWrite;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.connector.write.PhysicalWriteInfo;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Manages the batch write operation for creating Vortex files.
 *
 * <p>This class coordinates the distributed write operation across Spark executors, handling the creation of data
 * writers and managing commits/aborts.
 */
public final class VortexBatchWrite implements Write, BatchWrite, Serializable {

    private static final Logger log = LoggerFactory.getLogger(VortexBatchWrite.class);
    private final String outputPath;
    private final StructType schema;
    private final Map<String, String> options;
    private final boolean overwrite;
    // Resolved eagerly so that Spark Transform objects (Scala case classes that are not
    // Java-serializable) never reach the DataWriterFactory serialization boundary.
    private final PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms;

    /**
     * Creates a new VortexBatchWrite.
     *
     * @param outputPath the base path where Vortex files will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     * @param overwrite whether to overwrite existing files
     * @param partitionTransforms partition transforms (may be empty)
     */
    VortexBatchWrite(
            String outputPath,
            StructType schema,
            Map<String, String> options,
            boolean overwrite,
            Transform[] partitionTransforms) {
        this.outputPath = outputPath;
        this.schema = schema;
        this.options = options;
        this.overwrite = overwrite;
        this.resolvedTransforms = PartitionedVortexDataWriter.resolveTransforms(partitionTransforms, schema);
    }

    /**
     * Returns this object as a BatchWrite.
     *
     * <p>This method is required by the Write interface to support batch writes.
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
     * <p>This method is called once at the start of the write operation, making it the right place to handle overwrite
     * cleanup.
     *
     * @return a new VortexDataWriterFactory
     */
    @Override
    public DataWriterFactory createBatchWriterFactory(PhysicalWriteInfo info) {
        // Handle overwrite cleanup BEFORE writing starts
        if (overwrite) {
            var session = VortexSparkSession.get(options);
            var uris = NativeFiles.listFiles(session, outputPath, options);
            // Deleting the existing files is destructive and happens before the new data is written:
            // if the subsequent write fails, abort() only removes the newly written files and cannot
            // restore what was deleted here. Log loudly so operators can see what was removed.
            log.warn(
                    "Deleting {} existing file(s) under {} because of overwrite, before writing new data; "
                            + "this cannot be undone if the subsequent write fails",
                    uris.size(),
                    outputPath);
            NativeFiles.delete(session, uris.toArray(new String[0]), options);
        }

        return new VortexDataWriterFactory(outputPath, schema, options, resolvedTransforms);
    }

    /**
     * Called when a single data writer task completes successfully.
     *
     * <p>This is called for each successful task but individual file commits are handled in the data writer itself.
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
     *
     * <p>This finalizes the write operation and ensures all Vortex files are properly written.
     *
     * @param messages commit messages from all successful write tasks
     */
    @Override
    public void commit(WriterCommitMessage[] messages) {
        List<String> writtenFiles = extractFilePaths(messages);

        if (!writtenFiles.isEmpty()) {
            log.info("Successfully wrote {} Vortex files to {}", writtenFiles.size(), outputPath);
        }
    }

    /**
     * Aborts the write job due to failures.
     *
     * <p>Deletes the files the tasks reported, through the same native filesystem layer the writers used, so that URL
     * paths ({@code file://}, {@code s3://}) resolve the same way on cleanup as they did on write.
     *
     * @param messages commit messages from write tasks (may include failures)
     */
    @Override
    public void abort(WriterCommitMessage[] messages) {
        List<String> filePaths = extractFilePaths(messages);
        if (filePaths.isEmpty()) {
            return;
        }
        log.warn("Deleting {} file(s) written before the job failed, under {}", filePaths.size(), outputPath);
        try {
            NativeFiles.delete(VortexSparkSession.get(options), filePaths.toArray(new String[0]), options);
        } catch (RuntimeException e) {
            log.error("Failed to clean up {} file(s) under {}", filePaths.size(), outputPath, e);
        }
    }

    private static List<String> extractFilePaths(WriterCommitMessage[] messages) {
        return Arrays.stream(messages)
                .flatMap(msg -> {
                    if (msg instanceof VortexWriterCommitMessage) {
                        return Stream.of(((VortexWriterCommitMessage) msg).filePath());
                    } else if (msg instanceof PartitionedVortexDataWriter.PartitionedWriterCommitMessage) {
                        return ((PartitionedVortexDataWriter.PartitionedWriterCommitMessage) msg)
                                .getPartitionMessages().stream().map(VortexWriterCommitMessage::filePath);
                    }
                    return Stream.empty();
                })
                .collect(Collectors.toList());
    }
}
