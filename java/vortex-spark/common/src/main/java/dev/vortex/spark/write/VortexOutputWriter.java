// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.io.NativeWritable;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Map;
import org.apache.spark.TaskContext;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.datasources.OutputWriter;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Writes Spark InternalRow data to a Vortex file.
 *
 * <p>Each row is appended straight into a batch of Arrow vectors as it arrives, and a full batch is handed to the
 * Vortex writer.
 */
public final class VortexOutputWriter extends OutputWriter {
    private static final Logger logger = LoggerFactory.getLogger(VortexOutputWriter.class);

    /** Option sizing the row batch converted to Arrow and handed to Vortex at a time. */
    public static final String BATCH_SIZE_OPTION = "batch.size";

    /** Option sizing the write batch for Vortex alone, overriding {@value #BATCH_SIZE_OPTION}. */
    public static final String WRITE_BATCH_SIZE_OPTION = "vortex.write.batch.size";

    /**
     * Boolean option that writes with the compact strategy, which prefers a smaller file over read throughput. Off by
     * default.
     */
    public static final String COMPACT_OPTION = "vortex.write.compact";

    private static final int DEFAULT_BATCH_SIZE = 2048;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 65536; // 64K rows max per batch
    private static final int INITIAL_VECTOR_CAPACITY = 64;

    private final String filePath;
    private final VortexOptions options;
    private final int batchSize;
    private NativeWritable writable;

    private Session session;
    private VortexWriter vortexWriter;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private ArrowFieldWriter<?>[] fieldWriters;
    private int batchRowCount = 0;
    private Exception writeFailure;
    private boolean closed = false;

    /**
     * Creates a writer for the task path assigned by Spark's commit protocol.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     */
    public VortexOutputWriter(String filePath, StructType schema, VortexOptions options, NativeWritable writable) {
        this(filePath, schema, options, writable, Map.of());
    }

    /**
     * Creates a writer that also stores {@code metadata} in the file footer, for callers that write files outside
     * Spark's commit protocol.
     *
     * @param metadata footer segments, read back verbatim by {@code NativeFiles.readMetadata}
     */
    public VortexOutputWriter(
            String filePath,
            StructType schema,
            VortexOptions options,
            NativeWritable writable,
            Map<String, byte[]> metadata) {
        this.filePath = filePath;
        this.options = options;
        this.writable = writable;

        int configuredBatchSize = configuredBatchSize(options);
        if (configuredBatchSize < MIN_BATCH_SIZE || configuredBatchSize > MAX_BATCH_SIZE) {
            logger.warn(
                    "Batch size {} is out of valid range [{}, {}], using default: {}",
                    configuredBatchSize,
                    MIN_BATCH_SIZE,
                    MAX_BATCH_SIZE,
                    DEFAULT_BATCH_SIZE);
            this.batchSize = DEFAULT_BATCH_SIZE;
        } else {
            this.batchSize = configuredBatchSize;
            if (this.batchSize != DEFAULT_BATCH_SIZE) {
                logger.debug("Using configured batch size: {}", this.batchSize);
            }
        }

        try {
            this.allocator = new RootAllocator();
            var arrowSchema = SparkToArrowSchema.convert(schema);

            this.session = VortexSparkSession.get(options);
            this.vortexWriter = VortexWriter.builder(session, writable, arrowSchema, allocator)
                    .metadata(metadata)
                    .compact(options.getBoolean(COMPACT_OPTION, false))
                    .build();
            this.vectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator);
            // The native writer holds several batches at once, so a batch of few rows must not take Arrow's default
            // capacity. Vectors grow on demand and keep their largest capacity for later batches.
            int initialCapacity = Math.min(batchSize, INITIAL_VECTOR_CAPACITY);
            for (FieldVector vector : vectorSchemaRoot.getFieldVectors()) {
                vector.setInitialCapacity(initialCapacity);
            }
            StructField[] fields = schema.fields();
            this.fieldWriters = new ArrowFieldWriter<?>[fields.length];
            for (int i = 0; i < fields.length; i++) {
                fieldWriters[i] = ArrowFieldWriter.create(vectorSchemaRoot.getVector(i), fields[i].dataType());
            }

            logger.debug("Initialized VortexOutputWriter for {}", filePath);

        } catch (IOException e) {
            closeAfterInitializationFailure(e);
            throw new UncheckedIOException("Failed to initialize VortexOutputWriter for " + filePath, e);
        }
    }

    /**
     * The batch size this writer was asked for.
     *
     * <p>{@value #BATCH_SIZE_OPTION} is the generic name, shared with whatever else a job writes.
     * {@value #WRITE_BATCH_SIZE_OPTION} overrides it, so a job that sets one batch size across formats can still say
     * something different for Vortex.
     *
     * <p>Package-private so the precedence between the two can be tested without a native writer.
     */
    static int configuredBatchSize(VortexOptions options) {
        if (options.get(WRITE_BATCH_SIZE_OPTION).isPresent()) {
            return options.getInt(WRITE_BATCH_SIZE_OPTION, DEFAULT_BATCH_SIZE);
        }
        return options.getInt(BATCH_SIZE_OPTION, DEFAULT_BATCH_SIZE);
    }

    /**
     * Writes a single row to the Vortex file.
     *
     * <p>The row is consumed before this method returns: Spark reuses the row object for the next call.
     *
     * @param row the row to write
     */
    @Override
    public void write(InternalRow row) {
        if (writeFailure != null) {
            throw new IllegalStateException(
                    "VortexOutputWriter for " + filePath + " failed on an earlier write", writeFailure);
        }
        try {
            if (batchRowCount == 0) {
                // The native writer keeps the buffers of every batch it is handed, so each batch needs new ones.
                // ArrowFieldWriter also needs them zeroed, since it leaves null slots untouched.
                vectorSchemaRoot.allocateNew();
            }
            for (int i = 0; i < fieldWriters.length; i++) {
                fieldWriters[i].write(row, i);
            }
            batchRowCount++;
            if (batchRowCount >= batchSize) {
                writeBatch();
            }
        } catch (IOException e) {
            writeFailure = e;
            throw new UncheckedIOException("Failed to write a Vortex batch to " + filePath, e);
        } catch (RuntimeException e) {
            // A failure part-way through a row leaves the columns of the batch misaligned.
            writeFailure = e;
            throw e;
        }
    }

    /** Hands the rows appended since the last batch to the Vortex writer. */
    private void writeBatch() throws IOException {
        if (batchRowCount == 0) {
            return;
        }
        vectorSchemaRoot.setRowCount(batchRowCount);

        vortexWriter.writeBatch(vectorSchemaRoot, allocator);

        vectorSchemaRoot.clear();
        for (ArrowFieldWriter<?> writer : fieldWriters) {
            writer.reset();
        }
        batchRowCount = 0;
    }

    /**
     * Flushes the last batch and finalizes the file.
     *
     * <p>After a failed {@link #write}, or in a task that has already failed, the last batch is dropped because the
     * file is never committed. In both cases this releases everything and logs what fails instead of throwing, because
     * Spark's abort closes a task's writers one after another and stops at the first that throws.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }

        boolean aborting = writeFailure != null || taskFailed();
        IOException failure = null;

        if (!aborting) {
            try {
                writeBatch();
            } catch (IOException e) {
                failure = e;
            } catch (RuntimeException e) {
                failure = new IOException("Failed to write the last Vortex batch", e);
            }
        }

        if (vortexWriter != null) {
            try {
                vortexWriter.finish();
            } catch (IOException e) {
                failure = addFailure(failure, e);
            } finally {
                vortexWriter = null;
            }
        }

        failure = closeArrowResources(failure);

        if (writable != null) {
            try {
                writable.close();
            } catch (IOException e) {
                failure = addFailure(failure, e);
            } finally {
                writable = null;
            }
        }

        session = null;
        closed = true;

        if (failure != null && aborting) {
            logger.warn("Failed to close the writer for {} in a failed task", filePath, failure);
            return;
        }

        if (failure != null) {
            throw new UncheckedIOException("Failed to close Vortex output " + filePath, failure);
        }
    }

    @Override
    public String path() {
        return filePath;
    }

    /** Spark marks a task failed before its abort closes the task's writers. */
    private static boolean taskFailed() {
        TaskContext context = TaskContext.get();
        return context != null && context.isFailed();
    }

    /** Package-private so tests can see what close leaves allocated. */
    BufferAllocator allocator() {
        return allocator;
    }

    private IOException closeArrowResources(IOException failure) {
        if (vectorSchemaRoot != null) {
            try {
                vectorSchemaRoot.close();
            } catch (RuntimeException e) {
                failure = addFailure(failure, new IOException("Failed to close VectorSchemaRoot", e));
            } finally {
                vectorSchemaRoot = null;
            }
        }
        if (allocator != null) {
            long leftover = allocator.getAllocatedMemory();
            if (leftover == 0) {
                allocator.close();
            } else {
                // Left open so a buffer the native side releases late is still freed.
                logger.warn(
                        "{} bytes of Arrow memory were still allocated when the writer for {} closed",
                        leftover,
                        filePath);
            }
            allocator = null;
        }
        return failure;
    }

    private void closeAfterInitializationFailure(IOException failure) {
        closeArrowResources(failure);
        try {
            writable.close();
        } catch (IOException e) {
            failure.addSuppressed(e);
        }
        writable = null;
    }

    private static IOException addFailure(IOException failure, IOException additional) {
        if (failure == null) {
            return additional;
        }
        failure.addSuppressed(additional);
        return failure;
    }
}
