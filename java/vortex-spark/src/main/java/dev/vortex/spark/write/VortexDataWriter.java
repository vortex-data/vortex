// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.relocated.org.apache.arrow.c.ArrowArray;
import dev.vortex.relocated.org.apache.arrow.c.ArrowSchema;
import dev.vortex.relocated.org.apache.arrow.c.Data;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.BitVector;
import dev.vortex.relocated.org.apache.arrow.vector.DateDayVector;
import dev.vortex.relocated.org.apache.arrow.vector.DecimalVector;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.Float4Vector;
import dev.vortex.relocated.org.apache.arrow.vector.Float8Vector;
import dev.vortex.relocated.org.apache.arrow.vector.IntVector;
import dev.vortex.relocated.org.apache.arrow.vector.SmallIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMicroTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMicroVector;
import dev.vortex.relocated.org.apache.arrow.vector.TinyIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.SpecializedGetters;
import org.apache.spark.sql.catalyst.util.ArrayData;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.BinaryType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.unsafe.types.UTF8String;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Writes Spark InternalRow data to a Vortex file.
 *
 * <p>This writer converts Spark's internal row format to Arrow vectors and writes them to a Vortex file using the
 * Vortex writer API.
 */
public final class VortexDataWriter implements DataWriter<InternalRow>, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(VortexDataWriter.class);

    private static final int DEFAULT_BATCH_SIZE = 2048;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 65536; // 64K rows max per batch

    private final String filePath;
    private final StructType schema;
    private final CaseInsensitiveStringMap options;
    private final int batchSize;

    private Session session;
    private VortexWriter vortexWriter;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private final List<InternalRow> batchRows = new ArrayList<>();
    private long recordCount = 0;
    private long bytesWritten = 0;
    private boolean closed = false;

    /**
     * Creates a new VortexDataWriter.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     */
    VortexDataWriter(String filePath, StructType schema, CaseInsensitiveStringMap options) {
        this.filePath = filePath;
        this.schema = schema;
        this.options = options;

        // Get batch size from options with validation
        // Users can set this with: .option("vortex.write.batch.size", "4096")
        int configuredBatchSize =
                options.getInt("vortex.write.batch.size", options.getInt("batch.size", DEFAULT_BATCH_SIZE));
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

            this.session = VortexSparkSession.get(options.asCaseSensitiveMap());
            this.vortexWriter =
                    VortexWriter.create(session, filePath, arrowSchema, options.asCaseSensitiveMap(), allocator);
            this.vectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator);

            logger.debug("Initialized VortexDataWriter for {}", filePath);

        } catch (IOException e) {
            logger.error("Failed to initialize VortexDataWriter for {}", filePath, e);
            throw new RuntimeException("Failed to initialize VortexDataWriter", e);
        }
    }

    /**
     * Writes a single row to the Vortex file.
     *
     * <p>Rows are batched and converted to Arrow format before writing.
     *
     * @param row the row to write
     * @throws IOException if writing fails
     */
    @Override
    public void write(InternalRow row) throws IOException {
        // Add row to current batch
        batchRows.add(row.copy());
        recordCount++;

        // Write batch if it's full
        if (batchRows.size() >= batchSize) {
            writeBatch();
        }
    }

    /** Writes the current batch of rows to the Vortex file. */
    private void writeBatch() throws IOException {
        if (batchRows.isEmpty()) {
            return;
        }

        // Allocate vectors and populate with data from InternalRows
        vectorSchemaRoot.allocateNew();

        // Populate each field in the schema
        StructField[] fields = schema.fields();
        for (int fieldIndex = 0; fieldIndex < fields.length; fieldIndex++) {
            FieldVector vector = vectorSchemaRoot.getVector(fieldIndex);
            DataType dataType = fields[fieldIndex].dataType();
            boolean nullable = fields[fieldIndex].nullable();

            // Populate this vector with data from all rows
            for (int rowIndex = 0; rowIndex < batchRows.size(); rowIndex++) {
                InternalRow row = batchRows.get(rowIndex);

                if (nullable && row.isNullAt(fieldIndex)) {
                    // Set null value
                    vector.setNull(rowIndex);
                } else {
                    // Set actual value based on data type
                    populateVector(vector, dataType, row, fieldIndex, rowIndex);
                }
            }
        }

        vectorSchemaRoot.setRowCount(batchRows.size());

        // Export via Arrow C Data Interface and write to Vortex
        for (FieldVector vector : vectorSchemaRoot.getFieldVectors()) {
            bytesWritten += vector.getBufferSize();
        }
        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
            Data.exportVectorSchemaRoot(allocator, vectorSchemaRoot, null, arrowArray, arrowSchema);
            vortexWriter.writeBatch(arrowArray.memoryAddress(), arrowSchema.memoryAddress());
        }

        vectorSchemaRoot.clear();
        batchRows.clear();
    }

    /** Populates an Arrow vector with a value from an InternalRow. */
    private void populateVector(
            FieldVector vector, DataType dataType, SpecializedGetters row, int fieldIndex, int rowIndex) {
        if (dataType instanceof BooleanType) {
            ((BitVector) vector).setSafe(rowIndex, row.getBoolean(fieldIndex) ? 1 : 0);
        } else if (dataType instanceof ByteType) {
            ((TinyIntVector) vector).setSafe(rowIndex, row.getByte(fieldIndex));
        } else if (dataType instanceof ShortType) {
            ((SmallIntVector) vector).setSafe(rowIndex, row.getShort(fieldIndex));
        } else if (dataType instanceof IntegerType) {
            ((IntVector) vector).setSafe(rowIndex, row.getInt(fieldIndex));
        } else if (dataType instanceof LongType) {
            ((BigIntVector) vector).setSafe(rowIndex, row.getLong(fieldIndex));
        } else if (dataType instanceof FloatType) {
            ((Float4Vector) vector).setSafe(rowIndex, row.getFloat(fieldIndex));
        } else if (dataType instanceof DoubleType) {
            ((Float8Vector) vector).setSafe(rowIndex, row.getDouble(fieldIndex));
        } else if (dataType instanceof StringType) {
            UTF8String str = row.getUTF8String(fieldIndex);
            if (str != null) {
                ((VarCharVector) vector).setSafe(rowIndex, str.getBytes());
            }
        } else if (dataType instanceof BinaryType) {
            byte[] bytes = row.getBinary(fieldIndex);
            if (bytes != null) {
                ((VarBinaryVector) vector).setSafe(rowIndex, bytes);
            }
        } else if (dataType instanceof DateType) {
            ((DateDayVector) vector).setSafe(rowIndex, row.getInt(fieldIndex));
        } else if (dataType instanceof TimestampType) {
            ((TimeStampMicroTZVector) vector).setSafe(rowIndex, row.getLong(fieldIndex));
        } else if (dataType instanceof TimestampNTZType) {
            ((TimeStampMicroVector) vector).setSafe(rowIndex, row.getLong(fieldIndex));
        } else if (dataType instanceof DecimalType decType) {
            if (decType.precision() <= 38) {
                // Use Decimal type from InternalRow
                java.math.BigDecimal decimal = row.getDecimal(fieldIndex, decType.precision(), decType.scale())
                        .toJavaBigDecimal();
                ((DecimalVector) vector).setSafe(rowIndex, decimal);
            }
        } else if (dataType instanceof StructType structType) {
            populateStructVector(
                    (StructVector) vector, structType, row.getStruct(fieldIndex, structType.fields().length), rowIndex);
        } else if (dataType instanceof ArrayType arrayType) {
            ArrayData data = row.getArray(fieldIndex);
            ListVector listVector = ((ListVector) vector);
            int writtenElements = listVector.getElementEndIndex(listVector.getLastSet());
            listVector.startNewValue(rowIndex);
            for (int i = 0; i < data.numElements(); i++) {
                populateVector(listVector.getDataVector(), arrayType.elementType(), data, i, writtenElements + i);
            }
            listVector.endValue(rowIndex, data.numElements());
        } else {
            // For unsupported types, set null
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    private void populateStructVector(StructVector vector, StructType dataType, InternalRow row, int rowIndex) {
        vector.setIndexDefined(rowIndex);

        StructField[] fields = dataType.fields();
        for (int fieldIndex = 0; fieldIndex < fields.length; fieldIndex++) {
            FieldVector childVector = (FieldVector) vector.getVectorById(fieldIndex);
            if (row.isNullAt(fieldIndex)) {
                childVector.setNull(rowIndex);
                continue;
            }
            populateVector(childVector, fields[fieldIndex].dataType(), row, fieldIndex, rowIndex);
        }
    }

    /**
     * Commits the write operation and returns a commit message.
     *
     * <p>This flushes any remaining rows and closes the Vortex writer.
     *
     * @return a commit message with file information
     * @throws IOException if commit fails
     */
    @Override
    public WriterCommitMessage commit() throws IOException {
        if (!closed) {
            IOException exception = null;

            try {
                // Write any remaining rows
                if (!batchRows.isEmpty()) {
                    writeBatch();
                }

                // Close the Vortex writer to finalize the file
                if (vortexWriter != null) {
                    try {
                        vortexWriter.close();
                    } finally {
                        vortexWriter = null; // Always null out the reference
                    }
                }
            } catch (IOException e) {
                exception = e;
            }

            // Clean up Arrow resources - always attempt cleanup even if there was an error
            try {
                if (vectorSchemaRoot != null) {
                    vectorSchemaRoot.close();
                    vectorSchemaRoot = null;
                }
            } catch (Exception e) {
                if (exception == null) {
                    exception = new IOException("Failed to close VectorSchemaRoot", e);
                } else {
                    exception.addSuppressed(e);
                }
            }

            // The Arrow C Data Interface export (Data.exportVectorSchemaRoot) creates structural
            // allocations from this allocator. When writeBatch passes the ArrowArray to Rust,
            // FFI_ArrowArray::from_raw() takes ownership and nullifies the release callback on
            // the Java side. The Rust side calls release asynchronously on its own thread, so
            // small structural allocations may still be outstanding when the allocator is closed.
            // These are reclaimed when the allocator is garbage collected.
            if (allocator != null) {
                try {
                    allocator.close();
                } catch (IllegalStateException e) {
                    logger.debug("Allocator closed with outstanding FFI allocations: {}", e.getMessage());
                }
                allocator = null;
            }

            // Session is the JVM-wide singleton held by VortexSparkSession; we just
            // drop our local handle to it here.
            session = null;

            closed = true;

            // Throw any exception that occurred during cleanup
            if (exception != null) {
                throw exception;
            }
        }

        return new VortexWriterCommitMessage(filePath, recordCount, bytesWritten);
    }

    /**
     * Aborts the write operation and cleans up resources.
     *
     * <p>This deletes any partially written file.
     *
     * @throws IOException if abort fails
     */
    @Override
    public void abort() throws IOException {
        if (!closed) {
            // Close resources
            if (vortexWriter != null) {
                try {
                    vortexWriter.close();
                } catch (Exception e) {
                    // Ignore errors during abort
                } finally {
                    vortexWriter = null; // Always null out the reference
                }
            }

            if (vectorSchemaRoot != null) {
                vectorSchemaRoot.close();
                vectorSchemaRoot = null;
            }

            if (allocator != null) {
                try {
                    allocator.close();
                } catch (IllegalStateException e) {
                    logger.debug("Allocator closed with outstanding FFI allocations: {}", e.getMessage());
                }
                allocator = null;
            }

            // Session is the JVM-wide singleton held by VortexSparkSession; we just
            // drop our local handle to it here.
            session = null;

            // Delete the partial file if it exists
            try {
                Files.deleteIfExists(Paths.get(filePath));
            } catch (IOException e) {
                // Ignore - we're already aborting
            }

            closed = true;
        }
    }

    /**
     * Closes the writer and releases resources.
     *
     * <p>This method ensures resources are cleaned up even if commit() or abort() were not called, making the class
     * safe for use with try-with-resources.
     */
    @Override
    public void close() throws IOException {
        if (!closed) {
            logger.warn("VortexDataWriter.close() called without commit() or abort() - cleaning up");
            try {
                abort();
            } catch (IOException e) {
                logger.error("Error during cleanup in close()", e);
                // Suppress the exception as we're already in close()
            }
        }
    }
}
