// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.io.NativeWritable;
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
import dev.vortex.relocated.org.apache.arrow.vector.complex.MapVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.SpecializedGetters;
import org.apache.spark.sql.catalyst.util.ArrayData;
import org.apache.spark.sql.catalyst.util.MapData;
import org.apache.spark.sql.execution.datasources.OutputWriter;
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
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.unsafe.types.UTF8String;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Writes Spark InternalRow data to a Vortex file.
 *
 * <p>This writer converts Spark's internal row format to Arrow vectors and writes them to a Vortex file using the
 * Vortex writer API.
 */
public final class VortexOutputWriter extends OutputWriter {
    private static final Logger logger = LoggerFactory.getLogger(VortexOutputWriter.class);

    /** Option sizing the row batch converted to Arrow and handed to Vortex at a time. */
    public static final String BATCH_SIZE_OPTION = "batch.size";

    /** Option sizing the write batch for Vortex alone, overriding {@value #BATCH_SIZE_OPTION}. */
    public static final String WRITE_BATCH_SIZE_OPTION = "vortex.write.batch.size";

    private static final int DEFAULT_BATCH_SIZE = 2048;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 65536; // 64K rows max per batch

    private final String filePath;
    private final StructType schema;
    private final VortexOptions options;
    private final int batchSize;
    private NativeWritable writable;

    private Session session;
    private VortexWriter vortexWriter;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private final List<InternalRow> batchRows = new ArrayList<>();
    private boolean closed = false;

    /**
     * Creates a writer for the task path assigned by Spark's commit protocol.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     */
    public VortexOutputWriter(String filePath, StructType schema, VortexOptions options, NativeWritable writable) {
        this.filePath = filePath;
        this.schema = schema;
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
                    .build();
            this.vectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator);

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
     * <p>Rows are batched and converted to Arrow format before writing.
     *
     * @param row the row to write
     */
    @Override
    public void write(InternalRow row) {
        // Add row to current batch
        batchRows.add(row.copy());

        // Write batch if it's full
        if (batchRows.size() >= batchSize) {
            try {
                writeBatch();
            } catch (IOException e) {
                throw new UncheckedIOException("Failed to write a Vortex batch to " + filePath, e);
            }
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

            // Populate this vector with data from all rows
            for (int rowIndex = 0; rowIndex < batchRows.size(); rowIndex++) {
                InternalRow row = batchRows.get(rowIndex);

                if (row.isNullAt(fieldIndex)) {
                    vector.setNull(rowIndex);
                } else {
                    populateVector(vector, dataType, row, fieldIndex, rowIndex);
                }
            }
        }

        vectorSchemaRoot.setRowCount(batchRows.size());

        // Export via Arrow C Data Interface and write to Vortex
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
            java.math.BigDecimal decimal = row.getDecimal(fieldIndex, decType.precision(), decType.scale())
                    .toJavaBigDecimal();
            ((DecimalVector) vector).setSafe(rowIndex, decimal);
        } else if (dataType instanceof StructType structType) {
            populateStructVector(
                    (StructVector) vector, structType, row.getStruct(fieldIndex, structType.fields().length), rowIndex);
        } else if (dataType instanceof ArrayType arrayType) {
            ArrayData data = row.getArray(fieldIndex);
            ListVector listVector = ((ListVector) vector);
            int writtenElements = listVector.getElementEndIndex(listVector.getLastSet());
            listVector.startNewValue(rowIndex);
            FieldVector elementVector = listVector.getDataVector();
            for (int i = 0; i < data.numElements(); i++) {
                int elementIndex = writtenElements + i;
                if (data.isNullAt(i)) {
                    // Reading a null slot of a fixed-width element returns the zeroed slot, and the
                    // typed setters mark it valid, so a null element must be written explicitly.
                    elementVector.setNull(elementIndex);
                } else {
                    populateVector(elementVector, arrayType.elementType(), data, i, elementIndex);
                }
            }
            listVector.endValue(rowIndex, data.numElements());
        } else if (dataType instanceof MapType mapType) {
            MapData data = row.getMap(fieldIndex);
            MapVector mapVector = (MapVector) vector;
            int writtenEntries = mapVector.getElementEndIndex(mapVector.getLastSet());
            mapVector.startNewValue(rowIndex);

            StructVector entries = (StructVector) mapVector.getDataVector();
            FieldVector keyVector = entries.getChild(MapVector.KEY_NAME);
            FieldVector valueVector = entries.getChild(MapVector.VALUE_NAME);
            ArrayData keys = data.keyArray();
            ArrayData values = data.valueArray();

            for (int i = 0; i < data.numElements(); i++) {
                int entryIndex = writtenEntries + i;
                entries.setIndexDefined(entryIndex);
                populateVector(keyVector, mapType.keyType(), keys, i, entryIndex);
                if (values.isNullAt(i)) {
                    valueVector.setNull(entryIndex);
                } else {
                    populateVector(valueVector, mapType.valueType(), values, i, entryIndex);
                }
            }
            mapVector.endValue(rowIndex, data.numElements());
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

    @Override
    public void close() {
        if (closed) {
            return;
        }
        IOException failure = null;
        try {
            if (!batchRows.isEmpty()) {
                writeBatch();
            }
            if (vortexWriter != null) {
                vortexWriter.finish();
            }
        } catch (IOException e) {
            failure = e;
        } finally {
            vortexWriter = null;
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
        if (failure != null) {
            throw new UncheckedIOException("Failed to close Vortex output " + filePath, failure);
        }
    }

    @Override
    public String path() {
        return filePath;
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
            try {
                allocator.close();
            } catch (IllegalStateException e) {
                logger.debug("Allocator closed with outstanding FFI allocations: {}", e.getMessage());
            } finally {
                allocator = null;
            }
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
