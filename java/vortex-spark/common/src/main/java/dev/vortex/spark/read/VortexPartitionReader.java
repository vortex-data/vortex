// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.Expression;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/** Columnar reader over one Spark {@link PartitionedFile}. */
public final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final PartitionedFile file;
    private final StructType readDataSchema;
    private final StructType readPartitionSchema;
    private final BufferAllocator allocator;

    /**
     * For each field of {@code readDataSchema}, its position among the columns this file returns, or {@code -1} for a
     * field the file does not carry. A merged dataset schema holds every field any file carries, so a file written
     * before a column was added returns fewer columns than the query asked for.
     */
    private final int[] vectorSlots;

    private NativeReadable readable;
    private Session session;
    private DataSource dataSource;
    private Scan scan;
    private Partition currentPartition;
    private ArrowReader currentReader;
    private ColumnarBatch currentBatch;
    private boolean batchLoaded;
    private boolean exhausted;

    public VortexPartitionReader(
            PartitionedFile file,
            StructType dataSchema,
            StructType readDataSchema,
            StructType readPartitionSchema,
            VortexIo io,
            VortexOptions formatOptions,
            Filter[] pushedFilters) {
        this.file = file;
        this.readDataSchema = readDataSchema;
        this.readPartitionSchema = readPartitionSchema;
        this.allocator = ArrowAllocation.rootAllocator();
        try {
            session = VortexSparkSession.get(formatOptions);
            readable = io.openReadable(new VortexFile(file.toPath().toString(), file.fileSize()));
            dataSource = DataSource.open(session, List.of(readable), io.readConcurrency());

            Set<String> fileFields = fieldNames(dataSource);
            List<String> projection = new ArrayList<>(readDataSchema.length());
            this.vectorSlots = new int[readDataSchema.length()];
            StructField[] fields = readDataSchema.fields();
            for (int i = 0; i < fields.length; i++) {
                if (fileFields.contains(fields[i].name())) {
                    vectorSlots[i] = projection.size();
                    projection.add(fields[i].name());
                } else if (fields[i].nullable()) {
                    vectorSlots[i] = -1;
                } else {
                    throw new IllegalArgumentException(String.format(
                            Locale.ROOT,
                            "%s does not carry the non-nullable column %s that the query requires",
                            file.toPath(),
                            fields[i].name()));
                }
            }

            var options = ScanOptions.builder();
            // Always project, an empty read schema included: a query that needs only partition columns or only a
            // row count must not pull every data column off storage.
            options.projection(Expression.select(projection.toArray(new String[0]), Expression.root()));
            // Filters are converted against this file's own columns. Spark evaluates every data filter above the
            // scan anyway, so one that reads a column this file lacks is dropped rather than pushed.
            buildFilterExpression(pushedFilters, restrictTo(dataSchema, fileFields))
                    .ifPresent(options::filter);
            scan = dataSource.scan(options.build());
        } catch (RuntimeException e) {
            closeReadableAfterFailure(e);
            throw e;
        }
    }

    private Set<String> fieldNames(DataSource source) {
        Set<String> names = new HashSet<>();
        for (Field field : source.arrowSchema(allocator).getFields()) {
            names.add(field.getName());
        }
        return names;
    }

    private static StructType restrictTo(StructType schema, Set<String> names) {
        StructField[] present = Arrays.stream(schema.fields())
                .filter(field -> names.contains(field.name()))
                .toArray(StructField[]::new);
        return present.length == schema.length() ? schema : new StructType(present);
    }

    private static Optional<Expression> buildFilterExpression(Filter[] filters, StructType dataSchema) {
        Expression combined = null;
        if (filters != null) {
            for (Filter filter : filters) {
                Optional<Expression> converted = SparkFilterToVortexExpression.convert(filter, dataSchema);
                if (converted.isPresent()) {
                    combined = combined == null ? converted.get() : Expression.and(combined, converted.get());
                }
            }
        }
        return Optional.ofNullable(combined);
    }

    @Override
    public boolean next() {
        closeCurrentBatch();
        batchLoaded = false;
        if (exhausted) {
            return false;
        }
        while (true) {
            if (currentReader != null) {
                try {
                    if (currentReader.loadNextBatch()) {
                        batchLoaded = true;
                        return true;
                    }
                } catch (IOException e) {
                    throw failure("load a batch from", e);
                }
                closeCurrentReader();
            }
            if (!scan.hasNext()) {
                exhausted = true;
                return false;
            }
            currentPartition = scan.next();
            currentReader = currentPartition.scanArrow(allocator);
        }
    }

    @Override
    public ColumnarBatch get() {
        if (!batchLoaded) {
            throw new IllegalStateException("no batch loaded; call next() first");
        }
        batchLoaded = false;
        VectorSchemaRoot root;
        try {
            root = currentReader.getVectorSchemaRoot();
        } catch (IOException e) {
            throw failure("read the loaded batch of", e);
        }

        int rowCount = root.getRowCount();
        StructField[] fields = readDataSchema.fields();
        ColumnVector[] dataVectors = new ColumnVector[fields.length];
        for (int i = 0; i < dataVectors.length; i++) {
            dataVectors[i] = vectorSlots[i] < 0
                    ? nullColumn(rowCount, fields[i].dataType())
                    : new VortexArrowColumnVector(root.getFieldVectors().get(vectorSlots[i]));
        }
        ColumnVector[] partitionVectors =
                PartitionColumnVectors.create(rowCount, readPartitionSchema, file.partitionValues());
        ColumnVector[] vectors = Arrays.copyOf(dataVectors, dataVectors.length + partitionVectors.length);
        System.arraycopy(partitionVectors, 0, vectors, dataVectors.length, partitionVectors.length);
        currentBatch = new ColumnarBatch(vectors, rowCount);
        return currentBatch;
    }

    private static ColumnVector nullColumn(int rowCount, DataType type) {
        ConstantColumnVector vector = new ConstantColumnVector(rowCount, type);
        vector.setNull();
        return vector;
    }

    @Override
    public void close() {
        closeCurrentBatch();
        closeCurrentReader();
        scan = null;
        dataSource = null;
        session = null;
        if (readable != null) {
            try {
                readable.close();
            } catch (IOException e) {
                throw failure("close the readable for", e);
            } finally {
                readable = null;
            }
        }
    }

    private RuntimeException failure(String action, IOException cause) {
        return new UncheckedIOException(action + " " + file.toPath(), cause);
    }

    private void closeCurrentBatch() {
        if (currentBatch != null) {
            currentBatch.close();
            currentBatch = null;
        }
    }

    private void closeCurrentReader() {
        if (currentReader != null) {
            try {
                currentReader.close();
            } catch (IOException e) {
                throw failure("close the reader for", e);
            } finally {
                currentReader = null;
                currentPartition = null;
            }
        }
    }

    private void closeReadableAfterFailure(RuntimeException failure) {
        if (readable != null) {
            try {
                readable.close();
            } catch (IOException e) {
                failure.addSuppressed(e);
            }
            readable = null;
        }
    }
}
