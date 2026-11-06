// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.DataSource.RowCount;
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
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.sources.Filter;
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
     * How the columns this file returns are laid out as {@code readDataSchema}, or {@code null} if they already are. A
     * merged dataset schema holds every field any file carries, so a file written before a column was added returns
     * fewer columns than the query asked for.
     */
    private final NestedSchemaPruning.Shape shape;

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
            Filter[] pushedFilters,
            boolean caseSensitive) {
        this.file = file;
        this.readDataSchema = readDataSchema;
        this.readPartitionSchema = readPartitionSchema;
        this.allocator = ArrowAllocation.rootAllocator();
        try {
            session = VortexSparkSession.get(formatOptions);
            readable = io.openReadable(new VortexFile(file.toPath().toString(), file.fileSize()));
            dataSource = DataSource.open(session, List.of(readable), io.readConcurrency());

            List<Field> fileColumns = dataSource.arrowSchema(allocator).getFields();
            var options = ScanOptions.builder();
            // Always project, an empty read schema included: a query that needs only partition columns or only a
            // row count must not pull every data column off storage. Struct columns that Spark pruned read only the
            // nested fields the query uses.
            NestedSchemaPruning.Projection projection = NestedSchemaPruning.projection(
                    readDataSchema, fileColumns, caseSensitive, file.toPath().toString());
            options.projection(projection.expression());
            shape = projection.shape();
            // Filters are converted against this file's own columns. Spark evaluates every data filter above the
            // scan anyway, so one that reads a column this file lacks is dropped rather than pushed.
            buildFilterExpression(pushedFilters, restrictTo(dataSchema, fileColumns))
                    .ifPresent(options::filter);
            Optional<SplitRows> rows = splitRows(file, dataSource.rowCount());
            if (rows.isPresent() && rows.get().count() == 0) {
                // The native scan reads a range of 0 to 0 as the whole file.
                exhausted = true;
            } else {
                rows.ifPresent(range -> options.rowRangeBegin(range.begin()).rowRangeEnd(range.end()));
                scan = dataSource.scan(options.build());
            }
        } catch (RuntimeException e) {
            closeReadableAfterFailure(e);
            throw e;
        }
    }

    /** The rows this split of {@code file} reads, or empty when it reads them all. */
    private static Optional<SplitRows> splitRows(PartitionedFile file, RowCount rowCount) {
        if (file.start() == 0 && file.length() >= file.fileSize()) {
            return Optional.empty();
        }
        if (!(rowCount instanceof RowCount.Exact exact)) {
            // Without an exact count no two splits can agree on a boundary, so the first reads everything.
            return file.start() == 0 ? Optional.empty() : Optional.of(new SplitRows(0, 0));
        }
        return Optional.of(SplitRows.of(file.start(), file.length(), file.fileSize(), exact.value()));
    }

    private static StructType restrictTo(StructType schema, List<Field> fileColumns) {
        // Native filters use physical field names, at every level. Filters on other paths stay with Spark.
        Map<String, Field> byName = new HashMap<>();
        for (Field column : fileColumns) {
            byName.putIfAbsent(column.getName(), column);
        }

        List<StructField> present = new ArrayList<>(schema.length());
        for (StructField field : schema.fields()) {
            Field column = byName.get(field.name());
            if (column == null) {
                continue;
            }

            if (field.dataType() instanceof StructType struct && column.getType() instanceof ArrowType.Struct) {
                present.add(new StructField(
                        field.name(), restrictTo(struct, column.getChildren()), field.nullable(), field.metadata()));
            } else {
                present.add(field);
            }
        }

        return new StructType(present.toArray(new StructField[0]));
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
        VortexArrowColumnVector[] dataVectors = root.getFieldVectors().stream()
                .map(VortexArrowColumnVector::new)
                .toArray(VortexArrowColumnVector[]::new);
        if (shape != null) {
            dataVectors = shape.layout(dataVectors, rowCount);
        }
        ColumnVector[] partitionVectors =
                PartitionColumnVectors.create(rowCount, readPartitionSchema, file.partitionValues());
        ColumnVector[] vectors =
                Arrays.copyOf(dataVectors, dataVectors.length + partitionVectors.length, ColumnVector[].class);
        System.arraycopy(partitionVectors, 0, vectors, dataVectors.length, partitionVectors.length);
        currentBatch = new ColumnarBatch(vectors, rowCount);
        return currentBatch;
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
