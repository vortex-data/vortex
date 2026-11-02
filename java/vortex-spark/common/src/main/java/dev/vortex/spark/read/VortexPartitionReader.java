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
import dev.vortex.jni.NativeFiles;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.UInt8Vector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.SparkVariants;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.TreeMap;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.roaringbitmap.longlong.Roaring64NavigableMap;

/**
 * Columnar reader over one Spark {@link PartitionedFile}.
 *
 * <p>A file split by Spark into byte ranges is read as the matching range of rows: split {@code [start, end)} of a file
 * of {@code size} bytes and {@code rows} rows reads rows {@code [rows * start / size, rows * end / size)}. The splits
 * of a file tile its bytes, so their row ranges tile its rows, and the scan reads only the chunks a range overlaps.
 */
public final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    /**
     * The column through which Spark asks a file format for each row's index in its file, matching
     * {@code ParquetFileFormat.ROW_INDEX_TEMPORARY_COLUMN_NAME}. Spark moves it into {@code _metadata.row_index}.
     */
    public static final String ROW_INDEX_COLUMN = "_tmp_metadata_row_index";

    /** Prefixes the projected names of extracted variant parts, which no file column carries. */
    private static final String EXTRACTED_PREFIX = "__vortex_variant_";

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

    /** The field of {@code readDataSchema} that holds row indexes, or {@code -1}. */
    private final int rowIndexField;

    /** Whether decimal columns of at most 18 digits arrive as the longs of their unscaled values. */
    private final boolean unscaledSmallDecimals;

    /** For each field of {@code readDataSchema} with pushed variant extractions, the fields it extracts. */
    private final Map<Integer, List<SparkVariants.VariantField>> variantExtractions = new HashMap<>();

    /** The pushed-down top-N this reader tightens, or null. */
    private final TopNThreshold topN;

    /** The field of {@code readDataSchema} holding the top-N sort keys, or {@code -1}. */
    private int topNField = -1;

    private NativeReadable readable;
    private Session session;
    private DataSource dataSource;
    private Scan scan;
    private Partition currentPartition;
    private ArrowReader currentReader;
    private ColumnarBatch currentBatch;
    private boolean batchLoaded;
    private boolean exhausted;

    /** The metrics of the task reading this file, which this reader adds to. */
    private final VortexScanMetrics.Task metrics;

    /**
     * @param pushedFilters filters Vortex evaluates while reading. Spark evaluates every data filter again above the
     *     scan, so a filter here only narrows what is read.
     * @param limit the most rows any reader of the scan needs to return. Spark applies the limit again above the scan.
     * @param topN the pushed-down top-N of the task, shared by its readers, or null. Spark sorts and limits again above
     *     the scan.
     * @param sample the pushed-down table sample, or null. Spark does not sample again.
     * @param catalystPredicates data filters without a V1 form that {@link CatalystPredicates} converts. Spark
     *     evaluates them again above the scan.
     * @param firstRow the first row of an unsplit file to read, which a pushed-down {@code OFFSET} moves past the rows
     *     it skips, or 0
     * @param ordered whether batches must keep the file's row order, because the scan reports it to Spark
     * @param metrics the metrics of the task reading the file, which the reader adds to
     */
    public VortexPartitionReader(
            PartitionedFile file,
            StructType dataSchema,
            StructType readDataSchema,
            StructType readPartitionSchema,
            VortexIo io,
            VortexOptions formatOptions,
            Filter[] pushedFilters,
            boolean caseSensitive,
            OptionalLong limit,
            TopNThreshold topN,
            VortexSample sample,
            org.apache.spark.sql.catalyst.expressions.Expression[] catalystPredicates,
            long firstRow,
            boolean ordered,
            VortexScanMetrics.Task metrics) {
        this.file = file;
        this.topN = topN;
        this.readDataSchema = readDataSchema;
        this.readPartitionSchema = readPartitionSchema;
        this.allocator = ArrowAllocation.rootAllocator();
        this.metrics = metrics;
        metrics.fileOpened();
        long openStart = System.nanoTime();
        try {
            session = VortexSparkSession.get(formatOptions);
            readable = io.openReadable(new VortexFile(file.toPath().toString(), file.fileSize()));
            dataSource = DataSource.open(session, List.of(readable), io.readConcurrency());

            Map<String, Field> fileFields = fieldsByName(dataSource, caseSensitive);
            StructField[] fields = readDataSchema.fields();
            List<String> projection = new ArrayList<>(fields.length);
            List<StructField> projectedFields = new ArrayList<>(fields.length);
            // Computed columns follow the projected file columns. Their slots count from the first of them until
            // the number of projected columns is known.
            List<String> extraNames = new ArrayList<>();
            List<Expression> extraColumns = new ArrayList<>();
            boolean[] extra = new boolean[fields.length];
            int rowIndex = -1;
            this.vectorSlots = new int[fields.length];
            for (int i = 0; i < fields.length; i++) {
                if (isRowIndex(fields[i], dataSchema)) {
                    rowIndex = i;
                    extra[i] = true;
                    vectorSlots[i] = extraNames.size();
                    extraNames.add(ROW_INDEX_COLUMN);
                    extraColumns.add(Expression.rowIdx());
                    continue;
                }

                Field fileField = fileFields.get(fields[i].name());
                if (fileField == null) {
                    if (!fields[i].nullable()) {
                        throw new IllegalArgumentException(String.format(
                                Locale.ROOT,
                                "%s does not carry the non-nullable column %s that the query requires",
                                file.toPath(),
                                fields[i].name()));
                    }
                    vectorSlots[i] = -1;
                    continue;
                }

                Optional<List<SparkVariants.VariantField>> extracted = SparkVariants.isVariant(fileField)
                        ? SparkVariants.extractedFields(fields[i].dataType())
                        : Optional.empty();
                if (extracted.isPresent()) {
                    // Vortex extracts the variant at each path, so only those parts leave the file, and Spark's cast
                    // turns each into the requested type in `get`. The last slot says which variants are null.
                    Expression variant = Expression.column(fileField.getName());
                    extra[i] = true;
                    vectorSlots[i] = extraNames.size();
                    List<SparkVariants.VariantField> parts = extracted.get();
                    for (int part = 0; part < parts.size(); part++) {
                        extraNames.add(EXTRACTED_PREFIX + i + "_" + part);
                        extraColumns.add(Expression.variantGet(
                                variant,
                                parts.get(part).names(),
                                parts.get(part).indexes()));
                    }
                    extraNames.add(EXTRACTED_PREFIX + i + "_present");
                    extraColumns.add(Expression.isNotNull(variant));
                    variantExtractions.put(i, parts);
                    continue;
                }

                vectorSlots[i] = projection.size();
                projection.add(fileField.getName());
                projectedFields.add(fields[i]);
            }
            for (int i = 0; i < fields.length; i++) {
                if (extra[i]) {
                    vectorSlots[i] += projection.size();
                }
            }
            this.rowIndexField = rowIndex;
            this.unscaledSmallDecimals = smallDecimalsMatch(projectedFields, fileFields);

            var options = ScanOptions.builder();
            // Always project, an empty read schema included: a query that needs only partition columns or only a
            // row count must not pull every data column off storage. Struct columns that Spark pruned read only the
            // nested fields the query uses.
            options.projection(NestedSchemaPruning.projection(
                    projection.toArray(new String[0]),
                    new StructType(projectedFields.toArray(new StructField[0])),
                    dataSchema,
                    extraNames,
                    extraColumns));
            // Filters are converted against this file's own columns. Spark evaluates every data filter above the
            // scan anyway, so one that reads a column this file lacks is dropped rather than pushed.
            StructType fileDataSchema = restrictTo(dataSchema, fileFields);
            Optional<Expression> filter = buildFilterExpression(pushedFilters, fileDataSchema);
            for (org.apache.spark.sql.catalyst.expressions.Expression predicate : catalystPredicates) {
                Optional<Expression> converted = CatalystPredicates.convert(predicate, fileDataSchema);
                if (converted.isPresent()) {
                    filter = Optional.of(
                            filter.map(f -> Expression.and(f, converted.get())).orElse(converted.get()));
                }
            }
            Optional<Expression> topNFilter = topNFilter(fileFields, caseSensitive);
            if (topNFilter.isPresent()) {
                filter = Optional.of(
                        filter.map(f -> Expression.and(f, topNFilter.get())).orElse(topNFilter.get()));
            }
            filter.ifPresent(options::filter);
            limit.ifPresent(options::limit);
            options.ordered(ordered);

            boolean split = isSplit(file);
            if (split || sample != null || firstRow > 0) {
                long rowCount = exactRowCount(dataSource);
                long begin = split ? splitRow(rowCount, file.start(), file.fileSize()) : Math.min(firstRow, rowCount);
                long end = split ? splitRow(rowCount, file.start() + file.length(), file.fileSize()) : rowCount;
                if (begin == end) {
                    // No row starts in this split. The native scan reads an empty range as the whole file.
                    exhausted = true;
                    metrics.splitSkipped();
                    return;
                }
                if (split || firstRow > 0) {
                    options.rowRangeBegin(begin).rowRangeEnd(end);
                }
                if (sample != null) {
                    String path = file.toPath().toString();
                    Roaring64NavigableMap rows = sample.blocks()
                            ? sample.selectBlocks(path, NativeFiles.splits(session, readable), begin, end)
                            : sample.select(path, begin, end);
                    if (rows.isEmpty()) {
                        exhausted = true;
                        metrics.splitSkipped();
                        return;
                    }
                    options.selectionRoaringBitmap(VortexSample.portable(rows))
                            .selectionMode(ScanOptions.SelectionMode.INCLUDE_ROARING);
                }
            }
            scan = dataSource.scan(options.build());
        } catch (RuntimeException e) {
            closeReadableAfterFailure(e);
            throw e;
        } finally {
            metrics.addOpenNanos(System.nanoTime() - openStart);
        }
    }

    /** The filter of the top-N over this file's sort column, noting which field holds the keys. */
    private Optional<Expression> topNFilter(Map<String, Field> fileFields, boolean caseSensitive) {
        if (topN == null) {
            return Optional.empty();
        }
        StructField[] fields = readDataSchema.fields();
        for (int i = 0; i < fields.length; i++) {
            String name = fields[i].name();
            boolean matches = caseSensitive
                    ? name.equals(topN.topN().column())
                    : name.equalsIgnoreCase(topN.topN().column());
            Field fileField = fileFields.get(name);
            if (matches && fileField != null && i != rowIndexField) {
                topNField = i;
                return Optional.of(topN.filter(Expression.column(fileField.getName())));
            }
        }
        // A file without the column holds only nulls there, which a filter on values would drop.
        return Optional.empty();
    }

    private static boolean isRowIndex(StructField field, StructType dataSchema) {
        return field.name().equals(ROW_INDEX_COLUMN)
                && DataTypes.LongType.equals(field.dataType())
                && Arrays.stream(dataSchema.fieldNames()).noneMatch(ROW_INDEX_COLUMN::equals);
    }

    /**
     * Whether every decimal column of at most 18 digits among {@code projected} has, in the file, the precision and
     * scale Spark reads it as. The native scan then sends such a column as the longs of its unscaled values, which keep
     * no precision or scale of their own.
     */
    private static boolean smallDecimalsMatch(List<StructField> projected, Map<String, Field> fileFields) {
        for (StructField field : projected) {
            if (fileFields.get(field.name()).getType() instanceof ArrowType.Decimal fileDecimal
                    && fileDecimal.getPrecision() <= Decimal.MAX_LONG_DIGITS()
                    && !(field.dataType() instanceof DecimalType decimal
                            && decimal.precision() == fileDecimal.getPrecision()
                            && decimal.scale() == fileDecimal.getScale())) {
                return false;
            }
        }
        return true;
    }

    private static boolean isSplit(PartitionedFile file) {
        return file.start() != 0 || file.length() != file.fileSize();
    }

    private long exactRowCount(DataSource source) {
        if (source.rowCount() instanceof DataSource.RowCount.Exact exact) {
            return exact.value();
        }
        throw new IllegalStateException(String.format(
                Locale.ROOT,
                "%s states no exact row count, so its split at byte %d cannot be mapped to rows",
                file.toPath(),
                file.start()));
    }

    /** The first row of the split that starts at {@code offset}, of a file of {@code size} bytes. */
    static long splitRow(long rowCount, long offset, long size) {
        if (offset >= size) {
            return rowCount;
        }
        // Rows times bytes can exceed a long, while the quotient never exceeds the row count.
        return BigInteger.valueOf(rowCount)
                .multiply(BigInteger.valueOf(offset))
                .divide(BigInteger.valueOf(size))
                .longValueExact();
    }

    /**
     * The first byte of a split that starts at {@code row}, of a file of {@code size} bytes and {@code rowCount} rows:
     * the smallest offset that {@link #splitRow} maps to {@code row} or later. It maps to {@code row} itself when the
     * file has no more rows than bytes.
     */
    static long splitOffset(long rowCount, long row, long size) {
        if (row >= rowCount) {
            return size;
        }

        BigInteger[] quotient = BigInteger.valueOf(row)
                .multiply(BigInteger.valueOf(size))
                .divideAndRemainder(BigInteger.valueOf(rowCount));
        return quotient[0].longValueExact() + (quotient[1].signum() == 0 ? 0 : 1);
    }

    private Map<String, Field> fieldsByName(DataSource source, boolean caseSensitive) {
        Map<String, Field> fields = caseSensitive ? new HashMap<>() : new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        for (Field field : source.arrowSchema(allocator).getFields()) {
            Field previous = fields.putIfAbsent(field.getName(), field);
            if (previous != null) {
                throw new IllegalArgumentException(String.format(
                        Locale.ROOT,
                        "Ambiguous columns %s and %s in %s under Spark's case sensitivity setting",
                        previous.getName(),
                        field.getName(),
                        file.toPath()));
            }
        }
        return fields;
    }

    private static StructType restrictTo(StructType schema, Map<String, Field> fields) {
        // Native filters use physical field names. Filters with different casing stay with Spark.
        StructField[] present = Arrays.stream(schema.fields())
                .filter(field -> {
                    Field fileField = fields.get(field.name());
                    return fileField != null && field.name().equals(fileField.getName());
                })
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
        long start = System.nanoTime();
        try {
            while (true) {
                if (currentReader != null) {
                    try {
                        if (currentReader.loadNextBatch()) {
                            batchLoaded = true;
                            metrics.batchRead();
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
                currentReader = currentPartition.scanArrow(allocator, unscaledSmallDecimals);
            }
        } finally {
            metrics.addScanNanos(System.nanoTime() - start);
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
        List<FieldVector> vectors = root.getFieldVectors();
        for (int i = 0; i < dataVectors.length; i++) {
            if (vectorSlots[i] < 0) {
                dataVectors[i] = nullColumn(rowCount, fields[i].dataType());
            } else if (variantExtractions.containsKey(i)) {
                List<SparkVariants.VariantField> parts = variantExtractions.get(i);
                ColumnVector[] extracted = new ColumnVector[parts.size()];
                for (int part = 0; part < extracted.length; part++) {
                    extracted[part] = new VortexArrowColumnVector(vectors.get(vectorSlots[i] + part));
                }
                ColumnVector present = new VortexArrowColumnVector(vectors.get(vectorSlots[i] + parts.size()));
                dataVectors[i] =
                        SparkVariants.extract((StructType) fields[i].dataType(), parts, extracted, present, rowCount);
            } else if (i == rowIndexField) {
                dataVectors[i] = VortexArrowColumnVector.unsignedAsLong((UInt8Vector) vectors.get(vectorSlots[i]));
            } else if (unscaledSmallDecimals
                    && fields[i].dataType() instanceof DecimalType decimal
                    && vectors.get(vectorSlots[i]) instanceof BigIntVector unscaled) {
                dataVectors[i] = VortexArrowColumnVector.unscaledDecimal(unscaled, decimal);
            } else {
                dataVectors[i] = new VortexArrowColumnVector(vectors.get(vectorSlots[i]));
            }
        }
        if (topNField >= 0) {
            topN.offer(dataVectors[topNField], rowCount);
        }
        ColumnVector[] partitionVectors =
                PartitionColumnVectors.create(rowCount, readPartitionSchema, file.partitionValues());
        ColumnVector[] batchVectors = Arrays.copyOf(dataVectors, dataVectors.length + partitionVectors.length);
        System.arraycopy(partitionVectors, 0, batchVectors, dataVectors.length, partitionVectors.length);
        currentBatch = new ColumnarBatch(batchVectors, rowCount);
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
