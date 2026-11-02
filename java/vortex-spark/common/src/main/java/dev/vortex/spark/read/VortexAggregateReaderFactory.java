// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.FileAggregate;
import dev.vortex.api.FileAggregateResult;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.jni.NativeFiles;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.ArrowProperties;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.Serializable;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.aggregate.AggregateFunc;
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation;
import org.apache.spark.sql.connector.expressions.aggregate.Count;
import org.apache.spark.sql.connector.expressions.aggregate.CountStar;
import org.apache.spark.sql.connector.expressions.aggregate.Max;
import org.apache.spark.sql.connector.expressions.aggregate.Min;
import org.apache.spark.sql.connector.expressions.aggregate.Sum;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.datasources.v2.EmptyPartitionReader;
import org.apache.spark.sql.execution.datasources.v2.FilePartitionReaderFactory;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Produces one partial aggregate row per Vortex file, which Spark combines across files.
 *
 * <p>An aggregate over a data column is answered natively: from the footer statistics when they state it exactly, and
 * otherwise from a scan of that column alone. An aggregate over a partition column comes from the file's partition
 * value, which every row of the file shares. A data column a file does not carry holds only nulls for that file's rows.
 *
 * <p>A listed file without {@link VortexFile#EXTENSION} is not part of the dataset and produces no row.
 */
public final class VortexAggregateReaderFactory extends FilePartitionReaderFactory implements Serializable {
    static {
        // Executors load this class before any Vortex batch, so Arrow sees the settings before it reads them.
        ArrowProperties.configure();
    }

    private static final long serialVersionUID = 1L;

    private final FileSourceOptions fileOptions;
    private final VortexIo io;
    private final VortexOptions formatOptions;
    private final StructType aggregateSchema;
    private final boolean caseSensitive;
    private final int[] groupByOrdinals;
    private final FileAggregate.Kind[] kinds;

    /** For each aggregate, the data column it reads, or null for {@code COUNT(*)} and partition columns. */
    private final String[] dataColumns;

    /** For each aggregate, the ordinal of the partition column it reads, or {@code -1}. */
    private final int[] partitionOrdinals;

    public VortexAggregateReaderFactory(
            FileSourceOptions fileOptions,
            VortexIo io,
            VortexOptions formatOptions,
            StructType aggregateSchema,
            StructType partitionSchema,
            Aggregation aggregation,
            boolean caseSensitive) {
        this.fileOptions = fileOptions;
        this.io = io;
        this.formatOptions = formatOptions;
        this.aggregateSchema = aggregateSchema;
        this.caseSensitive = caseSensitive;

        Expression[] groupBy = aggregation.groupByExpressions();
        this.groupByOrdinals = new int[groupBy.length];
        for (int i = 0; i < groupBy.length; i++) {
            groupByOrdinals[i] = partitionOrdinal(partitionSchema, columnName(groupBy[i]));
            if (groupByOrdinals[i] < 0) {
                throw new IllegalArgumentException("Pushed aggregates can only group by partition columns");
            }
        }

        AggregateFunc[] aggregates = aggregation.aggregateExpressions();
        this.kinds = new FileAggregate.Kind[aggregates.length];
        this.dataColumns = new String[aggregates.length];
        this.partitionOrdinals = new int[aggregates.length];
        for (int i = 0; i < aggregates.length; i++) {
            AggregateFunc aggregate = aggregates[i];
            Expression column;
            if (aggregate instanceof CountStar) {
                kinds[i] = FileAggregate.Kind.COUNT_STAR;
                column = null;
            } else if (aggregate instanceof Count count) {
                kinds[i] = FileAggregate.Kind.COUNT;
                column = count.column();
            } else if (aggregate instanceof Min min) {
                kinds[i] = FileAggregate.Kind.MIN;
                column = min.column();
            } else if (aggregate instanceof Max max) {
                kinds[i] = FileAggregate.Kind.MAX;
                column = max.column();
            } else if (aggregate instanceof Sum sum) {
                kinds[i] = FileAggregate.Kind.SUM;
                column = sum.column();
            } else {
                throw new IllegalArgumentException("Unsupported pushed aggregate " + aggregate);
            }

            partitionOrdinals[i] = -1;
            if (column != null) {
                String name = columnName(column);
                partitionOrdinals[i] = partitionOrdinal(partitionSchema, name);
                dataColumns[i] = partitionOrdinals[i] < 0 ? name : null;
            }
        }
    }

    private static String columnName(Expression expression) {
        if (expression instanceof NamedReference reference && reference.fieldNames().length == 1) {
            return reference.fieldNames()[0];
        }
        throw new IllegalArgumentException("Pushed aggregates must reference top-level columns, not " + expression);
    }

    private int partitionOrdinal(StructType partitionSchema, String name) {
        StructField[] fields = partitionSchema.fields();
        for (int i = 0; i < fields.length; i++) {
            if (fields[i].name().equals(name)) {
                return i;
            }
        }
        if (!caseSensitive) {
            for (int i = 0; i < fields.length; i++) {
                if (fields[i].name().equalsIgnoreCase(name)) {
                    return i;
                }
            }
        }
        return -1;
    }

    @Override
    public FileSourceOptions options() {
        return fileOptions;
    }

    @Override
    public PartitionReader<InternalRow> buildReader(PartitionedFile file) {
        throw new UnsupportedOperationException("row-based aggregate reads are not supported");
    }

    @Override
    public PartitionReader<ColumnarBatch> buildColumnarReader(PartitionedFile file) {
        if (!VortexFile.hasVortexExtension(file.toPath().getName())) {
            return new EmptyPartitionReader<>();
        }
        return new AggregateReader(file, true);
    }

    /**
     * The partial aggregate row of {@code file} as its footer states it, copied out of native memory, or empty when the
     * footer does not state every aggregate exactly. Nothing is computed from the file's data. A file that is not part
     * of the dataset has no row.
     */
    public Optional<List<InternalRow>> footerRows(PartitionedFile file) {
        if (!VortexFile.hasVortexExtension(file.toPath().getName())) {
            return Optional.of(List.of());
        }

        try (AggregateReader reader = new AggregateReader(file, false)) {
            if (!reader.next()) {
                return Optional.empty();
            }
            return Optional.of(List.of(reader.get().getRow(0).copy()));
        }
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }

    /** Emits the single partial aggregate row of one file. */
    private final class AggregateReader implements PartitionReader<ColumnarBatch> {
        private final PartitionedFile file;

        /** Whether aggregates the footer does not state exactly are computed from the file's data. */
        private final boolean computeMissing;

        private boolean emitted;
        private FileAggregateResult result;
        private ColumnarBatch batch;

        private AggregateReader(PartitionedFile file, boolean computeMissing) {
            this.file = file;
            this.computeMissing = computeMissing;
        }

        /** Loads the file's row, which is missing only when the footer leaves an aggregate it may not compute. */
        @Override
        public boolean next() {
            if (emitted) {
                return false;
            }
            emitted = true;
            batch = aggregateBatch();
            return batch != null;
        }

        private ColumnarBatch aggregateBatch() {
            Session session = VortexSparkSession.get(formatOptions);
            NativeReadable readable =
                    io.openReadable(new VortexFile(file.toPath().toString(), file.fileSize()));
            int[] slots = new int[kinds.length];
            try {
                result = aggregate(session, readable, slots);
            } catch (RuntimeException | Error e) {
                try {
                    readable.close();
                } catch (IOException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
                throw e;
            }
            try {
                readable.close();
            } catch (IOException e) {
                throw new UncheckedIOException("close the readable for " + file.toPath(), e);
            }

            if (!result.allAnswered()) {
                if (!computeMissing) {
                    return null;
                }
                throw new IllegalStateException("Vortex left an aggregate over " + file.toPath() + " unanswered");
            }
            long rowCount = ((BigIntVector) result.vector(0)).get(0);

            StructField[] fields = aggregateSchema.fields();
            ColumnVector[] vectors = new ColumnVector[fields.length];
            for (int i = 0; i < groupByOrdinals.length; i++) {
                vectors[i] = PartitionColumnVectors.create(1, fields[i], file.partitionValues(), groupByOrdinals[i]);
            }
            for (int i = 0; i < kinds.length; i++) {
                StructField field = fields[groupByOrdinals.length + i];
                if (slots[i] >= 0) {
                    vectors[groupByOrdinals.length + i] = new VortexArrowColumnVector(result.vector(slots[i]));
                } else if (partitionOrdinals[i] >= 0) {
                    vectors[groupByOrdinals.length + i] = partitionAggregate(i, field, rowCount);
                } else {
                    // The file predates the column, so each of its rows holds null there.
                    vectors[groupByOrdinals.length + i] =
                            kinds[i] == FileAggregate.Kind.COUNT ? longConstant(0) : nullConstant(field.dataType());
                }
            }
            return new ColumnarBatch(vectors, 1);
        }

        /** Answers this file's data column aggregates, recording in {@code slots} where each one landed. */
        private FileAggregateResult aggregate(Session session, NativeReadable readable, int[] slots) {
            // Both reads key the footer cache by the readable's name, so the footer is read once.
            Map<String, String> fileColumns;
            try (DataSource source = DataSource.open(session, List.of(readable), io.readConcurrency())) {
                fileColumns = fileColumns(source);
            }

            List<FileAggregate> requests = new ArrayList<>(kinds.length + 1);
            // The row count answers aggregates over partition columns, and it is free from the footer.
            requests.add(FileAggregate.countStar());
            for (int i = 0; i < kinds.length; i++) {
                slots[i] = -1;
                if (kinds[i] == FileAggregate.Kind.COUNT_STAR) {
                    slots[i] = 0;
                } else if (dataColumns[i] != null) {
                    String physical = fileColumns.get(dataColumns[i]);
                    if (physical != null) {
                        slots[i] = requests.size();
                        requests.add(new FileAggregate(kinds[i], physical));
                    }
                }
            }
            return NativeFiles.aggregate(session, readable, requests, computeMissing, ArrowAllocation.rootAllocator());
        }

        /** An aggregate over a partition column, whose value every row of the file shares. */
        private ColumnVector partitionAggregate(int aggregate, StructField field, long rowCount) {
            int ordinal = partitionOrdinals[aggregate];
            boolean valueIsNull = file.partitionValues().isNullAt(ordinal);
            if (kinds[aggregate] == FileAggregate.Kind.COUNT) {
                return longConstant(valueIsNull ? 0 : rowCount);
            }
            if (rowCount == 0 || valueIsNull) {
                return nullConstant(field.dataType());
            }
            return PartitionColumnVectors.create(1, field, file.partitionValues(), ordinal);
        }

        private Map<String, String> fileColumns(DataSource source) {
            Map<String, String> names = caseSensitive ? new HashMap<>() : new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
            for (Field field :
                    source.arrowSchema(ArrowAllocation.rootAllocator()).getFields()) {
                String previous = names.putIfAbsent(field.getName(), field.getName());
                if (previous != null) {
                    throw new IllegalArgumentException(String.format(
                            Locale.ROOT,
                            "Ambiguous columns %s and %s in %s under Spark's case sensitivity setting",
                            previous,
                            field.getName(),
                            file.toPath()));
                }
            }
            return names;
        }

        @Override
        public ColumnarBatch get() {
            if (batch == null) {
                throw new IllegalStateException("no aggregate row loaded; call next() first");
            }
            return batch;
        }

        @Override
        public void close() {
            if (batch != null) {
                batch.close();
                batch = null;
            }
            if (result != null) {
                result.close();
                result = null;
            }
        }
    }

    private static ColumnVector longConstant(long value) {
        ConstantColumnVector vector = new ConstantColumnVector(1, DataTypes.LongType);
        vector.setNotNull();
        vector.setLong(value);
        return vector;
    }

    private static ColumnVector nullConstant(DataType type) {
        ConstantColumnVector vector = new ConstantColumnVector(1, type);
        vector.setNull();
        return vector;
    }
}
