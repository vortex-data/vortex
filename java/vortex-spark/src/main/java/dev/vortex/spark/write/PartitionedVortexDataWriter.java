// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import com.google.common.collect.ImmutableList;
import com.google.common.primitives.ImmutableIntArray;
import java.io.IOException;
import java.io.Serializable;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.hadoop.shaded.com.google.common.collect.Streams;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.BoundReference;
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Literal;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.BinaryType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
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
 * Writes Spark InternalRow data to Vortex files organized in Hive-style partition directories.
 *
 * <p>Supports the standard Spark partition transforms: {@code identity}, {@code years}, {@code months}, {@code days},
 * {@code hours}, and {@code bucket}. For each unique combination of evaluated transform values, a separate subdirectory
 * is created and a dedicated {@link VortexDataWriter} writes data within it.
 */
public final class PartitionedVortexDataWriter implements DataWriter<InternalRow>, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(PartitionedVortexDataWriter.class);
    private static final String HIVE_DEFAULT_PARTITION = "__HIVE_DEFAULT_PARTITION__";

    private final String baseOutputUri;
    private final StructType dataSchema;
    private final UnsafeProjection dataProjection;
    private final CaseInsensitiveStringMap options;
    private final ResolvedTransform[] resolvedTransforms;
    private final int partitionId;
    private final long taskId;

    private final Map<String, VortexDataWriter> writers = new HashMap<>();
    private boolean closed = false;

    /**
     * Creates a new PartitionedVortexDataWriter.
     *
     * @param baseOutputUri the base output path
     * @param schema the full schema of the data
     * @param options write options
     * @param resolvedTransforms pre-resolved partition transforms
     * @param partitionId the Spark partition ID
     * @param taskId the Spark task ID
     */
    PartitionedVortexDataWriter(
            String baseOutputUri,
            StructType schema,
            CaseInsensitiveStringMap options,
            ResolvedTransform[] resolvedTransforms,
            int partitionId,
            long taskId) {
        this.baseOutputUri = baseOutputUri.endsWith("/") ? baseOutputUri : baseOutputUri + "/";
        this.options = options;
        this.partitionId = partitionId;
        this.taskId = taskId;
        this.resolvedTransforms = resolvedTransforms;

        // Compute the data schema by removing identity partition columns.
        // Only identity transforms correspond to columns that should be stripped from the data,
        // since temporal/bucket transforms derive values from the source column.
        Set<Integer> identityPartitionIndices = new HashSet<>();
        for (ResolvedTransform rt : resolvedTransforms) {
            if ("identity".equals(rt.transformName())) {
                identityPartitionIndices.add(rt.columnIndices().get(0));
            }
        }

        StructField[] fields = schema.fields();
        List<StructField> dataFields = new ArrayList<>();
        List<org.apache.spark.sql.catalyst.expressions.Expression> projExprs = new ArrayList<>();
        for (int i = 0; i < fields.length; i++) {
            if (!identityPartitionIndices.contains(i)) {
                dataFields.add(fields[i]);
                projExprs.add(new BoundReference(i, fields[i].dataType(), fields[i].nullable()));
            }
        }
        this.dataSchema = new StructType(dataFields.toArray(new StructField[0]));
        this.dataProjection = UnsafeProjection.create(asScalaSeq(projExprs));
    }

    @SuppressWarnings("deprecation") // JavaConverters is deprecated in Scala 2.13 but works in both 2.12 and 2.13
    private static <T> scala.collection.immutable.Seq<T> asScalaSeq(List<T> list) {
        return scala.collection.JavaConverters.asScalaBufferConverter(list)
                .asScala()
                .toList();
    }

    @Override
    public void write(InternalRow row) throws IOException {
        String partitionPath = getPartitionPath(row);
        VortexDataWriter writer = writers.get(partitionPath);
        if (writer == null) {
            writer = createWriterForPartition(partitionPath);
            writers.put(partitionPath, writer);
        }
        writer.write(dataProjection.apply(row));
    }

    @Override
    public WriterCommitMessage commit() throws IOException {
        if (closed) {
            return new PartitionedWriterCommitMessage(List.of());
        }

        List<VortexWriterCommitMessage> messages = new ArrayList<>();
        IOException firstException = null;

        for (Map.Entry<String, VortexDataWriter> entry : writers.entrySet()) {
            try {
                WriterCommitMessage msg = entry.getValue().commit();
                if (msg instanceof VortexWriterCommitMessage) {
                    messages.add((VortexWriterCommitMessage) msg);
                }
            } catch (IOException e) {
                if (firstException == null) {
                    firstException = e;
                } else {
                    firstException.addSuppressed(e);
                }
            }
        }

        closed = true;

        if (firstException != null) {
            throw firstException;
        }

        logger.info("Committed {} partition writers", messages.size());
        return new PartitionedWriterCommitMessage(messages);
    }

    @Override
    public void abort() throws IOException {
        if (closed) {
            return;
        }

        for (VortexDataWriter writer : writers.values()) {
            try {
                writer.abort();
            } catch (IOException e) {
                logger.error("Error aborting partition writer", e);
            }
        }
        closed = true;
    }

    @Override
    public void close() throws IOException {
        if (!closed) {
            logger.warn("PartitionedVortexDataWriter.close() called without commit() or abort() - cleaning up");
            try {
                abort();
            } catch (IOException e) {
                logger.error("Error during cleanup in close()", e);
            }
        }
    }

    private VortexDataWriter createWriterForPartition(String partitionPath) {
        String fileName = String.format("part-%05d-%d.vortex", partitionId, taskId);
        String fileUri = baseOutputUri + partitionPath + "/" + fileName;
        logger.debug("Creating writer for partition path: {}", fileUri);
        return new VortexDataWriter(fileUri, dataSchema, options);
    }

    private String getPartitionPath(InternalRow row) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < resolvedTransforms.length; i++) {
            if (i > 0) {
                sb.append("/");
            }
            ResolvedTransform rt = resolvedTransforms[i];
            sb.append(URLEncoder.encode(rt.directoryKey, StandardCharsets.UTF_8));
            sb.append("=");

            String value = evaluateTransform(rt, row);
            sb.append(URLEncoder.encode(value, StandardCharsets.UTF_8));
        }
        return sb.toString();
    }

    // ------------------------------------------------------------------
    // Transform resolution: converts Transform[] into ResolvedTransform[]
    // ------------------------------------------------------------------

    static ResolvedTransform[] resolveTransforms(Transform[] transforms, StructType schema) {
        return Arrays.stream(transforms)
                .map(transform -> resolveOne(transform, schema))
                .toArray(ResolvedTransform[]::new);
    }

    private static ResolvedTransform resolveOne(Transform transform, StructType schema) {
        String transformName = transform.name();
        NamedReference[] refs = transform.references();

        if (refs.length == 0) {
            throw new IllegalArgumentException("Partition transform has no column references: " + transform);
        }

        // Primary column (all single-column transforms use this)
        String primaryColName = String.join(".", refs[0].fieldNames());
        int primaryColIdx = schema.fieldIndex(primaryColName);
        DataType primaryType = schema.fields()[primaryColIdx].dataType();

        switch (transformName) {
            case "identity":
                return new ResolvedTransform(primaryColName, transformName, primaryColIdx, primaryType);

            case "years":
                requireTemporalType(primaryType, transformName);
                return new ResolvedTransform(primaryColName + "_year", transformName, primaryColIdx, primaryType);

            case "months":
                requireTemporalType(primaryType, transformName);
                return new ResolvedTransform(primaryColName + "_month", transformName, primaryColIdx, primaryType);

            case "days":
                requireTemporalType(primaryType, transformName);
                return new ResolvedTransform(primaryColName + "_day", transformName, primaryColIdx, primaryType);

            case "hours":
                requireTimestampType(primaryType, transformName);
                return new ResolvedTransform(primaryColName + "_hour", transformName, primaryColIdx, primaryType);

            case "bucket": {
                int bucketCount = extractBucketCount(transform);
                String colNames = Arrays.stream(refs)
                        .map(r -> String.join(".", r.fieldNames()))
                        .collect(Collectors.joining("_"));

                // Resolve all referenced columns for multi-column bucket
                ImmutableIntArray.Builder allIndices = ImmutableIntArray.builder(refs.length);
                ImmutableList.Builder<DataType> allTypes = ImmutableList.builderWithExpectedSize(refs.length);
                for (NamedReference ref : refs) {
                    String colName = String.join(".", ref.fieldNames());
                    int idx = schema.fieldIndex(colName);
                    allIndices.add(idx);
                    allTypes.add(schema.fields()[idx].dataType());
                }
                return new ResolvedTransform(
                        colNames + "_bucket", transformName, allIndices.build(), allTypes.build(), bucketCount);
            }

            default:
                throw new IllegalArgumentException("Unsupported partition transform: " + transformName);
        }
    }

    private static int extractBucketCount(Transform transform) {
        for (Expression arg : transform.arguments()) {
            if (arg instanceof Literal<?>) {
                Object value = ((Literal<?>) arg).value();
                if (value instanceof Integer) {
                    return (Integer) value;
                }
            }
        }
        throw new IllegalArgumentException("bucket transform missing integer numBuckets argument");
    }

    private static void requireTemporalType(DataType type, String transformName) {
        if (!(type instanceof DateType || type instanceof TimestampType || type instanceof TimestampNTZType)) {
            throw new IllegalArgumentException(
                    transformName + " transform requires a date or timestamp column, got: " + type);
        }
    }

    private static void requireTimestampType(DataType type, String transformName) {
        if (!(type instanceof TimestampType || type instanceof TimestampNTZType)) {
            throw new IllegalArgumentException(transformName + " transform requires a timestamp column, got: " + type);
        }
    }

    // ------------------------------------------------------------------
    // Transform evaluation: produces partition values from rows
    // ------------------------------------------------------------------

    private static String evaluateTransform(ResolvedTransform rt, InternalRow row) {
        int colIdx = rt.columnIndices.get(0);

        if (row.isNullAt(colIdx)) {
            return HIVE_DEFAULT_PARTITION;
        }

        return switch (rt.transformName) {
            case "identity" -> extractIdentityValue(row, colIdx, rt.columnTypes.get(0));
            case "years" -> extractYearValue(row, colIdx, rt.columnTypes.get(0));
            case "months" -> extractMonthValue(row, colIdx, rt.columnTypes.get(0));
            case "days" -> extractDayValue(row, colIdx, rt.columnTypes.get(0));
            case "hours" -> extractHourValue(row, colIdx, rt.columnTypes.get(0));
            case "bucket" -> extractBucketValue(row, rt);
            default -> throw new IllegalArgumentException("Unsupported transform: " + rt.transformName);
        };
    }

    private static String extractIdentityValue(InternalRow row, int ordinal, DataType dataType) {
        if (dataType instanceof BooleanType) {
            return String.valueOf(row.getBoolean(ordinal));
        } else if (dataType instanceof ByteType) {
            return String.valueOf(row.getByte(ordinal));
        } else if (dataType instanceof ShortType) {
            return String.valueOf(row.getShort(ordinal));
        } else if (dataType instanceof IntegerType) {
            return String.valueOf(row.getInt(ordinal));
        } else if (dataType instanceof LongType) {
            return String.valueOf(row.getLong(ordinal));
        } else if (dataType instanceof FloatType) {
            return String.valueOf(row.getFloat(ordinal));
        } else if (dataType instanceof DoubleType) {
            return String.valueOf(row.getDouble(ordinal));
        } else if (dataType instanceof StringType) {
            UTF8String str = row.getUTF8String(ordinal);
            return str != null ? str.toString() : HIVE_DEFAULT_PARTITION;
        } else if (dataType instanceof DateType) {
            return String.valueOf(row.getInt(ordinal));
        } else if (dataType instanceof TimestampType || dataType instanceof TimestampNTZType) {
            return String.valueOf(row.getLong(ordinal));
        } else {
            throw new IllegalArgumentException("Unsupported partition column type: " + dataType);
        }
    }

    private static String extractYearValue(InternalRow row, int colIdx, DataType type) {
        if (type instanceof DateType) {
            return String.valueOf(LocalDate.ofEpochDay(row.getInt(colIdx)).getYear());
        } else {
            return String.valueOf(microsToDateTime(row.getLong(colIdx)).getYear());
        }
    }

    private static String extractMonthValue(InternalRow row, int colIdx, DataType type) {
        LocalDate date;
        if (type instanceof DateType) {
            date = LocalDate.ofEpochDay(row.getInt(colIdx));
        } else {
            date = microsToDateTime(row.getLong(colIdx)).toLocalDate();
        }
        return String.format("%04d-%02d", date.getYear(), date.getMonthValue());
    }

    private static String extractDayValue(InternalRow row, int colIdx, DataType type) {
        LocalDate date;
        if (type instanceof DateType) {
            date = LocalDate.ofEpochDay(row.getInt(colIdx));
        } else {
            date = microsToDateTime(row.getLong(colIdx)).toLocalDate();
        }
        return date.toString(); // YYYY-MM-DD
    }

    private static String extractHourValue(InternalRow row, int colIdx, DataType type) {
        LocalDateTime dt = microsToDateTime(row.getLong(colIdx));
        return String.format("%s-%02d", dt.toLocalDate(), dt.getHour());
    }

    /**
     * Computes the bucket value matching Spark's {@code InMemoryBaseTable} reference implementation: per-column values
     * are converted to longs (hashed for strings/binary), summed, then {@code Math.floorMod(sum, numBuckets)}.
     */
    private static String extractBucketValue(InternalRow row, ResolvedTransform rt) {
        long hash = Streams.zip(
                        rt.columnIndices.stream().boxed(),
                        rt.columnTypes.stream(),
                        (Integer idx, DataType dt) -> columnHashValue(row, idx, dt))
                .reduce(0L, Long::sum);
        int bucket = Math.floorMod(hash, rt.bucketCount);
        return String.valueOf(bucket);
    }

    private static long columnHashValue(InternalRow row, int ordinal, DataType dataType) {
        if (dataType instanceof ByteType) {
            return row.getByte(ordinal);
        } else if (dataType instanceof ShortType) {
            return row.getShort(ordinal);
        } else if (dataType instanceof IntegerType) {
            return row.getInt(ordinal);
        } else if (dataType instanceof LongType
                || dataType instanceof TimestampType
                || dataType instanceof TimestampNTZType) {
            return row.getLong(ordinal);
        } else if (dataType instanceof StringType) {
            return row.getUTF8String(ordinal).hashCode();
        } else if (dataType instanceof BinaryType) {
            return java.util.Arrays.hashCode(row.getBinary(ordinal));
        } else {
            throw new IllegalArgumentException("Unsupported bucket column type: " + dataType);
        }
    }

    private static LocalDateTime microsToDateTime(long micros) {
        long epochSecond = Math.floorDiv(micros, 1_000_000);
        int nanoOfSecond = (int) (Math.floorMod(micros, 1_000_000) * 1000);
        return LocalDateTime.ofInstant(Instant.ofEpochSecond(epochSecond, nanoOfSecond), ZoneOffset.UTC);
    }

    // ------------------------------------------------------------------
    // Internal types
    // ------------------------------------------------------------------

    /**
     * Pre-resolved representation of a partition transform, ready for per-row evaluation.
     *
     * @param bucketCount -1 if not a bucket transform
     */
    record ResolvedTransform(
            String directoryKey,
            String transformName,
            ImmutableIntArray columnIndices,
            List<DataType> columnTypes,
            int bucketCount)
            implements Serializable {
        ResolvedTransform(String directoryKey, String transformName, int columnIndex, DataType columnType) {
            this(directoryKey, transformName, ImmutableIntArray.of(columnIndex), List.of(columnType), -1);
        }
    }

    /** Commit message that aggregates results from multiple partition writers. */
    public static final class PartitionedWriterCommitMessage implements WriterCommitMessage, Serializable {
        private final List<VortexWriterCommitMessage> partitionMessages;

        PartitionedWriterCommitMessage(List<VortexWriterCommitMessage> partitionMessages) {
            this.partitionMessages = partitionMessages;
        }

        /** Returns the commit messages from each individual partition writer. */
        public List<VortexWriterCommitMessage> getPartitionMessages() {
            return partitionMessages;
        }
    }
}
