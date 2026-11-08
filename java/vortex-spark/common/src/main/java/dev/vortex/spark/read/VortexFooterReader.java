// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.DataSource.RowCount;
import dev.vortex.api.FileAggregate;
import dev.vortex.api.FileAggregateResult;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.jni.NativeFiles;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.ArrowUtils;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;
import org.apache.hadoop.fs.FileStatus;
import org.apache.spark.sql.types.ArrayType;
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
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;

/** Reads schema and row-count metadata from Vortex file footers. */
public final class VortexFooterReader {
    /** Option bounding how many footers scan statistics may read; {@code 0} removes the bound. */
    public static final String MAX_FILES_OPTION = "vortex.stats.maxFiles";

    /** Option turning schema merging off, leaving the first file's schema to stand for the dataset. */
    public static final String MERGE_SCHEMA_OPTION = "vortex.mergeSchema";

    /** Option bounding how many footers are read at once. */
    public static final String FOOTER_PARALLELISM_OPTION = "vortex.footerParallelism";

    private static final int DEFAULT_FOOTER_PARALLELISM = 8;
    private static final int DEFAULT_MAX_FILES = 1000;

    private VortexFooterReader() {}

    /**
     * Infers the data schema of a Vortex dataset, or returns null when the listing holds no files at all.
     *
     * <p>Listed files without {@link VortexFile#EXTENSION} are not part of the dataset and are skipped.
     *
     * <p>Every file's footer is read and the schemas are merged, so a column only some files carry is still part of the
     * dataset. A field missing from a file is nullable in the result, and the reader fills it with nulls for that
     * file's rows. Set {@value #MERGE_SCHEMA_OPTION} to false to read one footer and let the first file's schema stand
     * for the dataset; a dataset of uniform files then costs one footer read instead of one per file.
     *
     * @throws IllegalArgumentException if the listing holds files but none of them is a Vortex file, or if two files
     *     give a field types that cannot be merged
     */
    public static StructType inferSchema(List<FileStatus> files, VortexIo io, VortexOptions options) {
        List<VortexFile> vortexFiles = new ArrayList<>();
        boolean sawFile = false;
        for (FileStatus status : files) {
            if (!status.isFile()) {
                continue;
            }
            sawFile = true;
            if (VortexFile.hasVortexExtension(status.getPath().getName())) {
                vortexFiles.add(new VortexFile(status.getPath().toString(), status.getLen()));
            }
        }

        if (vortexFiles.isEmpty()) {
            if (sawFile) {
                throw new IllegalArgumentException(
                        "No Vortex file found to infer a schema from: every file in a Vortex dataset must end with "
                                + VortexFile.EXTENSION);
            }
            return null;
        }

        if (!options.getBoolean(MERGE_SCHEMA_OPTION, true)) {
            return withDataSource(vortexFiles.get(0), io, options, VortexFooterReader::sparkSchema);
        }

        List<StructType> schemas = mapFooters(vortexFiles, io, options, VortexFooterReader::sparkSchema);
        StructType merged = schemas.get(0);
        for (int i = 1; i < schemas.size(); i++) {
            try {
                merged = mergeStructs(merged, schemas.get(i));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        String.format(
                                Locale.ROOT,
                                "Cannot merge the schema of %s with the schemas of the files before it: %s. "
                                        + "Set the %s option to false to read the dataset with one file's schema.",
                                vortexFiles.get(i).path(),
                                e.getMessage(),
                                MERGE_SCHEMA_OPTION),
                        e);
            }
        }
        return merged;
    }

    /**
     * Returns the footer row count for one file, exact or estimated.
     *
     * <p>Only for Spark scan statistics, which are an estimate by contract. Anything that must answer a query with this
     * number needs {@link #exactRowCount}.
     */
    public static OptionalLong estimatedRowCount(VortexFile file, VortexIo io, VortexOptions options) {
        return withDataSource(file, io, options, source -> source.rowCount().asOptional());
    }

    /**
     * Returns the footer row count for one file, and only when the footer states it exactly.
     *
     * <p>{@code COUNT(*)} pushdown answers the query from this number instead of reading the file, so an estimate would
     * be returned to the user as fact.
     */
    public static OptionalLong exactRowCount(VortexFile file, VortexIo io, VortexOptions options) {
        return withDataSource(
                file,
                io,
                options,
                source -> source.rowCount() instanceof RowCount.Exact exact
                        ? OptionalLong.of(exact.value())
                        : OptionalLong.empty());
    }

    /**
     * Returns every file's exact footer row count, in listing order, read in a bounded pool.
     *
     * @throws IllegalStateException if a footer states no exact row count
     */
    public static List<Long> exactRowCounts(List<VortexFile> files, VortexIo io, VortexOptions options) {
        if (files.isEmpty()) {
            return List.of();
        }
        List<Long> counts = new ArrayList<>(files.size());
        List<OptionalLong> exact = mapFooters(
                files,
                io,
                options,
                source -> source.rowCount() instanceof RowCount.Exact count
                        ? OptionalLong.of(count.value())
                        : OptionalLong.empty());
        for (int i = 0; i < exact.size(); i++) {
            if (exact.get(i).isEmpty()) {
                throw new IllegalStateException("Vortex footer states no exact row count for "
                        + files.get(i).path());
            }
            counts.add(exact.get(i).getAsLong());
        }
        return counts;
    }

    /**
     * Sums footer row counts in a bounded pool.
     *
     * <p>Returns empty if any footer has no count at all, or if the dataset holds more files than
     * {@value #MAX_FILES_OPTION} allows. Each footer costs a read against storage on the driver, so a large dataset
     * would otherwise pay for the whole listing before the job starts.
     */
    public static OptionalLong sumRowCounts(List<VortexFile> files, VortexIo io, VortexOptions options) {
        if (files.isEmpty()) {
            return OptionalLong.of(0);
        }
        int maxFiles = options.getInt(MAX_FILES_OPTION, DEFAULT_MAX_FILES);
        if (maxFiles > 0 && files.size() > maxFiles) {
            return OptionalLong.empty();
        }

        long total = 0;
        for (OptionalLong count :
                mapFooters(files, io, options, source -> source.rowCount().asOptional())) {
            if (count.isEmpty()) {
                return OptionalLong.empty();
            }
            total = Math.addExact(total, count.getAsLong());
        }
        return OptionalLong.of(total);
    }

    /**
     * Gathers footer statistics for Spark's cost-based optimizer: the dataset's row count, and for each of
     * {@code columns} its null count and extrema, as Catalyst values.
     *
     * <p>Only what every footer states exactly is reported, so no data is read: a column whose null count one footer
     * leaves open has no null count, and a column whose extrema one footer leaves open, among files that hold values in
     * it, has no extrema. A file that predates a column holds only nulls in it. Like {@link #sumRowCounts}, this
     * reports nothing for a dataset of more than {@value #MAX_FILES_OPTION} files.
     *
     * @param columns data columns whose types order through {@link Comparable} in their Catalyst form
     */
    public static Statistics statistics(
            List<VortexFile> files, VortexIo io, VortexOptions options, List<String> columns, boolean caseSensitive) {
        if (files.isEmpty()) {
            return new Statistics(OptionalLong.of(0), Map.of());
        }
        int maxFiles = options.getInt(MAX_FILES_OPTION, DEFAULT_MAX_FILES);
        if (maxFiles > 0 && files.size() > maxFiles) {
            return new Statistics(OptionalLong.empty(), Map.of());
        }

        List<FileStatistics> perFile =
                mapFiles(files, io, options, footer -> fileStatistics(footer, columns, caseSensitive));

        long rowCount = 0;
        Map<String, ColumnAccumulator> merged = new LinkedHashMap<>();
        for (String column : columns) {
            merged.put(column, new ColumnAccumulator());
        }
        for (FileStatistics file : perFile) {
            rowCount = Math.addExact(rowCount, file.rowCount());
            file.columns().forEach((column, statistics) -> merged.get(column).add(statistics));
        }

        Map<String, ColumnStatistics> result = new LinkedHashMap<>();
        merged.forEach((column, accumulator) -> result.put(column, accumulator.finish()));
        return new Statistics(OptionalLong.of(rowCount), result);
    }

    /**
     * The footer statistics of each file, in the order of {@code files}, for {@code columns}. Like {@link #statistics},
     * it reads no data, and it reads nothing for a dataset of more than {@value #MAX_FILES_OPTION} files.
     *
     * @param columns data columns whose types order through {@link Comparable} in their Catalyst form
     */
    public static Optional<List<FileStatistics>> statisticsByFile(
            List<VortexFile> files, VortexIo io, VortexOptions options, List<String> columns, boolean caseSensitive) {
        int maxFiles = options.getInt(MAX_FILES_OPTION, DEFAULT_MAX_FILES);
        if (maxFiles > 0 && files.size() > maxFiles) {
            return Optional.empty();
        }

        return Optional.of(mapFiles(files, io, options, footer -> fileStatistics(footer, columns, caseSensitive)));
    }

    /**
     * The natural row splits of each file, in the order of {@code files}, as {@link NativeFiles#splits} states them.
     * Split {@code i} of a file holds rows {@code [boundaries[i], boundaries[i + 1])}, and the last boundary is the
     * file's row count.
     */
    public static List<long[]> splitBoundaries(
            List<VortexFile> files, List<String> columns, VortexIo io, VortexOptions options) {
        return mapFiles(
                files, io, options, footer -> NativeFiles.splits(footer.session(), footer.readable(), columns));
    }

    private static FileStatistics fileStatistics(OpenFooter footer, List<String> columns, boolean caseSensitive) {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Map<String, Field> fileFields = caseSensitive ? new HashMap<>() : new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        for (Field field : footer.source().arrowSchema(allocator).getFields()) {
            fileFields.putIfAbsent(field.getName(), field);
        }

        List<FileAggregate> requests = new ArrayList<>(1 + 3 * columns.size());
        requests.add(FileAggregate.countStar());
        List<String> present = new ArrayList<>(columns.size());
        for (String column : columns) {
            Field field = fileFields.get(column);
            if (field != null) {
                present.add(column);
                requests.add(FileAggregate.count(field.getName()));
                requests.add(FileAggregate.min(field.getName()));
                requests.add(FileAggregate.max(field.getName()));
            }
        }

        // Without a scan, the answers are what the footer states exactly, and nothing more.
        try (FileAggregateResult result =
                NativeFiles.aggregate(footer.session(), footer.readable(), requests, false, allocator)) {
            long rowCount = ((BigIntVector) result.vector(0)).get(0);
            Map<String, ColumnStatistics> statistics = new HashMap<>();
            for (String column : columns) {
                // A file that predates the column holds only nulls in it.
                statistics.put(
                        column,
                        new ColumnStatistics(
                                OptionalLong.of(rowCount), Optional.of(NO_VALUES), Optional.of(NO_VALUES)));
            }
            for (int i = 0; i < present.size(); i++) {
                int count = 1 + 3 * i;
                OptionalLong nullCount = result.answered(count)
                        ? OptionalLong.of(rowCount - ((BigIntVector) result.vector(count)).get(0))
                        : OptionalLong.empty();
                boolean holdsValues = nullCount.isEmpty() || nullCount.getAsLong() < rowCount;
                statistics.put(
                        present.get(i),
                        new ColumnStatistics(
                                nullCount,
                                holdsValues ? catalystValue(result, count + 1) : Optional.of(NO_VALUES),
                                holdsValues ? catalystValue(result, count + 2) : Optional.of(NO_VALUES)));
            }
            return new FileStatistics(rowCount, statistics);
        }
    }

    /**
     * A starting threshold for an {@code ORDER BY column ... LIMIT limit} from footer statistics alone, or empty.
     *
     * <p>A file holding at least {@code limit} non-null values of {@code column} proves that the result's last value is
     * at least that file's minimum when sorting descending, or at most its maximum when ascending. The best such bound
     * over every file lets each reader skip files and chunks before it has read a row. Rows equal to the bound may
     * still belong to the result. Like {@link #sumRowCounts}, this reads nothing for more than
     * {@value #MAX_FILES_OPTION} files.
     */
    public static Optional<Object> topNSeed(
            List<VortexFile> files,
            VortexIo io,
            VortexOptions options,
            String column,
            boolean descending,
            int limit,
            boolean caseSensitive) {
        int maxFiles = options.getInt(MAX_FILES_OPTION, DEFAULT_MAX_FILES);
        if (files.isEmpty() || (maxFiles > 0 && files.size() > maxFiles)) {
            return Optional.empty();
        }

        List<Optional<Object>> bounds =
                mapFiles(files, io, options, footer -> fileTopNBound(footer, column, descending, limit, caseSensitive));
        Comparable<Object> best = null;
        for (Optional<Object> bound : bounds) {
            if (bound.isEmpty()) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Comparable<Object> candidate = (Comparable<Object>) bound.get();
            if (best == null || (descending ? candidate.compareTo(best) > 0 : candidate.compareTo(best) < 0)) {
                best = candidate;
            }
        }
        return Optional.ofNullable(best);
    }

    /**
     * The first of {@code columns} that every file's footer states sorted ascending without nulls, or empty.
     *
     * <p>Like {@link #sumRowCounts}, this reads nothing for more than {@value #MAX_FILES_OPTION} files.
     */
    public static Optional<String> sortedColumn(
            List<VortexFile> files, VortexIo io, VortexOptions options, List<String> columns, boolean caseSensitive) {
        int maxFiles = options.getInt(MAX_FILES_OPTION, DEFAULT_MAX_FILES);
        if (files.isEmpty() || columns.isEmpty() || (maxFiles > 0 && files.size() > maxFiles)) {
            return Optional.empty();
        }

        List<List<String>> sortedPerFile =
                mapFiles(files, io, options, footer -> sortedColumns(footer, columns, caseSensitive));
        return columns.stream()
                .filter(column -> sortedPerFile.stream().allMatch(sorted -> sorted.contains(column)))
                .findFirst();
    }

    private static List<String> sortedColumns(OpenFooter footer, List<String> columns, boolean caseSensitive) {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Map<String, Field> fileFields = caseSensitive ? new HashMap<>() : new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        for (Field field : footer.source().arrowSchema(allocator).getFields()) {
            fileFields.putIfAbsent(field.getName(), field);
        }

        List<String> present = new ArrayList<>();
        List<FileAggregate> requests = new ArrayList<>();
        for (String column : columns) {
            Field field = fileFields.get(column);
            if (field != null) {
                present.add(column);
                requests.add(FileAggregate.isSorted(field.getName()));
            }
        }
        if (requests.isEmpty()) {
            return List.of();
        }

        try (FileAggregateResult result =
                NativeFiles.aggregate(footer.session(), footer.readable(), requests, false, allocator)) {
            List<String> sorted = new ArrayList<>();
            for (int i = 0; i < present.size(); i++) {
                if (result.answered(i) && Boolean.TRUE.equals(result.vector(i).getObject(0))) {
                    sorted.add(present.get(i));
                }
            }
            return sorted;
        }
    }

    private static Optional<Object> fileTopNBound(
            OpenFooter footer, String column, boolean descending, int limit, boolean caseSensitive) {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        String physical = null;
        for (Field field : footer.source().arrowSchema(allocator).getFields()) {
            if (caseSensitive ? field.getName().equals(column) : field.getName().equalsIgnoreCase(column)) {
                physical = field.getName();
                break;
            }
        }
        if (physical == null) {
            return Optional.empty();
        }

        List<FileAggregate> requests = List.of(
                FileAggregate.count(physical), descending ? FileAggregate.min(physical) : FileAggregate.max(physical));
        try (FileAggregateResult result =
                NativeFiles.aggregate(footer.session(), footer.readable(), requests, false, allocator)) {
            if (!result.answered(0) || ((BigIntVector) result.vector(0)).get(0) < limit) {
                return Optional.empty();
            }
            return catalystValue(result, 1).filter(value -> value != NO_VALUES);
        }
    }

    /** Marks the extrema of a file without values in a column, which leave the dataset's extrema as they are. */
    private static final Object NO_VALUES = new Object();

    /** The one value of aggregate {@code index} in its Catalyst form, or empty when it was left unanswered. */
    private static Optional<Object> catalystValue(FileAggregateResult result, int index) {
        if (!result.answered(index)) {
            return Optional.empty();
        }
        VortexArrowColumnVector vector = new VortexArrowColumnVector(result.vector(index));
        if (vector.isNullAt(0)) {
            return Optional.of(NO_VALUES);
        }
        DataType type = vector.dataType();
        Object value;
        if (type instanceof BooleanType) {
            value = vector.getBoolean(0);
        } else if (type instanceof ByteType) {
            value = vector.getByte(0);
        } else if (type instanceof ShortType) {
            value = vector.getShort(0);
        } else if (type instanceof IntegerType || type instanceof DateType) {
            value = vector.getInt(0);
        } else if (type instanceof LongType || type instanceof TimestampType || type instanceof TimestampNTZType) {
            value = vector.getLong(0);
        } else if (type instanceof FloatType) {
            value = vector.getFloat(0);
        } else if (type instanceof DoubleType) {
            value = vector.getDouble(0);
        } else if (type instanceof DecimalType decimal) {
            value = vector.getDecimal(0, decimal.precision(), decimal.scale());
        } else {
            return Optional.empty();
        }
        return Optional.of(value);
    }

    /** Footer statistics of a dataset. See {@link #statistics}. */
    public record Statistics(OptionalLong rowCount, Map<String, ColumnStatistics> columns) {}

    /**
     * Footer statistics of one column, with extrema in their Catalyst form. Extrema the footer leaves open are empty.
     */
    public record ColumnStatistics(OptionalLong nullCount, Optional<Object> min, Optional<Object> max) {
        /** Whether the footer proves that every value of the column is null, which leaves it without extrema. */
        public boolean holdsNoValues() {
            return min.isPresent() && min.get() == NO_VALUES;
        }
    }

    /** Footer statistics of one file: its row count, and the statistics of each column asked for. */
    public record FileStatistics(long rowCount, Map<String, ColumnStatistics> columns) {}

    /** Merges the statistics of one column across files. */
    private static final class ColumnAccumulator {
        private OptionalLong nullCount = OptionalLong.of(0);
        private boolean extremaKnown = true;
        private Comparable<Object> min;
        private Comparable<Object> max;

        @SuppressWarnings("unchecked")
        void add(ColumnStatistics file) {
            nullCount = nullCount.isPresent() && file.nullCount().isPresent()
                    ? OptionalLong.of(Math.addExact(
                            nullCount.getAsLong(), file.nullCount().getAsLong()))
                    : OptionalLong.empty();
            if (file.min().isEmpty() || file.max().isEmpty()) {
                extremaKnown = false;
                return;
            }
            if (file.min().get() == NO_VALUES || file.max().get() == NO_VALUES) {
                return;
            }
            Comparable<Object> fileMin = (Comparable<Object>) file.min().get();
            Comparable<Object> fileMax = (Comparable<Object>) file.max().get();
            min = min == null || fileMin.compareTo(min) < 0 ? fileMin : min;
            max = max == null || fileMax.compareTo(max) > 0 ? fileMax : max;
        }

        ColumnStatistics finish() {
            boolean known = extremaKnown && min != null;
            return new ColumnStatistics(
                    nullCount,
                    known ? Optional.<Object>of(min) : Optional.empty(),
                    known ? Optional.<Object>of(max) : Optional.empty());
        }
    }

    /**
     * Merges two schemas of the same dataset.
     *
     * <p>Top-level fields are unioned and keep the order they were first seen in. A field only one side carries is
     * nullable in the result, because the rows of the other side have no value for it, and the reader fills those rows
     * with nulls.
     *
     * <p>Below the top level only nullability is merged. A struct that gained a field cannot be merged: the reader
     * projects a struct column whole, as the file stores it, so it has no way to widen one file's struct to a shape
     * another file agreed on.
     *
     * @throws IllegalArgumentException if a field has types that cannot be merged
     */
    static StructType mergeStructs(StructType left, StructType right) {
        Map<String, StructField> rightFields = indexByName(right);
        Map<String, StructField> leftFields = indexByName(left);

        LinkedHashMap<String, StructField> merged = new LinkedHashMap<>();
        for (StructField field : left.fields()) {
            StructField other = rightFields.get(field.name());
            merged.put(field.name(), other == null ? asNullable(field) : mergeFields(field, other));
        }
        for (StructField field : right.fields()) {
            if (!leftFields.containsKey(field.name())) {
                merged.put(field.name(), asNullable(field));
            }
        }
        return new StructType(merged.values().toArray(new StructField[0]));
    }

    private static StructField mergeFields(StructField left, StructField right) {
        return new StructField(
                left.name(),
                mergeTypes(left.name(), left.dataType(), right.dataType()),
                left.nullable() || right.nullable(),
                left.metadata());
    }

    /** Merges the types of one field. Only nullability differences are reconcilable below the top level. */
    private static DataType mergeTypes(String field, DataType left, DataType right) {
        if (left.equals(right)) {
            return left;
        }
        if (left instanceof StructType leftStruct && right instanceof StructType rightStruct) {
            return mergeNestedStruct(field, leftStruct, rightStruct);
        }
        if (left instanceof ArrayType leftArray && right instanceof ArrayType rightArray) {
            return new ArrayType(
                    mergeTypes(field, leftArray.elementType(), rightArray.elementType()),
                    leftArray.containsNull() || rightArray.containsNull());
        }
        if (left instanceof MapType leftMap && right instanceof MapType rightMap) {
            return new MapType(
                    mergeTypes(field, leftMap.keyType(), rightMap.keyType()),
                    mergeTypes(field, leftMap.valueType(), rightMap.valueType()),
                    leftMap.valueContainsNull() || rightMap.valueContainsNull());
        }
        throw new IllegalArgumentException(String.format(
                Locale.ROOT,
                "field %s is %s in one file and %s in another",
                field,
                left.catalogString(),
                right.catalogString()));
    }

    private static StructType mergeNestedStruct(String field, StructType left, StructType right) {
        if (!Arrays.equals(left.fieldNames(), right.fieldNames())) {
            throw new IllegalArgumentException(String.format(
                    Locale.ROOT,
                    "nested field %s holds %s in one file and %s in another, and a struct that gained or lost a "
                            + "field cannot be merged",
                    field,
                    left.catalogString(),
                    right.catalogString()));
        }

        StructField[] fields = new StructField[left.fields().length];
        for (int i = 0; i < fields.length; i++) {
            fields[i] = mergeFields(left.fields()[i], right.fields()[i]);
        }
        return new StructType(fields);
    }

    private static StructField asNullable(StructField field) {
        return field.nullable() ? field : new StructField(field.name(), field.dataType(), true, field.metadata());
    }

    private static Map<String, StructField> indexByName(StructType schema) {
        Map<String, StructField> byName = new LinkedHashMap<>();
        for (StructField field : schema.fields()) {
            byName.put(field.name(), field);
        }
        return byName;
    }

    private static StructType sparkSchema(DataSource source) {
        StructField[] fields = source.arrowSchema(ArrowAllocation.rootAllocator()).getFields().stream()
                .map(field -> new StructField(
                        field.getName(), ArrowUtils.fromArrowField(field), field.isNullable(), Metadata.empty()))
                .toArray(StructField[]::new);
        return new StructType(fields);
    }

    /** Applies {@code function} to every file's footer in a bounded pool, returning results in listing order. */
    private static <T> List<T> mapFooters(
            List<VortexFile> files, VortexIo io, VortexOptions options, DataSourceFunction<T> function) {
        return mapFiles(files, io, options, footer -> function.apply(footer.source()));
    }

    /** Applies {@code function} to every open file in a bounded pool, returning results in listing order. */
    private static <T> List<T> mapFiles(
            List<VortexFile> files, VortexIo io, VortexOptions options, FooterFunction<T> function) {
        return mapInParallel(files, options, file -> withFooter(file, io, options, function));
    }

    /**
     * Applies {@code function} to every item in a pool bounded by {@value #FOOTER_PARALLELISM_OPTION}, returning
     * results in the order of {@code items}.
     */
    static <A, T> List<T> mapInParallel(List<A> items, VortexOptions options, Function<A, T> function) {
        int configured = options.getInt(FOOTER_PARALLELISM_OPTION, DEFAULT_FOOTER_PARALLELISM);
        if (configured < 1) {
            throw new IllegalArgumentException(FOOTER_PARALLELISM_OPTION + " must be at least 1, got " + configured);
        }
        if (items.isEmpty()) {
            return List.of();
        }

        ExecutorService executor = Executors.newFixedThreadPool(Math.min(configured, items.size()));
        try {
            List<Future<T>> futures = new ArrayList<>(items.size());
            for (A item : items) {
                futures.add(executor.submit(() -> function.apply(item)));
            }

            List<T> results = new ArrayList<>(items.size());
            for (Future<T> future : futures) {
                results.add(future.get());
            }
            return results;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while reading Vortex footers", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
            if (cause instanceof Error error) {
                throw error;
            }
            throw new RuntimeException("Failed to read Vortex footers", cause);
        } finally {
            executor.shutdownNow();
        }
    }

    private static <T> T withDataSource(
            VortexFile file, VortexIo io, VortexOptions options, DataSourceFunction<T> function) {
        return withFooter(file, io, options, footer -> function.apply(footer.source()));
    }

    private static <T> T withFooter(VortexFile file, VortexIo io, VortexOptions options, FooterFunction<T> function) {
        Session session = VortexSparkSession.get(options);
        try (NativeReadable readable = io.openReadable(file);
                DataSource source = DataSource.open(session, List.of(readable), io.readConcurrency())) {
            return function.apply(new OpenFooter(session, readable, source));
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to close footer readable for " + file.path(), e);
        }
    }

    @FunctionalInterface
    private interface DataSourceFunction<T> {
        T apply(DataSource source);
    }

    /** A file whose footer is open, with the readable it was opened through. */
    private record OpenFooter(Session session, NativeReadable readable, DataSource source) {}

    @FunctionalInterface
    private interface FooterFunction<T> {
        T apply(OpenFooter footer);
    }
}
