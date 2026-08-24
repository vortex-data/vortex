// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.DataSource.RowCount;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.spark.ArrowUtils;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.hadoop.fs.FileStatus;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

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
        int configured = options.getInt(FOOTER_PARALLELISM_OPTION, DEFAULT_FOOTER_PARALLELISM);
        if (configured < 1) {
            throw new IllegalArgumentException(FOOTER_PARALLELISM_OPTION + " must be at least 1, got " + configured);
        }

        ExecutorService executor = Executors.newFixedThreadPool(Math.min(configured, files.size()));
        try {
            List<Future<T>> futures = new ArrayList<>(files.size());
            for (VortexFile file : files) {
                futures.add(executor.submit(() -> withDataSource(file, io, options, function)));
            }

            List<T> results = new ArrayList<>(files.size());
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
        Session session = VortexSparkSession.get(options);
        NativeReadable readable = io.openReadable(file);

        T result;
        try {
            DataSource source = DataSource.open(session, List.of(readable), io.readConcurrency());
            result = function.apply(source);
        } catch (RuntimeException | Error e) {
            // The footer read already failed. A close failure on top of it is a detail of that
            // failure, never a replacement for it.
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
            throw new UncheckedIOException("Failed to close footer readable for " + file.path(), e);
        }
        return result;
    }

    @FunctionalInterface
    private interface DataSourceFunction<T> {
        T apply(DataSource source);
    }
}
