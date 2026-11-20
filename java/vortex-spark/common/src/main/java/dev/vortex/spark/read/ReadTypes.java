// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.ArrowUtils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Matches the types Spark reads against the types a file stores.
 *
 * <p>A column vector takes its type from the file and converts nothing, so a column read as another type would return
 * its stored values under that type: a count of microseconds read as a timestamp, for one. Only conversions Spark's
 * Parquet source also makes, and that the vectors get right as they are, are allowed.
 *
 * <p>{@link NestedSchemaPruning} resolves the read schema against the file and calls {@link #check} on the fields it
 * resolves, so there is one walk and one name resolution for each file.
 */
final class ReadTypes {
    // Spark 4 lets Parquet widen decimal scales and read timestamps with a time zone as timestamps without one.
    private static final boolean SPARK_4 =
            !org.apache.spark.package$.MODULE$.SPARK_VERSION().startsWith("3.");

    private ReadTypes() {}

    /**
     * Fails unless a vector of {@code file} values reads correctly as {@code read}.
     *
     * <p>{@link NestedSchemaPruning} calls this for each field that it resolves and that the scan returns as the file
     * stores it: a leaf, an array, a map, or an extension type stored as a struct. Struct fields below such a field
     * match by position under the same name, since the vectors return them as stored.
     *
     * @throws IllegalArgumentException naming {@code fileName}, the field at {@code path} and both types
     */
    static void check(DataType read, Field file, String path, boolean caseSensitive, String fileName) {
        new Check(caseSensitive, fileName).value(read, file, path);
    }

    private record Check(boolean caseSensitive, String fileName) {
        void value(DataType read, Field file, String path) {
            ArrowType type = file.getType();
            if (type instanceof ArrowType.Null) {
                // Every value is null, which reads the same as any type.
                return;
            }

            if (read instanceof StructType struct) {
                if (!(type instanceof ArrowType.Struct)) {
                    throw mismatch(read, file, path, fileName);
                }
                positional(struct, file, path);
            } else if (read instanceof ArrayType array) {
                if (!(type instanceof ArrowType.List || type instanceof ArrowType.ListView)) {
                    throw mismatch(read, file, path, fileName);
                }
                value(array.elementType(), file.getChildren().get(0), path + ".element");
            } else if (read instanceof MapType map) {
                if (!(type instanceof ArrowType.Map)) {
                    throw mismatch(read, file, path, fileName);
                }
                List<Field> entry = file.getChildren().get(0).getChildren();
                value(map.keyType(), entry.get(0), path + ".key");
                value(map.valueType(), entry.get(1), path + ".value");
            } else if (!readable(file, read)) {
                throw mismatch(read, file, path, fileName);
            }
        }

        /** Checks the fields of a struct that the scan returns as stored, which Spark then reads by position. */
        void positional(StructType read, Field file, String path) {
            StructField[] fields = read.fields();
            List<Field> children = file.getChildren();
            if (fields.length > children.size()) {
                throw mismatch(read, file, path, fileName);
            }

            for (int i = 0; i < fields.length; i++) {
                String name = children.get(i).getName();
                if (!(caseSensitive ? name.equals(fields[i].name()) : name.equalsIgnoreCase(fields[i].name()))) {
                    throw mismatch(read, file, path, fileName);
                }
                value(fields[i].dataType(), children.get(i), path + "." + fields[i].name());
            }
        }
    }

    /** Whether a vector of {@code fileField} values reads correctly as {@code read}. */
    private static boolean readable(Field fileField, DataType read) {
        DataType file = leafType(fileField);
        if (file == null) {
            return false;
        }
        // A collation only changes how strings compare.
        if (file.equals(read) || (file instanceof StringType && read instanceof StringType)) {
            return true;
        }
        // Dates are days since the epoch and timestamps microseconds since it. Parquet reads a timestamp of
        // another unit as BIGINT in that unit, while the vector normalizes it to microseconds.
        boolean micros = fileField.getType() instanceof ArrowType.Timestamp timestamp
                && timestamp.getUnit() == TimeUnit.MICROSECOND;
        if ((file.equals(DataTypes.IntegerType) && read.equals(DataTypes.DateType))
                || (file.equals(DataTypes.DateType) && read.equals(DataTypes.IntegerType))
                || (micros && read.equals(DataTypes.LongType))
                || (file.equals(DataTypes.TimestampNTZType) && read.equals(DataTypes.TimestampType))
                || (SPARK_4 && file.equals(DataTypes.TimestampType) && read.equals(DataTypes.TimestampNTZType))) {
            return true;
        }
        // Decimal values are rescaled to the type read, so it only has to keep every digit.
        if (file instanceof DecimalType stored && read instanceof DecimalType wanted) {
            int scaleIncrease = wanted.scale() - stored.scale();
            return (scaleIncrease == 0 || (SPARK_4 && scaleIncrease > 0))
                    && wanted.precision() - stored.precision() >= scaleIncrease;
        }
        return false;
    }

    /** The Spark type of a file leaf, or {@code null} for a nested or unsupported one. */
    private static DataType leafType(Field file) {
        ArrowType type = file.getType();
        if (type instanceof ArrowType.Struct
                || type instanceof ArrowType.List
                || type instanceof ArrowType.ListView
                || type instanceof ArrowType.Map) {
            return null;
        }
        return sparkType(file);
    }

    private static DataType sparkType(Field file) {
        try {
            return ArrowUtils.fromArrowField(file);
        } catch (UnsupportedOperationException e) {
            return null;
        }
    }

    private static IllegalArgumentException mismatch(DataType read, Field file, String path, String fileName) {
        DataType stored = sparkType(file);
        return new IllegalArgumentException(String.format(
                Locale.ROOT,
                "%s stores column %s as %s, which cannot be read as %s",
                fileName,
                path,
                stored != null ? stored.catalogString() : "Arrow " + file.getType(),
                read.catalogString()));
    }

    /**
     * Returns the columns {@code filters} reference that can be pushed down on for this file: those it carries under
     * the same name, at every level. A column stored in another type than filter literals are built in keeps only its
     * null checks, which hold for any type, as {@code NullType}, for which no literal is built.
     */
    static StructType filterable(StructType schema, List<Field> fileFields, Filter[] filters) {
        List<String[]> paths = new ArrayList<>();
        for (Filter filter : filters) {
            paths.addAll(Arrays.asList(filter.v2references()));
        }
        return filterable(schema, fileFields, paths, 0);
    }

    private static StructType filterable(StructType schema, List<Field> fileFields, List<String[]> paths, int depth) {
        Map<String, List<String[]>> referenced = new LinkedHashMap<>();
        for (String[] path : paths) {
            if (path.length > depth) {
                referenced
                        .computeIfAbsent(path[depth], name -> new ArrayList<>())
                        .add(path);
            }
        }
        // Filters name columns as the table does, and the native scan reads them under those names.
        Map<String, Field> fileByName = byName(fileFields);
        List<StructField> present = new ArrayList<>(referenced.size());
        for (Map.Entry<String, List<String[]>> entry : referenced.entrySet()) {
            Field fileField = fileByName.get(entry.getKey());
            if (fileField == null || schema.getFieldIndex(entry.getKey()).isEmpty()) {
                continue;
            }
            StructField field = schema.apply(entry.getKey());
            DataType type;
            // An extension type stored as a struct, such as Variant, has no native fields to filter on.
            if (field.dataType() instanceof StructType struct && !isExtension(fileField)) {
                if (!(fileField.getType() instanceof ArrowType.Struct)) {
                    continue;
                }
                type = filterable(struct, fileField.getChildren(), entry.getValue(), depth + 1);
            } else {
                type = literalType(fileField, field.dataType()) ? field.dataType() : DataTypes.NullType;
            }
            present.add(new StructField(field.name(), type, field.nullable(), field.metadata()));
        }
        return new StructType(present.toArray(new StructField[0]));
    }

    /** Whether the file stores {@code read} values as the native scan compares filter literals of that type. */
    private static boolean literalType(Field file, DataType read) {
        // The scan won't compare an extension type, such as WKB geometry, with literals of the type it exports as.
        if (isExtension(file) || !read.equals(leafType(file))) {
            return false;
        }
        // Timestamp literals are microseconds in UTC, or without a time zone, and the scan does not convert units.
        return !(file.getType() instanceof ArrowType.Timestamp timestamp)
                || (timestamp.getUnit() == TimeUnit.MICROSECOND
                        && (timestamp.getTimezone() == null
                                || timestamp.getTimezone().equals("UTC")));
    }

    private static boolean isExtension(Field file) {
        return file.getMetadata().containsKey(ArrowType.ExtensionType.EXTENSION_METADATA_KEY_NAME);
    }

    private static Map<String, Field> byName(List<Field> fields) {
        Map<String, Field> byName = new HashMap<>();
        for (Field field : fields) {
            byName.putIfAbsent(field.getName(), field);
        }
        return byName;
    }
}
