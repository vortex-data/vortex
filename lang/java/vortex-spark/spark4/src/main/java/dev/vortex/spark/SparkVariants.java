// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import dev.vortex.relocated.org.apache.arrow.vector.VarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.expressions.SpecializedGetters;
import org.apache.spark.sql.catalyst.expressions.variant.ArrayExtraction;
import org.apache.spark.sql.catalyst.expressions.variant.ObjectExtraction;
import org.apache.spark.sql.catalyst.expressions.variant.VariantCastArgs;
import org.apache.spark.sql.catalyst.expressions.variant.VariantGet$;
import org.apache.spark.sql.catalyst.expressions.variant.VariantPathSegment;
import org.apache.spark.sql.execution.RowToColumnConverter;
import org.apache.spark.sql.execution.datasources.VariantMetadata$;
import org.apache.spark.sql.execution.datasources.VariantMetadata;
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector;
import org.apache.spark.sql.execution.vectorized.WritableColumnVector;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.VariantType$;
import org.apache.spark.sql.types.VariantType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.unsafe.types.VariantVal;
import scala.Option;

/**
 * Spark's {@code VARIANT} type, which Spark 4 introduced, as Vortex stores it: Arrow's {@code arrow.parquet.variant}
 * extension, a struct of the binary {@code metadata} and {@code value} of each variant.
 *
 * <p>Spark 3.5 has no {@code VARIANT}, and its build carries a class of this name that supports none.
 */
public final class SparkVariants {
    /** The Arrow extension name Vortex reads as its variant type. */
    public static final String EXTENSION = "arrow.parquet.variant";

    private static final String EXTENSION_KEY = "ARROW:extension:name";

    private SparkVariants() {}

    /** Whether {@code dataType} is Spark's variant type. */
    public static boolean isVariant(DataType dataType) {
        return dataType instanceof VariantType;
    }

    /** Spark's variant type, when this Spark has one. */
    public static Optional<DataType> variantType() {
        return Optional.of(VariantType$.MODULE$);
    }

    /** Whether {@code field} is Arrow's variant extension. */
    public static boolean isVariant(Field field) {
        return EXTENSION.equals(field.getMetadata().get(EXTENSION_KEY));
    }

    /** The Arrow field holding a variant column named {@code name}. */
    public static Field arrowField(String name, boolean nullable) {
        // A null variant keeps empty parts, so the parts themselves are never null.
        List<Field> children = List.of(
                new Field("metadata", FieldType.notNullable(new ArrowType.Binary()), null),
                new Field("value", FieldType.notNullable(new ArrowType.Binary()), null));
        return new Field(
                name, new FieldType(nullable, new ArrowType.Struct(), null, Map.of(EXTENSION_KEY, EXTENSION)), children);
    }

    /** Writes the variant at {@code ordinal} of {@code row} into row {@code index} of {@code vector}. */
    public static void write(StructVector vector, SpecializedGetters row, int ordinal, int index) {
        VariantVal variant = row.getVariant(ordinal);
        ((VarBinaryVector) vector.getChild("metadata")).setSafe(index, variant.getMetadata());
        ((VarBinaryVector) vector.getChild("value")).setSafe(index, variant.getValue());
        vector.setIndexDefined(index);
    }

    /** Marks row {@code index} of {@code vector} null, with empty, non-null parts. */
    public static void writeNull(StructVector vector, int index) {
        ((VarBinaryVector) vector.getChild("metadata")).setSafe(index, new byte[0]);
        ((VarBinaryVector) vector.getChild("value")).setSafe(index, new byte[0]);
        vector.setNull(index);
    }

    /**
     * The fields Spark asks the scan to extract from a variant column, when {@code requested} is the struct Spark
     * substitutes for one: a field per extraction, holding its path, target type, and cast rules in its metadata.
     */
    public static Optional<List<VariantField>> extractedFields(DataType requested) {
        if (!(requested instanceof StructType struct) || !VariantMetadata$.MODULE$.isVariantStruct(struct)) {
            return Optional.empty();
        }
        List<VariantField> fields = new ArrayList<>(struct.fields().length);
        for (StructField field : struct.fields()) {
            VariantMetadata metadata = VariantMetadata$.MODULE$.fromMetadata(field.metadata());
            VariantPathSegment[] segments = metadata.parsedPath();
            String[] names = new String[segments.length];
            long[] indexes = new long[segments.length];
            for (int i = 0; i < segments.length; i++) {
                if (segments[i] instanceof ObjectExtraction object) {
                    names[i] = object.key();
                } else if (segments[i] instanceof ArrayExtraction array && array.index() >= 0) {
                    indexes[i] = array.index();
                } else {
                    return Optional.empty();
                }
            }
            ZoneId zone = ZoneId.of(metadata.timeZoneId());
            fields.add(new VariantField(
                    names,
                    indexes,
                    field.dataType(),
                    new VariantCastArgs(metadata.failOnError(), Option.apply(metadata.timeZoneId()), zone)));
        }
        return Optional.of(fields);
    }

    /**
     * The column of {@code type}, the struct Spark substitutes for a variant column, built from the variants found at
     * each field's path, cast as Spark's {@code variant_get} casts them. A row whose variant is null is null.
     *
     * @param extracted for each field, the variants found at its path, null where the path does not resolve
     * @param present whether each row's variant is not null
     */
    public static ColumnVector extract(
            StructType type, List<VariantField> fields, ColumnVector[] extracted, ColumnVector present, int rowCount) {
        OnHeapColumnVector column = new OnHeapColumnVector(rowCount, type);
        RowToColumnConverter converter = new RowToColumnConverter(new StructType().add("variant", type));
        WritableColumnVector[] target = {column};
        Object[] values = new Object[fields.size()];
        for (int row = 0; row < rowCount; row++) {
            if (present.isNullAt(row) || !present.getBoolean(row)) {
                converter.convert(new GenericInternalRow(new Object[] {null}), target);
                continue;
            }
            for (int i = 0; i < values.length; i++) {
                values[i] = extracted[i].isNullAt(row)
                        ? null
                        : VariantGet$.MODULE$.cast(
                                extracted[i].getVariant(row),
                                fields.get(i).dataType(),
                                fields.get(i).castArgs());
            }
            converter.convert(
                    new GenericInternalRow(new Object[] {new GenericInternalRow(values.clone())}), target);
        }
        return column;
    }

    /**
     * One field Spark asks the scan to extract from a variant: the path, where {@code names[i]} is an object field or
     * null for the array element {@code indexes[i]}, and the type and rules of the cast.
     */
    public static final class VariantField {
        private final String[] names;
        private final long[] indexes;
        private final DataType dataType;
        private final VariantCastArgs castArgs;

        VariantField(String[] names, long[] indexes, DataType dataType, VariantCastArgs castArgs) {
            this.names = names;
            this.indexes = indexes;
            this.dataType = dataType;
            this.castArgs = castArgs;
        }

        public String[] names() {
            return names.clone();
        }

        public long[] indexes() {
            return indexes.clone();
        }

        public DataType dataType() {
            return dataType;
        }

        VariantCastArgs castArgs() {
            return castArgs;
        }
    }
}
