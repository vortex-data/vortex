// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import dev.vortex.api.DType;
import java.util.Optional;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.types.*;

/**
 * Helpers for converting between Spark and Vortex type systems.
 */
public final class SparkTypes {
    private SparkTypes() {}

    public static DType toDType(StructType schema) {

        String[] fieldNames = new String[schema.length()];
        DType[] fieldTypes = new DType[schema.length()];

        for (int i = 0; i < schema.length(); i++) {
            StructField field = schema.fields()[i];
            fieldNames[i] = field.name();
            fieldTypes[i] = convertField(field.dataType(), field.nullable());
        }

        return DType.newStruct(fieldNames, fieldTypes, false);
    }

    // Convert field type to Vortex type.
    static DType convertField(DataType dataType, boolean isNullable) {
        if (dataType instanceof ByteType) {
            return DType.newByte(isNullable);
        } else if (dataType instanceof ShortType) {
            return DType.newShort(isNullable);
        } else if (dataType instanceof IntegerType) {
            return DType.newInt(isNullable);
        } else if (dataType instanceof LongType) {
            return DType.newLong(isNullable);
        } else if (dataType instanceof FloatType) {
            return DType.newFloat(isNullable);
        } else if (dataType instanceof DoubleType) {
            return DType.newDouble(isNullable);
        } else if (dataType instanceof DecimalType) {
            DecimalType decimalType = (DecimalType) dataType;
            return DType.newDecimal(decimalType.precision(), decimalType.scale(), isNullable);
        } else if (dataType instanceof BooleanType) {
            return DType.newBinary(isNullable);
        } else if (dataType instanceof StringType) {
            return DType.newUtf8(isNullable);
        } else if (dataType instanceof BinaryType) {
            return DType.newBinary(isNullable);
        } else if (dataType instanceof ArrayType) {
            ArrayType arrayType = ((ArrayType) dataType);
            DType elementType = convertField(arrayType.elementType(), arrayType.containsNull());
            return DType.newList(elementType, isNullable);
        } else if (dataType instanceof StructType) {
            StructType structType = (StructType) dataType;
            return toDType(structType);
        } else if (dataType instanceof TimestampType) {
            // Spark emits timestamps with UTC timezone and microsecond precision.
            return DType.newTimestamp(DType.TimeUnit.MICROSECONDS, Optional.of("UTC"), isNullable);
        } else if (dataType instanceof TimestampNTZType) {
            // TimestampNTZ is microsecond timestamp without zone
            return DType.newTimestamp(DType.TimeUnit.MICROSECONDS, Optional.empty(), isNullable);
        } else if (dataType instanceof DateType) {
            // TODO(aduffy): any problems with the date values since they're refed to
            //  gregorian proleptic?
            return DType.newDate(DType.TimeUnit.DAYS, isNullable);
        } else {
            throw new IllegalArgumentException("Unsupported data type for Vortex: " + dataType);
        }
    }

    /**
     * Convert a STRUCT Vortex type to a Spark {@link DataType}.
     */
    public static DataType toDataType(DType dType) {
        switch (dType.getVariant()) {
            case NULL:
                return DataTypes.NullType;
            case BOOL:
                return DataTypes.BooleanType;
            case PRIMITIVE_U8:
            case PRIMITIVE_I8:
                return DataTypes.ByteType;
            case PRIMITIVE_U16:
            case PRIMITIVE_I16:
                return DataTypes.ShortType;
            case PRIMITIVE_U32:
            case PRIMITIVE_I32:
                return DataTypes.IntegerType;
            case PRIMITIVE_U64:
            case PRIMITIVE_I64:
                return DataTypes.LongType;
            case PRIMITIVE_F16:
                throw new IllegalArgumentException("Spark does not support f16");
            case PRIMITIVE_F32:
                return DataTypes.FloatType;
            case PRIMITIVE_F64:
                return DataTypes.DoubleType;
            case UTF8:
                return DataTypes.StringType;
            case BINARY:
                return DataTypes.BinaryType;
            case STRUCT:
                // For each of the inner struct fields, we capture them together here.
                var fieldNames = dType.getFieldNames();
                var fieldTypes = dType.getFieldTypes();

                // NOTE: it's very important we do this with a for loop. Using the streams API can easily
                //  lead to StackOverflowError being thrown.
                var fields = new StructField[fieldNames.size()];
                for (int i = 0; i < fieldNames.size(); i++) {
                    var name = fieldNames.get(i);
                    try (var type = fieldTypes.get(i)) {
                        fields[i] = new StructField(name, toDataType(type), dType.isNullable(), Metadata.empty());
                    }
                }

                return DataTypes.createStructType(fields);
            case LIST:
            case FIXED_SIZE_LIST:
                return DataTypes.createArrayType(toDataType(dType.getElementType()), dType.isNullable());
            case EXTENSION:
                /*
                 * Spark does not have a direct equivalent for many of the temporal types we support in Vortex or Arrow.
                 * Notably, there is no DATE type, and timestamps can have at most µs-level precision.
                 * This means that we need to "cheat" a little in how we convert into Spark's type system. We support
                 * the following conversions:
                 *  1. Vortex DATE -> Spark TIMESTAMP (with 00:00:00 time and local timezone)
                 *  2. Vortex TIMESTAMP -> Spark TIMESTAMP, with precision truncated to µs
                 *  3. Vortex TIME -> not supported
                 */
                if (dType.isTime()) {
                    throw new IllegalArgumentException("Spark does not support Vortex TIME data type");
                }

                if (dType.isDate()) {
                    return DateType$.MODULE$;
                }

                if (dType.isTimestamp()) {
                    return TimestampType$.MODULE$;
                }

                // TODO(aduffy): other extension types
                throw new IllegalArgumentException("Unsupported non-temporal extension type");
            case DECIMAL:
                return DataTypes.createDecimalType(dType.getPrecision(), dType.getScale());
            default:
                throw new IllegalArgumentException("unreachable");
        }
    }

    /**
     * Convert a STRUCT Vortex type to a Spark {@link Column}.
     */
    public static Column[] toColumns(DType dType) {
        var fieldNames = dType.getFieldNames();
        var fieldTypes = dType.getFieldTypes();
        var columns = new Column[fieldNames.size()];

        for (int i = 0; i < columns.length; i++) {
            var name = fieldNames.get(i);
            try (var type = fieldTypes.get(i)) {
                columns[i] = Column.create(name, toDataType(type), type.isNullable());
            }
        }

        return columns;
    }
}
