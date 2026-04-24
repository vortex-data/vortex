// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import dev.vortex.relocated.org.apache.arrow.vector.types.DateUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.FloatingPointPrecision;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import java.util.stream.Collectors;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;

/**
 * Utility class for converting Arrow types to Spark SQL data types.
 * <p>
 * This class provides static methods to convert Arrow field definitions and type definitions
 * into their corresponding Spark SQL DataType representations. It handles the mapping between
 * Arrow's type system and Spark's type system, including complex types like structs and arrays.
 */
public final class ArrowUtils {
    private ArrowUtils() {}

    /**
     * Converts an Arrow Field to a Spark SQL DataType.
     * <p>
     * This method handles complex types like structs and arrays by recursively converting
     * their child fields. For primitive types, it delegates to {@link #fromArrowType(ArrowType)}.
     *
     * @param field the Arrow field to convert
     * @return the corresponding Spark SQL DataType
     * @throws UnsupportedOperationException if the Arrow type is not supported
     */
    public static DataType fromArrowField(Field field) {
        switch (field.getType().getTypeID()) {
            case Struct:
                return DataTypes.createStructType(field.getChildren().stream()
                        .map(child -> {
                            DataType dt = fromArrowField(child);
                            return new StructField(child.getName(), dt, child.isNullable(), Metadata.empty());
                        })
                        .collect(Collectors.toList()));
            case List: {
                Field elementField = field.getChildren().get(0);
                DataType elementType = fromArrowField(elementField);
                return DataTypes.createArrayType(elementType, elementField.isNullable());
            }
            default:
                return fromArrowType(field.getType());
        }
    }

    /**
     * Converts an Arrow type to a Spark SQL DataType.
     * <p>
     * This method maps primitive Arrow types to their corresponding Spark SQL types.
     * It supports most common Arrow types including integers, floating point numbers,
     * strings, binary data, dates, timestamps, decimals, and nulls.
     *
     * @param dt the Arrow type to convert
     * @return the corresponding Spark SQL DataType
     * @throws UnsupportedOperationException if the Arrow type configuration is not supported
     * @throws RuntimeException if the Arrow type is not recognized
     */
    public static DataType fromArrowType(ArrowType dt) {
        switch (dt.getTypeID()) {
            case Bool:
                return DataTypes.BooleanType;
            case Int: {
                ArrowType.Int intType = (ArrowType.Int) dt;
                if (intType.getIsSigned() && intType.getBitWidth() == 8) {
                    return DataTypes.ByteType;
                } else if (intType.getIsSigned() && intType.getBitWidth() == 16) {
                    return DataTypes.ShortType;
                } else if (intType.getIsSigned() && intType.getBitWidth() == 32) {
                    return DataTypes.IntegerType;
                } else if (intType.getIsSigned() && intType.getBitWidth() == 64) {
                    return DataTypes.LongType;
                } else {
                    throw new UnsupportedOperationException("Unsupported Arrow type: " + dt);
                }
            }
            case FloatingPoint: {
                ArrowType.FloatingPoint floatType = (ArrowType.FloatingPoint) dt;
                if (floatType.getPrecision() == FloatingPointPrecision.SINGLE) {
                    return DataTypes.FloatType;
                } else if (floatType.getPrecision() == FloatingPointPrecision.DOUBLE) {
                    return DataTypes.DoubleType;
                } else {
                    throw new UnsupportedOperationException("Unsupported Arrow type: " + dt);
                }
            }
            case Decimal: {
                ArrowType.Decimal decimalType = (ArrowType.Decimal) dt;
                return DataTypes.createDecimalType(decimalType.getPrecision(), decimalType.getScale());
            }
            case Utf8:
            case LargeUtf8:
                return DataTypes.StringType;
            case Binary:
            case LargeBinary:
                return DataTypes.BinaryType;
            case Date: {
                ArrowType.Date dateType = (ArrowType.Date) dt;
                if (dateType.getUnit() == DateUnit.DAY) {
                    return DataTypes.DateType;
                } else {
                    throw new UnsupportedOperationException("Unsupported Arrow type: " + dt);
                }
            }
            case Timestamp: {
                ArrowType.Timestamp ts = (ArrowType.Timestamp) dt;
                if (ts.getUnit() == TimeUnit.MICROSECOND) {
                    return ts.getTimezone() != null ? DataTypes.TimestampType : DataTypes.TimestampNTZType;
                } else {
                    throw new UnsupportedOperationException("Unsupported Arrow type: " + dt);
                }
            }
            case Null:
                return DataTypes.NullType;
            default:
                throw new IllegalArgumentException("Unsupported Arrow type: " + dt);
        }
    }
}
